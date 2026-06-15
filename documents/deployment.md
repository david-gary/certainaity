# Certainaity — Deployment Plan

## Overview

Certainaity v1.0 targets a self-hosted deployment on a single GPU server for initial use. The system is containerized with Docker and orchestrated with Docker Compose. A production-grade Kubernetes path is described for v1.1 and beyond.

---

## Hardware Requirements

### Minimum (development / low-volume)

| Component | Spec |
|-----------|------|
| CPU | 8-core x86-64 |
| RAM | 32 GB |
| GPU | NVIDIA RTX 3090 (24 GB VRAM) |
| Storage | 500 GB NVMe SSD |
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.2+ |

### Recommended (production, up to ~200 images/hour)

| Component | Spec |
|-----------|------|
| CPU | 32-core (e.g., AMD EPYC 7543) |
| RAM | 128 GB ECC |
| GPU | 2× NVIDIA A100 40 GB (or 1× A100 80 GB) |
| Storage | 2 TB NVMe RAID-1 |
| Network | 10 GbE |
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.2+ |

---

## Container Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Docker Compose                             │
│                                                                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐    │
│  │   nginx     │──▶│  api (x2)   │──▶│   worker (Celery)   │    │
│  │  (reverse   │   │  FastAPI    │   │   (GPU-bound tasks) │    │
│  │   proxy)    │   │  CPU only   │   └─────────┬───────────┘    │
│  └─────────────┘   └──────┬──────┘             │                 │
│                           │                    │                 │
│                    ┌──────▼──────┐   ┌─────────▼───────────┐    │
│                    │    Redis    │   │   model-server       │    │
│                    │  (queue +   │   │ (Triton / plain      │    │
│                    │  rate limit)│   │  torch serve)        │    │
│                    └─────────────┘   └─────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                      Volumes                             │     │
│  │   weights/  output/  logs/  mlflow-data/                │     │
│  └─────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Docker Images

### `certainaity-api`

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[api]"

COPY src/ src/
COPY weights/ weights/

USER nobody
EXPOSE 8000
CMD ["uvicorn", "certainaity.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--loop", "uvloop"]
```

The API container does NOT load ML models. It validates input, dispatches to Redis queue, and serves artifacts from the output volume. This keeps the API container lightweight (< 1 GB image) and independently scalable.

### `certainaity-worker`

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[worker]"

COPY src/ src/
COPY weights/ weights/

ENV CUDA_VISIBLE_DEVICES=0
CMD ["celery", "-A", "certainaity.worker.app", "worker", \
     "--concurrency=1", "--loglevel=info", "-Q", "analysis"]
```

`--concurrency=1` because GPU memory cannot be shared across concurrent model runs. Each worker processes one image at a time.

### `certainaity-model-server` (v1.1, optional)

Triton Inference Server with ONNX-exported models. Enables dynamic batching. Not required for v1.0.

---

## Docker Compose

```yaml
# docker-compose.yml
version: "3.9"

services:
  nginx:
    image: nginx:1.25-alpine
    ports: ["443:443", "80:80"]
    volumes:
      - ./nginx/certainaity.conf:/etc/nginx/conf.d/default.conf:ro
      - ./certs:/etc/ssl/certs:ro
      - output:/srv/output:ro
    depends_on: [api]

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    deploy:
      replicas: 2
    environment:
      REDIS_URL: redis://redis:6379/0
      OUTPUT_DIR: /output
      JWT_PUBLIC_KEY_PATH: /run/secrets/jwt_public_key
    volumes:
      - output:/output
    secrets: [jwt_public_key]
    depends_on: [redis]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    deploy:
      replicas: 1       # one per GPU; increase for multi-GPU
    environment:
      REDIS_URL: redis://redis:6379/0
      OUTPUT_DIR: /output
      CUDA_VISIBLE_DEVICES: "0"
    volumes:
      - output:/output
      - ./weights:/app/weights:ro
    runtime: nvidia
    depends_on: [redis]

  redis:
    image: redis:7.2-alpine
    command: redis-server --save 60 1 --loglevel warning
    volumes:
      - redis-data:/data

volumes:
  output:
  redis-data:

secrets:
  jwt_public_key:
    file: ./secrets/jwt_public.pem
```

---

## nginx Configuration

```nginx
# nginx/certainaity.conf
server {
    listen 443 ssl http2;
    server_name api.certainaity.com;

    ssl_certificate     /etc/ssl/certs/certainaity.crt;
    ssl_certificate_key /etc/ssl/certs/certainaity.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    client_max_body_size 55M;    # 50 MB limit + overhead

    location /v1/ {
        proxy_pass         http://api:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 30s;
    }

    # Serve PDF/JSON artifacts directly from volume
    location /v1/reports/ {
        alias /srv/output/;
        add_header Content-Disposition 'attachment';
        add_header X-Content-Type-Options nosniff;
    }
}

server {
    listen 80;
    server_name api.certainaity.com;
    return 301 https://$host$request_uri;
}
```

---

## Secrets Management

For v1.0 (self-hosted), secrets are managed as Docker secrets:

```
secrets/
  jwt_public.pem       ← RS256 public key for verifying JWTs
  jwt_private.pem      ← RS256 private key (admin CLI only; NOT in worker/api containers)
```

The private key never enters the API or worker containers. It is held only on the admin machine that issues tokens. For v1.1, migrate to HashiCorp Vault or AWS Secrets Manager.

---

## Deployment Procedure

### First-time Setup

```bash
# 1. Clone repo and configure
git clone https://github.com/david-gary/certainaity.git
cd certainaity
cp .env.example .env    # edit: set REDIS_URL, OUTPUT_DIR, etc.

# 2. Download model weights (requires credential)
python scripts/download_weights.py --token $WEIGHTS_TOKEN

# 3. Generate secrets
openssl genrsa -out secrets/jwt_private.pem 2048
openssl rsa -in secrets/jwt_private.pem -pubout -out secrets/jwt_public.pem

# 4. Build and start
docker compose build
docker compose up -d

# 5. Verify
curl http://localhost/v1/health
```

### Updating to a New Version

```bash
git pull origin main
docker compose build
docker compose up -d --no-deps --build api worker
# Rolling restart: compose handles it; nginx keeps serving during restart
```

### Rollback

```bash
docker compose up -d --no-deps --build api worker --scale worker=0
docker compose up -d --no-deps --build api worker   # previous image still cached
# If previous image was cleaned: git checkout <previous-tag> && docker compose build
```

---

## Monitoring

### Logging

All containers log to stdout in JSON format (structured logging with `structlog`). Docker Compose forwards to the host's syslog. In production, pipe to Loki or a SIEM.

Log fields included on every API request:
```json
{
  "timestamp": "2025-02-01T12:00:00Z",
  "level": "info",
  "event": "analyze_complete",
  "request_id": "req_7a8f2b...",
  "sha256": "a1b2c3...",
  "org": "org_acme",
  "duration_ms": 412,
  "manipulation_detected": true,
  "gpu_used": true
}
```

Sensitive fields (file contents, full EXIF, API keys) are never logged.

### Metrics (Prometheus)

The FastAPI app exposes `/metrics` (scrape target for Prometheus) via `prometheus-fastapi-instrumentator`:

| Metric | Description |
|--------|-------------|
| `certainaity_requests_total` | Counter by status code and org |
| `certainaity_request_duration_seconds` | Histogram |
| `certainaity_queue_depth` | Gauge (Celery queue length) |
| `certainaity_gpu_memory_used_bytes` | Gauge (from nvidia-smi) |
| `certainaity_manipulation_detected_total` | Counter |

A Grafana dashboard (`monitoring/grafana/certainaity.json`) visualizes these metrics with alerting rules for:
- Queue depth > 20 for > 5 min → PagerDuty alert
- GPU memory > 95% → warning
- HTTP 5xx rate > 1% → PagerDuty alert

---

## Backup Strategy

| Data | Backup Method | Frequency | Retention |
|------|--------------|-----------|-----------|
| `output/` (reports) | rsync to cold storage | Nightly | 90 days |
| `weights/` | S3 versioned bucket | On upload (immutable) | Forever |
| `redis-data/` | Redis RDB snapshot + S3 | Hourly | 7 days |
| `secrets/` | Encrypted offline backup | On rotation | Last 3 versions |

---

## v1.1 Kubernetes Path

When request volume exceeds what a single machine can handle, migrate to Kubernetes:

1. Export model weights to ONNX; serve via Triton Inference Server.
2. Deploy API pods (CPU-only, stateless) with HPA scaled on `certainaity_queue_depth`.
3. Deploy worker pods (GPU-enabled) with KEDA scaled on Redis queue length.
4. Move output artifacts to S3 (or compatible: MinIO, GCS).
5. Replace Docker secrets with Kubernetes Secrets or Vault Agent Injector.
6. Add Istio service mesh for mTLS between components.
