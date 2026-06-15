# Docker Deployment

## Production stack

Certainaity uses Docker Compose with five services: `nginx`, `api`, `worker`, `redis`, and `frontend`.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│  nginx       │───▶│  api (×2)    │───▶│  worker (Celery GPU) │
│  :443 / :80  │    │  FastAPI     │    │  PyTorch inference   │
└──────────────┘    └──────┬───────┘    └──────────────────────┘
       │                   │
┌──────▼──────┐    ┌───────▼──────┐
│  frontend   │    │   redis      │
│  React UI   │    │  queue+state │
│  :3000      │    └──────────────┘
└─────────────┘
```

## Prerequisites

| Requirement | Minimum |
|-------------|---------|
| Docker | 24.x |
| Docker Compose | v2.20+ |
| NVIDIA driver | 525+ (for GPU worker) |
| NVIDIA Container Toolkit | 1.14+ |
| Disk | 20 GB (weights + outputs) |

## Vercel + Railway (recommended split hosting)

Use Vercel for the React frontend and Railway for the FastAPI backend.

1. Deploy API to Railway from repo root.
1. Railway will use `railway.toml` and `nixpacks.toml` to build with Python 3.11 and install `.[api]` extras.
1. Configure Railway environment variables (minimum):

```bash
CERTAINAITY_REDIS_URL=redis://<your-redis-host>:6379/0
CERTAINAITY_OUTPUT_DIR=/tmp/certainaity-output
CERTAINAITY_JWT_PUBLIC_KEY="-----BEGIN PUBLIC KEY-----\\n...\\n-----END PUBLIC KEY-----"
```

1. Copy your Railway public API URL, for example:

```bash
https://your-api.up.railway.app
```

1. Deploy frontend to Vercel from repo root (the root `vercel.json` builds `frontend/`).
1. In Vercel project environment variables, set:

```bash
VITE_API_BASE_URL=https://your-api.up.railway.app
```

1. Redeploy Vercel after saving the variable so the value is embedded at build time.

## Step-by-step

### 1. Clone and configure

```bash
git clone https://github.com/david-gary/certainaity.git
cd certainaity
cp .env.example .env
```

Edit `.env` — the required fields are:

```bash
CERTAINAITY_REDIS_URL=redis://redis:6379/0
CERTAINAITY_JWT_PUBLIC_KEY_PATH=secrets/jwt_public.pem
CERTAINAITY_OUTPUT_DIR=/output
```

### 2. Generate TLS certificates

```bash
# Self-signed (development only):
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
  -keyout certs/server.key -out certs/server.crt
```

For production, use Let's Encrypt with certbot or mount certificates from your provider.

### 3. Place model weights

```bash
python scripts/download_weights.py --all
# or manually:
ls weights/
# patchforensic_v1.pt  mantranet_v1.pt  spsl_v1.pt  inpainting_v1.pt
# checksums.sha256
```

Verify checksums:

```bash
sha256sum -c weights/checksums.sha256
```

### 4. Build and start

```bash
docker compose build
docker compose up -d
```

### 5. Health check

```bash
curl http://localhost:8000/v1/health
# {"status": "ok"}
```

## CPU-only development

Use the included override file:

```bash
docker compose up -d   # auto-merges docker-compose.override.yml
```

The override:

- Replaces the NVIDIA runtime with `runc`
- Sets `CERTAINAITY_USE_CPU=true`
- Exposes ports 8000 (API) and 6379 (Redis) directly
- Skips nginx (talk to the API directly on port 8000)

## Monitoring

Add Prometheus and Grafana to the stack:

```bash
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

Then open Grafana at `http://localhost:3001` (default user: `admin` / `admin`).

The Certainaity dashboard is auto-provisioned from
`monitoring/grafana/dashboards/certainaity.json`.

## Upgrading

```bash
git pull
docker compose build --no-cache
docker compose up -d --remove-orphans
```

## Rollback

Images are tagged by version on GHCR. To roll back to v1.0.0:

```bash
CERTAINAITY_VERSION=1.0.0 docker compose up -d
```

(Requires `image:` overrides or environment variable expansion in your compose file.)
