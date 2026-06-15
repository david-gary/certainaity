# Getting Started

## Prerequisites

- Python 3.11+
- Docker + Docker Compose (for the API stack)
- An RS256 JWT key pair (for API authentication)

## Install the CLI

```bash
pip install certainaity            # from PyPI (v1.0+)
# or from source:
git clone https://github.com/david-gary/certainaity.git
cd certainaity && pip install -e ".[api]"
```

## Analyze an image locally

```bash
certainaity analyze suspicious.jpg
```

Example output:

```json
{
  "job_id": "suspicious",
  "file_name": "suspicious.jpg",
  "sha256": "a3f4...",
  "analysis_timestamp": "2026-03-01T10:23:45.123456+00:00",
  "manipulation_detected": false,
  "overall_confidence": 0.12,
  "anti_forensic_warning": false,
  "regions": 0,
  "execution_time_ms": 843
}
```

!!! note "Model weights"
    Without trained model weights in `weights/`, the pipeline falls back to
    handcrafted feature signals (ELA, noise, CFA, DCT). The `models_used` field
    in the report will list `handcrafted_*` rather than the deep model names.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--job-id ID` | filename stem | Override the `job_id` in the report |
| `--output-json PATH` | (none) | Write JSON report to file |
| `--output-pdf PATH` | (none) | Write PDF report to file |

## Start the API stack

### 1. Generate a JWT key pair

```bash
python scripts/generate_keys.py
# Creates secrets/jwt_private.pem and secrets/jwt_public.pem
```

### 2. Configure the environment

```bash
cp .env.example .env
# Edit .env: set CERTAINAITY_JWT_PUBLIC_KEY_PATH, CERTAINAITY_REDIS_URL, etc.
```

### 3. Start services

```bash
docker compose up -d
# API: http://localhost:8000
# Web UI: http://localhost:3000
# Metrics: http://localhost:8000/metrics
```

### 4. Submit an image

```bash
TOKEN=$(python scripts/generate_dev_token.py)

curl -X POST http://localhost:8000/v1/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@suspicious.jpg"
```

Response:

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "poll_url": "/v1/jobs/3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "message": "Analysis queued"
}
```

### 5. Poll for results

```bash
curl http://localhost:8000/v1/jobs/3fa85f64-... \
  -H "Authorization: Bearer $TOKEN"
```

When `state` is `SUCCESS`, fetch the report:

```bash
curl http://localhost:8000/v1/reports/3fa85f64-.../json \
  -H "Authorization: Bearer $TOKEN" | jq .
```

## Run the benchmark

```bash
python scripts/benchmark.py \
  --url http://localhost:8000 \
  --token "$TOKEN" \
  --count 50
```

Outputs min/median/p95/p99/max latency in milliseconds.
