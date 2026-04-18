# Architecture Overview

## System diagram

```
                          ┌──────────────┐
Browser / CLI             │   Client     │
                          └──────┬───────┘
                                 │ HTTPS
                          ┌──────▼───────┐
                          │   nginx      │  TLS termination, static assets,
                          │  :443        │  reverse proxy to API
                          └──────┬───────┘
                                 │
               ┌─────────────────▼─────────────────┐
               │          FastAPI (2 replicas)      │
               │          forenscope.api            │
               │                                   │
               │  POST /v1/analyze  ──────────────▶ enqueue(Celery)
               │  GET  /v1/jobs/:id ◀── poll state ─ Redis
               │  GET  /v1/reports/:id/json,pdf     │
               │  GET  /metrics  (Prometheus)        │
               └──────────────────┬────────────────┘
                                  │ Redis pub/sub
               ┌──────────────────▼────────────────┐
               │         Celery worker              │
               │         forenscope.worker          │
               │                                   │
               │  1. ingest_image()                 │
               │  2. _extract_features()  [4 threads]
               │  3. ensemble.localize()            │
               │  4. run_resilience_test()          │
               │  5. save_json_report()             │
               │     save_pdf_report()              │
               └───────────────────────────────────┘
```

## Service responsibilities

| Service | Language | Role |
|---------|----------|------|
| `nginx` | — | TLS, static file serving, reverse proxy |
| `api` | Python / FastAPI | HTTP interface, job submission, report serving |
| `worker` | Python / Celery | Forensic pipeline execution, GPU inference |
| `redis` | — | Celery broker, task result backend, rate-limit counters |
| `frontend` | React / TypeScript | Web UI (upload, polling, heatmap, reports) |

## Observability

The API emits four Prometheus metrics:

| Metric | Type | Labels |
|--------|------|--------|
| `forenscope_http_requests_total` | Counter | `method`, `endpoint`, `status_code` |
| `forenscope_http_request_duration_seconds` | Histogram | `endpoint` |
| `forenscope_jobs_submitted_total` | Counter | — |
| `forenscope_jobs_rejected_total` | Counter | `reason` |

`/v1/jobs/{job_id}` paths are normalised to `/v1/jobs/{job_id}` before labeling to prevent cardinality explosion.

The Grafana dashboard (`monitoring/grafana/dashboards/forenscope.json`) displays throughput, latency percentiles, and job queue health.

## Security boundaries

- All API routes except `/v1/health` and `/metrics` require a valid RS256 JWT bearer token.
- Uploaded files are validated (format, size, dimensions) before queuing.
- Worker runs as non-root UID 10001.
- Model weights are mounted read-only; never baked into images.
- JWT private key is never in the repository; the public key is provided via Docker secrets.
