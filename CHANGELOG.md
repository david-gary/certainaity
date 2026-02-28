# Changelog

All notable changes to ForenScope are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-02-28

### Added

**CLI**
- `forenscope analyze <image>` entrypoint: ingests an image, runs the full pipeline, and prints a JSON summary to stdout.
- `--output-json` and `--output-pdf` flags for writing report files.
- `--job-id` override to control the report's `job_id` field.

**API (FastAPI + Celery)**
- `POST /v1/analyze` — multipart image upload, enqueues a Celery task, returns a `poll_url`.
- `GET /v1/jobs/{job_id}` — returns task state and stage metadata.
- `GET /v1/reports/{job_id}/json` — serves the completed JSON report.
- `GET /v1/reports/{job_id}/pdf` — serves the completed PDF report.
- `GET /v1/health` — liveness probe (unauthenticated).
- JWT RS256 bearer authentication on protected routes.
- Rate limiting via slowapi: configurable per-IP per-minute limit on `POST /v1/analyze`.

**Observability**
- `GET /metrics` — Prometheus scrape endpoint (unauthenticated, excluded from OpenAPI schema).
- `forenscope_http_requests_total` counter with `method`, `endpoint`, `status_code` labels.
- `forenscope_http_request_duration_seconds` histogram (11 buckets, 5 ms–10 s).
- `forenscope_jobs_submitted_total` and `forenscope_jobs_rejected_total` counters.
- Path normalisation collapses `/v1/jobs/<uuid>` to `/v1/jobs/{job_id}` to prevent cardinality explosion.
- Grafana dashboard (`monitoring/grafana/dashboards/forenscope.json`) with throughput, latency p50/p95/p99, and job queue panels.

**Forensic Pipeline**
- Feature extraction: ELA (error level analysis), wavelet-based noise variance, CFA interpolation correlation, DCT block similarity — all run in parallel via `ThreadPoolExecutor`.
- Ensemble inference: weighted fusion of PatchForensic, MantraNet, SPSL, and InpaintingDetector; falls back to handcrafted feature mean when model weights are absent.
- Resilience test: re-compresses the image at configurable JPEG qualities; flags `anti_forensic_warning` if ensemble confidence drops > 25%.
- Chain-of-custody: SHA-256 computed from raw bytes before any decoding; preserved through task execution into report.

**Reports**
- JSON report with all chain-of-custody fields (`job_id`, `sha256`, `analysis_timestamp`, `manipulation_detected`, `overall_confidence`, `regions`, `models_used`).
- PDF report (ReportLab): header, metadata table, verdict, optional heatmap, detected-region table, chain-of-custody statement.

**Infrastructure**
- Multi-stage production Dockerfiles for api and worker (wheel-only runtime layer, non-root UID 10001, no build tools in final image).
- `docker-compose.override.yml` for CPU-only local development (single replicas, ports 8000/6379 exposed, nginx skipped).
- `.github/workflows/ci.yml`: lint (ruff), type-check (mypy), unit tests (≥ 85% coverage), integration tests, model architecture smoke tests, Trivy container security scan.
- `.github/workflows/publish.yml`: builds and pushes `forenscope-api` and `forenscope-worker` to GHCR on `v*` tags; creates GitHub Release from CHANGELOG.

### Changed

- `docker/Dockerfile.api` and `docker/Dockerfile.worker` converted to multi-stage builds.
- Worker image no longer bakes in model weights; weights are mounted as a read-only volume.

---

## [Unreleased]

### Added

- React + Tailwind web UI (in development — see v1.1.0).
- GAN-generated image detection (StyleGAN / DALL-E / Midjourney) — in development.
- Kubernetes deployment manifests — in development.
- MkDocs documentation site — in development.
