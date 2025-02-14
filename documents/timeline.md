# ForenScope — Project Timeline & Milestones

## Summary

ForenScope targets a v1.0 production release in Q2 2025. The project is divided into four phases: Foundation, Model Development, Integration & Testing, and Release.

---

## Phase Overview

| Phase | Duration | Target Dates | Status |
|-------|----------|-------------|--------|
| Phase 0: Foundation | 3 weeks | Dec 9 – Dec 27, 2024 | In Progress |
| Phase 1: Data & Feature Layer | 4 weeks | Jan 6 – Jan 31, 2025 | Planned |
| Phase 2: Model Development | 6 weeks | Feb 3 – Mar 14, 2025 | Planned |
| Phase 3: Integration & API | 3 weeks | Mar 17 – Apr 4, 2025 | Planned |
| Phase 4: Testing & Hardening | 3 weeks | Apr 7 – Apr 25, 2025 | Planned |
| Phase 5: Release | 2 weeks | Apr 28 – May 9, 2025 | Planned |

---

## Phase 0: Foundation (Dec 9 – Dec 27, 2024)

**Goal**: Establish the project skeleton, tooling, and design documents so all later work has a clear scaffold.

### Milestones

| # | Deliverable | Owner | Due |
|---|------------|-------|-----|
| 0.1 | Repository initialized, branch strategy documented | david-gary | Dec 10 |
| 0.2 | Architecture document complete (this set of docs) | david-gary | Dec 20 |
| 0.3 | `pyproject.toml` with dependency groups (`api`, `worker`, `dev`, `train`) | david-gary | Dec 20 |
| 0.4 | Docker Compose skeleton (nginx, api, worker, redis stubs) | david-gary | Dec 23 |
| 0.5 | CI pipeline: lint (ruff), type-check (mypy), unit test skeleton | david-gary | Dec 27 |

### Exit Criteria

- [ ] All six planning documents reviewed and stable.
- [ ] `docker compose up` starts without errors (stubs returning 501).
- [ ] CI passes on `main`.

---

## Phase 1: Data & Feature Layer (Jan 6 – Jan 31, 2025)

**Goal**: Have a working data pipeline producing training-ready patches, and validated handcrafted feature extractors.

### Milestones

| # | Deliverable | Due |
|---|------------|-----|
| 1.1 | Dataset download scripts for CASIA v2, DEFACTO, NIST 16, COVERage | Jan 10 |
| 1.2 | Data validation + quality filtering script | Jan 13 |
| 1.3 | Patch extraction pipeline (`build_dataset.py`) | Jan 17 |
| 1.4 | `ForensicDataset` PyTorch class + albumentations augmentations | Jan 20 |
| 1.5 | ELA extractor implemented and unit-tested | Jan 22 |
| 1.6 | Noise variance extractor implemented and unit-tested | Jan 24 |
| 1.7 | CFA artifact map extractor implemented and unit-tested | Jan 27 |
| 1.8 | DCT block similarity matrix implemented and unit-tested | Jan 29 |
| 1.9 | Inpainting synthetic dataset generation script (Stable Diffusion) | Jan 31 |

### Exit Criteria

- [ ] `data/processed/` contains train/val/test splits with correct distribution.
- [ ] All 4 feature extractors pass unit tests (see `tests/unit/`).
- [ ] Synthetic inpainting set: 22,000 images generated.
- [ ] Dataset manifest (`metadata.jsonl`) is complete.

### Risk: Dataset Access

DEFACTO and NIST 16 require signed data use agreements. Applications were submitted Nov 28, 2024. Expected approval: early January. If delayed, Phase 1 can begin with CASIA v2 + COVERage while waiting.

---

## Phase 2: Model Development (Feb 3 – Mar 14, 2025)

**Goal**: Train all four models to target benchmark metrics and complete ensemble weight optimization.

### Milestones

| # | Deliverable | Due |
|---|------------|-----|
| 2.1 | PatchForensic architecture implemented + training loop | Feb 7 |
| 2.2 | PatchForensic training complete (90 epochs) | Feb 17 |
| 2.3 | PatchForensic eval: F1 ≥ 0.80 on CASIA v2 test | Feb 18 |
| 2.4 | MantraNet pretrained weights integrated + fine-tuning loop | Feb 21 |
| 2.5 | MantraNet fine-tuning complete (60 epochs) | Feb 28 |
| 2.6 | MantraNet eval: F1 ≥ 0.77 on DEFACTO test | Mar 1 |
| 2.7 | SPSL Siamese architecture + contrastive training loop | Mar 3 |
| 2.8 | SPSL training complete; FAISS inference wrapper | Mar 7 |
| 2.9 | SPSL eval: pair AUC ≥ 0.88 on COVERage test | Mar 8 |
| 2.10 | Inpainting Detector: CLIP fine-tuning loop | Mar 10 |
| 2.11 | Inpainting Detector training complete (50 epochs) | Mar 13 |
| 2.12 | Inpainting Detector eval: AUC ≥ 0.94 on synthetic test | Mar 13 |
| 2.13 | Ensemble weight optimization (L-BFGS-B on val) | Mar 14 |

### Exit Criteria

- [ ] All four models meet or exceed target benchmark metrics.
- [ ] Weights exported to `weights/` and checksums committed to `weights/checksums.sha256`.
- [ ] MLflow experiment `forenscope/ensemble/v1` logged with final weights `[w1, w2, w3, w4]`.

### GPU Allocation Plan

Training is sequential (GPU memory can't fit all four models simultaneously):

| Model | Training GPU | Wall-clock Time (est.) |
|-------|-------------|----------------------|
| PatchForensic | GPU 0+1 (DDP) | ~10 hours |
| MantraNet | GPU 0 | ~6 hours |
| SPSL | GPU 0 | ~4 hours |
| Inpainting Det. | GPU 0+1 (DDP) | ~8 hours |
| Inpainting set gen. | GPU 2 (parallel) | ~4.5 hours |

---

## Phase 3: Integration & API (Mar 17 – Apr 4, 2025)

**Goal**: Wire all components into the FastAPI application and have the full pipeline running end-to-end via the API.

### Milestones

| # | Deliverable | Due |
|---|------------|-----|
| 3.1 | Ingestion layer (`ingest.py`) implemented | Mar 19 |
| 3.2 | Feature extraction orchestration (parallel `ProcessPoolExecutor`) | Mar 21 |
| 3.3 | Inference layer: `ForensicModel` base class + all 4 wrappers | Mar 25 |
| 3.4 | Ensemble fusion + connected component labeling | Mar 26 |
| 3.5 | Manipulation type classification head | Mar 27 |
| 3.6 | Resilience test module | Mar 28 |
| 3.7 | JSON report generator | Mar 28 |
| 3.8 | PDF report generator (ReportLab) | Apr 1 |
| 3.9 | FastAPI routes: `/analyze`, `/analyze/{id}/status`, `/reports/...`, `/quota`, `/health` | Apr 2 |
| 3.10 | Auth middleware (JWT RS256 verify) | Apr 2 |
| 3.11 | Rate limiter (Redis sliding window) | Apr 3 |
| 3.12 | Celery worker integration | Apr 4 |

### Exit Criteria

- [ ] `POST /v1/analyze` returns correct JSON report for test fixtures.
- [ ] Chain-of-custody test (`test_chain_of_custody.py`) passes.
- [ ] PDF report opens correctly in Acrobat and contains correct hash.
- [ ] Auth and rate limiting behave per spec.

---

## Phase 4: Testing & Hardening (Apr 7 – Apr 25, 2025)

**Goal**: Achieve full test coverage targets, run the complete benchmark suite, fix regressions, and harden the deployment.

### Milestones

| # | Deliverable | Due |
|---|------------|-----|
| 4.1 | Unit test suite: all extractors, ingest, report gen, auth | Apr 11 |
| 4.2 | Integration test suite: all API endpoints | Apr 14 |
| 4.3 | Coverage report: ≥ 85% overall | Apr 15 |
| 4.4 | Full benchmark run: ensemble F1 ≥ 0.84 | Apr 17 |
| 4.5 | Security review: input validation, path traversal, injection | Apr 18 |
| 4.6 | Load test: 60 req/min sustained for 10 min (no 5xx) | Apr 21 |
| 4.7 | Docker images built, scanned (Trivy), no HIGH/CRITICAL CVEs | Apr 22 |
| 4.8 | Monitoring: Prometheus + Grafana dashboards live on staging | Apr 23 |
| 4.9 | Runbook documented: deploy, rollback, alert response | Apr 25 |

### Exit Criteria

- [ ] Zero known HIGH/CRITICAL security issues.
- [ ] All CI checks green on `release/v1.0` branch.
- [ ] Benchmark results committed to `benchmarks/results/v1.0.json`.
- [ ] Load test report shows p99 latency < 2 s under sustained load.

---

## Phase 5: Release (Apr 28 – May 9, 2025)

### Milestones

| # | Deliverable | Due |
|---|------------|-----|
| 5.1 | `CHANGELOG.md` and release notes drafted | Apr 29 |
| 5.2 | Docker images tagged `v1.0.0` and pushed to registry | Apr 30 |
| 5.3 | Model weights published to versioned S3 bucket | Apr 30 |
| 5.4 | GitHub Release `v1.0.0` created with checksums | May 1 |
| 5.5 | API keys issued to initial pilot users (3 organizations) | May 5 |
| 5.6 | Pilot monitoring: active on-call for first week post-launch | May 5–9 |
| 5.7 | Post-launch retrospective | May 9 |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| DEFACTO/NIST 16 DUA approval delayed | Medium | Medium | Begin Phase 1 with CASIA v2; DEFACTO can be added in Phase 2 without blocking PatchForensic training |
| PatchForensic does not reach F1 ≥ 0.80 | Low | High | Budget 1 extra week; try: deeper decoder, attention gates, stronger augmentation |
| Inpainting detector AUC below 0.94 | Medium | Medium | Increase synthetic set to 40,000; try SD-XL for more diverse inpainting |
| GPU unavailable for training window | Low | High | AWS `p3.8xlarge` as backup (4× V100); pre-approve cost |
| MantraNet pretrained weights not public | Low | High | Contact original authors; fall back to training from scratch on DEFACTO only |

---

## v1.1 Scope (Post-Launch, tentative Q3 2025)

- Web UI (React + Tailwind): drag-and-drop, heatmap overlay visualization.
- Triton Inference Server with ONNX models: dynamic batching, higher throughput.
- S3-backed artifact storage.
- Kubernetes deployment manifests.
- Video frame analysis: extract frames at 1 fps, analyze each, aggregate results.
- GAN-generated image detection (StyleGAN, DALL-E, Midjourney).
