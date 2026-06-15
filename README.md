# Certainaity

Detects image manipulation for legal and investigative use. Identifies splicing, copy-move, AI inpainting, content removal, and GAN-generated images. Produces court-ready PDF reports with chain-of-custody metadata.

Live at [certainaity.com](https://certainaity.com).

---

## How it works

Upload an image → five-stage pipeline runs in Celery → JSON/PDF report with confidence scores and per-region heatmap.

1. **Ingest** — decode, validate dimensions, compute SHA-256
2. **Feature extraction** — ELA, wavelet noise variance, CFA interpolation, DCT block similarity (parallel threads)
3. **Ensemble inference** — PatchForensic + MantraNet + SPSL + InpaintingDetector + GANDetector; weighted fusion
4. **Resilience test** — re-compress at JPEG [70, 85, 95]; flag anti-forensic tampering if confidence drops >25%
5. **Report** — JSON + PDF with heatmap, region table, chain-of-custody statement

---

## Running locally

**Prerequisites:** Docker, Docker Compose, a Redis instance (included in compose).

```bash
cp .env.example .env
# edit .env — set CERTAINAITY_JWT_PUBLIC_KEY_PATH at minimum
docker compose up
```

API available at `http://localhost:8000`. Web UI at `http://localhost:3000`.

Generate a dev key pair:

```bash
python scripts/generate_keys.py
```

CPU-only dev (no GPU required):

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml up
```

---

## API

All routes require `Authorization: Bearer <token>` except `/v1/health`.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/v1/health` | Liveness probe |
| `POST` | `/v1/analyze` | Submit image (multipart) |
| `GET` | `/v1/jobs/{id}` | Poll task state |
| `GET` | `/v1/jobs/{id}/report` | JSON report |
| `GET` | `/v1/jobs/{id}/report.pdf` | PDF report |

Rate limit: 60 `POST /v1/analyze` per IP per minute (configurable).

---

## Configuration

All settings use the `CERTAINAITY_` prefix and can be set via environment or `.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `CERTAINAITY_REDIS_URL` | `redis://localhost:6379/0` | Celery broker |
| `CERTAINAITY_WEIGHTS_DIR` | `weights/` | Model weights directory |
| `CERTAINAITY_OUTPUT_DIR` | `output/` | Report output directory |
| `CERTAINAITY_JWT_PUBLIC_KEY_PATH` | `secrets/jwt_public.pem` | RS256 public key |
| `CERTAINAITY_ENSEMBLE_THRESHOLD` | `0.65` | Confidence threshold for positive detection |
| `CERTAINAITY_RATE_LIMIT_PER_MINUTE` | `60` | Requests per IP per minute |
| `CERTAINAITY_USE_CPU` | `false` | Force CPU inference |

---

## Models

Weights are not included — mount them at runtime to `CERTAINAITY_WEIGHTS_DIR`. Without weights, the pipeline falls back to handcrafted feature means and labels results accordingly.

| Model | File | Detects |
| --- | --- | --- |
| PatchForensic | `patchforensic_v2.pth` | Splicing, copy-move |
| MantraNet | `mantranet_finetuned.pth` | General manipulation |
| SPSL | `spsl_siamese.pth` | Frequency artifacts |
| InpaintingDetector | `inpainting_detector_clip.pth` | AI inpainting, removal |
| GANDetector | `gandec_v1.pt` | StyleGAN, DALL-E, Midjourney, SD |

---

## Deployment

See [docs.certainaity.com/guides/deployment](https://docs.certainaity.com/guides/deployment) for Docker and [docs.certainaity.com/guides/kubernetes](https://docs.certainaity.com/guides/kubernetes) for Kubernetes.

Production images are published to GHCR on version tags:

```text
ghcr.io/david-gary/certainaity-api:<version>
ghcr.io/david-gary/certainaity-worker:<version>
```

---

## Development

```bash
pip install -e ".[dev]"
pytest                        # unit + integration tests (≥85% coverage required)
ruff check src tests          # lint
mypy src                      # type check
```

```bash
cd frontend && npm install && npm run dev   # web UI dev server on :3000
```
