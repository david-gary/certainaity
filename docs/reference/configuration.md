# Configuration Reference

All settings use the `CERTAINAITY_` prefix and can be set via environment variables or a `.env` file.

## Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `CERTAINAITY_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL (broker + result backend) |

## File paths

| Variable | Default | Description |
|----------|---------|-------------|
| `CERTAINAITY_OUTPUT_DIR` | `output` | Directory where job output subdirectories are written |
| `CERTAINAITY_WEIGHTS_DIR` | `weights` | Directory containing model weight files |

## JWT authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `CERTAINAITY_JWT_PUBLIC_KEY_PATH` | `secrets/jwt_public.pem` | Path to RS256 public key for verifying bearer tokens |

## Upload limits

| Variable | Default | Description |
|----------|---------|-------------|
| `CERTAINAITY_MAX_FILE_BYTES` | `52428800` (50 MB) | Maximum upload size in bytes |
| `CERTAINAITY_MIN_IMAGE_DIMENSION` | `64` | Minimum image side length in pixels |
| `CERTAINAITY_MAX_IMAGE_DIMENSION` | `20000` | Maximum image side length in pixels |

## Rate limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `CERTAINAITY_RATE_LIMIT_PER_MINUTE` | `60` | Maximum `POST /v1/analyze` requests per source IP per minute |

## Inference

| Variable | Default | Description |
|----------|---------|-------------|
| `CERTAINAITY_ENSEMBLE_THRESHOLD` | `0.65` | Confidence threshold above which `manipulation_detected` is `true` |
| `CERTAINAITY_MIN_REGION_PX` | `4096` | Minimum connected-component area (pixels) to report as a region |
| `CERTAINAITY_USE_CPU` | `false` | Force CPU inference even when an NVIDIA GPU is available |

## Feature extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `CERTAINAITY_ELA_QUALITY` | `75` | JPEG resave quality for ELA computation |
| `CERTAINAITY_NOISE_BLOCK_SIZE` | `32` | Block size in pixels for wavelet noise variance estimation |
| `CERTAINAITY_DCT_BLOCK_SIZE` | `8` | Block size in pixels for DCT copy-move analysis |
| `CERTAINAITY_FEATURE_WORKERS` | `4` | ThreadPoolExecutor parallelism for feature extraction |

## Resilience test

| Variable | Default | Description |
|----------|---------|-------------|
| `CERTAINAITY_RESILIENCE_QUALITIES` | `[70, 85, 95]` | JPEG re-compression quality levels to test |
| `CERTAINAITY_RESILIENCE_DROP_THRESHOLD` | `0.25` | Confidence drop that triggers `anti_forensic_warning` |
