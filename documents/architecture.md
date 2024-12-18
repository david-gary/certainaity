# ForenScope — System Architecture

## Overview

ForenScope is a forensic image manipulation detection system designed for use in legal, investigative, and journalistic contexts. The system accepts an image as input and produces a structured forensic report identifying manipulated regions, the type of manipulation, confidence scores, and supporting evidence.

The architecture is split into five primary layers:

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│       REST API  /  CLI Tool  /  Web UI (future)             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Ingestion Layer                            │
│   Image validation, SHA-256 hash, metadata extraction        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Feature Extraction Layer                     │
│   ELA · Noise · CFA · DCT · YCbCr decomposition             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     Inference Layer                           │
│   PatchForensic · MantraNet · SPSL · InpaintingDetector      │
│               Weighted ensemble fusion                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Reporting Layer                            │
│          JSON report + PDF forensic summary                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### 1. Client Layer

**REST API** (primary interface)
- Built with FastAPI (Python 3.11+)
- Accepts `multipart/form-data` POST with the image file
- Returns JSON response inline for small images; presigned download URL for PDF reports
- Rate-limited: 60 requests/minute per API key (configurable)
- Auth via bearer token (JWT); admin tokens issued per organization

**CLI Tool**
- Single binary (`forenscope analyze <image_path> [--output json|pdf|both]`)
- Wraps the same Python pipeline; intended for batch processing on forensic workstations
- Supports reading from stdin for pipe-based automation

**Web UI (deferred)**
- React + Tailwind drag-and-drop interface
- Visualizes heatmaps overlaid on the original image
- Planned for v1.1; not in scope for v1.0

---

### 2. Ingestion Layer

**Responsibilities**
- Validate file type (JPEG, PNG, TIFF, WebP supported; GIF rejected)
- Reject files > 50 MB or > 20,000 px on any side
- Compute SHA-256 of the raw bytes immediately on arrival — this hash is the chain-of-custody anchor
- Extract EXIF metadata (camera make/model, GPS, software tag, creation/modification timestamps)
- Extract JPEG quantization tables and thumbnail (if present)
- Check thumbnail–main image consistency (thumbnail mismatch is itself a forensic signal)

**Libraries**
- `Pillow` for decoding and format validation
- `piexif` for EXIF extraction
- `jpegtran-cffi` for quantization table extraction
- `hashlib` (stdlib) for SHA-256

**Output schema (internal)**
```python
@dataclass
class IngestedImage:
    sha256: str
    width: int
    height: int
    format: str           # "JPEG" | "PNG" | "TIFF" | "WebP"
    exif: dict            # raw EXIF key-value pairs
    quantization_tables: list[list[int]] | None
    thumbnail_mismatch: bool
    pil_image: Image.Image
```

---

### 3. Feature Extraction Layer

All handcrafted features are computed in parallel using a `ProcessPoolExecutor` with 4 workers (configurable). Each feature extractor receives the `PIL.Image` and returns a `numpy.ndarray` of shape `(H, W)` normalized to `[0, 1]`, plus a short text summary for the report.

| Extractor | Key Implementation Detail |
|-----------|--------------------------|
| ELA | Resave at quality 75; compute absolute difference; normalize by percentile (99th) |
| Noise variance | Discrete wavelet transform (db8, 3 levels); residual HH subband variance per block |
| CFA artifact map | Compute 2D autocorrelation of green channel residual; peak at (0,2) and (2,0) indicates Bayer |
| DCT block similarity | 8×8 DCT of each block; cosine distance between all block pairs; mark pairs < 0.05 distance |

These maps are stacked into a `(4, H', W')` tensor at 1/8 resolution (downsampled to match model input) and passed to the inference layer as auxiliary input channels.

---

### 4. Inference Layer

#### Model Loading
Models are loaded once at startup into GPU memory (CUDA) or CPU fallback. Weights are stored in `weights/` directory (excluded from git; distributed separately via LFS or a presigned S3 URL).

```
weights/
  patchforensic_v2.pth
  mantranet_finetuned.pth
  spsl_siamese.pth
  inpainting_detector_clip.pth
```

#### Per-Model Input/Output Contract

Each model implements a `ForensicModel` abstract class:

```python
class ForensicModel(ABC):
    @abstractmethod
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (1, 3, H, W) float32 in [0, 1]
        returns: (1, 1, H, W) float32 in [0, 1] — manipulation probability map
        """
```

Images are padded to multiples of 32 before inference and cropped back afterward.

#### Ensemble Fusion

```python
weights = [0.35, 0.30, 0.20, 0.15]   # PatchForensic, MantraNet, SPSL, Inpainting
maps    = [m1, m2, m3, m4]            # each shape (H, W)

fused = sum(w * m for w, m in zip(weights, maps))
binary_mask = (fused > 0.65).astype(np.uint8)
```

Weights are learned by minimizing BCE loss on the val split of CASIA v2 + NIST 16 combined using `scipy.optimize.minimize` with L-BFGS-B. They are fixed at inference time.

#### Connected Component Labeling

After thresholding, `skimage.measure.label` extracts connected regions. Regions smaller than 64×64 px are discarded (noise floor). Each surviving region is assigned a bounding box, type classification, and per-region confidence.

**Type classification (per region)**:  
Run a lightweight 4-class head (Linear → ReLU → Linear → Softmax) on the mean feature vector within the region. Classes: `splicing`, `copy_move`, `removal`, `ai_inpainting`.

---

### 5. Reporting Layer

Two artifacts are generated per analysis:

**A. JSON report** — see `forenscope_plan.md` for full schema.

**B. PDF report** — generated with `reportlab`:
- Page 1: executive summary (hash, timestamp, overall verdict, confidence)
- Page 2: original image + binary mask overlay (red = manipulated)
- Page 3+: per-region detail (cropped region, heatmap, evidence text, model votes)
- Footer: "Generated by ForenScope vX.Y — for investigative use only"

Both artifacts are written to `output/<sha256_prefix>/` and the paths are returned in the API response.

---

## Data Flow (end-to-end)

```
POST /analyze
  → validate & hash
  → extract EXIF / quantization tables
  → decompose to YCbCr + residual
  → [parallel] ELA, Noise, CFA, DCT
  → [parallel] PatchForensic, MantraNet, SPSL, Inpainting inference
  → fuse ensemble
  → threshold + label connected components
  → classify manipulation types per region
  → [optional] resilience test (3× re-compress)
  → generate JSON + PDF
  → return 200 with report URL
```

Typical end-to-end latency on a single A100 GPU: ~400 ms for a 12 MP image.  
CPU-only fallback: ~8–12 s.

---

## Scalability Considerations

- **Horizontal scaling**: the API is stateless; scale out with multiple replicas behind a load balancer. Each replica loads all four models into its own GPU.
- **Model batching**: not implemented in v1.0 (single-image per request); planned for v1.1 with dynamic batching via Triton Inference Server.
- **Storage**: output artifacts stored in local filesystem in v1.0; S3-compatible object storage (MinIO or AWS S3) planned for v1.1.
- **Queue**: for batch CLI usage, a Celery + Redis task queue is available; API requests bypass the queue for low-latency response.

---

## Security Considerations

- All uploaded images are processed in an isolated subprocess; the main API process never executes image data directly.
- SHA-256 hash is computed before any processing to prevent TOCTOU attacks.
- Output directory names are derived from the SHA-256 prefix, not from user-supplied filenames, to prevent path traversal.
- API keys are stored as bcrypt hashes; raw keys are never logged.
- PDF generation runs in a sandboxed subprocess to mitigate potential ReportLab deserialization exploits.
