# Forensic Pipeline

Every image submitted to Certainaity passes through five stages executed by the Celery worker.

## Stage 1 — Ingest & validate

`certainaity.ingest.ingest_image()`

- SHA-256 is computed from the raw byte stream **before** any decoding, preserving chain-of-custody integrity.
- Format, size, and dimension constraints are enforced (see [Configuration](../reference/configuration.md)).
- EXIF metadata is extracted and flattened.
- JPEG quantization tables are parsed.
- Thumbnail aspect-ratio mismatch is flagged (a common indicator of re-save manipulation).

## Stage 2 — Feature extraction

`certainaity.worker.tasks._extract_features()`

Four handcrafted forensic feature extractors run in parallel via `ThreadPoolExecutor`:

| Extractor | Module | What it detects |
|-----------|--------|-----------------|
| **ELA** (Error Level Analysis) | `features.ela` | Regions compressed at a different JPEG quality level |
| **Noise variance** | `features.noise` | Sensor noise inconsistency across regions (wavelet HH subband) |
| **CFA correlation** | `features.cfa` | Missing or mismatched Bayer demosaicing pattern |
| **DCT similarity** | `features.dct` | Copy-move pairs via cosine distance in DCT coefficient space |

Each extractor returns a `(H, W) float32` map in `[0, 1]`. Higher values indicate stronger forensic signal.

## Stage 3 — Ensemble inference

`certainaity.models.Ensemble.localize()`

Four deep learning models produce pixel-level manipulation probability maps:

| Model | Architecture | Speciality |
|-------|-------------|------------|
| **PatchForensic** | 9-layer fully-convolutional network | General manipulation localization |
| **MantraNet** | VGG-16/BN + local anomaly detector | Manipulation type classification |
| **SPSL** | Siamese ResNet-50 + FAISS | Splicing (source/target pair detection) |
| **InpaintingDetector** | CLIP ViT-B/32 + segmentation head | AI-generated inpainting |

Fusion: `heatmap = Σ (wᵢ / Σwⱼ) × mapᵢ` with default weights `[0.35, 0.30, 0.20, 0.15]`.

Connected components smaller than `CERTAINAITY_MIN_REGION_PX` are discarded as noise.

### WeightsNotFoundError fallback

If model weight files are absent, `WeightsNotFoundError` is caught and the pipeline falls back to the mean of the four handcrafted feature maps as the overall confidence estimate. The `models_used` field in the report reflects this.

## Stage 4 — Resilience test

`certainaity.resilience.run_resilience_test()`

Only runs when Stage 3 produced a heatmap **and** `overall_confidence > ensemble_threshold`.

The image is re-compressed at each quality in `CERTAINAITY_RESILIENCE_QUALITIES` (default `[70, 85, 95]`). The ensemble runs on each recompressed version. If confidence drops by more than `CERTAINAITY_RESILIENCE_DROP_THRESHOLD` (default `0.25`) at any quality, `anti_forensic_warning = True` is set in the report.

This detects post-processing designed to make forensic signals appear only at specific compression settings — a known anti-forensic evasion technique.

## Stage 5 — Report generation

`certainaity.report.generate_report()`, `save_json_report()`, `save_pdf_report()`

Outputs are written to `CERTAINAITY_OUTPUT_DIR/{job_id}/`:

| File | Contents |
|------|----------|
| `report.json` | Full structured report with all custody fields |
| `report.pdf` | ReportLab PDF with verdict, heatmap (if available), region table, and chain-of-custody statement |
