# Forenscope Plan

---

## Detailed Modules

### 1. Preprocessing & Metadata Extraction

- Parse EXIF, JPEG quantization tables, and thumbnail mismatches.
- Convert image to YCbCr + RGB + residual (high‑pass filtered) domains.
- Detect multiple JPEG compression history (double quantization artifacts).
- Compute SHA‑256 hash immediately for chain‑of‑custody.

### 2. Handcrafted Forensic Feature Extraction

| Feature | Description | Output |
|---------|-------------|--------|
| **Error Level Analysis (ELA)** | Highlights regions resaved at a different JPEG quality. | ELA heatmap (8‑bit) |
| **Noise variance estimation** | Wavelet‑based denoising; inconsistent noise = splicing. | Noise inconsistency map |
| **CFA interpolation correlation** | Detects mismatched Bayer pattern interpolation. | CFA artifact map |
| **DCT coefficient histograms** | Block‑wise DCT histogram similarity for copy‑move. | Block similarity matrix |

These features are **not** used as final decisions but as inputs to the ensemble (by concatenating feature maps) and as evidence text in the final report.

### 3. Deep Learning Detectors (Ensemble)

| Model | Architecture | Training Data | Primary Role |
|-------|--------------|---------------|--------------|
| **PatchForensic** | Custom 9‑layer CNN + patch‑based | CASIA v2, COVERage | Splicing & copy‑move localization |
| **MantraNet** | Fine‑tuned (pretrained on ImageNet) | DEFACTO, NIST 16 | Multi‑class (splicing, copy‑move, removal) |
| **SPSL (Siamese)** | Dual‑branch Siamese network | Custom + augmented | Copy‑move with rotation & scaling |
| **Inpainting Detector** | CLIP‑based classifier + attention rollout | Self‑generated (Stable Diffusion inpainted) | AI‑generated removal/inpainting |

**Ensemble Voting**  
Each model outputs a manipulation probability map (0–1 per pixel or patch). The final score is a weighted average where weights are optimized on a validation set (e.g., 0.35, 0.30, 0.20, 0.15). A pixel is flagged as “manipulated” if the weighted score > 0.65.

### 4. AI‑Generated Inpainting Detector (Detailed)

**Motivation** – Standard forgery detectors are not trained on generative inpainting and often mistake smooth AI‑filled regions as authentic.

**Training Data Generation**  

1. Start with 10,000 pristine, high‑resolution images (e.g., DIV2K, COCO).  
2. For each image, randomly select 1–3 polygonal regions (5–20% of image area).  
3. Erase the region and inpaint using **Stable Diffusion 2.1** with prompt “fill realistically” (empty prompt = context‑aware fill).  
4. The ground truth mask is the erased region.

**Model** – Fine‑tune a CLIP vision encoder with a small segmentation head. Use contrastive loss between inpainted patch and original patch. Inference: attention rollout maps highlight fake areas.

**Performance** – Expected AUC > 0.94 on held‑out AI‑inpainted test set.

### 5. Post‑processing Resilience Test

Because real evidence images may be re‑compressed, resized, or filtered, ForenScope includes an automated resilience test:

1. Take the input image.
2. Re‑save it at JPEG quality levels **70, 85, 95**.
3. Run the full ensemble on each re‑compressed version.
4. If the confidence score drops by more than **0.25** (on a 0–1 scale) between the original and any re‑compressed version, add a warning: `"anti_forensic_possible": true` to the report.

### 6. Forensic Report Generator

Outputs two files per analysis:

**A. JSON report (machine‑readable)**

```json
{
  "file_name": "evidence_001.jpg",
  "sha256": "a1b2c3...",
  "analysis_timestamp": "2026-06-15T14:32:00Z",
  "manipulation_detected": true,
  "overall_confidence": 0.92,
  "regions": [
    {
      "bbox": [100, 200, 300, 400],
      "type": "splicing",
      "confidence": 0.96,
      "evidence": "CFA mismatch (p=0.003), noise variance discontinuity"
    },
    {
      "bbox": [500, 50, 600, 150],
      "type": "ai_inpainting",
      "confidence": 0.88,
      "evidence": "CLIP attention rollout > 0.82"
    }
  ],
  "anti_forensic_warning": false,
  "models_used": ["PatchForensic", "MantraNet", "SPSL", "InpaintingDetector"],
  "execution_time_ms": 412
}
