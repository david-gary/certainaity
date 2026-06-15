# Certainaity — Data Pipeline

## Overview

The data pipeline covers everything from raw dataset acquisition to the tensors consumed during model training. It is separate from the inference pipeline; during inference, only the preprocessing steps apply.

The pipeline is implemented in `src/data/` and is runnable as a standalone script:

```bash
python -m certainaity.data.build_dataset \
    --datasets casia_v2 defacto nist16 div2k_inpainted \
    --output_dir data/processed/ \
    --split 0.80/0.10/0.10
```

---

## Source Datasets

| Dataset | Size | Manipulation Types | License |
|---------|------|--------------------|---------|
| CASIA v2 | 12,614 images (7,491 authentic / 5,123 tampered) | Splicing, copy-move | Research only |
| DEFACTO | 149,000 images | Splicing, copy-move, removal | Research only |
| NIST 16 | 564 tampered (+ 1,000 authentic reference) | Splicing, copy-move, removal | Research only |
| COVERage | 2,188 pairs | Copy-move | Research only |
| DIV2K (authentic source) | 1,000 high-res images | None (used to create inpainting set) | CC BY 4.0 |
| COCO (authentic source) | ~10,000 sampled images | None (used to create inpainting set) | CC BY 4.0 |

### Synthetic Inpainting Set (self-generated)
- **Total images**: 22,000 (DIV2K + COCO subset)
- **Method**: for each source image, select 1–3 polygonal regions (5–20% of image area each). Erase region with a median-fill buffer (2 px feathering). Inpaint with Stable Diffusion 2.1 via `diffusers` library (50 DDIM steps, guidance scale 7.5, empty positive prompt).
- **Ground truth**: binary mask of the erased polygon(s).
- **Diversity augmentation**: vary polygon count, shape irregularity, and aspect ratio to prevent the model from keying on shape priors.

---

## Directory Structure (post-download)

```
data/
  raw/
    casia_v2/
      Tp/          ← tampered images
      Au/          ← authentic images
      masks/       ← ground truth binary masks
    defacto/
      ...
    nist16/
      ...
    coverage/
      ...
    div2k/
      HR/
    coco_sample/
  processed/
    train/
      images/      ← 256×256 crops (augmented)
      masks/       ← corresponding binary masks
      metadata.jsonl
    val/
    test/
  inpainting_synthetic/
    images/
    masks/
    source_info.jsonl
```

---

## Preprocessing Steps

### Step 1: Format Normalization

All images are loaded with `Pillow` and converted to RGB. TIFF and PNG are accepted as-is; JPEG files are decoded without re-encoding to preserve compression artifacts. If a ground truth mask is provided as a TIFF with multiple channels, only channel 0 is used and binarized at threshold 127.

### Step 2: Quality Filtering

Discard images where:
- Shorter side < 256 px (too small for meaningful 256×256 crops)
- File is corrupt (catches ~0.3% of DEFACTO)
- Ground truth mask is all-zero or all-one (ambiguous; ~1.2% of NIST 16 after re-inspection)

### Step 3: Patch Extraction

For training, images are not resized to a fixed resolution (which would distort forensic statistics). Instead, overlapping 256×256 patches are extracted:

```python
stride = 128   # 50% overlap
patches = []
for y in range(0, H - 256, stride):
    for x in range(0, W - 256, stride):
        img_patch  = image[y:y+256, x:x+256]
        mask_patch = mask[y:y+256, x:x+256]
        # include patch only if mask has ≥ 16 manipulated px
        # OR ≥ 200 authentic px, to balance classes
        if mask_patch.sum() >= 16 or (mask_patch.sum() == 0 and random() < 0.4):
            patches.append((img_patch, mask_patch))
```

This produces approximately **4.2 million** patches across all datasets.

### Step 4: Class Balancing

Without intervention, authentic patches outnumber manipulated ~6:1. Balancing strategy:
1. Keep all manipulated patches.
2. Randomly subsample authentic patches to achieve a 2:1 authentic-to-manipulated ratio (some authentic context is necessary; pure 1:1 causes over-prediction at inference).

Final balanced training set: ~2.1M patches.

### Step 5: Data Augmentation (train split only)

Applied on-the-fly during training via `albumentations`:

| Transform | Parameters | Rationale |
|-----------|------------|-----------|
| HorizontalFlip | p=0.5 | Symmetry invariance |
| VerticalFlip | p=0.3 | Less common in real forensics |
| RandomRotate90 | p=0.5 | Rotation invariance for copy-move |
| GaussNoise | var_limit=(5, 25), p=0.2 | Simulate re-scan noise |
| ISONoise | intensity=(0.1, 0.3), p=0.2 | Camera noise variation |
| RandomBrightnessContrast | limit=0.1, p=0.3 | Lighting normalization robustness |
| JpegCompression | quality_lower=75, quality_upper=95, p=0.3 | JPEG re-save resilience |

**Important**: augmentations that could plausibly create forensic artifacts (e.g., copy-paste within the patch) are NOT applied, as they would corrupt the ground truth mask.

### Step 6: Normalization

Images normalized to `[0, 1]` float32 (divide by 255). No ImageNet mean/std subtraction — forensic features depend on absolute pixel values, and mean subtraction destroys DC coefficient information.

---

## Metadata Manifest

Each split has a `metadata.jsonl` file where each line is a JSON object:

```json
{
  "image_path": "train/images/casia_0042_patch_003.npy",
  "mask_path": "train/masks/casia_0042_patch_003.npy",
  "source_dataset": "casia_v2",
  "source_image": "Tp/Tp_D_CND_M_N_ani00073_sec00150_11854.jpg",
  "manipulation_type": "splicing",
  "manipulated_fraction": 0.31,
  "patch_origin": [256, 128]
}
```

This manifest is used by the `torch.utils.data.Dataset` implementation to avoid storing all patches in memory.

---

## Dataset Class

```python
class ForensicDataset(Dataset):
    def __init__(self, manifest_path: str, transform=None):
        self.records = [json.loads(l) for l in open(manifest_path)]
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = np.load(rec["image_path"])   # (256, 256, 3) uint8
        mask  = np.load(rec["mask_path"])    # (256, 256)    uint8 {0, 1}

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask  = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask
```

---

## Inpainting Set Generation Pipeline

```bash
python -m certainaity.data.generate_inpainting \
    --source_dir data/raw/div2k/HR data/raw/coco_sample \
    --output_dir data/inpainting_synthetic/ \
    --n_images 22000 \
    --model_id stabilityai/stable-diffusion-2-inpainting \
    --device cuda:0
```

Key parameters:
- `--poly_count`: 1–3 polygons per image (uniform random)
- `--area_fraction`: 0.05–0.20 of total image area
- `--feather_px`: 2 px feathering at polygon edge
- `--steps`: 50 DDIM steps
- `--seed`: per-image deterministic seed (hash of source filename) for reproducibility

Expected runtime: ~4.5 hours on a single A100 for 22,000 images.

---

## Validation & Test Splits

- **Validation** (10%): stratified by dataset and manipulation type. Used for ensemble weight optimization and early stopping.
- **Test** (10%): held out entirely until final evaluation. Never used for hyperparameter selection.

Split is deterministic given a fixed random seed (42). Split indices are saved to `data/splits.json` so results are exactly reproducible.

---

## Storage Requirements

| Directory | Approx. Size |
|-----------|-------------|
| `data/raw/` (all datasets) | ~85 GB |
| `data/processed/` (patches as .npy) | ~210 GB |
| `data/inpainting_synthetic/` | ~18 GB |
| **Total** | **~313 GB** |

A minimum of 500 GB of fast NVMe storage is recommended for the training machine. The processed patches can be placed on a slower HDD if the DataLoader `num_workers` is set high enough (≥8) to prefetch ahead.
