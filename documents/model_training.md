# ForenScope — Model Training Plan

## Overview

Four models are trained independently and then combined via a learned weighted ensemble. Training for each model follows the same general loop but with model-specific loss functions, architectures, and hyperparameters. All training runs are logged to MLflow; experiment names follow the convention `forenscope/<model_name>/<run_id>`.

---

## Shared Training Infrastructure

**Hardware target**: single machine with 4× NVIDIA A100 80 GB GPUs (DDP training via `torch.distributed`).  
**Framework**: PyTorch 2.2 + `torch.compile` (Inductor backend).  
**Logging**: MLflow (local server, `mlflow ui --port 5000`).  
**Checkpointing**: save every 5 epochs; keep best-3 by val F1.  
**Mixed precision**: `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.

### Common Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR schedule | Cosine annealing with warmup (5% of steps) |
| Gradient clipping | max norm 1.0 |
| Weight decay | 1e-4 |
| Batch size (per GPU) | 32 (effective batch 128 across 4 GPUs) |
| Mixed precision | bfloat16 |
| Seed | 42 |

---

## Model 1: PatchForensic

### Architecture

9-layer CNN operating on 256×256 image patches. No pretrained backbone — trained from scratch to avoid ImageNet biases affecting forensic features.

```
Input: (3, 256, 256)
Conv2d(3, 64, 3, padding=1)   → BN → ReLU
Conv2d(64, 64, 3, padding=1)  → BN → ReLU → MaxPool2d(2)
Conv2d(64, 128, 3, padding=1) → BN → ReLU
Conv2d(128, 128, 3, padding=1)→ BN → ReLU → MaxPool2d(2)
Conv2d(128, 256, 3, padding=1)→ BN → ReLU
Conv2d(256, 256, 3, padding=1)→ BN → ReLU
Conv2d(256, 512, 3, padding=1)→ BN → ReLU
TransposedConv2d(512, 256, 4, stride=2, padding=1)  ← decoder
TransposedConv2d(256, 128, 4, stride=2, padding=1)
Conv2d(128, 1, 1) → Sigmoid
Output: (1, 256, 256)  ← pixel-wise manipulation probability
```

Skip connections from each encoder block to its mirror decoder block (U-Net style). The handcrafted feature maps (ELA, noise, CFA, DCT) are concatenated to the bottleneck features (8 extra channels).

### Loss Function

Combined loss:
```
L = 0.5 * BCE(pred, mask) + 0.5 * DiceLoss(pred, mask)
```

BCE handles the pixel-level binary classification; Dice loss compensates for class imbalance without explicit sample weighting.

### Training Schedule

| Phase | Epochs | LR | Notes |
|-------|--------|----|-------|
| Warmup | 1–5 | 1e-4 → 5e-4 | Linear ramp |
| Main | 6–80 | 5e-4 → 1e-6 | Cosine decay |
| Fine-tune on NIST 16 | 81–90 | 1e-5 | Domain-specific fine-tune |

Expected val F1 after epoch 90: ~0.82 on CASIA v2 val split.

### Key Training Details

- **Curriculum**: start with easy patches (large manipulated regions > 30% of patch area); introduce hard patches (small regions 5–15%) after epoch 20.
- **Hard negative mining**: at each epoch, run the current model on the authentic validation set; select the 10% of authentic patches with highest false-positive score and oversample them in the next epoch's dataloader.

---

## Model 2: MantraNet (Fine-tuned)

### Architecture

MantraNet uses a pretrained VGG-like backbone (ImageNet pretrained) with a Local Anomaly Detection (LAD) head that computes Z-score statistics per spatial location. We fine-tune the LAD head and the last two backbone blocks; the early backbone blocks are frozen.

**Pretrained weights**: `mantranet_pretrained_public.pth` — weights released by the original authors.

### Fine-tuning Strategy

Only the LAD head and blocks 4–5 of the VGG backbone are updated:

```python
for name, param in model.named_parameters():
    if "features.0" in name or "features.1" in name or "features.2" in name:
        param.requires_grad = False   # freeze early blocks
```

### Loss Function

Same combined BCE + Dice loss as PatchForensic.

### Training Schedule

| Phase | Epochs | LR |
|-------|--------|----|
| LAD head only | 1–20 | 1e-3 |
| LAD head + blocks 4-5 | 21–50 | 5e-4 |
| Fine-tune on DEFACTO | 51–60 | 1e-5 |

Expected val F1: ~0.79 on DEFACTO val split.

### Notes

MantraNet's pretraining gives it strong generalization to unseen manipulation types. The fine-tuning is intentionally conservative (low LR, few unfrozen blocks) to preserve this property while adapting to the modern DEFACTO distribution.

---

## Model 3: SPSL (Siamese)

### Architecture

Dual-branch Siamese network for copy-move detection. Both branches share weights (weight tying). The network learns to compare patch similarity in a forensically meaningful embedding space.

```
Branch A (query patch) ──┐
  ResNet-50 (ImageNet)   ├─→ L2-normalize → 128-dim embedding
Branch B (candidate)   ──┘

Distance head:
  |emb_A - emb_B| → Linear(128, 64) → ReLU → Linear(64, 1) → Sigmoid
```

**Important**: during copy-move detection, Branch B receives candidate patches from all 8×8 block positions in the image. Distance below 0.4 (learned threshold) indicates a copy-move pair.

### Training Data

Pairs extracted from COVERage dataset + copy-move images from CASIA v2 and DEFACTO:
- Positive pairs: original patch + its copied counterpart (with geometric transformation).
- Negative pairs: randomly sampled non-matching patches.
- Pair ratio: 1:3 positive-to-negative (hard negatives are near-duplicate patches from different images).

### Loss Function

Contrastive loss:
```
L = y * d^2 + (1-y) * max(0, margin - d)^2
```
where `d = ||emb_A - emb_B||_2`, `y = 1` for genuine pairs, `margin = 1.0`.

### Training Schedule

| Phase | Epochs | LR |
|-------|--------|----|
| Warmup | 1–5 | 1e-4 |
| Main | 6–60 | 5e-4 → 1e-5 |

Expected pair AUC: ~0.91 on COVERage test set.

### Inference Mode

At inference, SPSL does not operate pair-wise across all blocks (O(n²) patches). Instead:
1. Extract embeddings for all non-overlapping 64×64 blocks.
2. Build a FAISS flat L2 index over block embeddings.
3. For each block, query k=5 nearest neighbors.
4. If any neighbor has distance < 0.4 and is far away in image space (Manhattan > 5 blocks), flag both blocks as copy-move.

This reduces inference complexity from O(n²) to O(n log n).

---

## Model 4: Inpainting Detector (CLIP-based)

### Architecture

Fine-tuned CLIP ViT-B/32 with a small segmentation head. The CLIP vision encoder processes the image in 224×224 tiles and produces a 768-dim embedding per token. The segmentation head upsamples and projects:

```
CLIP ViT-B/32 (frozen initially)
  → patch embeddings (196 tokens, 768-dim)
  → Linear(768, 256) → GELU → reshape to (14, 14, 256)
  → 4× bilinear upsample to (224, 224, 256)
  → Conv2d(256, 1, 1) → Sigmoid
Output: (1, 224, 224)
```

The attention rollout from the last CLIP transformer layer is also extracted and added to the sigmoid output (weighted 0.3) to improve boundary sharpness.

### Training Data

Synthetic inpainting set (22,000 images), split 80/10/10. No other datasets used, as other manipulation types would confuse the inpainting-specific detector.

### Loss Function

Focal loss (addresses extreme class imbalance — inpainted regions are typically 10–20% of pixels):

```
FL(p, y) = -y * alpha * (1 - p)^gamma * log(p)
           - (1-y) * (1-alpha) * p^gamma * log(1-p)
```
`alpha = 0.25`, `gamma = 2.0` (standard focal loss parameters).

### Training Schedule

| Phase | Epochs | LR | Frozen |
|-------|--------|----|--------|
| Segmentation head only | 1–15 | 1e-3 | CLIP encoder |
| Last 4 CLIP blocks | 16–40 | 5e-5 | Blocks 0–7 |
| All CLIP blocks | 41–50 | 1e-5 | None |

Expected AUC on held-out synthetic inpainting test: >0.94.

### Notes on CLIP Fine-tuning

CLIP's visual representations are strong for semantic understanding but are not inherently sensitive to low-level forensic signals (e.g., noise statistics, DCT artifacts). The gradual unfreezing strategy ensures the segmentation head first learns to exploit CLIP's existing semantic understanding before the encoder is asked to adapt its representations. Catastrophic forgetting is mitigated by the very low LR in the all-unfreeze phase.

---

## Ensemble Weight Learning

After all four models are trained:

1. Run all four models on the **validation** split (not test) and collect per-pixel probability maps.
2. For each image, stack the 4 maps into a (4, H, W) tensor.
3. Optimize `weights = [w1, w2, w3, w4]` (constrained to sum to 1, each ≥ 0) to maximize pixel-level F1 on the validation set.

```python
from scipy.optimize import minimize

def neg_f1(w):
    fused = sum(wi * mi for wi, mi in zip(w, val_maps))
    pred_binary = (fused > 0.65).astype(float)
    tp = (pred_binary * val_masks).sum()
    fp = (pred_binary * (1 - val_masks)).sum()
    fn = ((1 - pred_binary) * val_masks).sum()
    f1 = 2*tp / (2*tp + fp + fn + 1e-8)
    return -f1

result = minimize(neg_f1, x0=[0.25, 0.25, 0.25, 0.25],
                  method="L-BFGS-B",
                  bounds=[(0, 1)] * 4,
                  constraints={"type": "eq", "fun": lambda w: sum(w) - 1})
```

Expected result: approximately `[0.35, 0.30, 0.20, 0.15]` (PatchForensic highest due to broad coverage).

---

## Evaluation Metrics

All models are evaluated on the **test** split using:

| Metric | Description |
|--------|-------------|
| Pixel F1 | Harmonic mean of pixel-level precision and recall at threshold 0.65 |
| AUC-ROC | Area under the ROC curve across all thresholds |
| AUC-PR | Area under the precision-recall curve (more informative under class imbalance) |
| Region IoU | Mean intersection-over-union of predicted vs ground truth regions |
| False Positive Rate on Authentic | % of unmanipulated images flagged (target < 5%) |

Results are written to `reports/eval/<model_name>_<date>.json` and aggregated in `reports/eval/summary.csv`.
