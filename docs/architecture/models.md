# Models

## Base class: `ForensicModel`

All four models extend `certainaity.models.base.ForensicModel`. The base class handles:

- **Lazy weight loading**: weights are loaded on the first `predict()` call, not at import time.
- **Padding**: images are padded to multiples of 32 before inference and cropped back.
- **Device management**: `_to_tensor()` moves inputs to the correct device.

Subclasses implement two methods:

```python
def _load_weights(self) -> None:
    """Load model weights from self._weights_dir."""

def _forward(self, x: torch.Tensor) -> torch.Tensor:
    """(1, 3, H, W) float32 → (1, 1, H, W) float32 in [0, 1]."""
```

## PatchForensic

**Weight file**: `patchforensic_v1.pt`  
**Default ensemble weight**: 0.35

A 9-layer fully-convolutional network trained on the CASIA v2 + DEFACTO datasets with a curriculum training strategy (easy negatives first, hard negatives after epoch 30). The decoder uses skip connections and a sigmoid output head.

Training details: 90 epochs, AdamW lr=1e-4, focal loss γ=2, MLflow experiment `certainaity/patchforensic/v1`.

## MantraNet

**Weight file**: `mantranet_v1.pt`  
**Default ensemble weight**: 0.30

A fine-tuned version of the original MantraNet architecture (VGG-16/BN backbone + local anomaly detection head). Fine-tuned on DEFACTO for 60 epochs with gradual backbone unfreezing.

## SPSL

**Weight file**: `spsl_v1.pt`  
**Default ensemble weight**: 0.20

Siamese ResNet-50 trained with contrastive loss to distinguish spliced image pairs. At inference time, a FAISS flat-L2 index over training set embeddings provides nearest-neighbour similarity scores that are converted to a pixel-level splicing map.

## InpaintingDetector

**Weight file**: `inpainting_v1.pt`  
**Default ensemble weight**: 0.15

CLIP ViT-B/32 with a fine-tuned segmentation head, trained on a synthetic dataset of 22,000 AI-inpainted images generated with Stable Diffusion v1.5 and v2.1. Specialised for detecting regions filled by text-to-image models.

## Ensemble fusion

Weights are configurable and can be optimized with `scripts/optimize_ensemble.py`:

```bash
python scripts/optimize_ensemble.py \
  --val-manifest data/processed/val/manifest.jsonl \
  --output weights/ensemble_weights.json
```

The optimizer uses L-BFGS-B to maximise F1 on the validation set. Load optimized weights at runtime:

```python
ensemble.load_optimized_weights(Path("weights/ensemble_weights.json"))
```
