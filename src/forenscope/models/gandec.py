"""GAN-generated image detector.

Detects images produced by generative models including StyleGAN2/3,
DALL-E 2/3, Midjourney v5/v6, Stable Diffusion, and other latent
diffusion models.

Architecture: CLIP ViT-L/14 backbone with a two-head output:
  - Global head: image-level binary GAN/real classification (sigmoid)
  - Local head: 16×16 patch-level authenticity map (upsampled to input resolution)

Unlike the InpaintingDetector which flags regions within otherwise authentic
photos, the GANDetector classifies whether the entire image is AI-generated.
The local head produces a localization map to highlight which image regions
most strongly exhibit GAN fingerprints (useful for partial composites).

Training data:
  - Real: COCO 2017 val (5,000 images), FFHQ (5,000 images)
  - GAN: StyleGAN2-ADA (5,000 faces), DALL-E 3 (3,000 scenes),
    Midjourney v6 (3,000 scenes), SD 2.1 (4,000 scenes)
  - Total: 25,000 images; 80/10/10 train/val/test split

Target metric: AUC ≥ 0.97 on the held-out test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from forenscope.exceptions import WeightsNotFoundError
from forenscope.models.base import ForensicModel, ModelName


class _GANDetectorNet:
    """Placeholder network used when torch is available but weights are absent."""

    def __init__(self) -> None:
        pass

    def forward(self, x: object) -> tuple[object, object]:
        raise WeightsNotFoundError(
            "GANDetector weights not found. "
            "Download with: python scripts/download_weights.py --model gandec"
        )


class GANDetector(ForensicModel):
    """Detect fully AI-generated images (StyleGAN, DALL-E, Midjourney, SD).

    Returns a (H, W) float32 map in [0, 1]; values close to 1.0 indicate
    regions that strongly exhibit GAN-generation fingerprints.

    For fully synthetic images the map will be uniformly high; for composites
    (an AI-generated element pasted into a real photo) the map will be
    elevated only in the synthetic region.

    Example::

        detector = GANDetector(Path("weights"))
        prob_map = detector.predict(image_np)
        is_gan = prob_map.mean() > 0.5
    """

    WEIGHT_FILE = "gandec_v1.pt"

    def _load_weights(self) -> None:
        weight_path = self._weights_dir / self.WEIGHT_FILE
        if not weight_path.exists():
            raise WeightsNotFoundError(
                f"GANDetector weight file not found: {weight_path}. "
                "Download with: python scripts/download_weights.py --model gandec"
            )

        try:
            import torch

            state = torch.load(str(weight_path), map_location=self._device)
            self._model = _build_gandec_net().to(self._device)
            self._model.load_state_dict(state)
            self._model.eval()
        except Exception as exc:
            raise WeightsNotFoundError(
                f"Failed to load GANDetector weights from {weight_path}: {exc}"
            ) from exc

    def _forward(self, x: object) -> object:
        import torch

        _global, local_map = self._model(x)  # type: ignore[misc]
        # Upsample the 16×16 patch map to input resolution with bilinear interpolation.
        import torch.nn.functional as F

        _, _, H, W = x.shape  # type: ignore[union-attr]
        upsampled = F.interpolate(local_map, size=(H, W), mode="bilinear", align_corners=False)
        return upsampled  # (1, 1, H, W)


def _build_gandec_net() -> object:
    """Build the CLIP ViT-L/14 + dual-head GAN detection network.

    Returns a torch.nn.Module. Raises ImportError if torch/transformers are
    not installed.
    """
    import torch
    import torch.nn as nn

    try:
        from transformers import CLIPVisionModel
    except ImportError as exc:
        raise ImportError(
            "GANDetector requires the `transformers` package. "
            "Install with: pip install -e '.[worker]'"
        ) from exc

    class GANDetectorNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

            # Global classification head (image-level)
            hidden = self.backbone.config.hidden_size  # 1024 for ViT-L/14
            self.global_head = nn.Sequential(
                nn.Linear(hidden, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

            # Local segmentation head (patch-level → 16×16 map)
            self.local_head = nn.Sequential(
                nn.Linear(hidden, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = self.backbone(pixel_values=x)
            # last_hidden_state: (B, 1 + num_patches, hidden)
            cls_token = outputs.last_hidden_state[:, 0, :]      # (B, hidden)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B, 256, hidden)

            global_score = self.global_head(cls_token)           # (B, 1)
            local_scores = self.local_head(patch_tokens)         # (B, 256, 1)

            # Reshape patch scores to 16×16 spatial map.
            B = x.shape[0]
            local_map = local_scores.squeeze(-1).view(B, 1, 16, 16)  # (B, 1, 16, 16)

            return global_score, local_map

    return GANDetectorNet()
