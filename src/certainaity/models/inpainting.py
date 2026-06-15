"""InpaintingDetector: CLIP ViT-B/32 + segmentation head stub."""

from __future__ import annotations

from pathlib import Path

from certainaity.exceptions import WeightsNotFoundError
from certainaity.models.base import ForensicModel, ModelName

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# CLIP ViT-B/32 operates on 224×224 images divided into 16×16 patches → 14×14 grid.
_CLIP_SIZE = 224
_PATCH_GRID = 14     # 224 // 16 = 14
_CLIP_EMB_DIM = 768  # ViT-B/32 transformer hidden dimension


class _SegmentationHead(nn.Module):
    """Per-patch linear projection → sigmoid → bilinear upsample to full resolution.

    Accepts the 196 (14×14) non-CLS patch tokens from CLIP's last hidden state
    and produces a per-pixel manipulation probability map at the original image size.
    """

    def __init__(self, emb_dim: int = _CLIP_EMB_DIM) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        patch_tokens: "torch.Tensor",
        target_size: tuple[int, int],
    ) -> "torch.Tensor":
        """
        Args:
            patch_tokens: (B, N, D) — N = _PATCH_GRID² = 196
            target_size:  (H, W) to upsample the output map to

        Returns:
            (B, 1, H, W) float32 sigmoid probability map
        """
        B = patch_tokens.shape[0]
        # (B, N, D) → (B, N, 1) via per-token linear projection
        scores = self.proj(patch_tokens)
        # Reshape to spatial grid: (B, 1, 14, 14)
        scores = scores.view(B, 1, _PATCH_GRID, _PATCH_GRID)
        # Bilinear upsample to original image resolution
        out = F.interpolate(scores, size=target_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(out)


class _InpaintingDetectorModel(nn.Module):
    """CLIP ViT-B/32 visual encoder + lightweight segmentation head.

    CLIP parameters are frozen initially.  The training script gradually
    unfreezes the last N transformer encoder layers via
    :func:`unfreeze_clip_layers`.

    Input:  (B, 3, H, W) float32 in [0, 1]
    Output: (B, 1, H, W) float32 sigmoid probability map
    """

    def __init__(self) -> None:
        super().__init__()
        from transformers import CLIPVisionModel
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        # All CLIP parameters frozen until gradual unfreezing during training.
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.head = _SegmentationHead(_CLIP_EMB_DIM)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        H, W = x.shape[-2:]
        # Resize to CLIP input resolution.
        x_resized = F.interpolate(
            x, size=(_CLIP_SIZE, _CLIP_SIZE), mode="bilinear", align_corners=False
        )
        # CLIP expects pixel values in [−1, 1].
        x_clip = x_resized * 2.0 - 1.0
        outputs = self.clip(pixel_values=x_clip)
        # last_hidden_state: (B, N+1, D) — index 0 is the CLS token.
        patch_tokens = outputs.last_hidden_state[:, 1:, :]   # (B, 196, 768)
        return self.head(patch_tokens, target_size=(H, W))

    def unfreeze_clip_layers(self, num_layers: int) -> None:
        """Unfreeze the last ``num_layers`` transformer encoder layers in CLIP."""
        encoder_layers = self.clip.vision_model.encoder.layers
        for layer in encoder_layers[-num_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)


class InpaintingDetector(ForensicModel):
    """CLIP ViT-B/32 fine-tuned with focal loss for AI-inpainting detection.

    Weights are loaded lazily from ``weights/inpainting_detector_clip.pth`` on
    first call to :meth:`predict`. Download with ``scripts/download_weights.py``.
    Note: ``transformers`` (HuggingFace) is required at inference time.
    """

    WEIGHT_FILE = "inpainting_detector_clip.pth"
    MODEL_NAME = ModelName.INPAINTING_DETECTOR

    def __init__(self, weights_dir: Path, device: str = "cpu") -> None:
        super().__init__(weights_dir, device)
        if _TORCH_AVAILABLE:
            self._model: "_InpaintingDetectorModel | None" = None

    def _load_weights(self) -> None:
        import torch
        if not self.weight_path.exists():
            raise WeightsNotFoundError(
                f"InpaintingDetector weights not found at {self.weight_path}. "
                "Run: python scripts/download_weights.py --models inpainting_detector_clip.pth"
            )
        self._model = _InpaintingDetectorModel().to(self._device)
        state = torch.load(self.weight_path, map_location=self._device, weights_only=True)
        # Load only the head weights; CLIP base weights come from HuggingFace.
        self._model.load_state_dict(state, strict=False)
        self._model.eval()

    def _forward(self, x: "torch.Tensor") -> "torch.Tensor":
        assert self._model is not None
        return self._model(x)
