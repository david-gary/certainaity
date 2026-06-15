"""PatchForensic: 9-layer FCN stub for pixel-level manipulation detection."""

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


class _EncBlock(nn.Module):
    """Two-conv encoder block: Conv→BN→ReLU→Conv→BN→ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.conv(x)


class _DecBlock(nn.Module):
    """One-conv decoder block: bilinear upsample → cat(skip) → Conv→BN→ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: "torch.Tensor", skip: "torch.Tensor"
    ) -> "torch.Tensor":
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class _PatchForensicNet(nn.Module):
    """
    9 conv layers total:
        enc1 (2) + enc2 (2) + enc3 (2) + dec2 (1) + dec1 (1) + head (1) = 9

    Input:  (B, 3, H, W) float32 in [0, 1]
    Output: (B, 1, H, W) float32 sigmoid probability map
    """

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = _EncBlock(3, 32)
        self.enc2 = _EncBlock(32, 64)
        self.enc3 = _EncBlock(64, 128)
        self.dec2 = _DecBlock(128 + 64, 64)
        self.dec1 = _DecBlock(64 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        d = self.dec2(s3, s2)
        d = self.dec1(d, s1)
        return torch.sigmoid(self.head(d))


class PatchForensic(ForensicModel):
    """9-layer FCN trained on CASIA v2, DEFACTO, NIST 16, and COVERage.

    Weights are loaded lazily from ``weights/patchforensic_v2.pth`` on first
    call to :meth:`predict`. Download with ``scripts/download_weights.py``.
    """

    WEIGHT_FILE = "patchforensic_v2.pth"
    MODEL_NAME = ModelName.PATCH_FORENSIC

    def __init__(self, weights_dir: Path, device: str = "cpu") -> None:
        super().__init__(weights_dir, device)
        if _TORCH_AVAILABLE:
            self._model: "_PatchForensicNet | None" = None

    def _load_weights(self) -> None:
        import torch
        if not self.weight_path.exists():
            raise WeightsNotFoundError(
                f"PatchForensic weights not found at {self.weight_path}. "
                "Run: python scripts/download_weights.py --models patchforensic_v2.pth"
            )
        self._model = _PatchForensicNet().to(self._device)
        state = torch.load(self.weight_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

    def _forward(self, x: "torch.Tensor") -> "torch.Tensor":
        assert self._model is not None
        return self._model(x)
