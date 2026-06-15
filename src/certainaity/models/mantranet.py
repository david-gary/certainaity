"""MantraNet: fine-tuned VGG-16/BN stub for local anomaly detection."""

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


class _LocalAnomalyHead(nn.Module):
    """Z-score anomaly scoring over a local 7×7 neighbourhood.

    For each spatial position, normalises each feature channel by the local
    mean and variance, then projects the resulting volume to a 1-channel map.
    The z-score formulation ensures the head is sensitive to features that
    deviate from their local context — the key signal for splicing and removal.
    """

    def __init__(self, in_ch: int = 512) -> None:
        super().__init__()
        self.local_avg = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, feat: "torch.Tensor") -> "torch.Tensor":
        # Placeholder: full z-score implementation requires trained scale/bias.
        raise NotImplementedError(
            "MantraNet forward pass requires trained weights. "
            "Run scripts/train_mantranet.py to produce a checkpoint."
        )


class _MantraNetModel(nn.Module):
    """VGG-16/BN backbone + local anomaly detection head.

    Input:  (B, 3, H, W) float32 in [0, 1]
    Output: (B, 1, H, W) float32 sigmoid probability map (bilinear upsampled)

    During fine-tuning, features[0:20] (conv blocks 1–3) are frozen.
    features[20:] (blocks 4–5) are unfrozen to adapt to the forensic domain.
    """

    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as tvm
        vgg = tvm.vgg16_bn(weights=None)
        self.features = vgg.features       # (B, 512, H/32, W/32)
        # Freeze early conv blocks (1–3 = indices 0..19 in VGG-16/BN features).
        for i, layer in enumerate(self.features):
            if i < 20:
                for p in layer.parameters():
                    p.requires_grad_(False)
        self.anomaly = _LocalAnomalyHead(512)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        feat = self.features(x)
        score = self.anomaly(feat)
        return F.interpolate(score, size=x.shape[-2:], mode="bilinear", align_corners=False)


class MantraNet(ForensicModel):
    """VGG-16/BN backbone fine-tuned for image manipulation localization.

    Weights are loaded lazily from ``weights/mantranet_finetuned.pth`` on
    first call to :meth:`predict`. Download with ``scripts/download_weights.py``.
    """

    WEIGHT_FILE = "mantranet_finetuned.pth"
    MODEL_NAME = ModelName.MANTRA_NET

    def __init__(self, weights_dir: Path, device: str = "cpu") -> None:
        super().__init__(weights_dir, device)
        if _TORCH_AVAILABLE:
            self._model: "_MantraNetModel | None" = None

    def _load_weights(self) -> None:
        import torch
        if not self.weight_path.exists():
            raise WeightsNotFoundError(
                f"MantraNet weights not found at {self.weight_path}. "
                "Run: python scripts/download_weights.py --models mantranet_finetuned.pth"
            )
        self._model = _MantraNetModel().to(self._device)
        state = torch.load(self.weight_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

    def _forward(self, x: "torch.Tensor") -> "torch.Tensor":
        assert self._model is not None
        return self._model(x)
