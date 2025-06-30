"""SPSL: Siamese ResNet-50 stub for copy-move and splicing detection."""

from __future__ import annotations

from pathlib import Path

from forenscope.exceptions import WeightsNotFoundError
from forenscope.models.base import ForensicModel, ModelName

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_PATCH_SIZE = 32     # sliding-window patch size during inference
_PATCH_STRIDE = 16   # stride for patch extraction
_EMB_DIM = 256       # projection head output dimension


class _ProjectionHead(nn.Module):
    """Maps ResNet-50 features to a normalised embedding space for contrastive training."""

    def __init__(self, in_features: int = 1024, out_features: int = _EMB_DIM) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return F.normalize(self.layers(x), dim=1)


class _SPSLBackbone(nn.Module):
    """ResNet-50 truncated after layer3 + projection head.

    Accepts (B, 3, 32, 32) patch tensors and returns (B, _EMB_DIM) L2-normalised
    embeddings. Using layer3 (rather than layer4) keeps the receptive field tight
    enough to capture local copy-move patterns without over-smoothing.
    """

    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as tvm
        resnet = tvm.resnet50(weights=None)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3   # output: (B, 1024, 4, 4) for 32×32 input
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = _ProjectionHead(1024, _EMB_DIM)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Placeholder: full forward pass requires trained weights.
        raise NotImplementedError(
            "SPSL forward pass requires trained weights. "
            "Run scripts/train_spsl.py to produce a checkpoint."
        )


class _SPSLModel(nn.Module):
    """Full SPSL model for inference-time anomaly scoring.

    Inference strategy: extract overlapping patches via a sliding window, embed
    each patch with the shared backbone, then score every patch by its cosine
    distance to its k nearest neighbours in the same image.  Patches with low
    self-similarity are flagged as anomalous (possible manipulation boundary).

    A FAISS index (``faiss_index_path``) built from authentic training patches
    can optionally be used for cross-image retrieval.  When no index is loaded
    the model falls back to within-image similarity.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _SPSLBackbone()
        # Optional FAISS index for cross-image nearest-neighbour retrieval.
        self._faiss_index = None   # type: ignore[assignment]

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        raise NotImplementedError(
            "SPSL forward pass requires trained weights and, optionally, a "
            "FAISS index. Run scripts/train_spsl.py to produce both."
        )

    def load_faiss_index(self, index_path: Path) -> None:
        """Load a pre-built FAISS flat-IP index from disk."""
        import faiss  # type: ignore[import]
        self._faiss_index = faiss.read_index(str(index_path))


class SPSL(ForensicModel):
    """Siamese ResNet-50 trained with contrastive loss on patch pairs.

    Weights are loaded lazily from ``weights/spsl_siamese.pth`` on first call
    to :meth:`predict`. A matching FAISS index (``weights/spsl_faiss.index``)
    is loaded automatically if present.  Download with ``scripts/download_weights.py``.
    """

    WEIGHT_FILE = "spsl_siamese.pth"
    MODEL_NAME = ModelName.SPSL

    def __init__(self, weights_dir: Path, device: str = "cpu") -> None:
        super().__init__(weights_dir, device)
        if _TORCH_AVAILABLE:
            self._model: "_SPSLModel | None" = None

    def _load_weights(self) -> None:
        import torch
        if not self.weight_path.exists():
            raise WeightsNotFoundError(
                f"SPSL weights not found at {self.weight_path}. "
                "Run: python scripts/download_weights.py --models spsl_siamese.pth"
            )
        self._model = _SPSLModel().to(self._device)
        state = torch.load(self.weight_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
        # Load FAISS index if co-located with the checkpoint.
        index_path = self._weights_dir / "spsl_faiss.index"
        if index_path.exists():
            self._model.load_faiss_index(index_path)

    def _forward(self, x: "torch.Tensor") -> "torch.Tensor":
        assert self._model is not None
        return self._model(x)
