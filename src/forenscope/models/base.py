"""Abstract base class for ForenScope forensic detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class ModelName(StrEnum):
    PATCH_FORENSIC = "PatchForensic"
    MANTRA_NET = "MantraNet"
    SPSL = "SPSL"
    INPAINTING_DETECTOR = "InpaintingDetector"
    GAN_DETECTOR = "GANDetector"


class ForensicModel(ABC):
    """Contract that every forensic detection model must satisfy.

    Subclasses are responsible for:
      - Loading weights from a well-known path in ``weights/``.
      - Accepting a (1, 3, H, W) float32 tensor in [0, 1] (or a PIL image).
      - Returning a (H, W) float32 numpy array in [0, 1] as the
        per-pixel manipulation probability map.

    The base class handles:
      - Lazy weight loading on first call to ``predict``.
      - Padding images to multiples of 32 and cropping back.
      - Moving inputs to the correct device.
    """

    #: Override in subclass; corresponds to a file under weights/.
    WEIGHT_FILE: str = ""

    def __init__(self, weights_dir: Path, device: str = "cpu") -> None:
        self._weights_dir = weights_dir
        self._device = device
        self._loaded = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, image: "np.ndarray") -> "np.ndarray":
        """Run inference on a (H, W, 3) uint8 or (3, H, W) float32 array.

        Returns:
            (H, W) float32 ndarray in [0, 1].
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for model inference. "
                "Install the [worker] extras: pip install -e '.[worker]'"
            )

        if not self._loaded:
            self._load_weights()
            self._loaded = True

        import torch
        tensor = self._to_tensor(image)           # (1, 3, H, W)
        padded, (pad_h, pad_w) = self._pad(tensor)
        with torch.inference_mode():
            out = self._forward(padded)           # (1, 1, H_pad, W_pad)
        out_np = out.squeeze().cpu().numpy()      # (H_pad, W_pad)
        return out_np[: out_np.shape[0] - pad_h, : out_np.shape[1] - pad_w]

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_weights(self) -> None:
        """Load model weights from self._weights_dir into self._model."""

    @abstractmethod
    def _forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run the model forward pass.

        Args:
            x: (1, 3, H, W) float32 tensor on self._device.

        Returns:
            (1, 1, H, W) float32 tensor in [0, 1].
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, image: "np.ndarray") -> "torch.Tensor":
        import torch
        import numpy as np

        if image.ndim == 3 and image.shape[2] == 3:
            # (H, W, 3) uint8 or float
            arr = image.astype(np.float32)
            if arr.max() > 1.0:
                arr /= 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        elif image.ndim == 3 and image.shape[0] == 3:
            # (3, H, W) float
            t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected image shape {image.shape}")

        return t.to(self._device)

    @staticmethod
    def _pad(
        x: "torch.Tensor", multiple: int = 32
    ) -> tuple["torch.Tensor", tuple[int, int]]:
        import torch.nn.functional as F

        _, _, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        # Pad bottom and right with reflection.
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return padded, (pad_h, pad_w)

    @property
    def device(self) -> str:
        return self._device

    @property
    def weight_path(self) -> Path:
        return self._weights_dir / self.WEIGHT_FILE
