"""PyTorch dataset class for forensic patch data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class ForensicDataset:
    """Map-style dataset over a JSONL manifest of image/mask patch pairs.

    Each line in the manifest is a JSON object with at minimum:
        image_path (str): path to a (256, 256, 3) uint8 .npy array
        mask_path  (str): path to a (256, 256)    uint8 {0, 1} .npy array

    Optional fields used for sampling or analysis:
        source_dataset, manipulation_type, manipulated_fraction, patch_origin

    Args:
        manifest_path: Path to the JSONL manifest file.
        root: If provided, image_path and mask_path in the manifest are
              interpreted relative to this directory.
        transform: Optional callable applied to the augmented ``(image, mask)``
                   albumentations-style dict before tensor conversion.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        root: Path | str | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._manifest = Path(manifest_path)
        self._root = Path(root) if root else None
        self.transform = transform
        self._records: list[dict[str, Any]] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        with self._manifest.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        rec = self._records[idx]
        image_path = self._resolve(rec["image_path"])
        mask_path = self._resolve(rec["mask_path"])

        image: np.ndarray = np.load(image_path)  # (256, 256, 3) uint8
        mask: np.ndarray = np.load(mask_path)    # (256, 256)    uint8

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if _TORCH_AVAILABLE:
            import torch
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            msk_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
            return img_tensor, msk_tensor

        # Without torch return raw numpy arrays (useful for testing).
        return image.astype(np.float32) / 255.0, mask.astype(np.float32)

    def _resolve(self, rel_path: str) -> Path:
        p = Path(rel_path)
        return self._root / p if self._root else p

    def record(self, idx: int) -> dict[str, Any]:
        """Return the raw manifest record for an index."""
        return self._records[idx]

    @classmethod
    def manipulation_types(cls) -> list[str]:
        return ["splicing", "copy_move", "removal", "ai_inpainting"]
