"""Ensemble fusion of all four Certainaity models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import label as cc_label

from certainaity.config import get_settings
from certainaity.models.base import ForensicModel, ModelName
from certainaity.models.gandec import GANDetector
from certainaity.models.inpainting import InpaintingDetector
from certainaity.models.mantranet import MantraNet
from certainaity.models.patchforensic import PatchForensic
from certainaity.models.spsl import SPSL

# Default ensemble weights from architecture.md §4.3 ensemble fusion.
# GANDetector weight is 0 by default so it does not affect v1.0 results;
# set to a positive value (e.g. 0.10) once gandec_v1.pt weights are available.
_DEFAULT_WEIGHTS: dict[ModelName, float] = {
    ModelName.PATCH_FORENSIC: 0.35,
    ModelName.MANTRA_NET: 0.30,
    ModelName.SPSL: 0.20,
    ModelName.INPAINTING_DETECTOR: 0.15,
    ModelName.GAN_DETECTOR: 0.0,
}


@dataclass
class LocalizationResult:
    """Output of :meth:`Ensemble.localize`."""

    heatmap: np.ndarray           # (H, W) float32 in [0, 1] — weighted fusion map
    binary_mask: np.ndarray       # (H, W) bool — thresholded, noise-cleaned mask
    num_regions: int              # number of connected components
    region_labels: np.ndarray     # (H, W) int — 0 = background, 1…N = regions
    model_maps: dict[str, np.ndarray] = field(default_factory=dict)


class Ensemble:
    """Weighted fusion of PatchForensic, MantraNet, SPSL, and InpaintingDetector.

    Model weights are loaded lazily on the first :meth:`predict` call.
    Use :meth:`load_optimized_weights` to override the defaults with weights
    produced by ``scripts/optimize_ensemble.py``.

    Example::

        ensemble = Ensemble(Path("weights"), device="cuda")
        result = ensemble.localize(image_np)
        print(result.num_regions, result.heatmap.max())
    """

    def __init__(
        self,
        weights_dir: Path,
        device: str = "cpu",
        model_weights: dict[ModelName, float] | None = None,
    ) -> None:
        self._weights_dir = weights_dir
        self._device = device
        self._model_weights: dict[ModelName, float] = (
            dict(model_weights) if model_weights else dict(_DEFAULT_WEIGHTS)
        )
        self._models: dict[ModelName, ForensicModel] = {
            ModelName.PATCH_FORENSIC: PatchForensic(weights_dir, device),
            ModelName.MANTRA_NET: MantraNet(weights_dir, device),
            ModelName.SPSL: SPSL(weights_dir, device),
            ModelName.INPAINTING_DETECTOR: InpaintingDetector(weights_dir, device),
            ModelName.GAN_DETECTOR: GANDetector(weights_dir, device),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return weighted-average heatmap.

        Args:
            image: (H, W, 3) uint8 or float32 in [0, 1]

        Returns:
            (H, W) float32 fusion map in [0, 1]
        """
        total = sum(self._model_weights.values())
        fused = np.zeros(image.shape[:2], dtype=np.float32)
        for name, model in self._models.items():
            w = self._model_weights.get(name, 0.0)
            if w == 0.0:
                continue
            fused += (w / total) * model.predict(image)
        return fused

    def localize(
        self,
        image: np.ndarray,
        threshold: float | None = None,
        min_region_px: int = 64,
        return_model_maps: bool = False,
    ) -> LocalizationResult:
        """Run ensemble inference and return labelled manipulation regions.

        Args:
            image: (H, W, 3) uint8 or float32 in [0, 1]
            threshold: Override the ``ensemble_threshold`` config value.
            min_region_px: Connected components smaller than this (in pixels)
                are removed as noise.
            return_model_maps: If True, include per-model heatmaps in the result.

        Returns:
            :class:`LocalizationResult`
        """
        settings = get_settings()
        if threshold is None:
            threshold = settings.ensemble_threshold

        total = sum(self._model_weights.values())
        model_maps: dict[str, np.ndarray] = {}
        fused = np.zeros(image.shape[:2], dtype=np.float32)

        for name, model in self._models.items():
            w = self._model_weights.get(name, 0.0)
            prob_map = model.predict(image)
            fused += (w / total) * prob_map
            if return_model_maps:
                model_maps[name.value] = prob_map

        binary = fused >= threshold

        # Remove small noise regions.
        labeled, _ = cc_label(binary)
        for region_id in range(1, labeled.max() + 1):
            if (labeled == region_id).sum() < min_region_px:
                binary[labeled == region_id] = False

        labeled, num_features = cc_label(binary)
        return LocalizationResult(
            heatmap=fused,
            binary_mask=binary,
            num_regions=num_features,
            region_labels=labeled.astype(np.int32),
            model_maps=model_maps,
        )

    # ------------------------------------------------------------------
    # Weight optimisation
    # ------------------------------------------------------------------

    @classmethod
    def optimize_weights(
        cls,
        predictions: dict[ModelName, np.ndarray],
        ground_truth: np.ndarray,
    ) -> dict[ModelName, float]:
        """Find optimal ensemble weights by maximising F1 on a validation set.

        Uses scipy L-BFGS-B to minimise negative F1 over the simplex.

        Args:
            predictions: ``{model_name: (N, H, W) float32 probability maps}``
            ground_truth: ``(N, H, W)`` binary masks

        Returns:
            Optimised weight dict normalised to sum to 1.
        """
        from scipy.optimize import minimize

        keys = list(predictions.keys())
        preds = np.stack([predictions[k] for k in keys], axis=0)  # (M, N, H, W)
        gt = (ground_truth > 0.5).astype(np.float32)

        def _neg_f1(raw_w: np.ndarray) -> float:
            w = np.abs(raw_w) / (np.abs(raw_w).sum() + 1e-9)
            fused = np.einsum("m,mnhw->nhw", w, preds)
            binary = fused >= 0.5
            tp = float((binary & (gt > 0.5)).sum())
            fp = float((binary & (gt <= 0.5)).sum())
            fn = float((~binary & (gt > 0.5)).sum())
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            return -f1

        x0 = np.array([_DEFAULT_WEIGHTS.get(k, 0.25) for k in keys], dtype=np.float64)
        result = minimize(_neg_f1, x0, method="L-BFGS-B",
                          bounds=[(0.0, 1.0)] * len(keys))
        opt_w = np.abs(result.x) / (np.abs(result.x).sum() + 1e-9)
        return {k: float(opt_w[i]) for i, k in enumerate(keys)}

    def load_optimized_weights(self, weights_path: Path) -> None:
        """Load ensemble weights from a JSON file produced by optimize_ensemble.py."""
        data = json.loads(weights_path.read_text())
        self._model_weights = {ModelName(k): float(v) for k, v in data.items()}

    def save_weights(self, path: Path) -> None:
        """Persist the current ensemble weights to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({k.value: v for k, v in self._model_weights.items()}, indent=2)
        )

    @property
    def model_weights(self) -> dict[ModelName, float]:
        return dict(self._model_weights)
