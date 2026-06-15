"""Handcrafted forensic feature extraction."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass

import numpy as np
from PIL import Image

from certainaity.config import get_settings
from certainaity.features.cfa import compute_cfa_map
from certainaity.features.dct import compute_dct_similarity
from certainaity.features.ela import compute_ela
from certainaity.features.noise import compute_noise_map


@dataclass
class FeatureMaps:
    """All four handcrafted feature maps, each normalized to [0, 1]."""

    ela: np.ndarray
    noise: np.ndarray
    cfa: np.ndarray
    dct: np.ndarray

    @property
    def stacked(self) -> np.ndarray:
        """Return (4, H, W) array at the resolution of the ELA map."""
        return np.stack([self.ela, self.noise, self.cfa, self.dct], axis=0)


def extract_features(image: Image.Image) -> FeatureMaps:
    """Run all four feature extractors in parallel and return their maps."""
    settings = get_settings()

    tasks = {
        "ela": (compute_ela, (image, settings.ela_quality)),
        "noise": (compute_noise_map, (image,)),
        "cfa": (compute_cfa_map, (image,)),
        "dct": (compute_dct_similarity, (image, settings.dct_block_size)),
    }

    results: dict[str, np.ndarray] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=settings.feature_workers) as pool:
        futures = {
            pool.submit(fn, *args): name for name, (fn, args) in tasks.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            results[name] = future.result()

    return FeatureMaps(
        ela=results["ela"],
        noise=results["noise"],
        cfa=results["cfa"],
        dct=results["dct"],
    )
