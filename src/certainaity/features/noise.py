"""Wavelet-based noise variance estimation."""

from __future__ import annotations

import numpy as np
import pywt
from PIL import Image


def compute_noise_map(image: Image.Image, block_size: int = 32) -> np.ndarray:
    """Estimate local noise variance using the HH wavelet subband.

    Genuine camera images have spatially consistent sensor noise. Spliced regions
    typically originate from a different sensor or acquisition condition, producing
    discontinuities in the local noise variance map.

    The image is decomposed via a one-level db8 DWT. The HH (diagonal) detail
    subband captures high-frequency noise. We then estimate the variance of HH
    in non-overlapping block_size//2 × block_size//2 windows (the DWT halves each
    dimension), upsample the variance map back to the original resolution, and
    normalize to [0, 1].

    Args:
        image: RGB PIL image.
        block_size: Block size in *original* image pixels. Must be even.

    Returns:
        (H, W) float32 ndarray in [0, 1].
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    H, W = gray.shape

    _, (_, _, hh) = pywt.dwt2(gray, "db8")
    hh_h, hh_w = hh.shape

    half_block = block_size // 2
    var_map_small = np.zeros((hh_h, hh_w), dtype=np.float32)

    for y in range(0, hh_h - half_block + 1, half_block):
        for x in range(0, hh_w - half_block + 1, half_block):
            block = hh[y : y + half_block, x : x + half_block]
            var = float(np.var(block))
            var_map_small[y : y + half_block, x : x + half_block] = var

    # Upsample to original resolution using nearest-neighbour (preserves block edges).
    scale_y = H / hh_h
    scale_x = W / hh_w
    rows = (np.arange(H) / scale_y).astype(int).clip(0, hh_h - 1)
    cols = (np.arange(W) / scale_x).astype(int).clip(0, hh_w - 1)
    var_map = var_map_small[np.ix_(rows, cols)]

    max_var = float(var_map.max())
    if max_var < 1e-10:
        return var_map
    return (var_map / max_var).astype(np.float32)


def noise_summary(noise_map: np.ndarray) -> str:
    """Return a human-readable summary of a noise map for report text."""
    std = float(noise_map.std())
    high_fraction = float((noise_map > 0.7).mean())
    return (
        f"Noise variance std={std:.4f} (high std suggests inconsistency), "
        f"high-variance fraction={high_fraction:.2%}"
    )
