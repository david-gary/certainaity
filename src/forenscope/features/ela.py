"""Error Level Analysis (ELA) feature extractor."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def compute_ela(image: Image.Image, quality: int = 75) -> np.ndarray:
    """Compute an ELA heatmap highlighting regions saved at a different JPEG quality.

    Works by resaving the image at a fixed quality, then measuring the absolute
    per-pixel difference. Authentic regions—already compressed at that quality—
    show low residual; spliced or manipulated regions that were compressed at a
    different quality (or not compressed at all) show high residual.

    Args:
        image: RGB PIL image.
        quality: JPEG resave quality (lower = more sensitive; 75 is conventional).

    Returns:
        (H, W) float32 ndarray in [0, 1]. Higher values indicate likely manipulation.
    """
    rgb = image.convert("RGB")

    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    orig = np.asarray(rgb, dtype=np.float32)
    resv = np.asarray(resaved, dtype=np.float32)

    diff = np.abs(orig - resv)
    # Max across channels so a single-channel spike doesn't get diluted.
    ela_map = diff.max(axis=2)

    # Normalize by 99th percentile to suppress isolated hot pixels.
    p99 = float(np.percentile(ela_map, 99))
    if p99 < 1.0:
        return np.zeros_like(ela_map)

    return np.clip(ela_map / p99, 0.0, 1.0).astype(np.float32)


def ela_summary(ela_map: np.ndarray) -> str:
    """Return a human-readable summary of an ELA map for report text."""
    mean = float(ela_map.mean())
    p95 = float(np.percentile(ela_map, 95))
    high_fraction = float((ela_map > 0.5).mean())
    return (
        f"ELA mean={mean:.3f}, p95={p95:.3f}, "
        f"high-signal fraction={high_fraction:.2%}"
    )
