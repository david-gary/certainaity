"""CFA (Color Filter Array) interpolation correlation analysis."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.signal import correlate2d


def compute_cfa_map(image: Image.Image) -> np.ndarray:
    """Detect mismatched Bayer-pattern CFA interpolation artifacts.

    Genuine camera images undergo a specific demosaicing interpolation that leaves
    predictable periodic correlations in the green channel. When an image region
    is copy-pasted from a different source—or is AI-generated—these correlations
    are absent or have a different periodicity, producing a forensic signal.

    Method:
        1. Extract the green channel and compute its local linear prediction
           residual (each pixel predicted from its four cardinal neighbours).
        2. Compute the 2-D normalized autocorrelation of the residual in
           non-overlapping 64×64 blocks.
        3. Measure the peak at (Δy=2, Δx=0) and (Δy=0, Δx=2)—the Bayer period.
           A high peak indicates consistent CFA interpolation; a low peak
           (relative to surrounding area) indicates a mismatch.

    Returns:
        (H, W) float32 ndarray in [0, 1].
        High values flag blocks where CFA correlation is *absent* (anomalous).
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    green = arr[:, :, 1]
    H, W = green.shape

    residual = _linear_prediction_residual(green)
    block_size = 64
    score_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - block_size + 1, block_size):
        for x in range(0, W - block_size + 1, block_size):
            block = residual[y : y + block_size, x : x + block_size]
            score = _bayer_correlation_score(block)
            # Invert: high score = CFA pattern present = authentic
            score_map[y : y + block_size, x : x + block_size] = 1.0 - score

    return score_map.astype(np.float32)


def _linear_prediction_residual(channel: np.ndarray) -> np.ndarray:
    """Predict each pixel as the mean of its four cardinal neighbours."""
    padded = np.pad(channel, 1, mode="reflect")
    pred = (
        padded[:-2, 1:-1]   # top
        + padded[2:, 1:-1]  # bottom
        + padded[1:-1, :-2] # left
        + padded[1:-1, 2:]  # right
    ) / 4.0
    return channel - pred


def _bayer_correlation_score(block: np.ndarray) -> float:
    """Score in [0, 1]; 1.0 = strong Bayer period-2 correlation, 0.0 = none."""
    if block.std() < 1e-6:
        return 0.5

    acf = correlate2d(block, block, mode="full", boundary="wrap")
    cy, cx = np.array(acf.shape) // 2

    # Peak at (±2, 0) and (0, ±2) indicates Bayer RGGB period.
    period_peaks = (
        abs(acf[cy + 2, cx])
        + abs(acf[cy - 2, cx])
        + abs(acf[cy, cx + 2])
        + abs(acf[cy, cx - 2])
    ) / 4.0

    # Normalize by the zero-lag autocorrelation.
    zero_lag = float(acf[cy, cx])
    if zero_lag < 1e-10:
        return 0.0

    score = float(period_peaks / zero_lag)
    return float(np.clip(score, 0.0, 1.0))


def cfa_summary(cfa_map: np.ndarray) -> str:
    """Return a human-readable summary of a CFA map for report text."""
    anomalous_fraction = float((cfa_map > 0.5).mean())
    return f"CFA anomaly fraction={anomalous_fraction:.2%}"
