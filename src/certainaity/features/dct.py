"""DCT coefficient block similarity for copy-move detection."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fft import dctn


def compute_dct_similarity(image: Image.Image, block_size: int = 8) -> np.ndarray:
    """Produce a copy-move suspicion map using DCT block similarity.

    Each non-overlapping block_size × block_size block of the luminance channel
    is transformed to the DCT domain and flattened into a feature vector. The
    cosine distance between all block pairs is computed. Blocks with cosine
    distance below a threshold are considered copy-move candidates; both blocks
    are then marked in the output map.

    For efficiency, the top-32 AC DCT coefficients (zig-zag order, skipping DC)
    are used. This is robust to minor brightness adjustments while remaining
    sensitive to structural copy-move.

    Args:
        image: RGB PIL image.
        block_size: DCT block size in pixels (8 matches JPEG native blocks).

    Returns:
        (H, W) float32 ndarray in [0, 1]. High values flag suspected copy-move
        source or destination blocks.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    H, W = gray.shape

    blocks, positions = _extract_blocks(gray, block_size)
    if len(blocks) < 2:
        return np.zeros((H, W), dtype=np.float32)

    features = _dct_features(blocks)
    similarity_map = np.zeros((H, W), dtype=np.float32)

    _mark_copy_move_pairs(features, positions, similarity_map, block_size)

    return similarity_map.astype(np.float32)


_ZIGZAG_32 = [
    (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1),
    (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4),
    (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3),
    (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3),
]


def _extract_blocks(
    gray: np.ndarray, block_size: int
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    H, W = gray.shape
    blocks = []
    positions = []
    for y in range(0, H - block_size + 1, block_size):
        for x in range(0, W - block_size + 1, block_size):
            blocks.append(gray[y : y + block_size, x : x + block_size])
            positions.append((y, x))
    return blocks, positions


def _dct_features(blocks: list[np.ndarray]) -> np.ndarray:
    """Return (N, 32) float32 array of normalized DCT feature vectors."""
    features = np.zeros((len(blocks), 32), dtype=np.float32)
    for i, block in enumerate(blocks):
        dct = dctn(block, norm="ortho")
        vec = np.array([dct[r, c] for r, c in _ZIGZAG_32], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        features[i] = vec
    return features


def _mark_copy_move_pairs(
    features: np.ndarray,
    positions: list[tuple[int, int]],
    out: np.ndarray,
    block_size: int,
    threshold: float = 0.05,
    min_spatial_distance: int = 2,
) -> None:
    """Mark block pairs with low cosine distance and sufficient spatial separation."""
    N = len(features)
    # Cosine distance matrix: D = 1 - F @ F^T  (features are already L2-normalized)
    cosine_sim = features @ features.T  # (N, N)

    for i in range(N):
        for j in range(i + 1, N):
            dist = 1.0 - float(cosine_sim[i, j])
            if dist > threshold:
                continue
            yi, xi = positions[i]
            yj, xj = positions[j]
            # Require blocks to be spatially separated.
            block_dist = abs(yi - yj) / block_size + abs(xi - xj) / block_size
            if block_dist < min_spatial_distance:
                continue
            # Mark both blocks.
            out[yi : yi + block_size, xi : xi + block_size] = 1.0
            out[yj : yj + block_size, xj : xj + block_size] = 1.0


def dct_summary(dct_map: np.ndarray) -> str:
    """Return a human-readable summary of a DCT similarity map for report text."""
    flagged = float((dct_map > 0.5).mean())
    return f"DCT copy-move flagged fraction={flagged:.2%}"
