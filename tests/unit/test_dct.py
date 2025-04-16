"""Unit tests for DCT block similarity feature extractor."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from forenscope.features.dct import (
    compute_dct_similarity,
    dct_summary,
    _dct_features,
    _extract_blocks,
)


class TestExtractBlocks:
    def test_correct_block_count(self) -> None:
        gray = np.zeros((256, 256), dtype=np.float32)
        blocks, positions = _extract_blocks(gray, block_size=8)
        expected = (256 // 8) * (256 // 8)
        assert len(blocks) == expected
        assert len(positions) == expected

    def test_block_shape(self) -> None:
        gray = np.zeros((128, 128), dtype=np.float32)
        blocks, _ = _extract_blocks(gray, block_size=16)
        assert all(b.shape == (16, 16) for b in blocks)


class TestDctFeatures:
    def test_output_shape(self) -> None:
        gray = np.zeros((64, 64), dtype=np.float32)
        blocks, _ = _extract_blocks(gray, block_size=8)
        feats = _dct_features(blocks)
        assert feats.shape == (len(blocks), 32)

    def test_features_are_unit_normalized(self) -> None:
        rng = np.random.default_rng(0)
        gray = rng.random((64, 64)).astype(np.float32)
        blocks, _ = _extract_blocks(gray, block_size=8)
        feats = _dct_features(blocks)
        norms = np.linalg.norm(feats, axis=1)
        # Non-zero blocks should have unit norm.
        nonzero = norms > 1e-6
        np.testing.assert_allclose(norms[nonzero], 1.0, atol=1e-5)


class TestComputeDctSimilarity:
    def test_output_shape_matches_input(self, authentic_rgb: Image.Image) -> None:
        dct_map = compute_dct_similarity(authentic_rgb)
        assert dct_map.shape == (authentic_rgb.height, authentic_rgb.width)

    def test_output_dtype_float32(self, authentic_rgb: Image.Image) -> None:
        dct_map = compute_dct_similarity(authentic_rgb)
        assert dct_map.dtype == np.float32

    def test_output_range_zero_to_one(self, authentic_rgb: Image.Image) -> None:
        dct_map = compute_dct_similarity(authentic_rgb)
        assert float(dct_map.min()) >= 0.0
        assert float(dct_map.max()) <= 1.0

    def test_copy_move_detected(self, copy_move_rgb: Image.Image) -> None:
        dct_map = compute_dct_similarity(copy_move_rgb)
        # At least some blocks should be flagged.
        assert float((dct_map > 0.5).mean()) > 0.0

    def test_authentic_has_low_copy_move_signal(
        self, authentic_rgb: Image.Image
    ) -> None:
        # A smooth gradient should have very few copy-move flags.
        dct_map = compute_dct_similarity(authentic_rgb)
        flagged = float((dct_map > 0.5).mean())
        assert flagged < 0.20, f"Too many blocks flagged on authentic image: {flagged:.2%}"

    def test_single_block_image_returns_zeros(self) -> None:
        small = Image.new("RGB", (8, 8), color=(128, 100, 80))
        dct_map = compute_dct_similarity(small)
        assert dct_map.shape == (8, 8)
        assert float(dct_map.max()) == 0.0

    def test_dct_summary_returns_string(self, authentic_rgb: Image.Image) -> None:
        dct_map = compute_dct_similarity(authentic_rgb)
        summary = dct_summary(dct_map)
        assert isinstance(summary, str)
        assert "DCT copy-move flagged fraction=" in summary
