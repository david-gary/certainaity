"""Unit tests for CFA interpolation correlation feature extractor."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from certainaity.features.cfa import compute_cfa_map, cfa_summary, _linear_prediction_residual


class TestLinearPredictionResidual:
    def test_flat_image_zero_residual(self) -> None:
        flat = np.full((64, 64), 0.5, dtype=np.float32)
        residual = _linear_prediction_residual(flat)
        np.testing.assert_allclose(residual, 0.0, atol=1e-6)

    def test_residual_shape_preserved(self, authentic_rgb: Image.Image) -> None:
        gray = np.asarray(authentic_rgb.convert("L"), dtype=np.float32) / 255.0
        residual = _linear_prediction_residual(gray)
        assert residual.shape == gray.shape


class TestComputeCfaMap:
    def test_output_shape_matches_input(self, authentic_rgb: Image.Image) -> None:
        cfa = compute_cfa_map(authentic_rgb)
        assert cfa.shape == (authentic_rgb.height, authentic_rgb.width)

    def test_output_dtype_float32(self, authentic_rgb: Image.Image) -> None:
        cfa = compute_cfa_map(authentic_rgb)
        assert cfa.dtype == np.float32

    def test_output_range_zero_to_one(self, authentic_rgb: Image.Image) -> None:
        cfa = compute_cfa_map(authentic_rgb)
        assert float(cfa.min()) >= 0.0
        assert float(cfa.max()) <= 1.0

    def test_flat_image_handled_gracefully(self) -> None:
        flat = Image.new("RGB", (256, 256), color=(100, 100, 100))
        cfa = compute_cfa_map(flat)
        # Should not raise; values indeterminate but in range.
        assert cfa.shape == (256, 256)
        assert float(cfa.min()) >= 0.0 and float(cfa.max()) <= 1.0

    def test_spliced_has_higher_anomaly_than_authentic(
        self, authentic_rgb: Image.Image, spliced_rgb: Image.Image
    ) -> None:
        cfa_auth = compute_cfa_map(authentic_rgb)
        cfa_splice = compute_cfa_map(spliced_rgb)
        # Spliced image should have more blocks flagged as CFA anomalous.
        assert float((cfa_splice > 0.5).mean()) >= float((cfa_auth > 0.5).mean())

    def test_cfa_summary_returns_string(self, authentic_rgb: Image.Image) -> None:
        cfa = compute_cfa_map(authentic_rgb)
        summary = cfa_summary(cfa)
        assert isinstance(summary, str)
        assert "CFA anomaly fraction=" in summary

    def test_small_image_does_not_crash(self) -> None:
        small = Image.new("RGB", (64, 64), color=(80, 130, 200))
        cfa = compute_cfa_map(small)
        assert cfa.shape == (64, 64)
