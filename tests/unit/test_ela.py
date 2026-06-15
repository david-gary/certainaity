"""Unit tests for ELA feature extractor."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from certainaity.features.ela import compute_ela, ela_summary


class TestComputeEla:
    def test_output_shape_matches_input(self, authentic_rgb: Image.Image) -> None:
        ela = compute_ela(authentic_rgb)
        assert ela.shape == (authentic_rgb.height, authentic_rgb.width)

    def test_output_dtype_float32(self, authentic_rgb: Image.Image) -> None:
        ela = compute_ela(authentic_rgb)
        assert ela.dtype == np.float32

    def test_output_range_zero_to_one(self, authentic_rgb: Image.Image) -> None:
        ela = compute_ela(authentic_rgb)
        assert float(ela.min()) >= 0.0
        assert float(ela.max()) <= 1.0

    def test_authentic_image_low_mean_ela(self, authentic_rgb: Image.Image) -> None:
        ela = compute_ela(authentic_rgb, quality=75)
        # A gradient image resaved at the same quality should show minimal ELA signal.
        assert float(ela.mean()) < 0.15, f"ELA mean too high for authentic image: {ela.mean():.3f}"

    def test_spliced_region_higher_ela(
        self, authentic_rgb: Image.Image, spliced_rgb: Image.Image
    ) -> None:
        ela_auth = compute_ela(authentic_rgb)
        ela_splice = compute_ela(spliced_rgb)
        # The spliced region (200:300, 200:300) should have higher ELA than the rest.
        splice_region = ela_splice[200:300, 200:300]
        auth_region = ela_auth[200:300, 200:300]
        assert float(splice_region.mean()) > float(auth_region.mean()), (
            "Spliced region should have higher ELA than the same region in an authentic image."
        )

    def test_uniform_black_image_returns_zeros(self) -> None:
        black = Image.new("RGB", (256, 256), color=(0, 0, 0))
        ela = compute_ela(black)
        assert float(ela.max()) == 0.0

    def test_custom_quality_accepted(self, authentic_rgb: Image.Image) -> None:
        ela_low = compute_ela(authentic_rgb, quality=50)
        ela_high = compute_ela(authentic_rgb, quality=95)
        # Both should produce valid maps; lower quality = stronger ELA sensitivity.
        assert ela_low.shape == ela_high.shape
        assert float(ela_low.max()) <= 1.0

    def test_accepts_rgba_via_rgb_conversion(self) -> None:
        rgba = Image.new("RGBA", (128, 128), color=(100, 150, 200, 255))
        ela = compute_ela(rgba)
        assert ela.shape == (128, 128)

    def test_ela_summary_returns_string(self, authentic_rgb: Image.Image) -> None:
        ela = compute_ela(authentic_rgb)
        summary = ela_summary(ela)
        assert isinstance(summary, str)
        assert "ELA mean=" in summary
