"""Unit tests for wavelet noise variance feature extractor."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from forenscope.features.noise import compute_noise_map, noise_summary


class TestComputeNoiseMap:
    def test_output_shape_matches_input(self, authentic_rgb: Image.Image) -> None:
        noise = compute_noise_map(authentic_rgb)
        assert noise.shape == (authentic_rgb.height, authentic_rgb.width)

    def test_output_dtype_float32(self, authentic_rgb: Image.Image) -> None:
        noise = compute_noise_map(authentic_rgb)
        assert noise.dtype == np.float32

    def test_output_range_zero_to_one(self, authentic_rgb: Image.Image) -> None:
        noise = compute_noise_map(authentic_rgb)
        assert float(noise.min()) >= 0.0
        assert float(noise.max()) <= 1.0 + 1e-6

    def test_flat_image_near_zero_variance(self) -> None:
        flat = Image.new("RGB", (256, 256), color=(128, 128, 128))
        noise = compute_noise_map(flat)
        assert float(noise.max()) < 1e-6, (
            f"Flat image should produce near-zero noise map, got max={noise.max():.6f}"
        )

    def test_noisy_image_nonzero_map(self) -> None:
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        noisy = Image.fromarray(arr, mode="RGB")
        noise = compute_noise_map(noisy)
        assert float(noise.max()) > 0.0

    def test_splice_boundary_increases_variance(
        self, spliced_rgb: Image.Image, authentic_rgb: Image.Image
    ) -> None:
        noise_auth = compute_noise_map(authentic_rgb)
        noise_splice = compute_noise_map(spliced_rgb)
        # Spliced image should have higher overall noise variance std.
        assert noise_splice.std() >= noise_auth.std()

    def test_custom_block_size(self, authentic_rgb: Image.Image) -> None:
        noise_16 = compute_noise_map(authentic_rgb, block_size=16)
        noise_64 = compute_noise_map(authentic_rgb, block_size=64)
        assert noise_16.shape == noise_64.shape == (512, 512)

    def test_noise_summary_returns_string(self, authentic_rgb: Image.Image) -> None:
        noise = compute_noise_map(authentic_rgb)
        summary = noise_summary(noise)
        assert isinstance(summary, str)
        assert "Noise variance std=" in summary
