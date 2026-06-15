"""Unit tests for the FeatureMaps dataclass and feature extraction pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from certainaity.features import FeatureMaps, extract_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def maps_128() -> FeatureMaps:
    """FeatureMaps with 128×128 random float32 arrays."""
    rng = np.random.default_rng(0)
    return FeatureMaps(
        ela=rng.random((128, 128)).astype(np.float32),
        noise=rng.random((128, 128)).astype(np.float32),
        cfa=rng.random((128, 128)).astype(np.float32),
        dct=rng.random((128, 128)).astype(np.float32),
    )


@pytest.fixture()
def small_image() -> Image.Image:
    """128×128 gradient image suitable for extract_features."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    for y in range(128):
        arr[y, :, 0] = y * 2
        arr[y, :, 1] = 128
        arr[y, :, 2] = 255 - y * 2
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Tests: FeatureMaps dataclass
# ---------------------------------------------------------------------------


class TestFeatureMaps:
    def test_stacked_shape(self, maps_128: FeatureMaps) -> None:
        stacked = maps_128.stacked
        assert stacked.shape == (4, 128, 128)

    def test_stacked_channel_order(self, maps_128: FeatureMaps) -> None:
        stacked = maps_128.stacked
        np.testing.assert_array_equal(stacked[0], maps_128.ela)
        np.testing.assert_array_equal(stacked[1], maps_128.noise)
        np.testing.assert_array_equal(stacked[2], maps_128.cfa)
        np.testing.assert_array_equal(stacked[3], maps_128.dct)

    def test_stacked_dtype_preserved(self, maps_128: FeatureMaps) -> None:
        assert maps_128.stacked.dtype == np.float32

    def test_each_map_accessible(self, maps_128: FeatureMaps) -> None:
        assert maps_128.ela.shape == (128, 128)
        assert maps_128.noise.shape == (128, 128)
        assert maps_128.cfa.shape == (128, 128)
        assert maps_128.dct.shape == (128, 128)

    def test_stacked_is_fresh_array(self, maps_128: FeatureMaps) -> None:
        s1 = maps_128.stacked
        s2 = maps_128.stacked
        assert s1 is not s2


# ---------------------------------------------------------------------------
# Tests: extract_features (integration with real extractors)
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_returns_feature_maps(self, small_image: Image.Image) -> None:
        result = extract_features(small_image)
        assert isinstance(result, FeatureMaps)

    def test_ela_shape_matches_image(self, small_image: Image.Image) -> None:
        result = extract_features(small_image)
        assert result.ela.shape[:2] == (small_image.height, small_image.width)

    def test_all_maps_have_same_height(self, small_image: Image.Image) -> None:
        result = extract_features(small_image)
        heights = {result.ela.shape[0], result.noise.shape[0], result.cfa.shape[0], result.dct.shape[0]}
        assert len(heights) == 1

    def test_stacked_first_dim_is_four(self, small_image: Image.Image) -> None:
        result = extract_features(small_image)
        assert result.stacked.shape[0] == 4

    def test_maps_are_float(self, small_image: Image.Image) -> None:
        result = extract_features(small_image)
        for arr in (result.ela, result.noise, result.cfa, result.dct):
            assert np.issubdtype(arr.dtype, np.floating), f"Expected float, got {arr.dtype}"
