"""Unit tests for Ensemble fusion and LocalizationResult."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from certainaity.models.base import ModelName
from certainaity.models.ensemble import Ensemble, LocalizationResult, _DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(prob_map: np.ndarray) -> MagicMock:
    """Return a mock ForensicModel whose predict() returns ``prob_map``."""
    mock = MagicMock()
    mock.predict.return_value = prob_map
    return mock


def _make_ensemble(
    weights_dir: Path,
    maps: dict[ModelName, np.ndarray],
) -> Ensemble:
    """Build an Ensemble with mocked sub-models producing ``maps``."""
    ensemble = Ensemble.__new__(Ensemble)
    ensemble._weights_dir = weights_dir
    ensemble._device = "cpu"
    ensemble._model_weights = dict(_DEFAULT_WEIGHTS)
    ensemble._models = {name: _make_mock_model(m) for name, m in maps.items()}
    return ensemble


# ---------------------------------------------------------------------------
# Tests: default weights
# ---------------------------------------------------------------------------

class TestDefaultWeights:
    def test_weights_sum_to_one(self) -> None:
        total = sum(_DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_four_models_in_defaults(self) -> None:
        assert len(_DEFAULT_WEIGHTS) == 4
        assert ModelName.PATCH_FORENSIC in _DEFAULT_WEIGHTS
        assert ModelName.MANTRA_NET in _DEFAULT_WEIGHTS
        assert ModelName.SPSL in _DEFAULT_WEIGHTS
        assert ModelName.INPAINTING_DETECTOR in _DEFAULT_WEIGHTS

    def test_patch_forensic_highest_weight(self) -> None:
        assert (
            _DEFAULT_WEIGHTS[ModelName.PATCH_FORENSIC]
            == max(_DEFAULT_WEIGHTS.values())
        )


# ---------------------------------------------------------------------------
# Tests: Ensemble.predict
# ---------------------------------------------------------------------------

class TestEnsemblePredict:
    def _maps(self) -> dict[ModelName, np.ndarray]:
        return {
            ModelName.PATCH_FORENSIC: np.full((64, 64), 0.8, dtype=np.float32),
            ModelName.MANTRA_NET: np.full((64, 64), 0.6, dtype=np.float32),
            ModelName.SPSL: np.full((64, 64), 0.4, dtype=np.float32),
            ModelName.INPAINTING_DETECTOR: np.full((64, 64), 0.2, dtype=np.float32),
        }

    def test_output_shape(self, tmp_path: Path) -> None:
        ens = _make_ensemble(tmp_path, self._maps())
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        out = ens.predict(image)
        assert out.shape == (64, 64)

    def test_output_dtype_float32(self, tmp_path: Path) -> None:
        ens = _make_ensemble(tmp_path, self._maps())
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        out = ens.predict(image)
        assert out.dtype == np.float32

    def test_weighted_average_correctness(self, tmp_path: Path) -> None:
        # With uniform predictions, the fused output should equal those values.
        maps = {k: np.full((32, 32), 0.5, dtype=np.float32) for k in ModelName}
        ens = _make_ensemble(tmp_path, maps)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        out = ens.predict(image)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_each_model_called_once(self, tmp_path: Path) -> None:
        ens = _make_ensemble(tmp_path, self._maps())
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        ens.predict(image)
        for model in ens._models.values():
            model.predict.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Ensemble.localize
# ---------------------------------------------------------------------------

class TestEnsembleLocalize:
    def _uniform_maps(self, val: float, shape: tuple = (128, 128)) -> dict[ModelName, np.ndarray]:
        return {name: np.full(shape, val, dtype=np.float32) for name in ModelName}

    def test_returns_localization_result(self, tmp_path: Path) -> None:
        ens = _make_ensemble(tmp_path, self._uniform_maps(0.8))
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        result = ens.localize(image)
        assert isinstance(result, LocalizationResult)

    def test_heatmap_shape(self, tmp_path: Path) -> None:
        ens = _make_ensemble(tmp_path, self._uniform_maps(0.5))
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        result = ens.localize(image)
        assert result.heatmap.shape == (128, 128)

    def test_high_confidence_mask_is_true(self, tmp_path: Path) -> None:
        ens = _make_ensemble(tmp_path, self._uniform_maps(0.9))
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        result = ens.localize(image, threshold=0.65, min_region_px=1)
        assert result.binary_mask.all()

    def test_low_confidence_mask_is_false(self, tmp_path: Path) -> None:
        ens = _make_ensemble(tmp_path, self._uniform_maps(0.1))
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        result = ens.localize(image, threshold=0.65)
        assert not result.binary_mask.any()

    def test_model_maps_empty_by_default(self, tmp_path: Path) -> None:
        ens = _make_ensemble(tmp_path, self._uniform_maps(0.5))
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        result = ens.localize(image)
        assert result.model_maps == {}

    def test_model_maps_populated_when_requested(self, tmp_path: Path) -> None:
        ens = _make_ensemble(tmp_path, self._uniform_maps(0.5))
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        result = ens.localize(image, return_model_maps=True)
        assert len(result.model_maps) == 4

    def test_min_region_px_filters_noise(self, tmp_path: Path) -> None:
        # Create a map with a tiny bright region in a dark field.
        base = np.full((128, 128), 0.1, dtype=np.float32)
        base[60:63, 60:63] = 0.9    # 9-pixel region
        maps = {name: base.copy() for name in ModelName}
        ens = _make_ensemble(tmp_path, maps)
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        # min_region_px=50 should eliminate the 9-pixel spot.
        result = ens.localize(image, threshold=0.65, min_region_px=50)
        assert result.num_regions == 0


# ---------------------------------------------------------------------------
# Tests: Ensemble.optimize_weights
# ---------------------------------------------------------------------------

class TestOptimizeWeights:
    def test_returns_dict_of_four_weights(self) -> None:
        rng = np.random.default_rng(3)
        shape = (10, 32, 32)
        preds = {name: rng.random(shape).astype(np.float32) for name in ModelName}
        gt = rng.integers(0, 2, shape).astype(np.float32)
        result = Ensemble.optimize_weights(preds, gt)
        assert len(result) == 4

    def test_weights_sum_to_one(self) -> None:
        rng = np.random.default_rng(4)
        shape = (5, 16, 16)
        preds = {name: rng.random(shape).astype(np.float32) for name in ModelName}
        gt = rng.integers(0, 2, shape).astype(np.float32)
        result = Ensemble.optimize_weights(preds, gt)
        assert abs(sum(result.values()) - 1.0) < 1e-5

    def test_all_model_names_present(self) -> None:
        rng = np.random.default_rng(5)
        shape = (5, 16, 16)
        preds = {name: rng.random(shape).astype(np.float32) for name in ModelName}
        gt = rng.integers(0, 2, shape).astype(np.float32)
        result = Ensemble.optimize_weights(preds, gt)
        for name in ModelName:
            assert name in result


# ---------------------------------------------------------------------------
# Tests: weight persistence
# ---------------------------------------------------------------------------

class TestWeightPersistence:
    def test_save_and_load_round_trips(self, tmp_path: Path) -> None:
        ens = _make_ensemble(tmp_path, {name: np.zeros((8, 8)) for name in ModelName})
        out_path = tmp_path / "weights.json"
        ens.save_weights(out_path)
        ens2 = _make_ensemble(tmp_path, {name: np.zeros((8, 8)) for name in ModelName})
        ens2.load_optimized_weights(out_path)
        assert ens._model_weights == ens2._model_weights

    def test_model_weights_property_returns_copy(self, tmp_path: Path) -> None:
        ens = _make_ensemble(tmp_path, {name: np.zeros((8, 8)) for name in ModelName})
        w = ens.model_weights
        w[ModelName.PATCH_FORENSIC] = 999.0
        assert ens._model_weights[ModelName.PATCH_FORENSIC] != 999.0
