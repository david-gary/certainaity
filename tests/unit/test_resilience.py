"""Unit tests for the anti-forensic resilience detection module."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from certainaity.exceptions import WeightsNotFoundError
from certainaity.resilience import run_resilience_test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def jpeg_bytes() -> bytes:
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture()
def settings() -> SimpleNamespace:
    return SimpleNamespace(
        resilience_qualities=[70, 85, 95],
        resilience_drop_threshold=0.25,
        ensemble_threshold=0.65,
        min_region_px=64,
    )


def _mock_ensemble(heatmap_value: float) -> MagicMock:
    result = MagicMock()
    result.heatmap = np.full((128, 128), heatmap_value, dtype=np.float32)
    ensemble = MagicMock()
    ensemble.localize.return_value = result
    return ensemble


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunResilienceTest:
    def test_returns_false_when_confidence_stable(
        self, jpeg_bytes: bytes, settings: SimpleNamespace
    ) -> None:
        ensemble = _mock_ensemble(0.8)
        assert run_resilience_test(jpeg_bytes, ensemble, 0.8, settings) is False

    def test_returns_true_when_confidence_drops_sharply(
        self, jpeg_bytes: bytes, settings: SimpleNamespace
    ) -> None:
        ensemble = _mock_ensemble(0.3)
        assert run_resilience_test(jpeg_bytes, ensemble, 0.8, settings) is True

    def test_drop_exactly_at_threshold_is_not_flagged(
        self, jpeg_bytes: bytes, settings: SimpleNamespace
    ) -> None:
        # drop == threshold is not > threshold, so should not trigger
        ensemble = _mock_ensemble(0.55)
        assert run_resilience_test(jpeg_bytes, ensemble, 0.8, settings) is False

    def test_returns_false_when_weights_not_found(
        self, jpeg_bytes: bytes, settings: SimpleNamespace
    ) -> None:
        ensemble = MagicMock()
        ensemble.localize.side_effect = WeightsNotFoundError("no weights")
        assert run_resilience_test(jpeg_bytes, ensemble, 0.9, settings) is False

    def test_continues_on_general_exception(
        self, jpeg_bytes: bytes, settings: SimpleNamespace
    ) -> None:
        stable_result = MagicMock()
        stable_result.heatmap = np.full((128, 128), 0.8, dtype=np.float32)
        ensemble = MagicMock()
        ensemble.localize.side_effect = [
            RuntimeError("gpu exploded"),
            stable_result,
            stable_result,
        ]
        assert run_resilience_test(jpeg_bytes, ensemble, 0.8, settings) is False

    def test_empty_qualities_returns_false(self, jpeg_bytes: bytes) -> None:
        empty_settings = SimpleNamespace(
            resilience_qualities=[],
            resilience_drop_threshold=0.25,
            ensemble_threshold=0.65,
            min_region_px=64,
        )
        assert run_resilience_test(jpeg_bytes, MagicMock(), 0.9, empty_settings) is False

    def test_qualities_sorted_before_iteration(
        self, jpeg_bytes: bytes
    ) -> None:
        # Quality list is unsorted; function should sort before iterating.
        unsorted_settings = SimpleNamespace(
            resilience_qualities=[95, 70, 85],
            resilience_drop_threshold=0.25,
            ensemble_threshold=0.65,
            min_region_px=64,
        )
        ensemble = _mock_ensemble(0.8)
        # Should not raise; result doesn't matter here, just verifying it runs.
        run_resilience_test(jpeg_bytes, ensemble, 0.8, unsorted_settings)
        assert ensemble.localize.call_count == 3

    def test_ensemble_called_for_each_quality(
        self, jpeg_bytes: bytes, settings: SimpleNamespace
    ) -> None:
        ensemble = _mock_ensemble(0.8)
        run_resilience_test(jpeg_bytes, ensemble, 0.8, settings)
        assert ensemble.localize.call_count == 3
