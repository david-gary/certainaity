"""Unit tests for runtime configuration / Settings."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic_settings")

import certainaity.config as config_module
from certainaity.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    config_module._settings = None
    yield
    config_module._settings = None


class TestSettingsDefaults:
    def test_max_file_bytes_is_50mb(self) -> None:
        assert Settings().max_file_bytes == 50 * 1024 * 1024

    def test_min_image_dimension(self) -> None:
        assert Settings().min_image_dimension == 64

    def test_max_image_dimension(self) -> None:
        assert Settings().max_image_dimension == 20_000

    def test_ensemble_threshold(self) -> None:
        assert Settings().ensemble_threshold == pytest.approx(0.65)

    def test_weights_dir(self) -> None:
        assert Settings().weights_dir == Path("weights")

    def test_output_dir(self) -> None:
        assert Settings().output_dir == Path("output")

    def test_ela_quality(self) -> None:
        assert Settings().ela_quality == 75

    def test_dct_block_size(self) -> None:
        assert Settings().dct_block_size == 8

    def test_noise_block_size(self) -> None:
        assert Settings().noise_block_size == 32

    def test_feature_workers(self) -> None:
        assert Settings().feature_workers == 4

    def test_resilience_qualities(self) -> None:
        assert Settings().resilience_qualities == [70, 85, 95]

    def test_resilience_drop_threshold(self) -> None:
        assert Settings().resilience_drop_threshold == pytest.approx(0.25)

    def test_redis_url(self) -> None:
        assert Settings().redis_url == "redis://localhost:6379/0"

    def test_use_cpu_defaults_false(self) -> None:
        assert Settings().use_cpu is False


class TestSettingsEnvOverrides:
    def test_max_file_bytes_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CERTAINAITY_MAX_FILE_BYTES", "1048576")
        assert Settings().max_file_bytes == 1_048_576

    def test_use_cpu_true_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CERTAINAITY_USE_CPU", "true")
        assert Settings().use_cpu is True

    def test_redis_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CERTAINAITY_REDIS_URL", "redis://myhost:6380/2")
        assert Settings().redis_url == "redis://myhost:6380/2"

    def test_ensemble_threshold_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CERTAINAITY_ENSEMBLE_THRESHOLD", "0.8")
        assert Settings().ensemble_threshold == pytest.approx(0.8)

    def test_weights_dir_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("CERTAINAITY_WEIGHTS_DIR", str(tmp_path))
        assert Settings().weights_dir == tmp_path


class TestGetSettings:
    def test_returns_settings_instance(self) -> None:
        assert isinstance(get_settings(), Settings)

    def test_singleton_same_object_on_repeat_calls(self) -> None:
        assert get_settings() is get_settings()

    def test_singleton_reset_by_fixture(self) -> None:
        s1 = get_settings()
        config_module._settings = None
        s2 = get_settings()
        assert s1 is not s2
