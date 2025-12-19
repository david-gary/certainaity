"""Shared helpers and fixtures for the integration test suite.

Fixtures defined here are auto-discovered by pytest for all files under
``tests/integration/`` without explicit import.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Image factories (importable functions — not fixtures, to avoid name clashes
# with the per-file helpers already defined in test_api.py and test_worker.py)
# ---------------------------------------------------------------------------

def make_jpeg(size: tuple[int, int] = (128, 128), color: tuple[int, int, int] = (100, 150, 200)) -> bytes:
    """Return the raw bytes of a synthetic JPEG at *size*."""
    buf = io.BytesIO()
    PILImage.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


def make_png(size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (200, 100, 50)) -> bytes:
    """Return the raw bytes of a synthetic PNG at *size*."""
    buf = io.BytesIO()
    PILImage.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def jpeg_bytes() -> bytes:
    """Minimal valid JPEG bytes for use in tests."""
    return make_jpeg()


@pytest.fixture()
def png_bytes() -> bytes:
    """Minimal valid PNG bytes for use in tests."""
    return make_png()


@pytest.fixture()
def mock_settings(tmp_path: Path):
    """MagicMock settings object pointing all paths into tmp_path."""
    from unittest.mock import MagicMock

    s = MagicMock()
    s.output_dir = tmp_path
    s.max_file_bytes = 50 * 1024 * 1024
    s.max_image_dimension = 20_000
    s.min_image_dimension = 64
    s.weights_dir = tmp_path / "weights"
    s.redis_url = "redis://localhost:6379/0"
    return s
