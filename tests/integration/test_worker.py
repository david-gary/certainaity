"""Integration tests for the Celery analyze_image task.

Tasks are called via ``.run()`` to bypass the broker entirely, running
synchronously in the test process.  ``update_state`` is patched so there's
no dependency on a live Redis backend.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jpeg_b64(size: tuple[int, int] = (128, 128)) -> str:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color=(90, 120, 150)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_mock_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.output_dir = tmp_path
    s.max_file_bytes = 50 * 1024 * 1024
    s.max_image_dimension = 20_000
    s.min_image_dimension = 64
    s.weights_dir = tmp_path / "weights"
    return s


# ---------------------------------------------------------------------------
# Tests: task registration
# ---------------------------------------------------------------------------

class TestTaskRegistration:
    def test_task_is_in_registry(self) -> None:
        from forenscope.worker import tasks  # noqa: F401 — ensures task registers
        from forenscope.worker.app import celery_app

        assert "forenscope.analyze_image" in celery_app.tasks

    def test_task_has_correct_name(self) -> None:
        from forenscope.worker.tasks import analyze_image

        assert analyze_image.name == "forenscope.analyze_image"

    def test_task_max_retries(self) -> None:
        from forenscope.worker.tasks import analyze_image

        assert analyze_image.max_retries == 2


# ---------------------------------------------------------------------------
# Tests: task execution
# ---------------------------------------------------------------------------

class TestAnalyzeImageTask:
    def _run_task(self, job_id: str, image_b64: str, tmp_path: Path) -> dict:
        from forenscope.worker.tasks import analyze_image

        mock_settings = _make_mock_settings(tmp_path)
        with (
            patch("forenscope.worker.tasks.get_settings", return_value=mock_settings),
            patch("forenscope.ingest.get_settings", return_value=mock_settings),
            patch.object(analyze_image, "update_state"),
        ):
            return analyze_image.run(job_id, image_b64, "test.jpg")

    def test_returns_job_id(self, tmp_path: Path) -> None:
        pytest.importorskip("reportlab")
        job_id = "worker-test-001"
        result = self._run_task(job_id, _jpeg_b64(), tmp_path)
        assert result["job_id"] == job_id

    def test_returns_sha256(self, tmp_path: Path) -> None:
        pytest.importorskip("reportlab")
        result = self._run_task("worker-test-002", _jpeg_b64(), tmp_path)
        assert "sha256" in result
        assert len(result["sha256"]) == 64

    def test_json_report_written(self, tmp_path: Path) -> None:
        pytest.importorskip("reportlab")
        job_id = "worker-test-003"
        self._run_task(job_id, _jpeg_b64(), tmp_path)
        report_path = tmp_path / job_id / "report.json"
        assert report_path.exists()

    def test_json_report_valid(self, tmp_path: Path) -> None:
        pytest.importorskip("reportlab")
        job_id = "worker-test-004"
        self._run_task(job_id, _jpeg_b64(), tmp_path)
        data = json.loads((tmp_path / job_id / "report.json").read_text())
        assert "sha256" in data
        assert "manipulation_detected" in data
        assert "overall_confidence" in data

    def test_pdf_report_written(self, tmp_path: Path) -> None:
        pytest.importorskip("reportlab")
        job_id = "worker-test-005"
        self._run_task(job_id, _jpeg_b64(), tmp_path)
        pdf_path = tmp_path / job_id / "report.pdf"
        assert pdf_path.exists()

    def test_pdf_has_magic_bytes(self, tmp_path: Path) -> None:
        pytest.importorskip("reportlab")
        job_id = "worker-test-006"
        self._run_task(job_id, _jpeg_b64(), tmp_path)
        assert (tmp_path / job_id / "report.pdf").read_bytes()[:4] == b"%PDF"

    def test_output_dir_created_per_job(self, tmp_path: Path) -> None:
        pytest.importorskip("reportlab")
        job_id = "worker-test-007"
        self._run_task(job_id, _jpeg_b64(), tmp_path)
        assert (tmp_path / job_id).is_dir()

    def test_no_manipulation_in_stub(self, tmp_path: Path) -> None:
        pytest.importorskip("reportlab")
        result = self._run_task("worker-test-008", _jpeg_b64(), tmp_path)
        # Stub pipeline always returns 0.0 confidence → no manipulation detected.
        assert result["overall_confidence"] == 0.0
