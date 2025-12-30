"""End-to-end chain-of-custody integration tests.

These tests exercise the full path from image submission through task
execution to report retrieval, verifying that the SHA-256 digest and
job ID remain consistent across every layer of the system.

Tasks are run synchronously via ``.run()`` (bypassing the Celery broker)
so the test suite requires no live Redis instance.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image as PILImage

from forenscope.api.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg(size: tuple[int, int] = (128, 128), color: tuple[int, int, int] = (42, 128, 200)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _mock_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.output_dir = tmp_path
    s.max_file_bytes = 50 * 1024 * 1024
    s.max_image_dimension = 20_000
    s.min_image_dimension = 64
    s.weights_dir = tmp_path / "weights"
    return s


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def stub_jwt():
    with patch("forenscope.api.auth.verify_jwt", return_value={"sub": "investigator-1"}):
        yield


@pytest.fixture()
async def api_client(stub_jwt):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChainOfCustody:
    async def test_sha256_consistent_from_upload_to_task_result(
        self, api_client, tmp_path: Path
    ) -> None:
        """The SHA-256 computed at upload must match the one in the task result."""
        pytest.importorskip("reportlab")
        raw = _make_jpeg()
        expected_sha256 = hashlib.sha256(raw).hexdigest()

        captured_job_id: list[str] = []

        def _record_async(args, task_id, **kwargs):
            captured_job_id.append(task_id)

        with patch("forenscope.worker.tasks.analyze_image") as mock_task:
            mock_task.apply_async.side_effect = _record_async
            response = await api_client.post(
                "/v1/analyze",
                files={"file": ("evidence.jpg", io.BytesIO(raw), "image/jpeg")},
            )

        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert job_id == captured_job_id[0]

        from forenscope.worker.tasks import analyze_image

        mock_s = _mock_settings(tmp_path)
        with (
            patch("forenscope.worker.tasks.get_settings", return_value=mock_s),
            patch("forenscope.ingest.get_settings", return_value=mock_s),
            patch.object(analyze_image, "update_state"),
        ):
            result = analyze_image.run(job_id, base64.b64encode(raw).decode(), "evidence.jpg")

        assert result["sha256"] == expected_sha256

    async def test_sha256_consistent_from_task_result_to_json_report(
        self, tmp_path: Path
    ) -> None:
        """The SHA-256 in the task result dict must match the value in report.json."""
        pytest.importorskip("reportlab")
        raw = _make_jpeg()
        expected_sha256 = hashlib.sha256(raw).hexdigest()
        job_id = "e2e-coc-001"

        from forenscope.worker.tasks import analyze_image

        mock_s = _mock_settings(tmp_path)
        with (
            patch("forenscope.worker.tasks.get_settings", return_value=mock_s),
            patch("forenscope.ingest.get_settings", return_value=mock_s),
            patch.object(analyze_image, "update_state"),
        ):
            result = analyze_image.run(job_id, base64.b64encode(raw).decode(), "photo.jpg")

        report_data = json.loads((tmp_path / job_id / "report.json").read_text())
        assert result["sha256"] == expected_sha256
        assert report_data["sha256"] == expected_sha256

    async def test_job_id_consistent_across_all_layers(
        self, api_client, tmp_path: Path
    ) -> None:
        """The job_id returned by the API must appear in report.json."""
        pytest.importorskip("reportlab")
        raw = _make_jpeg()
        captured: list[str] = []

        with patch("forenscope.worker.tasks.analyze_image") as mock_task:
            mock_task.apply_async.side_effect = lambda args, task_id, **kw: captured.append(task_id)
            response = await api_client.post(
                "/v1/analyze",
                files={"file": ("img.jpg", io.BytesIO(raw), "image/jpeg")},
            )

        job_id = response.json()["job_id"]

        from forenscope.worker.tasks import analyze_image

        mock_s = _mock_settings(tmp_path)
        with (
            patch("forenscope.worker.tasks.get_settings", return_value=mock_s),
            patch("forenscope.ingest.get_settings", return_value=mock_s),
            patch.object(analyze_image, "update_state"),
        ):
            analyze_image.run(job_id, base64.b64encode(raw).decode(), "img.jpg")

        data = json.loads((tmp_path / job_id / "report.json").read_text())
        assert data["job_id"] == job_id

    async def test_poll_url_matches_job_id(self, api_client) -> None:
        raw = _make_jpeg()
        with patch("forenscope.worker.tasks.analyze_image") as mock_task:
            mock_task.apply_async.return_value = None
            response = await api_client.post(
                "/v1/analyze",
                files={"file": ("img.jpg", io.BytesIO(raw), "image/jpeg")},
            )
        body = response.json()
        assert body["poll_url"] == f"/v1/jobs/{body['job_id']}"

    async def test_report_json_contains_all_custody_fields(self, tmp_path: Path) -> None:
        """All fields required by the chain-of-custody specification must be present."""
        pytest.importorskip("reportlab")
        from forenscope.worker.tasks import analyze_image

        raw = _make_jpeg()
        job_id = "e2e-coc-002"

        mock_s = _mock_settings(tmp_path)
        with (
            patch("forenscope.worker.tasks.get_settings", return_value=mock_s),
            patch("forenscope.ingest.get_settings", return_value=mock_s),
            patch.object(analyze_image, "update_state"),
        ):
            analyze_image.run(job_id, base64.b64encode(raw).decode(), "exhibit.jpg")

        data = json.loads((tmp_path / job_id / "report.json").read_text())
        required_fields = (
            "job_id",
            "file_name",
            "sha256",
            "analysis_timestamp",
            "manipulation_detected",
            "overall_confidence",
        )
        for field in required_fields:
            assert field in data, f"Chain-of-custody field missing: {field!r}"

        assert data["job_id"] == job_id
        assert data["file_name"] == "exhibit.jpg"
        assert len(data["sha256"]) == 64

    async def test_pdf_report_produced_alongside_json(self, tmp_path: Path) -> None:
        pytest.importorskip("reportlab")
        from forenscope.worker.tasks import analyze_image

        raw = _make_jpeg()
        job_id = "e2e-coc-003"

        mock_s = _mock_settings(tmp_path)
        with (
            patch("forenscope.worker.tasks.get_settings", return_value=mock_s),
            patch("forenscope.ingest.get_settings", return_value=mock_s),
            patch.object(analyze_image, "update_state"),
        ):
            analyze_image.run(job_id, base64.b64encode(raw).decode(), "scan.jpg")

        assert (tmp_path / job_id / "report.json").exists()
        pdf = tmp_path / job_id / "report.pdf"
        assert pdf.exists()
        assert pdf.read_bytes()[:4] == b"%PDF"
