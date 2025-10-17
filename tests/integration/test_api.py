"""Integration tests for the ForenScope FastAPI application.

All external I/O (Redis, Celery broker, JWT key file) is mocked so the
tests run in CI without a running Redis instance.
"""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg(size: tuple[int, int] = (128, 128)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_png(size: tuple[int, int] = (128, 128)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def app():
    from forenscope.api.main import create_app
    return create_app()


@pytest.fixture()
def _stub_jwt():
    """Patch verify_jwt to always succeed without a real key file."""
    with patch("forenscope.api.auth.verify_jwt", return_value={"sub": "test-user"}):
        yield


# ---------------------------------------------------------------------------
# Tests: GET /v1/health
# ---------------------------------------------------------------------------

class TestHealth:
    @pytest.mark.asyncio
    async def test_returns_200(self, app) -> None:
        with patch("redis.from_url") as mock_r:
            mock_r.return_value.ping.return_value = True
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_body_has_status_ok(self, app) -> None:
        with patch("redis.from_url") as mock_r:
            mock_r.return_value.ping.return_value = True
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/v1/health")
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_redis_unavailable_still_returns_200(self, app) -> None:
        with patch("redis.from_url") as mock_r:
            mock_r.return_value.ping.side_effect = Exception("connection refused")
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json()["redis"] == "unavailable"


# ---------------------------------------------------------------------------
# Tests: POST /v1/analyze
# ---------------------------------------------------------------------------

class TestSubmitImage:
    @pytest.mark.asyncio
    async def test_missing_auth_returns_403(self, app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/analyze",
                files={"file": ("test.jpg", _make_jpeg(), "image/jpeg")},
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_valid_jpeg_returns_202(self, app, _stub_jwt) -> None:
        with patch("forenscope.worker.tasks.analyze_image.apply_async"):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/analyze",
                    files={"file": ("test.jpg", _make_jpeg(), "image/jpeg")},
                    headers={"Authorization": "Bearer tok"},
                )
        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_response_has_job_id(self, app, _stub_jwt) -> None:
        with patch("forenscope.worker.tasks.analyze_image.apply_async"):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/analyze",
                    files={"file": ("test.jpg", _make_jpeg(), "image/jpeg")},
                    headers={"Authorization": "Bearer tok"},
                )
        body = resp.json()
        assert "job_id" in body
        assert "poll_url" in body

    @pytest.mark.asyncio
    async def test_poll_url_contains_job_id(self, app, _stub_jwt) -> None:
        with patch("forenscope.worker.tasks.analyze_image.apply_async"):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/analyze",
                    files={"file": ("test.jpg", _make_jpeg(), "image/jpeg")},
                    headers={"Authorization": "Bearer tok"},
                )
        body = resp.json()
        assert body["job_id"] in body["poll_url"]

    @pytest.mark.asyncio
    async def test_tiny_image_returns_422(self, app, _stub_jwt) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/analyze",
                files={"file": ("tiny.jpg", _make_jpeg((10, 10)), "image/jpeg")},
                headers={"Authorization": "Bearer tok"},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_valid_png_returns_202(self, app, _stub_jwt) -> None:
        with patch("forenscope.worker.tasks.analyze_image.apply_async"):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/analyze",
                    files={"file": ("test.png", _make_png(), "image/png")},
                    headers={"Authorization": "Bearer tok"},
                )
        assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Tests: GET /v1/jobs/{job_id}
# ---------------------------------------------------------------------------

class TestJobStatus:
    @pytest.mark.asyncio
    async def test_pending_job(self, app, _stub_jwt) -> None:
        job_id = str(uuid.uuid4())
        with patch("forenscope.api.routes.AsyncResult") as mock_result:
            mock_result.return_value.state = "PENDING"
            mock_result.return_value.info = None
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    f"/v1/jobs/{job_id}",
                    headers={"Authorization": "Bearer tok"},
                )
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    @pytest.mark.asyncio
    async def test_started_job_has_stage(self, app, _stub_jwt) -> None:
        job_id = str(uuid.uuid4())
        with patch("forenscope.api.routes.AsyncResult") as mock_result:
            mock_result.return_value.state = "STARTED"
            mock_result.return_value.info = {"stage": "inference"}
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    f"/v1/jobs/{job_id}",
                    headers={"Authorization": "Bearer tok"},
                )
        body = resp.json()
        assert body["status"] == "started"
        assert body["stage"] == "inference"

    @pytest.mark.asyncio
    async def test_successful_job(self, app, _stub_jwt) -> None:
        job_id = str(uuid.uuid4())
        with patch("forenscope.api.routes.AsyncResult") as mock_result:
            mock_result.return_value.state = "SUCCESS"
            mock_result.return_value.info = {"job_id": job_id, "overall_confidence": 0.9}
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    f"/v1/jobs/{job_id}",
                    headers={"Authorization": "Bearer tok"},
                )
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    async def test_job_status_requires_auth(self, app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(f"/v1/jobs/{uuid.uuid4()}")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Tests: report endpoints
# ---------------------------------------------------------------------------

class TestReportEndpoints:
    @pytest.mark.asyncio
    async def test_json_report_not_found(self, app, _stub_jwt) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                f"/v1/jobs/{uuid.uuid4()}/report",
                headers={"Authorization": "Bearer tok"},
            )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_pdf_report_not_found(self, app, _stub_jwt) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                f"/v1/jobs/{uuid.uuid4()}/report.pdf",
                headers={"Authorization": "Bearer tok"},
            )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_json_report_served_from_disk(
        self, app, _stub_jwt, tmp_path: Path
    ) -> None:
        job_id = str(uuid.uuid4())
        report_dir = tmp_path / job_id
        report_dir.mkdir()
        (report_dir / "report.json").write_text(
            json.dumps({"job_id": job_id, "overall_confidence": 0.85})
        )
        with patch("forenscope.api.routes.get_settings") as mock_cfg:
            s = MagicMock()
            s.output_dir = tmp_path
            mock_cfg.return_value = s
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    f"/v1/jobs/{job_id}/report",
                    headers={"Authorization": "Bearer tok"},
                )
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job_id
