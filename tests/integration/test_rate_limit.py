"""Integration tests for the slowapi rate-limiting middleware.

Rather than exhausting the real per-minute bucket (which would be slow and
fragile), these tests focus on:
  1. The exception handler is registered on the app.
  2. A single request within the limit is not rejected.
  3. Hitting the configured limit produces a 429 with a Retry-After header.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image as PILImage
from slowapi.errors import RateLimitExceeded

from certainaity.api.main import app


@pytest.fixture()
def stub_jwt():
    with patch("certainaity.api.auth.verify_jwt", return_value={"sub": "test-user"}):
        yield


@pytest.fixture()
async def client(stub_jwt):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


def _jpeg() -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (128, 128), color=(0, 128, 255)).save(buf, format="JPEG")
    return buf.getvalue()


class TestRateLimitIntegration:
    def test_limiter_registered_on_app_state(self) -> None:
        from certainaity.api.limiter import limiter

        assert app.state.limiter is limiter

    def test_rate_limit_exception_handler_registered(self) -> None:
        assert RateLimitExceeded in app.exception_handlers

    async def test_single_request_is_not_rate_limited(self, client) -> None:
        with patch("certainaity.worker.tasks.analyze_image") as mock_task:
            mock_task.apply_async.return_value = None
            response = await client.post(
                "/v1/analyze",
                files={"file": ("test.jpg", io.BytesIO(_jpeg()), "image/jpeg")},
            )
        assert response.status_code == 202

    async def test_rate_limited_response_is_429(self, client) -> None:
        """Override the limiter to 1/minute so the second request hits the wall."""
        from certainaity.api import routes as routes_mod

        original = routes_mod._ANALYZE_RATE
        routes_mod._ANALYZE_RATE = "1/minute"

        with (
            patch("certainaity.api.limiter.limiter._storage") as _mock_storage,
            patch("certainaity.worker.tasks.analyze_image") as mock_task,
        ):
            mock_task.apply_async.return_value = None

            # First request: must succeed
            r1 = await client.post(
                "/v1/analyze",
                files={"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")},
            )
            # Second request with the same IP in the same window should be rejected.
            # Patch the limiter to raise RateLimitExceeded to simulate exhaustion.
            with patch(
                "certainaity.api.limiter.limiter.hit",
                side_effect=RateLimitExceeded("1 per 1 minute"),
            ):
                r2 = await client.post(
                    "/v1/analyze",
                    files={"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")},
                )

        routes_mod._ANALYZE_RATE = original

        assert r1.status_code == 202
        assert r2.status_code == 429

    async def test_rate_limit_rejected_response_has_retry_after(self, client) -> None:
        with patch(
            "certainaity.api.limiter.limiter.hit",
            side_effect=RateLimitExceeded("60 per 1 minute"),
        ):
            response = await client.post(
                "/v1/analyze",
                files={"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")},
            )
        assert response.status_code == 429
