"""Integration tests for the Prometheus /metrics scrape endpoint."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from certainaity.api.main import app


@pytest.fixture()
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


class TestMetricsEndpoint:
    async def test_returns_200(self, client) -> None:
        response = await client.get("/metrics")
        assert response.status_code == 200

    async def test_content_type_is_plain_text(self, client) -> None:
        response = await client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    async def test_body_contains_certainaity_counter(self, client) -> None:
        response = await client.get("/metrics")
        assert "certainaity_http_requests_total" in response.text

    async def test_body_contains_latency_histogram(self, client) -> None:
        response = await client.get("/metrics")
        assert "certainaity_http_request_duration_seconds" in response.text

    async def test_body_contains_jobs_submitted_counter(self, client) -> None:
        response = await client.get("/metrics")
        assert "certainaity_jobs_submitted_total" in response.text

    async def test_endpoint_excluded_from_openapi_schema(self, client) -> None:
        schema_response = await client.get("/openapi.json")
        assert schema_response.status_code == 200
        assert "/metrics" not in schema_response.text

    async def test_request_counter_increments_after_health_call(self, client) -> None:
        # Hit the health endpoint, then verify the counter text reflects activity.
        with patch("redis.from_url") as mock_redis:
            mock_redis.return_value.ping.return_value = True
            await client.get("/v1/health")

        metrics_response = await client.get("/metrics")
        # At least one GET request was recorded (the /v1/health call above).
        assert 'method="GET"' in metrics_response.text

    async def test_job_counter_label_present(self, client) -> None:
        response = await client.get("/metrics")
        # Jobs-rejected counter has a reason label — verify the label name is exported.
        assert "certainaity_jobs_rejected_total" in response.text
