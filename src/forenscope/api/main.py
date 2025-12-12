"""FastAPI application factory.

The Dockerfile resolves ``forenscope.api.main:app`` as the ASGI entrypoint.
"""

from __future__ import annotations

import re
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from forenscope.api.metrics import REQUEST_LATENCY, REQUESTS_TOTAL
from forenscope.api.routes import router
from forenscope.config import get_settings

log = structlog.get_logger()

# Collapse /v1/jobs/<uuid> variants into a single label to prevent cardinality explosion.
_JOB_PATH_RE = re.compile(
    r"/v1/jobs/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def _endpoint_label(path: str) -> str:
    return _JOB_PATH_RE.sub("/v1/jobs/{job_id}", path)


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    log.info(
        "forenscope_api_starting",
        weights_dir=str(settings.weights_dir),
        output_dir=str(settings.output_dir),
        redis_url=settings.redis_url,
    )
    yield
    log.info("forenscope_api_stopping")


def create_app() -> FastAPI:
    """Construct and return the FastAPI application."""
    application = FastAPI(
        title="ForenScope API",
        description="Forensic image manipulation detection — REST API",
        version="1.0.0",
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["Content-Disposition"],
    )

    @application.middleware("http")
    async def _access_log(request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start

        endpoint = _endpoint_label(request.url.path)
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()

        log.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=int(elapsed * 1000),
            client=request.client.host if request.client else None,
        )
        return response

    @application.get("/metrics", include_in_schema=False)
    async def _prometheus_metrics() -> Response:
        """Prometheus scrape endpoint — not authenticated; internal use only."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    application.include_router(router)
    return application


# Module-level singleton consumed by uvicorn and the Dockerfile CMD.
app = create_app()
