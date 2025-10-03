"""FastAPI application factory.

The Dockerfile resolves ``forenscope.api.main:app`` as the ASGI entrypoint.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from forenscope.api.routes import router
from forenscope.config import get_settings

log = structlog.get_logger()


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
        elapsed_ms = int((time.monotonic() - start) * 1000)
        log.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
            client=request.client.host if request.client else None,
        )
        return response

    application.include_router(router)
    return application


# Module-level singleton consumed by uvicorn and the Dockerfile CMD.
app = create_app()
