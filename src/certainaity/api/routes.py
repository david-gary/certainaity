"""FastAPI route handlers for the Certainaity v1 API.

Routes
------
GET  /v1/health                     — liveness probe (no auth)
POST /v1/analyze                    — submit an image for analysis
GET  /v1/jobs/{job_id}              — poll job status
GET  /v1/jobs/{job_id}/report       — retrieve JSON report
GET  /v1/jobs/{job_id}/report.pdf   — retrieve PDF report
"""

from __future__ import annotations

import base64
import json
import os
import uuid

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from certainaity.api.auth import verify_jwt
from certainaity.api.limiter import limiter
from certainaity.api.metrics import JOBS_REJECTED, JOBS_SUBMITTED
from certainaity.api.schemas import (
    AnalysisReportResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    SubmitResponse,
)
from certainaity.config import get_settings
from certainaity.exceptions import (
    CorruptImageError,
    FileTooLargeError,
    ImageTooSmallError,
    UnsupportedFormatError,
)

log = structlog.get_logger()
router = APIRouter(prefix="/v1")

# Read from env at import time; restart the process to pick up changes.
_ANALYZE_RATE = f"{os.environ.get('CERTAINAITY_RATE_LIMIT_PER_MINUTE', '60')}/minute"


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness probe — called every 30 s by nginx and ECS health checks."""
    import redis as redis_lib

    settings = get_settings()
    try:
        r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
        r.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "unavailable"

    return HealthResponse(redis=redis_status)


@router.post(
    "/analyze",
    response_model=SubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["analysis"],
)
@limiter.limit(_ANALYZE_RATE)
async def submit_image(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, TIFF, WEBP ≤ 50 MB)"),
    _token: dict = Depends(verify_jwt),
) -> SubmitResponse:
    """Accept an image upload, validate it, and enqueue a Celery analysis task.

    The file is read entirely into memory here so that format and size
    validation can happen synchronously before the 202 response is returned.
    """
    from certainaity.ingest import ingest_image
    from certainaity.worker.tasks import analyze_image

    raw = await file.read()

    try:
        ingest_image(raw)
    except FileTooLargeError as exc:
        JOBS_REJECTED.labels(reason="too_large").inc()
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except (UnsupportedFormatError, CorruptImageError, ImageTooSmallError) as exc:
        JOBS_REJECTED.labels(reason="invalid_image").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    job_id = str(uuid.uuid4())
    image_b64 = base64.b64encode(raw).decode()
    analyze_image.apply_async(
        args=[job_id, image_b64, file.filename or "upload"],
        task_id=job_id,
    )
    JOBS_SUBMITTED.inc()
    log.info("job_enqueued", job_id=job_id, filename=file.filename)
    return SubmitResponse(job_id=job_id, poll_url=f"/v1/jobs/{job_id}")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["analysis"])
async def get_job_status(
    job_id: str,
    _token: dict = Depends(verify_jwt),
) -> JobStatusResponse:
    """Return the current Celery task state for *job_id*."""
    from celery.result import AsyncResult

    from certainaity.worker.app import celery_app

    result = AsyncResult(job_id, app=celery_app)
    celery_state = result.state

    state_map: dict[str, JobStatus] = {
        "PENDING": JobStatus.PENDING,
        "STARTED": JobStatus.STARTED,
        "SUCCESS": JobStatus.SUCCESS,
        "FAILURE": JobStatus.FAILURE,
        "RETRY": JobStatus.STARTED,
    }
    job_status = state_map.get(celery_state, JobStatus.PENDING)

    meta = result.info or {}
    stage = meta.get("stage") if isinstance(meta, dict) else None
    error = str(meta) if celery_state == "FAILURE" else None

    return JobStatusResponse(job_id=job_id, status=job_status, stage=stage, error=error)


@router.get("/jobs/{job_id}/report", tags=["analysis"])
async def get_report_json(
    job_id: str,
    _token: dict = Depends(verify_jwt),
):
    """Return the structured JSON report for a completed job."""
    settings = get_settings()
    report_path = settings.output_dir / job_id / "report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report not found for job {job_id!r}. "
            "The job may still be running or may have failed.",
        )
    return json.loads(report_path.read_text())


@router.get("/jobs/{job_id}/report.pdf", tags=["analysis"])
async def get_report_pdf(
    job_id: str,
    _token: dict = Depends(verify_jwt),
) -> FileResponse:
    """Stream the PDF forensic report for a completed job."""
    settings = get_settings()
    pdf_path = settings.output_dir / job_id / "report.pdf"
    if not pdf_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF report not found for job {job_id!r}.",
        )
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"certainaity_{job_id}.pdf",
    )
