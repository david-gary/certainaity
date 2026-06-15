"""Pydantic request / response schemas for the Certainaity REST API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"


class SubmitResponse(BaseModel):
    """Response returned immediately after a successful image submission."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    poll_url: str = Field(description="URL to poll for job progress")


class JobStatusResponse(BaseModel):
    """Current state of an analysis job."""

    job_id: str
    status: JobStatus
    stage: str | None = Field(
        default=None,
        description="Pipeline stage currently executing (ingest / features / inference / …)",
    )
    error: str | None = None


class ReportRegionSchema(BaseModel):
    """One detected manipulation region inside an AnalysisReportResponse."""

    bbox: tuple[int, int, int, int] = Field(description="Bounding box as (x, y, w, h) in pixels")
    type: str = Field(description="splicing | copy_move | removal | ai_inpainting")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class AnalysisReportResponse(BaseModel):
    """Full forensic analysis result (mirrors AnalysisReport dataclass)."""

    job_id: str
    file_name: str
    sha256: str
    analysis_timestamp: str
    manipulation_detected: bool
    overall_confidence: float = Field(ge=0.0, le=1.0)
    regions: list[ReportRegionSchema] = Field(default_factory=list)
    anti_forensic_warning: bool
    models_used: list[str]
    execution_time_ms: int


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = "ok"
    version: str = "1.0.0"
    redis: str = Field(default="unknown", description="ok | unavailable")
