"""Unit tests for API request/response Pydantic schemas."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError

from certainaity.api.schemas import (
    AnalysisReportResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    ReportRegionSchema,
    SubmitResponse,
)


class TestJobStatus:
    def test_string_values(self) -> None:
        assert JobStatus.PENDING == "pending"
        assert JobStatus.STARTED == "started"
        assert JobStatus.SUCCESS == "success"
        assert JobStatus.FAILURE == "failure"

    def test_is_string_type(self) -> None:
        assert isinstance(JobStatus.SUCCESS, str)

    def test_roundtrip_from_string(self) -> None:
        assert JobStatus("success") is JobStatus.SUCCESS


class TestSubmitResponse:
    def test_default_status_is_pending(self) -> None:
        r = SubmitResponse(job_id="abc", poll_url="/v1/jobs/abc")
        assert r.status == JobStatus.PENDING

    def test_job_id_stored(self) -> None:
        r = SubmitResponse(job_id="xyz-123", poll_url="/v1/jobs/xyz-123")
        assert r.job_id == "xyz-123"

    def test_poll_url_stored(self) -> None:
        r = SubmitResponse(job_id="j", poll_url="/v1/jobs/j")
        assert r.poll_url == "/v1/jobs/j"


class TestJobStatusResponse:
    def test_stage_defaults_to_none(self) -> None:
        r = JobStatusResponse(job_id="j", status=JobStatus.PENDING)
        assert r.stage is None

    def test_error_defaults_to_none(self) -> None:
        r = JobStatusResponse(job_id="j", status=JobStatus.PENDING)
        assert r.error is None

    def test_stage_propagated(self) -> None:
        r = JobStatusResponse(job_id="j", status=JobStatus.STARTED, stage="inference")
        assert r.stage == "inference"

    def test_error_propagated(self) -> None:
        r = JobStatusResponse(job_id="j", status=JobStatus.FAILURE, error="OOM")
        assert r.error == "OOM"


class TestReportRegionSchema:
    def test_valid_region(self) -> None:
        r = ReportRegionSchema(bbox=(0, 0, 100, 80), type="splicing", confidence=0.9, evidence="ELA")
        assert r.confidence == pytest.approx(0.9)

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReportRegionSchema(bbox=(0, 0, 10, 10), type="splicing", confidence=1.1, evidence="x")

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReportRegionSchema(bbox=(0, 0, 10, 10), type="splicing", confidence=-0.1, evidence="x")

    def test_confidence_at_boundaries_accepted(self) -> None:
        ReportRegionSchema(bbox=(0, 0, 10, 10), type="copy_move", confidence=0.0, evidence="none")
        ReportRegionSchema(bbox=(0, 0, 10, 10), type="copy_move", confidence=1.0, evidence="full")

    def test_bbox_tuple(self) -> None:
        r = ReportRegionSchema(bbox=(5, 10, 20, 30), type="removal", confidence=0.5, evidence="noise")
        assert r.bbox == (5, 10, 20, 30)


class TestAnalysisReportResponse:
    def test_valid_report(self) -> None:
        r = AnalysisReportResponse(
            job_id="j",
            file_name="f.jpg",
            sha256="a" * 64,
            analysis_timestamp="2026-01-01T00:00:00+00:00",
            manipulation_detected=True,
            overall_confidence=0.8,
            anti_forensic_warning=False,
            models_used=["patchforensic"],
            execution_time_ms=1000,
        )
        assert r.manipulation_detected is True

    def test_overall_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisReportResponse(
                job_id="j", file_name="f", sha256="x",
                analysis_timestamp="ts", manipulation_detected=False,
                overall_confidence=1.5, anti_forensic_warning=False,
                models_used=[], execution_time_ms=0,
            )

    def test_regions_default_to_empty(self) -> None:
        r = AnalysisReportResponse(
            job_id="j", file_name="f", sha256="x",
            analysis_timestamp="ts", manipulation_detected=False,
            overall_confidence=0.1, anti_forensic_warning=False,
            models_used=[], execution_time_ms=0,
        )
        assert r.regions == []


class TestHealthResponse:
    def test_defaults(self) -> None:
        h = HealthResponse()
        assert h.status == "ok"
        assert h.version == "1.0.0"
        assert h.redis == "unknown"

    def test_redis_override(self) -> None:
        h = HealthResponse(redis="ok")
        assert h.redis == "ok"

    def test_redis_unavailable(self) -> None:
        h = HealthResponse(redis="unavailable")
        assert h.redis == "unavailable"
