"""Unit tests for report generation and serialisation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from certainaity.report import (
    AnalysisReport,
    ReportRegion,
    generate_report,
    save_json_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def region() -> ReportRegion:
    return ReportRegion(bbox=(10, 20, 50, 60), type="splicing", confidence=0.85, evidence="ELA delta")


@pytest.fixture()
def clean_report() -> AnalysisReport:
    return generate_report(
        job_id="job-001",
        file_name="photo.jpg",
        sha256="a" * 64,
        overall_confidence=0.2,
        regions=[],
        models_used=["patchforensic"],
        anti_forensic_warning=False,
        execution_time_ms=500,
    )


@pytest.fixture()
def manipulated_report(region: ReportRegion) -> AnalysisReport:
    return generate_report(
        job_id="job-002",
        file_name="tampered.jpg",
        sha256="b" * 64,
        overall_confidence=0.9,
        regions=[region],
        models_used=["patchforensic", "mantranet"],
        anti_forensic_warning=True,
        execution_time_ms=1200,
    )


# ---------------------------------------------------------------------------
# Tests: generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_manipulation_detected_above_threshold(self, manipulated_report: AnalysisReport) -> None:
        assert manipulated_report.manipulation_detected is True

    def test_no_manipulation_below_threshold(self, clean_report: AnalysisReport) -> None:
        assert clean_report.manipulation_detected is False

    def test_threshold_at_exactly_half(self) -> None:
        r = generate_report(
            job_id="j", file_name="f.jpg", sha256="x",
            overall_confidence=0.5, regions=[], models_used=[],
            anti_forensic_warning=False, execution_time_ms=0,
        )
        assert r.manipulation_detected is False

    def test_fields_populated_correctly(self, manipulated_report: AnalysisReport) -> None:
        assert manipulated_report.job_id == "job-002"
        assert manipulated_report.file_name == "tampered.jpg"
        assert manipulated_report.overall_confidence == 0.9
        assert manipulated_report.anti_forensic_warning is True
        assert manipulated_report.execution_time_ms == 1200
        assert manipulated_report.models_used == ["patchforensic", "mantranet"]

    def test_timestamp_is_iso_format(self, clean_report: AnalysisReport) -> None:
        ts = clean_report.analysis_timestamp
        assert "T" in ts

    def test_timestamp_is_utc(self, clean_report: AnalysisReport) -> None:
        ts = clean_report.analysis_timestamp
        assert ts.endswith("+00:00") or "Z" in ts

    def test_regions_preserved(self, manipulated_report: AnalysisReport, region: ReportRegion) -> None:
        assert len(manipulated_report.regions) == 1
        assert manipulated_report.regions[0].type == "splicing"
        assert manipulated_report.regions[0].confidence == pytest.approx(0.85)

    def test_empty_regions_default(self, clean_report: AnalysisReport) -> None:
        assert clean_report.regions == []


# ---------------------------------------------------------------------------
# Tests: save_json_report
# ---------------------------------------------------------------------------


class TestSaveJsonReport:
    def test_writes_valid_json(self, tmp_path: Path, clean_report: AnalysisReport) -> None:
        out = tmp_path / "report.json"
        save_json_report(clean_report, out)
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

    def test_job_id_serialised(self, tmp_path: Path, clean_report: AnalysisReport) -> None:
        out = tmp_path / "report.json"
        save_json_report(clean_report, out)
        assert json.loads(out.read_text())["job_id"] == "job-001"

    def test_creates_parent_directories(self, tmp_path: Path, clean_report: AnalysisReport) -> None:
        out = tmp_path / "jobs" / "job-001" / "report.json"
        save_json_report(clean_report, out)
        assert out.exists()

    def test_all_custody_fields_present(self, tmp_path: Path, clean_report: AnalysisReport) -> None:
        out = tmp_path / "report.json"
        save_json_report(clean_report, out)
        data = json.loads(out.read_text())
        for field in ("job_id", "file_name", "sha256", "analysis_timestamp",
                      "manipulation_detected", "overall_confidence", "regions"):
            assert field in data, f"Missing field: {field!r}"

    def test_regions_serialised(self, tmp_path: Path, manipulated_report: AnalysisReport) -> None:
        out = tmp_path / "report.json"
        save_json_report(manipulated_report, out)
        data = json.loads(out.read_text())
        assert len(data["regions"]) == 1
        assert data["regions"][0]["type"] == "splicing"
        assert data["regions"][0]["bbox"] == [10, 20, 50, 60]

    def test_overwrite_existing_file(self, tmp_path: Path, clean_report: AnalysisReport) -> None:
        out = tmp_path / "report.json"
        out.write_text("old content")
        save_json_report(clean_report, out)
        data = json.loads(out.read_text())
        assert data["job_id"] == "job-001"


# ---------------------------------------------------------------------------
# Tests: ReportRegion
# ---------------------------------------------------------------------------


class TestReportRegion:
    def test_bbox_stored(self, region: ReportRegion) -> None:
        assert region.bbox == (10, 20, 50, 60)

    def test_type_stored(self, region: ReportRegion) -> None:
        assert region.type == "splicing"

    def test_confidence_stored(self, region: ReportRegion) -> None:
        assert region.confidence == pytest.approx(0.85)
