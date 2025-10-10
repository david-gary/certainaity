"""Integration tests for the forensic report generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from forenscope.report import (
    AnalysisReport,
    ReportRegion,
    generate_report,
    save_json_report,
    save_pdf_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_report() -> AnalysisReport:
    return generate_report(
        job_id="test-job-001",
        file_name="evidence_001.jpg",
        sha256="a1b2c3d4" * 8,
        overall_confidence=0.92,
        regions=[
            ReportRegion(
                bbox=(100, 200, 300, 400),
                type="splicing",
                confidence=0.96,
                evidence="CFA mismatch (p=0.003), noise variance discontinuity",
            ),
            ReportRegion(
                bbox=(500, 50, 80, 80),
                type="ai_inpainting",
                confidence=0.88,
                evidence="CLIP attention rollout > 0.82",
            ),
        ],
        models_used=["PatchForensic", "MantraNet", "SPSL", "InpaintingDetector"],
        anti_forensic_warning=False,
        execution_time_ms=412,
    )


@pytest.fixture()
def heatmap_128() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((128, 128)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_manipulation_detected_above_threshold(self, sample_report: AnalysisReport) -> None:
        assert sample_report.manipulation_detected is True

    def test_no_manipulation_below_threshold(self) -> None:
        r = generate_report(
            job_id="x",
            file_name="clean.jpg",
            sha256="b" * 64,
            overall_confidence=0.2,
            regions=[],
            models_used=[],
            anti_forensic_warning=False,
            execution_time_ms=50,
        )
        assert r.manipulation_detected is False

    def test_timestamp_is_utc(self, sample_report: AnalysisReport) -> None:
        assert sample_report.analysis_timestamp.endswith("+00:00")

    def test_timestamp_has_t_separator(self, sample_report: AnalysisReport) -> None:
        assert "T" in sample_report.analysis_timestamp

    def test_region_count(self, sample_report: AnalysisReport) -> None:
        assert len(sample_report.regions) == 2

    def test_anti_forensic_flag_propagated(self) -> None:
        r = generate_report(
            job_id="y",
            file_name="suspicious.jpg",
            sha256="c" * 64,
            overall_confidence=0.7,
            regions=[],
            models_used=[],
            anti_forensic_warning=True,
            execution_time_ms=200,
        )
        assert r.anti_forensic_warning is True


# ---------------------------------------------------------------------------
# Tests: save_json_report
# ---------------------------------------------------------------------------

class TestSaveJsonReport:
    def test_file_is_created(self, tmp_path: Path, sample_report: AnalysisReport) -> None:
        out = tmp_path / "report.json"
        save_json_report(sample_report, out)
        assert out.exists()

    def test_json_is_valid(self, tmp_path: Path, sample_report: AnalysisReport) -> None:
        out = tmp_path / "report.json"
        save_json_report(sample_report, out)
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

    def test_sha256_round_trips(self, tmp_path: Path, sample_report: AnalysisReport) -> None:
        out = tmp_path / "report.json"
        save_json_report(sample_report, out)
        data = json.loads(out.read_text())
        assert data["sha256"] == sample_report.sha256

    def test_overall_confidence_round_trips(
        self, tmp_path: Path, sample_report: AnalysisReport
    ) -> None:
        out = tmp_path / "report.json"
        save_json_report(sample_report, out)
        data = json.loads(out.read_text())
        assert abs(data["overall_confidence"] - 0.92) < 1e-6

    def test_regions_serialized(self, tmp_path: Path, sample_report: AnalysisReport) -> None:
        out = tmp_path / "report.json"
        save_json_report(sample_report, out)
        data = json.loads(out.read_text())
        assert len(data["regions"]) == 2
        assert data["regions"][0]["type"] == "splicing"

    def test_creates_parent_directory(self, tmp_path: Path, sample_report: AnalysisReport) -> None:
        nested = tmp_path / "jobs" / "test-job-001" / "report.json"
        save_json_report(sample_report, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Tests: save_pdf_report
# ---------------------------------------------------------------------------

class TestSavePdfReport:
    def test_pdf_is_created(self, tmp_path: Path, sample_report: AnalysisReport) -> None:
        pytest.importorskip("reportlab")
        out = tmp_path / "report.pdf"
        save_pdf_report(sample_report, heatmap=None, path=out)
        assert out.exists()

    def test_pdf_has_nonzero_size(self, tmp_path: Path, sample_report: AnalysisReport) -> None:
        pytest.importorskip("reportlab")
        out = tmp_path / "report.pdf"
        save_pdf_report(sample_report, heatmap=None, path=out)
        assert out.stat().st_size > 2048

    def test_pdf_magic_bytes(self, tmp_path: Path, sample_report: AnalysisReport) -> None:
        pytest.importorskip("reportlab")
        out = tmp_path / "report.pdf"
        save_pdf_report(sample_report, heatmap=None, path=out)
        assert out.read_bytes()[:4] == b"%PDF"

    def test_pdf_with_heatmap(
        self, tmp_path: Path, sample_report: AnalysisReport, heatmap_128: np.ndarray
    ) -> None:
        pytest.importorskip("reportlab")
        out = tmp_path / "report_heatmap.pdf"
        save_pdf_report(sample_report, heatmap=heatmap_128, path=out)
        assert out.exists()
        assert out.stat().st_size > 2048

    def test_pdf_larger_with_heatmap(
        self, tmp_path: Path, sample_report: AnalysisReport, heatmap_128: np.ndarray
    ) -> None:
        pytest.importorskip("reportlab")
        no_hm = tmp_path / "no_hm.pdf"
        with_hm = tmp_path / "with_hm.pdf"
        save_pdf_report(sample_report, heatmap=None, path=no_hm)
        save_pdf_report(sample_report, heatmap=heatmap_128, path=with_hm)
        assert with_hm.stat().st_size > no_hm.stat().st_size

    def test_pdf_anti_forensic_report(self, tmp_path: Path) -> None:
        pytest.importorskip("reportlab")
        r = generate_report(
            job_id="af-job",
            file_name="suspect.jpg",
            sha256="d" * 64,
            overall_confidence=0.71,
            regions=[],
            models_used=["PatchForensic"],
            anti_forensic_warning=True,
            execution_time_ms=999,
        )
        out = tmp_path / "af_report.pdf"
        save_pdf_report(r, heatmap=None, path=out)
        assert out.exists()
