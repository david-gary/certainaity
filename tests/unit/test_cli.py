"""Unit tests for the certainaity CLI entrypoint."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image as PILImage

from certainaity.cli import _build_parser, main


def _write_jpeg(path: Path, size: tuple[int, int] = (128, 128)) -> None:
    PILImage.new("RGB", size, color=(100, 150, 200)).save(path, format="JPEG")


# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_analyze_subcommand_parses_image_path(self) -> None:
        args = _build_parser().parse_args(["analyze", "img.jpg"])
        assert args.command == "analyze"
        assert args.image == "img.jpg"

    def test_job_id_defaults_to_none(self) -> None:
        args = _build_parser().parse_args(["analyze", "img.jpg"])
        assert args.job_id is None

    def test_job_id_override(self) -> None:
        args = _build_parser().parse_args(["analyze", "img.jpg", "--job-id", "case-001"])
        assert args.job_id == "case-001"

    def test_output_json_defaults_to_none(self) -> None:
        args = _build_parser().parse_args(["analyze", "img.jpg"])
        assert args.output_json is None

    def test_output_pdf_defaults_to_none(self) -> None:
        args = _build_parser().parse_args(["analyze", "img.jpg"])
        assert args.output_pdf is None

    def test_no_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])


# ---------------------------------------------------------------------------
# End-to-end CLI tests
# ---------------------------------------------------------------------------

class TestAnalyzeCmd:
    def test_missing_file_exits_with_1(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            sys, "argv", ["certainaity", "analyze", str(tmp_path / "ghost.jpg")]
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_corrupt_file_exits_with_1(self, tmp_path: Path, monkeypatch) -> None:
        bad = tmp_path / "corrupt.jpg"
        bad.write_bytes(b"not an image")
        monkeypatch.setattr(sys, "argv", ["certainaity", "analyze", str(bad)])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_valid_jpeg_exits_with_0(self, tmp_path: Path, monkeypatch) -> None:
        img = tmp_path / "photo.jpg"
        _write_jpeg(img)
        monkeypatch.setattr(sys, "argv", ["certainaity", "analyze", str(img)])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0

    def test_stdout_is_valid_json(self, tmp_path: Path, monkeypatch, capsys) -> None:
        img = tmp_path / "photo.jpg"
        _write_jpeg(img)
        monkeypatch.setattr(sys, "argv", ["certainaity", "analyze", str(img)])
        with pytest.raises(SystemExit):
            main()
        result = json.loads(capsys.readouterr().out)
        assert isinstance(result, dict)

    def test_output_contains_sha256(self, tmp_path: Path, monkeypatch, capsys) -> None:
        img = tmp_path / "evidence.jpg"
        _write_jpeg(img)
        monkeypatch.setattr(sys, "argv", ["certainaity", "analyze", str(img)])
        with pytest.raises(SystemExit):
            main()
        result = json.loads(capsys.readouterr().out)
        assert len(result["sha256"]) == 64

    def test_default_job_id_is_filename_stem(self, tmp_path: Path, monkeypatch, capsys) -> None:
        img = tmp_path / "exhibit_a.jpg"
        _write_jpeg(img)
        monkeypatch.setattr(sys, "argv", ["certainaity", "analyze", str(img)])
        with pytest.raises(SystemExit):
            main()
        assert json.loads(capsys.readouterr().out)["job_id"] == "exhibit_a"

    def test_custom_job_id_propagates(self, tmp_path: Path, monkeypatch, capsys) -> None:
        img = tmp_path / "photo.jpg"
        _write_jpeg(img)
        monkeypatch.setattr(
            sys, "argv", ["certainaity", "analyze", str(img), "--job-id", "case-999"]
        )
        with pytest.raises(SystemExit):
            main()
        assert json.loads(capsys.readouterr().out)["job_id"] == "case-999"

    def test_stub_pipeline_reports_no_manipulation(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        img = tmp_path / "clean.jpg"
        _write_jpeg(img)
        monkeypatch.setattr(sys, "argv", ["certainaity", "analyze", str(img)])
        with pytest.raises(SystemExit):
            main()
        result = json.loads(capsys.readouterr().out)
        assert result["manipulation_detected"] is False
        assert result["overall_confidence"] == 0.0

    def test_output_json_flag_writes_file(self, tmp_path: Path, monkeypatch, capsys) -> None:
        img = tmp_path / "photo.jpg"
        _write_jpeg(img)
        out = tmp_path / "report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["certainaity", "analyze", str(img), "--output-json", str(out)],
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "sha256" in data

    def test_output_json_contains_all_custody_fields(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        img = tmp_path / "photo.jpg"
        _write_jpeg(img)
        out = tmp_path / "report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["certainaity", "analyze", str(img), "--output-json", str(out)],
        )
        with pytest.raises(SystemExit):
            main()
        data = json.loads(out.read_text())
        for field in (
            "job_id", "sha256", "file_name", "analysis_timestamp",
            "manipulation_detected", "overall_confidence",
        ):
            assert field in data, f"Missing field: {field!r}"
