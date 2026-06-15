"""Certainaity command-line tool for local forensic image analysis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _cmd_analyze(args: argparse.Namespace) -> int:
    from certainaity.ingest import ingest_image
    from certainaity.report import generate_report, save_json_report, save_pdf_report

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"error: file not found: {image_path}", file=sys.stderr)
        return 1

    t0 = time.monotonic()

    try:
        ingested = ingest_image(image_path)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Feature extraction and model inference are TODO stubs; confidence is 0.0
    # until weights are loaded and the pipeline stages are wired.
    report = generate_report(
        job_id=args.job_id or image_path.stem,
        file_name=image_path.name,
        sha256=ingested.sha256,
        overall_confidence=0.0,
        regions=[],
        models_used=[],
        anti_forensic_warning=False,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    )

    result = {
        "job_id": report.job_id,
        "file_name": report.file_name,
        "sha256": report.sha256,
        "analysis_timestamp": report.analysis_timestamp,
        "manipulation_detected": report.manipulation_detected,
        "overall_confidence": report.overall_confidence,
        "anti_forensic_warning": report.anti_forensic_warning,
        "regions": len(report.regions),
        "execution_time_ms": report.execution_time_ms,
    }
    print(json.dumps(result, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        save_json_report(report, out)
        print(f"JSON report written to {out}", file=sys.stderr)

    if args.output_pdf:
        out = Path(args.output_pdf)
        try:
            save_pdf_report(report, None, out)
            print(f"PDF report written to {out}", file=sys.stderr)
        except ImportError:
            print("warning: reportlab not installed; PDF output skipped", file=sys.stderr)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="certainaity",
        description="Certainaity forensic image manipulation detector",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Run local forensic analysis on a single image file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analyze.add_argument("image", help="Path to the image file (JPEG, PNG, TIFF, WEBP)")
    analyze.add_argument(
        "--job-id",
        metavar="ID",
        default=None,
        help="Override the job ID (default: image filename stem)",
    )
    analyze.add_argument(
        "--output-json",
        metavar="PATH",
        default=None,
        help="Write full JSON report to PATH",
    )
    analyze.add_argument(
        "--output-pdf",
        metavar="PATH",
        default=None,
        help="Write PDF report to PATH (requires reportlab)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "analyze":
        sys.exit(_cmd_analyze(args))


if __name__ == "__main__":
    main()
