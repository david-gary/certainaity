"""Forensic analysis report generation: structured JSON and ReportLab PDF."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class ReportRegion:
    bbox: tuple[int, int, int, int]  # x, y, w, h in pixels
    type: str  # "splicing" | "copy_move" | "removal" | "ai_inpainting"
    confidence: float
    evidence: str


@dataclass
class AnalysisReport:
    job_id: str
    file_name: str
    sha256: str
    analysis_timestamp: str
    manipulation_detected: bool
    overall_confidence: float
    regions: list[ReportRegion] = field(default_factory=list)
    anti_forensic_warning: bool = False
    models_used: list[str] = field(default_factory=list)
    execution_time_ms: int = 0


def generate_report(
    job_id: str,
    file_name: str,
    sha256: str,
    overall_confidence: float,
    regions: list[ReportRegion],
    models_used: list[str],
    anti_forensic_warning: bool,
    execution_time_ms: int,
) -> AnalysisReport:
    """Build an AnalysisReport dataclass from pipeline outputs."""
    return AnalysisReport(
        job_id=job_id,
        file_name=file_name,
        sha256=sha256,
        analysis_timestamp=datetime.now(tz=timezone.utc).isoformat(),
        manipulation_detected=overall_confidence > 0.5,
        overall_confidence=overall_confidence,
        regions=regions,
        anti_forensic_warning=anti_forensic_warning,
        models_used=models_used,
        execution_time_ms=execution_time_ms,
    )


def save_json_report(report: AnalysisReport, path: Path) -> None:
    """Serialise the report to a JSON file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2))
    log.info("json_report_saved", path=str(path))


def save_pdf_report(
    report: AnalysisReport,
    heatmap: np.ndarray | None,
    path: Path,
) -> None:
    """Render a ReportLab PDF containing all report sections.

    Sections: header, metadata table, verdict, optional heatmap image,
    detected-regions table, and chain-of-custody statement.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("ForenScope Forensic Analysis Report", styles["h1"]))
    story.append(Spacer(1, 0.3 * cm))

    # ── Metadata table ──────────────────────────────────────────────────────
    meta_rows = [
        ["File", report.file_name],
        ["Job ID", report.job_id],
        ["SHA-256", report.sha256],
        ["Timestamp", report.analysis_timestamp],
        ["Execution time", f"{report.execution_time_ms} ms"],
    ]
    meta_table = Table(meta_rows, colWidths=[4 * cm, 12 * cm])
    meta_table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
                ("FONT", (1, 0), (1, -1), "Helvetica", 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 0.5 * cm))

    # ── Verdict ─────────────────────────────────────────────────────────────
    if report.manipulation_detected:
        verdict_html = (
            f'<font color="#c0392b"><b>MANIPULATION DETECTED</b></font>'
            f' &mdash; overall confidence {report.overall_confidence:.1%}'
        )
    else:
        verdict_html = (
            f'<font color="#27ae60"><b>NO MANIPULATION DETECTED</b></font>'
            f' &mdash; overall confidence {report.overall_confidence:.1%}'
        )
    story.append(Paragraph(verdict_html, styles["h2"]))

    if report.anti_forensic_warning:
        story.append(
            Paragraph(
                '<font color="#e67e22"><b>WARNING:</b></font> Anti-forensic processing '
                "suspected &mdash; confidence dropped &gt;25% under re-compression.",
                styles["Normal"],
            )
        )
    story.append(Spacer(1, 0.4 * cm))

    # ── Heatmap ─────────────────────────────────────────────────────────────
    if heatmap is not None:
        story.append(Paragraph("Manipulation Heatmap", styles["h3"]))
        from PIL import Image as PILImage

        hm_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(hm_uint8, mode="L").convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        story.append(RLImage(buf, width=12 * cm, height=12 * cm))
        story.append(Spacer(1, 0.4 * cm))

    # ── Regions table ───────────────────────────────────────────────────────
    if report.regions:
        story.append(Paragraph("Detected Regions", styles["h3"]))
        header = [["#", "Bounding Box (x,y,w,h)", "Type", "Confidence", "Evidence"]]
        rows = [
            [
                str(i),
                f"({r.bbox[0]}, {r.bbox[1]}, {r.bbox[2]}, {r.bbox[3]})",
                r.type,
                f"{r.confidence:.1%}",
                r.evidence,
            ]
            for i, r in enumerate(report.regions, 1)
        ]
        table_data = header + rows
        col_widths = [0.8 * cm, 3.8 * cm, 3 * cm, 2.2 * cm, 6.2 * cm]
        region_table = Table(table_data, colWidths=col_widths)

        region_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 7),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        for row_idx in range(2, len(table_data), 2):
            region_style.append(
                ("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#f8f8f8"))
            )
        region_table.setStyle(TableStyle(region_style))
        story.append(region_table)
        story.append(Spacer(1, 0.4 * cm))

    # ── Chain-of-custody ────────────────────────────────────────────────────
    story.append(Paragraph("Chain of Custody", styles["h3"]))
    models_str = ", ".join(report.models_used) if report.models_used else "none"
    custody = (
        f"This report was generated by ForenScope v1.0.0 at {report.analysis_timestamp}. "
        f"The source file SHA-256 digest ({report.sha256}) was computed from the raw byte "
        "stream prior to any decoding or processing, preserving evidentiary integrity. "
        f"Analysis models applied: {models_str}. "
        "This document is a machine-generated forensic summary and must be reviewed by a "
        "qualified digital forensics examiner before use in legal proceedings."
    )
    story.append(Paragraph(custody, styles["Normal"]))

    doc.build(story)
    log.info("pdf_report_saved", path=str(path))
