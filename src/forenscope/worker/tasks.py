"""Celery task definitions for the ForenScope analysis pipeline.

The ``analyze_image`` task orchestrates the full forensic pipeline:
  1. Ingest & validate (hash, format check, EXIF extraction)
  2. Feature extraction (ELA, noise variance, CFA, DCT)   — TODO: wire up
  3. Ensemble inference (4 models + weighted fusion)        — TODO: wire up
  4. Resilience test (re-compress × 3 qualities)            — TODO: wire up
  5. Report generation (JSON + PDF written to output_dir)
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import structlog

from forenscope.config import get_settings
from forenscope.ingest import ingest_image
from forenscope.report import ReportRegion, generate_report, save_json_report, save_pdf_report
from forenscope.worker.app import celery_app

log = structlog.get_logger()


class _AnalysisTask(celery_app.Task):  # type: ignore[misc]
    """Base task that holds ensemble as a class-level singleton (lazy init)."""

    abstract = True
    _ensemble = None

    @property
    def ensemble(self):
        if self._ensemble is None:
            from forenscope.models import Ensemble

            settings = get_settings()
            self._ensemble = Ensemble(settings.weights_dir)
        return self._ensemble


@celery_app.task(
    bind=True,
    base=_AnalysisTask,
    name="forenscope.analyze_image",
    max_retries=2,
    default_retry_delay=10,
    soft_time_limit=300,
    time_limit=360,
)
def analyze_image(
    self,  # type: ignore[override]
    job_id: str,
    image_b64: str,
    file_name: str,
) -> dict:
    """Run a full forensic analysis and write JSON + PDF reports.

    Args:
        job_id:    UUID assigned by the API at submission time.
        image_b64: Base64-encoded raw image bytes (JSON-serializable).
        file_name: Original filename from the multipart upload.

    Returns:
        Dict with ``job_id``, ``sha256``, and ``overall_confidence`` keys.
    """
    settings = get_settings()
    started_at = time.monotonic()

    def _update(stage: str) -> None:
        try:
            self.update_state(state="STARTED", meta={"stage": stage})
        except Exception:
            pass

    # ── Stage 1: ingest & validate ──────────────────────────────────────────
    _update("ingest")
    image_bytes = base64.b64decode(image_b64)
    ingested = ingest_image(image_bytes)
    log.info(
        "image_ingested",
        job_id=job_id,
        sha256=ingested.sha256,
        format=ingested.format,
        size=(ingested.width, ingested.height),
    )

    # ── Stage 2: feature extraction ─────────────────────────────────────────
    _update("features")
    # TODO: features = extract_all_features(ingested, settings)

    # ── Stage 3: ensemble inference ─────────────────────────────────────────
    _update("inference")
    # TODO: localization = self.ensemble.localize(
    #     np.array(ingested.image), return_model_maps=True
    # )
    # Stub result used until weights are available:
    overall_confidence: float = 0.0
    regions: list[ReportRegion] = []
    heatmap = None

    # ── Stage 4: resilience test ─────────────────────────────────────────────
    _update("resilience")
    # TODO: anti_forensic = _run_resilience_test(image_bytes, self.ensemble, settings)
    anti_forensic_warning: bool = False

    # ── Stage 5: report generation ──────────────────────────────────────────
    _update("report")
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    report = generate_report(
        job_id=job_id,
        file_name=file_name,
        sha256=ingested.sha256,
        overall_confidence=overall_confidence,
        regions=regions,
        models_used=["PatchForensic", "MantraNet", "SPSL", "InpaintingDetector"],
        anti_forensic_warning=anti_forensic_warning,
        execution_time_ms=elapsed_ms,
    )

    output_dir: Path = settings.output_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_report(report, output_dir / "report.json")
    save_pdf_report(report, heatmap, output_dir / "report.pdf")

    log.info("analysis_complete", job_id=job_id, elapsed_ms=elapsed_ms)
    return {
        "job_id": job_id,
        "sha256": ingested.sha256,
        "overall_confidence": overall_confidence,
    }
