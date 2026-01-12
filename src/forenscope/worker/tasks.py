"""Celery task definitions for the ForenScope analysis pipeline.

The ``analyze_image`` task orchestrates the full forensic pipeline:
  1. Ingest & validate (hash, format check, EXIF extraction)
  2. Feature extraction (ELA, noise variance, CFA, DCT) — parallel threads
  3. Ensemble inference (4 models + weighted fusion)
  4. Resilience test (re-compress × N qualities; detect anti-forensic evasion)
  5. Report generation (JSON + PDF written to output_dir)
"""

from __future__ import annotations

import base64
import concurrent.futures
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

from forenscope.config import get_settings
from forenscope.exceptions import WeightsNotFoundError
from forenscope.ingest import ingest_image
from forenscope.report import ReportRegion, generate_report, save_json_report, save_pdf_report
from forenscope.worker.app import celery_app

if TYPE_CHECKING:
    from forenscope.models.ensemble import LocalizationResult

log = structlog.get_logger()


def _extract_features(image: object, settings: object) -> dict[str, np.ndarray]:
    """Run all four handcrafted feature extractors in parallel threads.

    Returns a dict with keys ``ela``, ``noise``, ``cfa``, ``dct``, each
    mapping to a (H, W) float32 ndarray in [0, 1].
    """
    from forenscope.features.cfa import compute_cfa_map
    from forenscope.features.dct import compute_dct_similarity
    from forenscope.features.ela import compute_ela
    from forenscope.features.noise import compute_noise_map

    ela_q = getattr(settings, "ela_quality", 75)
    noise_bs = getattr(settings, "noise_block_size", 32)
    dct_bs = getattr(settings, "dct_block_size", 8)
    workers = getattr(settings, "feature_workers", 4)

    extractors: dict[str, object] = {
        "ela": lambda img: compute_ela(img, quality=ela_q),
        "noise": lambda img: compute_noise_map(img, block_size=noise_bs),
        "cfa": compute_cfa_map,
        "dct": lambda img: compute_dct_similarity(img, block_size=dct_bs),
    }

    results: dict[str, np.ndarray] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {name: pool.submit(fn, image) for name, fn in extractors.items()}  # type: ignore[operator]
        for name, future in futures.items():
            results[name] = future.result()
    return results


def _localize_to_regions(
    localization: LocalizationResult,
    feature_maps: dict[str, np.ndarray],
) -> list[ReportRegion]:
    """Convert ensemble LocalizationResult to a list of ReportRegion objects.

    Each connected component in ``localization.region_labels`` becomes one
    region. Manipulation type is inferred from whichever per-signal score is
    highest inside the region.
    """
    regions: list[ReportRegion] = []
    labels = localization.region_labels
    heatmap = localization.heatmap

    for region_id in range(1, localization.num_regions + 1):
        mask = labels == region_id
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
        confidence = float(heatmap[mask].mean())

        ela_sig = float(feature_maps["ela"][mask].mean())
        dct_sig = float(feature_maps["dct"][mask].mean())
        cfa_sig = float(feature_maps["cfa"][mask].mean())
        inp_map = localization.model_maps.get("InpaintingDetector", heatmap)
        inp_sig = float(inp_map[mask].mean())

        signal_scores: dict[str, float] = {
            "splicing": cfa_sig,
            "copy_move": dct_sig,
            "ai_inpainting": inp_sig,
            "removal": ela_sig,
        }
        manip_type = max(signal_scores, key=lambda k: signal_scores[k])

        evidence_parts: list[str] = []
        if ela_sig > 0.3:
            evidence_parts.append(f"ELA={ela_sig:.2f}")
        if dct_sig > 0.3:
            evidence_parts.append(f"DCT copy-move={dct_sig:.2f}")
        if cfa_sig > 0.3:
            evidence_parts.append(f"CFA anomaly={cfa_sig:.2f}")
        if inp_sig > 0.3:
            evidence_parts.append(f"inpainting={inp_sig:.2f}")

        regions.append(
            ReportRegion(
                bbox=bbox,
                type=manip_type,
                confidence=confidence,
                evidence="; ".join(evidence_parts) or "ensemble heatmap signal",
            )
        )

    return regions


class _AnalysisTask(celery_app.Task):  # type: ignore[misc]
    """Base task that holds the ensemble as a class-level singleton (lazy init)."""

    abstract = True
    _ensemble = None

    @property
    def ensemble(self) -> object:
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
    feature_maps = _extract_features(ingested.image, settings)
    log.info("features_extracted", job_id=job_id, extractors=list(feature_maps.keys()))

    # ── Stage 3: ensemble inference ─────────────────────────────────────────
    _update("inference")
    overall_confidence: float = 0.0
    regions: list[ReportRegion] = []
    heatmap: np.ndarray | None = None
    models_used: list[str] = ["PatchForensic", "MantraNet", "SPSL", "InpaintingDetector"]

    try:
        localization = self.ensemble.localize(  # type: ignore[union-attr]
            np.asarray(ingested.image),
            threshold=settings.ensemble_threshold,
            min_region_px=settings.min_region_px,
            return_model_maps=True,
        )
        overall_confidence = float(localization.heatmap.mean())
        heatmap = localization.heatmap
        regions = _localize_to_regions(localization, feature_maps)
        log.info(
            "inference_complete",
            job_id=job_id,
            num_regions=localization.num_regions,
            overall_confidence=overall_confidence,
        )
    except WeightsNotFoundError as exc:
        # Weights not yet deployed; fall back to handcrafted feature signals.
        log.warning("weights_missing_feature_fallback", job_id=job_id, error=str(exc))
        overall_confidence = float(np.mean([m.mean() for m in feature_maps.values()]))
        models_used = [
            "handcrafted_ELA",
            "handcrafted_noise",
            "handcrafted_CFA",
            "handcrafted_DCT",
        ]

    # ── Stage 4: resilience test ─────────────────────────────────────────────
    _update("resilience")
    anti_forensic_warning: bool = False
    if heatmap is not None and overall_confidence > settings.ensemble_threshold:
        try:
            from forenscope.resilience import run_resilience_test

            anti_forensic_warning = run_resilience_test(
                image_bytes, self.ensemble, overall_confidence, settings
            )
            if anti_forensic_warning:
                log.warning("anti_forensic_detected", job_id=job_id)
        except Exception as exc:
            log.warning("resilience_test_failed", job_id=job_id, error=str(exc))

    # ── Stage 5: report generation ──────────────────────────────────────────
    _update("report")
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    report = generate_report(
        job_id=job_id,
        file_name=file_name,
        sha256=ingested.sha256,
        overall_confidence=overall_confidence,
        regions=regions,
        models_used=models_used,
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
