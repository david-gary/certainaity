"""Resilience test: detect anti-forensic JPEG re-compression attacks.

A manipulation that survives one specific JPEG quality setting but not others
suggests that post-processing was deliberately applied to evade forensic
detection at a particular quality level—a classic anti-forensic technique.

The test re-compresses the image at each quality in ``settings.resilience_qualities``
and re-runs the ensemble. If the confidence drops by more than
``settings.resilience_drop_threshold`` at any quality, the image is flagged.
"""

from __future__ import annotations

import io

import numpy as np
import structlog
from PIL import Image

log = structlog.get_logger()


def run_resilience_test(
    image_bytes: bytes,
    ensemble: object,
    original_confidence: float,
    settings: object,
) -> bool:
    """Return True if manipulation confidence collapses under JPEG re-compression.

    Args:
        image_bytes:          Raw bytes of the image under analysis.
        ensemble:             Loaded :class:`~certainaity.models.Ensemble` instance.
        original_confidence:  Overall confidence from the full-quality inference run.
        settings:             :class:`~certainaity.config.Settings` instance.

    Returns:
        True if anti-forensic post-processing is suspected, False otherwise.
    """
    from certainaity.exceptions import WeightsNotFoundError

    qualities: list[int] = getattr(settings, "resilience_qualities", [70, 85, 95])
    drop_threshold: float = getattr(settings, "resilience_drop_threshold", 0.25)
    ensemble_threshold: float = getattr(settings, "ensemble_threshold", 0.65)
    min_region_px: int = getattr(settings, "min_region_px", 64)

    base_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    base_arr = np.asarray(base_image)

    for quality in sorted(qualities):
        buf = io.BytesIO()
        base_image.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        recompressed = np.asarray(Image.open(buf).convert("RGB"))

        try:
            result = ensemble.localize(  # type: ignore[union-attr]
                recompressed,
                threshold=ensemble_threshold,
                min_region_px=min_region_px,
            )
            recompressed_confidence = float(result.heatmap.mean())
        except WeightsNotFoundError:
            # Weights unavailable; cannot perform resilience test.
            return False
        except Exception as exc:
            log.warning("resilience_quality_failed", quality=quality, error=str(exc))
            continue

        drop = original_confidence - recompressed_confidence
        log.debug(
            "resilience_quality_result",
            quality=quality,
            original=round(original_confidence, 4),
            recompressed=round(recompressed_confidence, 4),
            drop=round(drop, 4),
        )

        if drop > drop_threshold:
            log.info(
                "anti_forensic_signal_detected",
                quality=quality,
                confidence_drop=round(drop, 4),
                threshold=drop_threshold,
            )
            return True

    return False
