"""Skeleton for ensemble weight optimisation via scipy L-BFGS-B.

Usage
-----
    python scripts/optimize_ensemble.py \\
        --val-dir data/processed \\
        --weights-dir weights/ \\
        --output weights/ensemble_weights.json \\
        --device cuda

Loads validation-set images, runs all four models, then calls
:meth:`Ensemble.optimize_weights` to find the weight vector that maximises
pixel-level F1.  The result is written to ``--output`` in JSON format and
can be loaded at inference time with :meth:`Ensemble.load_optimized_weights`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise ForenScope ensemble weights")
    parser.add_argument(
        "--val-dir", type=Path, required=True,
        help="Root of processed dataset — val.jsonl must be present",
    )
    parser.add_argument(
        "--weights-dir", type=Path, default=Path("weights"),
        help="Directory containing trained model weights",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("weights/ensemble_weights.json"),
        help="Path to write optimised weights JSON",
    )
    parser.add_argument(
        "--max-samples", type=int, default=500,
        help="Maximum number of val images to use (default: 500)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cuda", "cpu", "mps"],
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    from torch.utils.data import DataLoader

    from forenscope.data.dataset import ForensicDataset
    from forenscope.models.base import ModelName
    from forenscope.models.ensemble import Ensemble
    from forenscope.models.inpainting import InpaintingDetector
    from forenscope.models.mantranet import MantraNet
    from forenscope.models.patchforensic import PatchForensic
    from forenscope.models.spsl import SPSL

    val_manifest = args.val_dir / "val.jsonl"
    if not val_manifest.exists():
        raise FileNotFoundError(f"Val manifest not found: {val_manifest}")

    val_ds = ForensicDataset(val_manifest, root=args.val_dir)
    n = min(len(val_ds), args.max_samples)
    log.info("Using %d / %d val samples", n, len(val_ds))

    # TODO: load each model and collect (N, H, W) prediction arrays.
    #
    #   models = {
    #       ModelName.PATCH_FORENSIC:      PatchForensic(args.weights_dir, args.device),
    #       ModelName.MANTRA_NET:          MantraNet(args.weights_dir, args.device),
    #       ModelName.SPSL:               SPSL(args.weights_dir, args.device),
    #       ModelName.INPAINTING_DETECTOR: InpaintingDetector(args.weights_dir, args.device),
    #   }
    #
    #   predictions: dict[ModelName, list[np.ndarray]] = {k: [] for k in models}
    #   ground_truth: list[np.ndarray] = []
    #
    #   for i in range(n):
    #       image, mask = val_ds[i]
    #       image_np = (np.asarray(image) * 255).astype(np.uint8)
    #       for name, model in models.items():
    #           predictions[name].append(model.predict(image_np))
    #       ground_truth.append(np.asarray(mask))
    #
    #   pred_arrays = {k: np.stack(v) for k, v in predictions.items()}
    #   gt_array    = np.stack(ground_truth)

    # Placeholder until models are trained:
    log.warning("Model predictions not yet available — using default ensemble weights.")
    optimal_weights = {
        ModelName.PATCH_FORENSIC: 0.35,
        ModelName.MANTRA_NET: 0.30,
        ModelName.SPSL: 0.20,
        ModelName.INPAINTING_DETECTOR: 0.15,
    }

    # TODO: uncomment once models are trained:
    # optimal_weights = Ensemble.optimize_weights(pred_arrays, gt_array)
    # log.info("Optimised weights: %s", optimal_weights)

    out = {k.value: v for k, v in optimal_weights.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    log.info("Ensemble weights written to %s", args.output)


if __name__ == "__main__":
    main()
