"""Rigorous evaluation harness for Certainaity forensic models.

Unlike the per-epoch ``val_f1`` logged during training (a pixel-F1 averaged over
*all* val samples, including authentic ones that score 0 by construction), this
harness reports the metrics that actually matter for commercial viability:

  * **Localization** (manipulated samples only):
      - pixel F1 @ 0.5 and @ best-threshold (sweep)
      - pixel IoU @ best-threshold
  * **Detection** (manipulated vs authentic, image level):
      - ROC-AUC and Average Precision using the image-level score
        (max-pooled probability map)
  * **Per-manipulation-type** localization breakdown.

The same harness runs on any processed manifest, so it doubles as the
**cross-dataset generalization** probe: train on CASIA, evaluate on NIST16 by
pointing ``--manifest`` at the NIST16 test split.

Usage
-----
    python scripts/evaluate.py \\
        --weights-dir weights/ \\
        --manifest data/processed/val.jsonl \\
        --data-dir data/processed \\
        --models patchforensic,mantranet,inpainting,spsl \\
        --output reports/eval_casia_val.json

On Modal:
    modal run scripts/modal_train.py::evaluate --manifest /data/processed/val.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_THRESHOLDS = [i / 20 for i in range(1, 20)]  # 0.05 … 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forensic models on a manifest")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root the manifest's relative paths resolve against")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSONL manifest of the evaluation split")
    parser.add_argument("--models", type=str, default="patchforensic,mantranet,inpainting,spsl")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N samples (0 = all; for smoke tests)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the JSON report here (also prints a markdown summary)")
    parser.add_argument("--score-reduce", type=str, default="max", choices=["max", "mean"],
                        help="How to reduce a prob map to one image-level detection score")
    return parser.parse_args()


# ─── Metrics (no sklearn dependency) ──────────────────────────────────────────

def _roc_auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based ROC-AUC (Mann–Whitney U). Returns NaN if one class is absent."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based, averaged over ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _average_precision(scores: list[float], labels: list[int]) -> float:
    """Average Precision (area under precision-recall), interpolation-free."""
    if not any(labels):
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    total_pos = sum(labels)
    ap = 0.0
    prev_recall = 0.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def _f1_iou(pred_bin, mask) -> tuple[float, float]:
    import numpy as np
    tp = float((pred_bin * mask).sum())
    fp = float((pred_bin * (1 - mask)).sum())
    fn = float(((1 - pred_bin) * mask).sum())
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return f1, iou


# ─── Evaluation ───────────────────────────────────────────────────────────────

def _build_model(name: str, weights_dir: Path, device: str):
    from certainaity.models.patchforensic import PatchForensic
    from certainaity.models.mantranet import MantraNet
    from certainaity.models.inpainting import InpaintingDetector
    from certainaity.models.spsl import SPSL
    registry = {
        "patchforensic": PatchForensic,
        "mantranet": MantraNet,
        "inpainting": InpaintingDetector,
        "spsl": SPSL,
    }
    return registry[name](weights_dir, device=device)


def _evaluate_model(name: str, model, records: list[dict], data_dir: Path, reduce: str) -> dict:
    import numpy as np

    # Per-threshold accumulators over manipulated samples (localization).
    f1_by_t: dict[float, list[float]] = {t: [] for t in _THRESHOLDS}
    iou_by_t: dict[float, list[float]] = {t: [] for t in _THRESHOLDS}
    f1_05: list[float] = []
    per_type_f1: dict[str, list[float]] = {}
    # Image-level detection.
    img_scores: list[float] = []
    img_labels: list[int] = []

    for rec in records:
        img = np.load(data_dir / rec["image_path"])              # (H, W, 3) uint8
        mask = np.load(data_dir / rec["mask_path"]).astype(np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0.5).astype(np.float32)
        prob = model.predict(img).astype(np.float32)             # (H, W) in [0,1]
        if prob.shape != mask.shape:
            from skimage.transform import resize
            prob = resize(prob, mask.shape, order=1, preserve_range=True).astype(np.float32)

        is_manip = mask.sum() > 0
        score = float(prob.max() if reduce == "max" else prob.mean())
        img_scores.append(score)
        img_labels.append(1 if is_manip else 0)

        if is_manip:
            best_f1 = 0.0
            for t in _THRESHOLDS:
                pred_bin = (prob > t).astype(np.float32)
                f1, iou = _f1_iou(pred_bin, mask)
                f1_by_t[t].append(f1)
                iou_by_t[t].append(iou)
                best_f1 = max(best_f1, f1)
            f1_05.append(_f1_iou((prob > 0.5).astype(np.float32), mask)[0])
            mt = rec.get("manipulation_type", "unknown")
            per_type_f1.setdefault(mt, []).append(best_f1)

    # Pick the threshold maximizing mean F1 across manipulated samples.
    mean_f1_by_t = {t: (sum(v) / len(v) if v else 0.0) for t, v in f1_by_t.items()}
    best_t = max(mean_f1_by_t, key=mean_f1_by_t.get) if mean_f1_by_t else 0.5
    return {
        "n_samples": len(records),
        "n_manipulated": int(sum(img_labels)),
        "n_authentic": int(len(img_labels) - sum(img_labels)),
        "localization": {
            "pixel_f1@0.5": round(sum(f1_05) / len(f1_05), 4) if f1_05 else None,
            "pixel_f1@best": round(mean_f1_by_t[best_t], 4) if mean_f1_by_t else None,
            "best_threshold": best_t,
            "pixel_iou@best": round(sum(iou_by_t[best_t]) / len(iou_by_t[best_t]), 4)
            if iou_by_t[best_t] else None,
        },
        "detection": {
            "roc_auc": round(_roc_auc(img_scores, img_labels), 4),
            "average_precision": round(_average_precision(img_scores, img_labels), 4),
            "score_reduce": reduce,
        },
        "per_type_pixel_f1@best": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(per_type_f1.items())
        },
    }


def _markdown(report: dict) -> str:
    lines = [
        f"# Evaluation — {report['manifest']}",
        "",
        f"- Samples: {report['n_records']} "
        f"(eval'd {report['n_evaluated']}), device={report['device']}",
        "",
        "| Model | Detect AUC | Detect AP | Loc F1@0.5 | Loc F1@best | IoU@best |",
        "|-------|-----------:|----------:|-----------:|------------:|---------:|",
    ]
    for name, r in report["models"].items():
        if "error" in r:
            lines.append(f"| {name} | ERROR | — | — | — | — |")
            continue
        d, loc = r["detection"], r["localization"]
        lines.append(
            f"| {name} | {d['roc_auc']} | {d['average_precision']} | "
            f"{loc['pixel_f1@0.5']} | {loc['pixel_f1@best']} | {loc['pixel_iou@best']} |"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    records: list[dict] = []
    with args.manifest.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]

    report: dict = {
        "manifest": str(args.manifest),
        "device": args.device,
        "n_records": len(records),
        "n_evaluated": len(records),
        "models": {},
    }

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        log.info("Evaluating %s on %d samples…", name, len(records))
        try:
            model = _build_model(name, args.weights_dir, args.device)
            report["models"][name] = _evaluate_model(
                name, model, records, args.data_dir, args.score_reduce
            )
            log.info("  %s: %s", name, report["models"][name]["detection"])
        except Exception as exc:  # noqa: BLE001 — one model failing shouldn't abort the rest
            log.exception("  %s failed: %s", name, exc)
            report["models"][name] = {"error": str(exc)}

    md = _markdown(report)
    print("\n" + md + "\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        args.output.with_suffix(".md").write_text(md)
        log.info("Wrote report: %s", args.output)


if __name__ == "__main__":
    main()
