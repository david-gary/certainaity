"""Training for PatchForensic FCN.

Usage
-----
    python scripts/train_patchforensic.py \\
        --data-dir data/processed \\
        --weights-dir weights/ \\
        --epochs 120 \\
        --batch-size 32 \\
        --lr 1e-3 \\
        --augment

Run ``pip install -e ".[train]"`` before using this script.

Loss: combined BCE + Dice with equal weighting (alpha=0.5).
Curriculum: first ``--curriculum-warmup`` epochs train only on samples whose
``manipulated_fraction`` >= 0.10 (easy, large regions); subsequent epochs use
the full dataset (hard samples included).
AMP (fp16) is enabled automatically on CUDA.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PatchForensic FCN")
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root of processed dataset (output of scripts/build_dataset.py)",
    )
    parser.add_argument(
        "--extra-data-dirs", type=Path, nargs="*", default=[],
        help="Additional processed-dataset roots to UNION into training "
             "(each must have train.jsonl/val.jsonl). Enables cross-source "
             "diversity training without clobbering any single source.",
    )
    parser.add_argument(
        "--weights-dir", type=Path, default=Path("weights"),
        help="Directory to write trained weights (default: weights/)",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--curriculum-warmup", type=int, default=30,
        help="Epochs to train on easy samples (large manipulated region) only",
    )
    parser.add_argument(
        "--easy-threshold", type=float, default=0.10,
        help="Minimum manipulated_fraction to qualify as an 'easy' sample",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience in epochs (measured on val F1)",
    )
    parser.add_argument(
        "--mlflow-uri", type=str, default="mlruns",
        help="MLflow tracking URI (default: local ./mlruns)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
    )
    parser.add_argument(
        "--preload", action="store_true",
        help="Preload all patches into RAM at startup (fast epochs on network volumes)",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Apply albumentations augmentation to training data",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed-precision (fp16). This from-scratch FCN can diverge "
             "under AMP (sudden val_f1 collapse to 0); fp32 is more stable.",
    )
    parser.add_argument(
        "--ckpt-name", type=str, default="patchforensic_v2.pth",
        help="Checkpoint filename under --weights-dir (use a distinct name for "
             "union/experiment runs so the baseline is not overwritten).",
    )
    return parser.parse_args()


def _build_augment() -> object:
    import albumentations as A
    # Geometric-only augmentation. Flips/rotations preserve the pixel-level
    # statistical artifacts (JPEG grids, resampling, noise fingerprints) that
    # forensic models depend on. Photometric shifts (brightness/contrast/hue)
    # corrupt those cues and empirically degrade val F1, so they are excluded.
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


def _bce_dice_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    alpha: float = 0.5,
    smooth: float = 1e-6,
) -> "torch.Tensor":
    import torch.nn.functional as F
    bce = F.binary_cross_entropy(pred, target)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + smooth) / (
        pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth
    )
    return alpha * bce + (1.0 - alpha) * dice.mean()


def _pixel_f1(pred: "torch.Tensor", target: "torch.Tensor", threshold: float = 0.5) -> float:
    p = (pred > threshold).float()
    tp = (p * target).sum()
    fp = (p * (1 - target)).sum()
    fn = ((1 - p) * target).sum()
    return (2 * tp / (2 * tp + fp + fn + 1e-8)).item()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    import torch
    import mlflow
    from torch.utils.data import ConcatDataset, DataLoader, Subset

    from certainaity.data.dataset import ForensicDataset
    from certainaity.models.patchforensic import _PatchForensicNet

    torch.manual_seed(args.seed)

    transform = _build_augment() if args.augment else None
    data_dirs = [args.data_dir, *args.extra_data_dirs]

    def _load_split(split: str, with_transform: bool) -> list[ForensicDataset]:
        out: list[ForensicDataset] = []
        for d in data_dirs:
            manifest = d / f"{split}.jsonl"
            if not manifest.exists():
                if split == "train":
                    raise FileNotFoundError(
                        f"Train manifest not found: {manifest}. Run build_dataset first."
                    )
                continue  # a source may legitimately have an empty val
            out.append(ForensicDataset(
                manifest, root=d,
                transform=transform if with_transform else None,
                preload=args.preload,
            ))
        return out

    train_sources = _load_split("train", with_transform=True)
    val_sources = _load_split("val", with_transform=False)
    train_ds = ConcatDataset(train_sources)
    val_ds = ConcatDataset(val_sources)

    if len(data_dirs) > 1:
        log.info("UNION training over %d sources: %s", len(data_dirs),
                 ", ".join(str(d) for d in data_dirs))
    log.info("Train samples: %d  Val samples: %d", len(train_ds), len(val_ds))

    def _build_curriculum_subset(sources: list[ForensicDataset], threshold: float) -> Subset:
        """Global indices (into the ConcatDataset) of easy samples across all sources."""
        indices: list[int] = []
        offset = 0
        for ds in sources:
            for i in range(len(ds)):
                if ds.record(i).get("manipulated_fraction", 1.0) >= threshold:
                    indices.append(offset + i)
            offset += len(ds)
        return Subset(train_ds, indices)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = _PatchForensicNet().to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = args.device == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log.info("Mixed precision (AMP): %s", "enabled" if use_amp else "disabled")

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.weights_dir / args.ckpt_name
    resume_path = args.weights_dir / (args.ckpt_name + ".resume")

    # ── Resume support ────────────────────────────────────────────────────
    # Long union runs get interrupted (timeouts, spot preemption, budget caps).
    # A per-epoch resume checkpoint lets a relaunch continue instead of
    # restarting from epoch 1.
    start_epoch = 1
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    if resume_path.exists():
        st = torch.load(resume_path, map_location=args.device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        start_epoch = st["epoch"] + 1
        best_val_f1 = st["best_val_f1"]
        epochs_without_improvement = st["epochs_without_improvement"]
        log.info("Resumed from %s → starting epoch %d (best_val_f1=%.4f)",
                 resume_path, start_epoch, best_val_f1)

    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(run_name="patchforensic"):
        mlflow.log_params(vars(args))

        for epoch in range(start_epoch, args.epochs + 1):
            if epoch <= args.curriculum_warmup:
                active_ds = _build_curriculum_subset(train_sources, args.easy_threshold)
                log.info(
                    "Epoch %d/%d  [curriculum: %d easy samples]",
                    epoch, args.epochs, len(active_ds),
                )
            else:
                active_ds = train_ds  # type: ignore[assignment]
                log.info("Epoch %d/%d  [full dataset: %d samples]",
                         epoch, args.epochs, len(active_ds))

            train_loader = DataLoader(
                active_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=args.device == "cuda",
            )

            # ── Training ──────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for imgs, masks in train_loader:
                imgs = imgs.to(args.device)
                masks = masks.to(args.device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(imgs)
                loss = _bce_dice_loss(preds.float(), masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # ── Validation ────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            val_f1_sum = 0.0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(args.device)
                    masks = masks.to(args.device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        preds = model(imgs)
                    preds = preds.float()
                    val_loss += _bce_dice_loss(preds, masks).item()
                    val_f1_sum += _pixel_f1(preds, masks)
            n_val = max(len(val_loader), 1)
            val_loss /= n_val
            val_f1 = val_f1_sum / n_val
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            log.info("  train_loss=%.4f  val_loss=%.4f  val_f1=%.4f", train_loss, val_loss, val_f1)

            # ── Early stopping ────────────────────────────────────────────
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), ckpt_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                log.info("Early stopping at epoch %d (best val_f1=%.4f)", epoch, best_val_f1)
                break

            scheduler.step()

            # Persist full training state so an interrupted run can resume.
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "epochs_without_improvement": epochs_without_improvement,
            }, resume_path)

        log.info("Best val F1: %.4f", best_val_f1)
        log.info("Checkpoint path: %s", ckpt_path)
        # Training finished cleanly — drop the resume file so a future run with
        # the same checkpoint name starts fresh.
        resume_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
