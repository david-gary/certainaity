"""Training for MantraNet (VGG-16/BN fine-tuning).

Usage
-----
    python scripts/train_mantranet.py \\
        --data-dir data/processed \\
        --weights-dir weights/ \\
        --epochs 60 \\
        --batch-size 16 \\
        --lr 1e-4 \\
        --augment

VGG-16/BN base weights are downloaded automatically via torchvision on the
first run. Early conv blocks (1–3) are frozen by default; gradual unfreezing
of blocks 4 and 5 is scheduled via ``--unfreeze-epoch``.

Loss: binary cross-entropy on the per-pixel manipulation probability.
LR schedule: CosineAnnealingLR over the full training run.
AMP (fp16) is enabled automatically on CUDA.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MantraNet")
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root of processed dataset (output of scripts/build_dataset.py)",
    )
    parser.add_argument(
        "--weights-dir", type=Path, default=Path("weights"),
        help="Directory to write trained weights (default: weights/)",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unfreeze-epoch", type=int, default=10,
        help="Epoch at which to begin unfreezing VGG conv block 4 (then 5 at +5 epochs)",
    )
    parser.add_argument(
        "--patience", type=int, default=8,
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


def _unfreeze_block(model: "torch.nn.Module", block_start: int) -> None:
    for i, layer in enumerate(model.features):  # type: ignore[attr-defined]
        if i >= block_start:
            for p in layer.parameters():
                p.requires_grad_(True)


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
    import torch.nn.functional as F
    import mlflow
    from torch.utils.data import DataLoader

    from certainaity.data.dataset import ForensicDataset
    from certainaity.models.mantranet import _MantraNetModel

    torch.manual_seed(args.seed)

    train_manifest = args.data_dir / "train.jsonl"
    val_manifest = args.data_dir / "val.jsonl"

    if not train_manifest.exists():
        raise FileNotFoundError(
            f"Train manifest not found: {train_manifest}. "
            "Run scripts/build_dataset.py first."
        )

    transform = _build_augment() if args.augment else None
    train_ds = ForensicDataset(train_manifest, root=args.data_dir, transform=transform, preload=args.preload)
    val_ds = ForensicDataset(val_manifest, root=args.data_dir, preload=args.preload)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    model = _MantraNetModel().to(args.device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.weights_dir / "mantranet_finetuned.pth"

    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(run_name="mantranet"):
        mlflow.log_params(vars(args))

        best_val_f1 = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            if epoch == args.unfreeze_epoch:
                log.info("Epoch %d: unfreezing VGG block 4", epoch)
                _unfreeze_block(model, block_start=20)
                trainable = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(
                    trainable, lr=args.lr * 0.1, weight_decay=args.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - epoch
                )
            elif epoch == args.unfreeze_epoch + 5:
                log.info("Epoch %d: unfreezing VGG block 5", epoch)
                _unfreeze_block(model, block_start=30)

            log.info("Epoch %d/%d", epoch, args.epochs)

            # ── Training ──────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for imgs, masks in train_loader:
                imgs = imgs.to(args.device)
                masks = masks.to(args.device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(imgs)
                loss = F.binary_cross_entropy(preds.float(), masks)
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
                    val_loss += F.binary_cross_entropy(preds, masks).item()
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

        log.info("Checkpoint saved: %s  (best val_f1=%.4f)", ckpt_path, best_val_f1)


if __name__ == "__main__":
    main()
