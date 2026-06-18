"""Training for InpaintingDetector (CLIP ViT-B/32 fine-tuning).

Usage
-----
    python scripts/train_inpainting.py \\
        --data-dir data/processed \\
        --weights-dir weights/ \\
        --epochs 35 \\
        --batch-size 8 \\
        --lr 5e-5 \\
        --augment

CLIP ViT-B/32 weights are downloaded from HuggingFace on the first run
(``openai/clip-vit-base-patch32``).  Only the segmentation head is trained
for the first ``--freeze-epochs``; thereafter, the last ``--unfreeze-layers``
transformer encoder blocks are unfrozen at a reduced learning rate.

Loss: focal loss with gamma=2.0, alpha=0.25 (handles class imbalance between
authentic and manipulated pixels without requiring strict class-ratio balancing).
AMP (fp16) is enabled automatically on CUDA.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune InpaintingDetector (CLIP + head)")
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root of processed dataset — must include ai_inpainting samples",
    )
    parser.add_argument(
        "--weights-dir", type=Path, default=Path("weights"),
        help="Directory to write trained weights (default: weights/)",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Small batch recommended: CLIP ViT-B/32 is ~340 MB")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for the segmentation head")
    parser.add_argument("--clip-lr", type=float, default=5e-6,
                        help="Reduced LR for unfrozen CLIP encoder layers")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Epochs to train only the segmentation head (CLIP frozen)")
    parser.add_argument("--unfreeze-layers", type=int, default=4,
                        help="Number of CLIP encoder layers to unfreeze after freeze-epochs")
    parser.add_argument("--seed", type=int, default=42)
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


def _focal_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> "torch.Tensor":
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target >= 0.5, pred, 1.0 - pred)
    alpha_t = torch.where(
        target >= 0.5,
        torch.full_like(target, alpha),
        torch.full_like(target, 1.0 - alpha),
    )
    return (alpha_t * (1.0 - pt) ** gamma * bce).mean()


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
    from torch.utils.data import DataLoader

    from certainaity.data.dataset import ForensicDataset
    from certainaity.models.inpainting import _InpaintingDetectorModel

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

    model = _InpaintingDetectorModel().to(args.device)
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    use_amp = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.weights_dir / "inpainting_detector_clip.pth"

    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(run_name="inpainting_detector"):
        mlflow.log_params(vars(args))

        best_val_f1 = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            if epoch == args.freeze_epochs + 1:
                log.info(
                    "Epoch %d: unfreezing last %d CLIP encoder layers",
                    epoch, args.unfreeze_layers,
                )
                model.unfreeze_clip_layers(args.unfreeze_layers)
                clip_params = [p for p in model.clip.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(
                    [
                        {"params": model.head.parameters(), "lr": args.lr},
                        {"params": clip_params, "lr": args.clip_lr},
                    ],
                    weight_decay=args.weight_decay,
                )
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            phase = "head only" if epoch <= args.freeze_epochs else f"head + {args.unfreeze_layers} CLIP layers"
            log.info("Epoch %d/%d  [%s]", epoch, args.epochs, phase)

            # ── Training ──────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for imgs, masks in train_loader:
                imgs = imgs.to(args.device)
                masks = masks.to(args.device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(imgs)
                loss = _focal_loss(preds.float(), masks, gamma=args.focal_gamma, alpha=args.focal_alpha)
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
                    val_loss += _focal_loss(preds, masks, gamma=args.focal_gamma, alpha=args.focal_alpha).item()
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
                torch.save(model.head.state_dict(), ckpt_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                log.info("Early stopping at epoch %d (best val_f1=%.4f)", epoch, best_val_f1)
                break

        log.info("Head checkpoint saved: %s  (best val_f1=%.4f)", ckpt_path, best_val_f1)


if __name__ == "__main__":
    main()
