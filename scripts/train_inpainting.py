"""Training skeleton for InpaintingDetector (CLIP ViT-B/32 fine-tuning).

Usage
-----
    python scripts/train_inpainting.py \\
        --data-dir data/processed \\
        --weights-dir weights/ \\
        --epochs 20 \\
        --batch-size 8 \\
        --lr 5e-5

CLIP ViT-B/32 weights are downloaded from HuggingFace on the first run
(``openai/clip-vit-base-patch32``).  Only the segmentation head is trained
for the first ``--freeze-epochs``; thereafter, the last ``--unfreeze-layers``
transformer encoder blocks are unfrozen at a reduced learning rate.

Loss: focal loss with gamma=2.0, alpha=0.25 (handles class imbalance between
authentic and manipulated pixels without requiring strict class-ratio balancing).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


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
    parser.add_argument("--epochs", type=int, default=20)
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
        "--patience", type=int, default=5,
        help="Early-stopping patience in epochs (measured on val F1)",
    )
    parser.add_argument(
        "--mlflow-uri", type=str, default="mlruns",
        help="MLflow tracking URI (default: local ./mlruns)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
    )
    return parser.parse_args()


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

    train_ds = ForensicDataset(train_manifest, root=args.data_dir)
    val_ds = ForensicDataset(val_manifest, root=args.data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,   # fewer workers: CLIP tokenisation is CPU-heavy
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    model = _InpaintingDetectorModel().to(args.device)
    # Phase 1: train only the segmentation head.
    head_optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.weights_dir / "inpainting_detector_clip.pth"

    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(run_name="inpainting_detector"):
        mlflow.log_params(vars(args))

        optimizer = head_optimizer

        for epoch in range(1, args.epochs + 1):
            if epoch == args.freeze_epochs + 1:
                # Phase 2: unfreeze the last N CLIP encoder layers.
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

            phase = "head only" if epoch <= args.freeze_epochs else f"head + {args.unfreeze_layers} CLIP layers"
            log.info("Epoch %d/%d  [%s]", epoch, args.epochs, phase)

            # TODO: training loop (focal loss)
            #   model.train()
            #   for imgs, masks in train_loader:
            #       imgs, masks = imgs.to(args.device), masks.to(args.device)
            #       preds = model(imgs).squeeze(1)
            #       loss  = focal_loss(preds, masks.float(),
            #                          gamma=args.focal_gamma, alpha=args.focal_alpha)
            #       optimizer.zero_grad()
            #       loss.backward()
            #       optimizer.step()
            #       mlflow.log_metric("train_loss", loss.item(), step=epoch)

            # TODO: validation loop + early stopping

        # Save only head weights (CLIP base weights remain at HuggingFace defaults).
        torch.save(model.head.state_dict(), ckpt_path)
        log.info("Head checkpoint saved: %s", ckpt_path)


if __name__ == "__main__":
    main()
