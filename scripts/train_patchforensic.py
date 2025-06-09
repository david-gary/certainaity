"""Training skeleton for PatchForensic.

Usage
-----
    python scripts/train_patchforensic.py \\
        --data-dir data/processed \\
        --weights-dir weights/ \\
        --epochs 90 \\
        --batch-size 32 \\
        --lr 1e-3

Run ``pip install -e ".[train]"`` before using this script.

Loss: combined BCE + Dice with equal weighting (alpha=0.5).
Curriculum: first ``--curriculum-warmup`` epochs train only on samples whose
``manipulated_fraction`` >= 0.10 (easy, large regions); subsequent epochs use
the full dataset (hard samples included).
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
        "--weights-dir", type=Path, default=Path("weights"),
        help="Directory to write trained weights (default: weights/)",
    )
    parser.add_argument("--epochs", type=int, default=90)
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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    # Deferred imports so the CLI fails fast if train extras are missing.
    import torch
    import mlflow
    from torch.utils.data import DataLoader, Subset

    from forenscope.data.dataset import ForensicDataset
    from forenscope.models.patchforensic import _PatchForensicNet

    torch.manual_seed(args.seed)

    train_manifest = args.data_dir / "train.jsonl"
    val_manifest = args.data_dir / "val.jsonl"

    if not train_manifest.exists():
        raise FileNotFoundError(
            f"Train manifest not found: {train_manifest}. "
            "Run scripts/build_dataset.py first."
        )

    # TODO: wrap with an albumentations augmentation pipeline for the train split.
    train_ds = ForensicDataset(train_manifest, root=args.data_dir)
    val_ds = ForensicDataset(val_manifest, root=args.data_dir)

    log.info("Train samples: %d  Val samples: %d", len(train_ds), len(val_ds))

    def _build_curriculum_subset(ds: ForensicDataset, threshold: float) -> Subset:
        """Return indices whose manipulated_fraction >= threshold."""
        indices = [
            i for i in range(len(ds))
            if ds.record(i).get("manipulated_fraction", 1.0) >= threshold
        ]
        return Subset(ds, indices)

    model = _PatchForensicNet().to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.weights_dir / "patchforensic_v2.pth"

    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(run_name="patchforensic"):
        mlflow.log_params(vars(args))

        best_val_f1 = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            if epoch <= args.curriculum_warmup:
                active_ds = _build_curriculum_subset(train_ds, args.easy_threshold)
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
                num_workers=4,
                pin_memory=args.device == "cuda",
            )

            # TODO: training loop
            #   model.train()
            #   for imgs, masks in train_loader:
            #       imgs  = imgs.to(args.device)
            #       masks = masks.to(args.device)
            #       preds = model(imgs)
            #       loss  = bce_dice_loss(preds, masks)
            #       optimizer.zero_grad()
            #       loss.backward()
            #       optimizer.step()
            #       mlflow.log_metric("train_loss", loss.item(), step=epoch)

            # TODO: validation loop
            #   val_f1 = _evaluate(model, val_loader, args.device)
            #   mlflow.log_metric("val_f1", val_f1, step=epoch)
            #   if val_f1 > best_val_f1:
            #       best_val_f1 = val_f1
            #       torch.save(model.state_dict(), ckpt_path)
            #       epochs_without_improvement = 0
            #   else:
            #       epochs_without_improvement += 1
            #   if epochs_without_improvement >= args.patience:
            #       log.info("Early stopping at epoch %d", epoch)
            #       break

            scheduler.step()

        log.info("Best val F1: %.4f", best_val_f1)
        log.info("Checkpoint path: %s", ckpt_path)


if __name__ == "__main__":
    main()
