"""Training skeleton for MantraNet (VGG-16/BN fine-tuning).

Usage
-----
    python scripts/train_mantranet.py \\
        --data-dir data/processed \\
        --weights-dir weights/ \\
        --epochs 30 \\
        --batch-size 16 \\
        --lr 1e-4

VGG-16/BN base weights are downloaded automatically via torchvision on the
first run. Early conv blocks (1–3) are frozen by default; use
``--unfreeze-epoch`` to schedule gradual unfreezing of subsequent blocks.

Loss: binary cross-entropy on the per-pixel manipulation probability.
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unfreeze-epoch", type=int, default=10,
        help="Epoch at which to begin unfreezing VGG conv block 4 (then 5 at +5 epochs)",
    )
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


def _unfreeze_block(model: "torch.nn.Module", block_start: int) -> None:
    """Unfreeze VGG feature layers starting at ``block_start``."""
    for i, layer in enumerate(model.features):  # type: ignore[attr-defined]
        if i >= block_start:
            for p in layer.parameters():
                p.requires_grad_(True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    import torch
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

    train_ds = ForensicDataset(train_manifest, root=args.data_dir)
    val_ds = ForensicDataset(val_manifest, root=args.data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    model = _MantraNetModel().to(args.device)

    # Only fine-tune the unfrozen portion and the anomaly head initially.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.weights_dir / "mantranet_finetuned.pth"

    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(run_name="mantranet"):
        mlflow.log_params(vars(args))

        for epoch in range(1, args.epochs + 1):
            # Gradual unfreezing: unfreeze VGG block 4 at unfreeze_epoch,
            # block 5 five epochs later.
            if epoch == args.unfreeze_epoch:
                log.info("Epoch %d: unfreezing VGG block 4", epoch)
                _unfreeze_block(model, block_start=20)
                trainable = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(
                    trainable, lr=args.lr * 0.1, weight_decay=args.weight_decay
                )
            elif epoch == args.unfreeze_epoch + 5:
                log.info("Epoch %d: unfreezing VGG block 5", epoch)
                _unfreeze_block(model, block_start=30)

            log.info("Epoch %d/%d", epoch, args.epochs)

            # TODO: training loop
            #   model.train()
            #   for imgs, masks in train_loader:
            #       imgs, masks = imgs.to(args.device), masks.to(args.device)
            #       preds = model(imgs)
            #       loss  = F.binary_cross_entropy(preds.squeeze(1), masks.float())
            #       optimizer.zero_grad()
            #       loss.backward()
            #       optimizer.step()
            #       mlflow.log_metric("train_loss", loss.item(), step=epoch)

            # TODO: validation loop + early stopping (same pattern as train_patchforensic.py)

            scheduler.step()

        log.info("Checkpoint path: %s", ckpt_path)


if __name__ == "__main__":
    main()
