"""Training skeleton for SPSL (Siamese ResNet-50 with contrastive loss).

Usage
-----
    python scripts/train_spsl.py \\
        --data-dir data/processed \\
        --weights-dir weights/ \\
        --epochs 60 \\
        --batch-size 64 \\
        --lr 1e-4

After training, a FAISS flat-IP index is built from embeddings of all
authentic patches in the training set and saved alongside the model weights
as ``weights/spsl_faiss.index``.

Loss: contrastive loss on (anchor, positive, negative) triplets sampled
from the dataset.  Positive = same manipulation type; negative = authentic.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SPSL Siamese ResNet-50")
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root of processed dataset (output of scripts/build_dataset.py)",
    )
    parser.add_argument(
        "--weights-dir", type=Path, default=Path("weights"),
        help="Directory to write trained weights and FAISS index",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Contrastive loss margin")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience in epochs (measured on val contrastive loss)",
    )
    parser.add_argument(
        "--skip-faiss", action="store_true",
        help="Skip building the FAISS index after training",
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
    from certainaity.models.spsl import _SPSLBackbone

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

    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    # TODO: replace ForensicDataset with a TripletDataset that yields
    # (anchor, positive, negative) patch triples for contrastive training.
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=args.device == "cuda",
    )

    backbone = _SPSLBackbone().to(args.device)
    optimizer = torch.optim.Adam(
        backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.weights_dir / "spsl_siamese.pth"
    index_path = args.weights_dir / "spsl_faiss.index"

    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(run_name="spsl"):
        mlflow.log_params(vars(args))

        for epoch in range(1, args.epochs + 1):
            log.info("Epoch %d/%d", epoch, args.epochs)

            # TODO: training loop (contrastive / triplet loss)
            #   backbone.train()
            #   for anchor, positive, negative in train_loader:
            #       emb_a = backbone(anchor.to(args.device))
            #       emb_p = backbone(positive.to(args.device))
            #       emb_n = backbone(negative.to(args.device))
            #       loss = contrastive_loss(emb_a, emb_p, emb_n, margin=args.margin)
            #       optimizer.zero_grad()
            #       loss.backward()
            #       optimizer.step()
            #       mlflow.log_metric("train_loss", loss.item(), step=epoch)

            # TODO: validation loop (measure mean intra-class vs inter-class distance)

            scheduler.step()

        torch.save(backbone.state_dict(), ckpt_path)
        log.info("Backbone checkpoint saved: %s", ckpt_path)

    if not args.skip_faiss:
        # TODO: build FAISS flat-IP index from authentic patch embeddings.
        #
        #   import faiss, numpy as np
        #   authentic_ds = [s for s in train_ds if s.manipulation_type is None]
        #   embeddings = []
        #   backbone.eval()
        #   with torch.no_grad():
        #       for patch, _ in DataLoader(authentic_ds, batch_size=256):
        #           embeddings.append(backbone(patch.to(args.device)).cpu().numpy())
        #   embeddings = np.concatenate(embeddings)
        #   index = faiss.IndexFlatIP(args.embedding_dim)
        #   index.add(embeddings)
        #   faiss.write_index(index, str(index_path))
        #   log.info("FAISS index saved: %s  (%d vectors)", index_path, len(embeddings))
        log.info("FAISS index build: not yet implemented (TODO)")


if __name__ == "__main__":
    main()
