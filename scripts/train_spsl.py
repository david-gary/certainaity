"""Training for SPSL (Siamese ResNet-50 with triplet loss).

Usage
-----
    python scripts/train_spsl.py \\
        --data-dir data/processed \\
        --weights-dir weights/ \\
        --epochs 90 \\
        --batch-size 64 \\
        --lr 1e-4 \\
        --augment

After training, a FAISS flat-IP index is built from embeddings of all
authentic patches in the training set and saved alongside the model weights
as ``weights/spsl_faiss.index``.

Loss: triplet loss on (anchor, positive, negative) tuples sampled from the
dataset.  Positive = same manipulation type; negative = authentic.
AMP (fp16) is enabled automatically on CUDA.
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
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Triplet loss margin")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience in epochs (measured on val triplet loss)",
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
    parser.add_argument(
        "--preload", action="store_true",
        help="Preload all patches into RAM at startup (fast epochs on network volumes)",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Apply albumentations augmentation to training data",
    )
    return parser.parse_args()


def _is_authentic(record: dict) -> bool:
    """True if a manifest record is an unmanipulated (authentic) patch.

    build_dataset.py labels authentic patches with ``manipulation_type ==
    "authentic"``; older manifests may omit the field (None). Both count as
    the negative class for triplet sampling and the FAISS reference index.
    """
    return record.get("manipulation_type") in (None, "authentic")


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    import random
    import torch
    import torch.nn.functional as F
    import mlflow
    from torch.utils.data import DataLoader, Dataset, Subset

    from certainaity.data.dataset import ForensicDataset
    from certainaity.models.spsl import _SPSLBackbone

    torch.manual_seed(args.seed)
    random.seed(args.seed)

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

    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    class TripletDataset(Dataset):
        """Yields (anchor, positive, negative) triples for contrastive training.

        Anchor and positive share the same manipulation type; the negative is
        drawn from authentic (unmanipulated) samples.
        """

        def __init__(self, base_ds: ForensicDataset) -> None:
            super().__init__()
            authentic: list[int] = []
            by_type: dict[str, list[int]] = {}
            for i in range(len(base_ds)):
                if _is_authentic(base_ds.record(i)):
                    authentic.append(i)
                else:
                    mt = base_ds.record(i)["manipulation_type"]
                    by_type.setdefault(mt, []).append(i)
            self._base = base_ds
            self._authentic = authentic
            self._by_type = by_type
            self._pool = [
                i for i in range(len(base_ds))
                if not _is_authentic(base_ds.record(i))
                and len(by_type.get(base_ds.record(i)["manipulation_type"], [])) > 1
                and len(authentic) > 0
            ]

        def __len__(self) -> int:
            return len(self._pool)

        def __getitem__(self, idx: int) -> tuple:
            anc_idx = self._pool[idx]
            anc_type = self._base.record(anc_idx)["manipulation_type"]
            pos_idx = anc_idx
            while pos_idx == anc_idx:
                pos_idx = random.choice(self._by_type[anc_type])
            neg_idx = random.choice(self._authentic)
            anc, _ = self._base[anc_idx]
            pos, _ = self._base[pos_idx]
            neg, _ = self._base[neg_idx]
            return anc, pos, neg

    train_triplets = TripletDataset(train_ds)
    val_triplets = TripletDataset(val_ds)
    log.info("Triplet pool — Train: %d  Val: %d", len(train_triplets), len(val_triplets))

    train_loader = DataLoader(
        train_triplets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(
        val_triplets,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device == "cuda",
    )

    backbone = _SPSLBackbone().to(args.device)
    optimizer = torch.optim.Adam(
        backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.weights_dir / "spsl_siamese.pth"
    index_path = args.weights_dir / "spsl_faiss.index"

    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(run_name="spsl"):
        mlflow.log_params(vars(args))

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            log.info("Epoch %d/%d", epoch, args.epochs)

            # ── Training ──────────────────────────────────────────────────
            backbone.train()
            train_loss = 0.0
            for anc, pos, neg in train_loader:
                anc = anc.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    emb_a = backbone(anc)
                    emb_p = backbone(pos)
                    emb_n = backbone(neg)
                    d_ap = (emb_a - emb_p).pow(2).sum(1)
                    d_an = (emb_a - emb_n).pow(2).sum(1)
                    loss = F.relu(d_ap - d_an + args.margin).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # ── Validation ────────────────────────────────────────────────
            backbone.eval()
            val_loss = 0.0
            if len(val_loader) > 0:
                with torch.no_grad():
                    for anc, pos, neg in val_loader:
                        anc = anc.to(args.device)
                        pos = pos.to(args.device)
                        neg = neg.to(args.device)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            emb_a = backbone(anc)
                            emb_p = backbone(pos)
                            emb_n = backbone(neg)
                            d_ap = (emb_a - emb_p).pow(2).sum(1)
                            d_an = (emb_a - emb_n).pow(2).sum(1)
                            val_loss += F.relu(d_ap - d_an + args.margin).mean().item()
                val_loss /= len(val_loader)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                log.info("  train_loss=%.4f  val_loss=%.4f", train_loss, val_loss)
            else:
                val_loss = train_loss
                log.info("  train_loss=%.4f  (no val triplets)", train_loss)

            # ── Early stopping ────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(backbone.state_dict(), ckpt_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                log.info("Early stopping at epoch %d (best val_loss=%.4f)", epoch, best_val_loss)
                break

            scheduler.step()

        log.info("Backbone checkpoint saved: %s  (best val_loss=%.4f)", ckpt_path, best_val_loss)

    if not args.skip_faiss:
        import faiss
        import numpy as np
        log.info("Building FAISS index from authentic training patches...")
        # Use a fresh dataset without augmentation for deterministic embeddings.
        faiss_ds = ForensicDataset(train_manifest, root=args.data_dir, preload=args.preload)
        authentic_indices = [
            i for i in range(len(faiss_ds))
            if _is_authentic(faiss_ds.record(i))
        ]
        if not authentic_indices:
            log.warning("No authentic samples found; skipping FAISS index build.")
        else:
            authentic_subset = Subset(faiss_ds, authentic_indices)
            embeddings = []
            backbone.eval()
            with torch.no_grad():
                for patch, _ in DataLoader(authentic_subset, batch_size=256, num_workers=0):
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        embeddings.append(backbone(patch.to(args.device)).cpu().float().numpy())
            emb_arr = np.concatenate(embeddings)
            index = faiss.IndexFlatIP(args.embedding_dim)
            index.add(emb_arr)
            faiss.write_index(index, str(index_path))
            log.info("FAISS index saved: %s  (%d vectors)", index_path, len(emb_arr))


if __name__ == "__main__":
    main()
