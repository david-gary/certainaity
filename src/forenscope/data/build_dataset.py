"""Build training/val/test manifests from raw forensic datasets.

Usage
-----
    python -m forenscope.data.build_dataset \\
        --datasets casia_v2 defacto nist16 coverage \\
        --raw_dir  data/raw/ \\
        --output   data/processed/ \\
        --split    0.80/0.10/0.10 \\
        --seed     42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

PATCH_SIZE = 256
STRIDE = 128
MIN_MANIPULATED_PX = 16
AUTHENTIC_KEEP_PROB = 0.4


@dataclass
class DatasetConfig:
    name: str
    tampered_dir: str
    authentic_dir: str
    mask_dir: str
    manipulation_type: str
    mask_suffix: str = "_gt.png"


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "casia_v2": DatasetConfig(
        name="casia_v2",
        tampered_dir="Tp",
        authentic_dir="Au",
        mask_dir="masks",
        manipulation_type="splicing",
    ),
    "defacto": DatasetConfig(
        name="defacto",
        tampered_dir="tampered",
        authentic_dir="pristine",
        mask_dir="masks",
        manipulation_type="splicing",
        mask_suffix="_mask.png",
    ),
    "nist16": DatasetConfig(
        name="nist16",
        tampered_dir="tampered",
        authentic_dir="pristine",
        mask_dir="masks",
        manipulation_type="removal",
    ),
    "coverage": DatasetConfig(
        name="coverage",
        tampered_dir="image",
        authentic_dir="image",
        mask_dir="mask",
        manipulation_type="copy_move",
        mask_suffix="t.png",
    ),
}


@dataclass
class Record:
    image_path: str
    mask_path: str
    source_dataset: str
    source_image: str
    manipulation_type: str
    manipulated_fraction: float
    patch_origin: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "source_dataset": self.source_dataset,
            "source_image": self.source_image,
            "manipulation_type": self.manipulation_type,
            "manipulated_fraction": round(self.manipulated_fraction, 4),
            "patch_origin": list(self.patch_origin),
        }


@dataclass
class Stats:
    total: int = 0
    manipulated: int = 0
    authentic: int = 0
    skipped_small: int = 0
    skipped_corrupt: int = 0
    datasets: dict[str, int] = field(default_factory=dict)


def build(
    datasets: list[str],
    raw_dir: Path,
    output_dir: Path,
    split: tuple[float, float, float],
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "masks").mkdir(parents=True, exist_ok=True)

    all_records: list[Record] = []
    stats = Stats()

    for ds_name in datasets:
        if ds_name not in DATASET_CONFIGS:
            log.warning("Unknown dataset %r — skipping.", ds_name)
            continue
        cfg = DATASET_CONFIGS[ds_name]
        ds_dir = raw_dir / ds_name

        if not ds_dir.exists():
            log.warning("Dataset directory %s not found — skipping.", ds_dir)
            continue

        ds_records = _process_dataset(cfg, ds_dir, output_dir, stats)
        all_records.extend(ds_records)
        stats.datasets[ds_name] = len(ds_records)
        log.info("%s: %d patches", ds_name, len(ds_records))

    random.shuffle(all_records)
    n = len(all_records)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    splits = {
        "train": all_records[:n_train],
        "val": all_records[n_train : n_train + n_val],
        "test": all_records[n_train + n_val :],
    }

    for split_name, records in splits.items():
        manifest = output_dir / split_name / "metadata.jsonl"
        with manifest.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec.to_dict()) + "\n")
        log.info("%s split: %d records → %s", split_name, len(records), manifest)

    _write_stats(stats, output_dir)


def _process_dataset(
    cfg: DatasetConfig,
    ds_dir: Path,
    output_dir: Path,
    stats: Stats,
) -> list[Record]:
    records: list[Record] = []
    tampered_dir = ds_dir / cfg.tampered_dir
    mask_dir = ds_dir / cfg.mask_dir

    if not tampered_dir.exists():
        return records

    image_paths = sorted(
        p for p in tampered_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )

    for img_path in image_paths:
        mask_stem = img_path.stem + cfg.mask_suffix
        mask_path = mask_dir / mask_stem
        if not mask_path.exists():
            mask_path = mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            log.debug("No mask for %s — skipping.", img_path.name)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")
        except Exception as exc:
            log.warning("Could not open %s: %s", img_path.name, exc)
            stats.skipped_corrupt += 1
            continue

        w, h = image.size
        if min(w, h) < PATCH_SIZE:
            stats.skipped_small += 1
            continue

        img_arr = np.asarray(image, dtype=np.uint8)
        msk_arr = (np.asarray(mask_img, dtype=np.uint8) > 127).astype(np.uint8)

        patch_records = _extract_patches(
            img_arr, msk_arr, img_path, cfg, output_dir, stats
        )
        records.extend(patch_records)

    return records


def _extract_patches(
    img: np.ndarray,
    msk: np.ndarray,
    source_path: Path,
    cfg: DatasetConfig,
    output_dir: Path,
    stats: Stats,
) -> list[Record]:
    H, W = img.shape[:2]
    records: list[Record] = []

    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            img_patch = img[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            msk_patch = msk[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

            n_manipulated = int(msk_patch.sum())
            is_manipulated = n_manipulated >= MIN_MANIPULATED_PX

            if not is_manipulated:
                if random.random() >= AUTHENTIC_KEEP_PROB:
                    continue
                manipulation_type = "authentic"
            else:
                manipulation_type = cfg.manipulation_type

            stem = f"{cfg.name}_{source_path.stem}_y{y:04d}_x{x:04d}"
            for split_name in ("train",):  # saved once; split assigned later
                img_out = output_dir / split_name / "images" / f"{stem}.npy"
                msk_out = output_dir / split_name / "masks" / f"{stem}.npy"
                np.save(img_out, img_patch)
                np.save(msk_out, msk_patch)

            stats.total += 1
            if is_manipulated:
                stats.manipulated += 1
            else:
                stats.authentic += 1

            records.append(
                Record(
                    image_path=str(Path("images") / f"{stem}.npy"),
                    mask_path=str(Path("masks") / f"{stem}.npy"),
                    source_dataset=cfg.name,
                    source_image=source_path.name,
                    manipulation_type=manipulation_type,
                    manipulated_fraction=n_manipulated / (PATCH_SIZE * PATCH_SIZE),
                    patch_origin=(y, x),
                )
            )

    return records


def _write_stats(stats: Stats, output_dir: Path) -> None:
    report = {
        "total_patches": stats.total,
        "manipulated": stats.manipulated,
        "authentic": stats.authentic,
        "skipped_small": stats.skipped_small,
        "skipped_corrupt": stats.skipped_corrupt,
        "by_dataset": stats.datasets,
    }
    with (output_dir / "build_stats.json").open("w") as fh:
        json.dump(report, fh, indent=2)
    log.info("Build complete: %d total patches (%d manip / %d authentic)",
             stats.total, stats.manipulated, stats.authentic)


def _parse_split(s: str) -> tuple[float, float, float]:
    parts = [float(p) for p in s.split("/")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Split must be three floats separated by '/', e.g. 0.80/0.10/0.10")
    if abs(sum(parts) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"Split fractions must sum to 1.0, got {sum(parts)}")
    return (parts[0], parts[1], parts[2])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build forensic patch dataset")
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_CONFIGS),
        choices=list(DATASET_CONFIGS),
        help="Which source datasets to include",
    )
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", type=_parse_split, default="0.80/0.10/0.10")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build(
        datasets=args.datasets,
        raw_dir=args.raw_dir,
        output_dir=args.output,
        split=args.split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
