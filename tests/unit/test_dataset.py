"""Unit tests for ForensicDataset."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from forenscope.data.dataset import ForensicDataset


@pytest.fixture()
def manifest_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with a manifest and stub .npy files."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    records = []
    for i in range(5):
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        msk = np.zeros((256, 256), dtype=np.uint8)
        msk[100:150, 100:150] = 1

        img_name = f"patch_{i:03d}.npy"
        msk_name = f"patch_{i:03d}.npy"
        np.save(images_dir / img_name, img)
        np.save(masks_dir / msk_name, msk)

        records.append({
            "image_path": f"images/{img_name}",
            "mask_path": f"masks/{msk_name}",
            "source_dataset": "casia_v2",
            "source_image": f"img_{i}.jpg",
            "manipulation_type": "splicing",
            "manipulated_fraction": 0.076,
            "patch_origin": [0, 0],
        })

    manifest = tmp_path / "metadata.jsonl"
    with manifest.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    return tmp_path


class TestForensicDataset:
    def test_length_matches_manifest(self, manifest_dir: Path) -> None:
        ds = ForensicDataset(manifest_dir / "metadata.jsonl", root=manifest_dir)
        assert len(ds) == 5

    def test_getitem_returns_pair(self, manifest_dir: Path) -> None:
        ds = ForensicDataset(manifest_dir / "metadata.jsonl", root=manifest_dir)
        image, mask = ds[0]
        assert image is not None
        assert mask is not None

    def test_image_normalized_to_0_1(self, manifest_dir: Path) -> None:
        ds = ForensicDataset(manifest_dir / "metadata.jsonl", root=manifest_dir)
        image, _ = ds[0]
        img_arr = np.asarray(image)
        assert float(img_arr.max()) <= 1.0 + 1e-6
        assert float(img_arr.min()) >= 0.0

    def test_mask_binary(self, manifest_dir: Path) -> None:
        ds = ForensicDataset(manifest_dir / "metadata.jsonl", root=manifest_dir)
        _, mask = ds[0]
        unique = np.unique(np.asarray(mask))
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_record_returns_metadata(self, manifest_dir: Path) -> None:
        ds = ForensicDataset(manifest_dir / "metadata.jsonl", root=manifest_dir)
        rec = ds.record(0)
        assert "source_dataset" in rec
        assert rec["source_dataset"] == "casia_v2"

    def test_transform_is_applied(self, manifest_dir: Path) -> None:
        calls: list[int] = []

        def counting_transform(image: np.ndarray, mask: np.ndarray) -> dict:
            calls.append(1)
            return {"image": image, "mask": mask}

        # albumentations-style: transform receives keyword args
        def albu_transform(**kwargs: object) -> dict:
            calls.append(1)
            return kwargs

        ds = ForensicDataset(
            manifest_dir / "metadata.jsonl",
            root=manifest_dir,
            transform=albu_transform,
        )
        ds[0]
        assert len(calls) == 1

    def test_manipulation_types_list(self) -> None:
        types = ForensicDataset.manipulation_types()
        assert "splicing" in types
        assert "copy_move" in types
        assert "ai_inpainting" in types

    def test_empty_manifest_has_zero_length(self, tmp_path: Path) -> None:
        manifest = tmp_path / "empty.jsonl"
        manifest.write_text("")
        ds = ForensicDataset(manifest)
        assert len(ds) == 0

    def test_all_indices_accessible(self, manifest_dir: Path) -> None:
        ds = ForensicDataset(manifest_dir / "metadata.jsonl", root=manifest_dir)
        for i in range(len(ds)):
            image, mask = ds[i]
            assert image is not None
