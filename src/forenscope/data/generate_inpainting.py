"""Generate a synthetic AI-inpainting dataset using Stable Diffusion.

This script creates training data for the InpaintingDetector model by:
  1. Loading pristine high-resolution images (DIV2K, COCO subset).
  2. Randomly selecting 1–3 polygonal regions per image (5–20% of area).
  3. Feathering the polygon edge (2 px Gaussian blur on the mask).
  4. Running Stable Diffusion 2 inpainting with an empty prompt.
  5. Saving the inpainted image and the binary mask.

Usage
-----
    python -m forenscope.data.generate_inpainting \\
        --source_dirs data/raw/div2k/HR data/raw/coco_sample \\
        --output_dir  data/inpainting_synthetic/ \\
        --n_images    22000 \\
        --device      cuda:0 \\
        --seed        0

Runtime: ~4.5 hours on a single A100 for 22,000 images.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

log = logging.getLogger(__name__)

_SD_MODEL = "stabilityai/stable-diffusion-2-inpainting"


def generate(
    source_dirs: list[Path],
    output_dir: Path,
    n_images: int,
    device: str,
    seed: int,
    poly_count_range: tuple[int, int] = (1, 3),
    area_fraction_range: tuple[float, float] = (0.05, 0.20),
    feather_px: int = 2,
    ddim_steps: int = 50,
    guidance_scale: float = 7.5,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    source_paths = _collect_sources(source_dirs)
    if not source_paths:
        raise FileNotFoundError(f"No images found in {source_dirs}")

    log.info("Found %d source images; generating %d inpainted samples.", len(source_paths), n_images)

    pipe = _load_pipeline(device)

    manifest_path = output_dir / "source_info.jsonl"
    with manifest_path.open("w") as manifest:
        generated = 0
        while generated < n_images:
            src_path = random.choice(source_paths)
            # Per-image seed is deterministic so generation is reproducible.
            img_seed = seed + generated
            rng = random.Random(img_seed)
            np_rng = np.random.default_rng(img_seed)

            try:
                result = _generate_one(
                    src_path, pipe, device, rng, np_rng,
                    poly_count_range, area_fraction_range, feather_px,
                    ddim_steps, guidance_scale,
                )
            except Exception as exc:
                log.warning("Failed to process %s: %s", src_path.name, exc)
                continue

            stem = f"{generated:06d}_{src_path.stem}"
            inpainted_out = output_dir / "images" / f"{stem}.png"
            mask_out = output_dir / "masks" / f"{stem}.png"

            result["image"].save(inpainted_out)
            result["mask"].save(mask_out)

            manifest.write(json.dumps({
                "stem": stem,
                "source": str(src_path),
                "seed": img_seed,
                "poly_count": result["poly_count"],
                "manipulated_fraction": result["manipulated_fraction"],
            }) + "\n")

            generated += 1
            if generated % 100 == 0:
                log.info("Generated %d / %d", generated, n_images)

    log.info("Done. Wrote %d samples to %s", generated, output_dir)


def _collect_sources(dirs: list[Path]) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    paths: list[Path] = []
    for d in dirs:
        if d.exists():
            paths.extend(p for p in sorted(d.iterdir()) if p.suffix.lower() in exts)
    return paths


def _load_pipeline(device: str) -> object:
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            _SD_MODEL,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        return pipe
    except ImportError as exc:
        raise ImportError(
            "diffusers and torch are required for inpainting generation. "
            "Install the [train] extras: pip install -e '.[train]'"
        ) from exc


def _generate_one(
    src_path: Path,
    pipe: object,
    device: str,
    rng: random.Random,
    np_rng: np.random.Generator,
    poly_count_range: tuple[int, int],
    area_fraction_range: tuple[float, float],
    feather_px: int,
    ddim_steps: int,
    guidance_scale: float,
) -> dict[str, object]:
    import torch

    image = Image.open(src_path).convert("RGB")
    w, h = image.size

    # Resize to a multiple of 8 ≤ 768 for SD2 compatibility.
    target = min(768, min(w, h))
    target = target - (target % 8)
    scale = target / min(w, h)
    new_w = int(w * scale) - (int(w * scale) % 8)
    new_h = int(h * scale) - (int(h * scale) % 8)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    n_polys = rng.randint(*poly_count_range)
    mask_pil = Image.new("L", (new_w, new_h), 0)
    draw = ImageDraw.Draw(mask_pil)

    for _ in range(n_polys):
        poly = _random_polygon(rng, new_w, new_h, area_fraction_range)
        draw.polygon(poly, fill=255)

    if feather_px > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_px))
        mask_pil = mask_pil.point(lambda p: 255 if p > 127 else 0)

    generator = torch.Generator(device=device)

    inpainted = pipe(
        prompt="",
        image=image,
        mask_image=mask_pil,
        num_inference_steps=ddim_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    msk_arr = np.asarray(mask_pil, dtype=np.uint8)
    fraction = float(msk_arr.sum()) / (255 * new_w * new_h)

    return {
        "image": inpainted,
        "mask": mask_pil,
        "poly_count": n_polys,
        "manipulated_fraction": fraction,
    }


def _random_polygon(
    rng: random.Random,
    w: int,
    h: int,
    area_fraction_range: tuple[float, float],
) -> list[tuple[int, int]]:
    """Generate a random convex-ish polygon covering area_fraction of the image."""
    target_area = rng.uniform(*area_fraction_range) * w * h

    # Approximate radius from target area assuming circle.
    r = math.sqrt(target_area / math.pi)
    cx = rng.randint(int(r), w - int(r))
    cy = rng.randint(int(r), h - int(r))

    n_verts = rng.randint(5, 12)
    angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n_verts))
    poly = []
    for angle in angles:
        rad = rng.uniform(r * 0.6, r * 1.4)
        px = int(cx + rad * math.cos(angle))
        py = int(cy + rad * math.sin(angle))
        poly.append((max(0, min(w - 1, px)), max(0, min(h - 1, py))))

    return poly


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate synthetic inpainting dataset")
    parser.add_argument("--source_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/inpainting_synthetic"))
    parser.add_argument("--n_images", type=int, default=22_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poly_count_min", type=int, default=1)
    parser.add_argument("--poly_count_max", type=int, default=3)
    parser.add_argument("--area_min", type=float, default=0.05)
    parser.add_argument("--area_max", type=float, default=0.20)
    parser.add_argument("--feather_px", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    generate(
        source_dirs=args.source_dirs,
        output_dir=args.output_dir,
        n_images=args.n_images,
        device=args.device,
        seed=args.seed,
        poly_count_range=(args.poly_count_min, args.poly_count_max),
        area_fraction_range=(args.area_min, args.area_max),
        feather_px=args.feather_px,
        ddim_steps=args.steps,
    )


if __name__ == "__main__":
    main()
