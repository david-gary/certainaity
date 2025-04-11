"""Shared fixtures for unit tests."""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture()
def authentic_rgb() -> Image.Image:
    """A 512×512 smooth gradient image — low forensic signal expected."""
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        arr[y, :, 0] = y // 2
        arr[y, :, 1] = 128
        arr[y, :, 2] = 255 - y // 2
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture()
def spliced_rgb(authentic_rgb: Image.Image) -> Image.Image:
    """512×512 image with a 100×100 region replaced by a solid flat patch."""
    img = authentic_rgb.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 200, 300, 300], fill=(200, 80, 40))
    return img


@pytest.fixture()
def copy_move_rgb(authentic_rgb: Image.Image) -> Image.Image:
    """512×512 image with a 64×64 block copied to a different location."""
    img = authentic_rgb.copy()
    arr = np.asarray(img).copy()
    # Copy block from top-left to bottom-right.
    arr[350:414, 350:414] = arr[50:114, 50:114]
    return Image.fromarray(arr)


@pytest.fixture()
def authentic_jpeg_bytes(authentic_rgb: Image.Image) -> bytes:
    buf = io.BytesIO()
    authentic_rgb.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture()
def tiny_rgb() -> Image.Image:
    """32×32 image — below minimum dimension for ingestion."""
    return Image.new("RGB", (32, 32), color=(128, 128, 128))


@pytest.fixture()
def gif_bytes() -> bytes:
    img = Image.new("P", (64, 64))
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    return buf.getvalue()
