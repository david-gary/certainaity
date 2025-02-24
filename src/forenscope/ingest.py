"""Image ingestion: validation, hashing, and metadata extraction."""

from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass, field
from pathlib import Path

import piexif
from PIL import Image, UnidentifiedImageError

from forenscope.config import get_settings
from forenscope.exceptions import (
    CorruptImageError,
    FileTooLargeError,
    ImageTooSmallError,
    UnsupportedFormatError,
)

SUPPORTED_FORMATS = {"JPEG", "PNG", "TIFF", "WEBP"}


@dataclass
class IngestedImage:
    sha256: str
    width: int
    height: int
    format: str
    exif: dict[str, object]
    quantization_tables: list[list[int]] | None
    thumbnail_mismatch: bool
    image: Image.Image = field(repr=False)


def ingest_image(source: Path | bytes) -> IngestedImage:
    """Validate, hash, and extract metadata from an image.

    SHA-256 is computed from the raw bytes before any decoding to preserve
    chain-of-custody integrity.
    """
    settings = get_settings()

    raw = _read_raw(source, settings.max_file_bytes)
    sha256 = hashlib.sha256(raw).hexdigest()

    image = _decode(raw)

    if image.format not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Format {image.format!r} is not supported. "
            f"Accepted: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    w, h = image.size
    min_dim = min(w, h)
    max_dim = max(w, h)

    if min_dim < settings.min_image_dimension:
        raise ImageTooSmallError(
            f"Shortest side {min_dim} px is below minimum {settings.min_image_dimension} px."
        )
    if max_dim > settings.max_image_dimension:
        raise ImageTooSmallError(
            f"Longest side {max_dim} px exceeds maximum {settings.max_image_dimension} px."
        )

    exif = _extract_exif(raw, image)
    qtables = _extract_quantization_tables(raw) if image.format == "JPEG" else None
    thumbnail_mismatch = _check_thumbnail_mismatch(raw, image) if image.format == "JPEG" else False

    rgb = image.convert("RGB")

    return IngestedImage(
        sha256=sha256,
        width=w,
        height=h,
        format=image.format or "UNKNOWN",
        exif=exif,
        quantization_tables=qtables,
        thumbnail_mismatch=thumbnail_mismatch,
        image=rgb,
    )


def _read_raw(source: Path | bytes, max_bytes: int) -> bytes:
    if isinstance(source, bytes):
        raw = source
    else:
        size = source.stat().st_size
        if size > max_bytes:
            raise FileTooLargeError(
                f"File is {size:,} bytes; limit is {max_bytes:,} bytes."
            )
        raw = source.read_bytes()

    if len(raw) > max_bytes:
        raise FileTooLargeError(
            f"Upload is {len(raw):,} bytes; limit is {max_bytes:,} bytes."
        )

    return raw


def _decode(raw: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        # verify() closes the file; reopen for actual use
        image = Image.open(io.BytesIO(raw))
        return image
    except UnidentifiedImageError as exc:
        raise CorruptImageError("Could not identify image format.") from exc
    except Exception as exc:
        raise CorruptImageError(f"Image decoding failed: {exc}") from exc


def _extract_exif(raw: bytes, image: Image.Image) -> dict[str, object]:
    try:
        exif_bytes = image.info.get("exif", b"")
        if not exif_bytes:
            return {}
        exif_dict = piexif.load(exif_bytes)
        return _flatten_exif(exif_dict)
    except Exception:
        return {}


def _flatten_exif(exif_dict: dict) -> dict[str, object]:
    flat: dict[str, object] = {}
    ifd_names = {
        "0th": piexif.ImageIFD,
        "Exif": piexif.ExifIFD,
        "GPS": piexif.GPSIFD,
        "1st": piexif.ImageIFD,
    }
    for ifd_name, tag_map in ifd_names.items():
        if ifd_name not in exif_dict:
            continue
        for tag_id, value in exif_dict[ifd_name].items():
            try:
                name = piexif.TAGS[ifd_name][tag_id]["name"]
            except (KeyError, TypeError):
                name = f"{ifd_name}_{tag_id}"
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="replace").rstrip("\x00")
                except Exception:
                    value = value.hex()
            flat[name] = value
    return flat


def _extract_quantization_tables(raw: bytes) -> list[list[int]] | None:
    """Parse DQT (Define Quantization Table) markers from a JPEG byte stream."""
    tables: list[list[int]] = []
    i = 0
    # JPEG starts with FFD8
    if raw[:2] != b"\xff\xd8":
        return None

    i = 2
    while i < len(raw) - 1:
        if raw[i] != 0xFF:
            break
        marker = raw[i + 1]
        if marker == 0xD9:  # EOI
            break
        if marker in (0x00, 0xFF):
            i += 1
            continue

        if i + 3 >= len(raw):
            break
        length = struct.unpack(">H", raw[i + 2 : i + 4])[0]
        segment = raw[i + 4 : i + 2 + length]

        if marker == 0xDB:  # DQT
            offset = 0
            while offset < len(segment):
                precision_and_id = segment[offset]
                precision = (precision_and_id >> 4) & 0xF
                offset += 1
                table_len = 128 if precision else 64
                if offset + table_len > len(segment):
                    break
                if precision:
                    table = list(
                        struct.unpack(">64H", segment[offset : offset + 128])
                    )
                else:
                    table = list(segment[offset : offset + 64])
                tables.append(table)
                offset += table_len

        i += 2 + length

    return tables if tables else None


def _check_thumbnail_mismatch(raw: bytes, image: Image.Image) -> bool:
    """Return True if the embedded JPEG thumbnail differs from the full image in aspect ratio."""
    try:
        exif_bytes = image.info.get("exif", b"")
        if not exif_bytes:
            return False
        exif_dict = piexif.load(exif_bytes)
        thumb_bytes = exif_dict.get("thumbnail")
        if not thumb_bytes:
            return False
        thumb = Image.open(io.BytesIO(thumb_bytes))
        tw, th = thumb.size
        iw, ih = image.size
        # Allow 5% tolerance on aspect ratio
        full_ar = iw / ih
        thumb_ar = tw / th
        return abs(full_ar - thumb_ar) / full_ar > 0.05
    except Exception:
        return False
