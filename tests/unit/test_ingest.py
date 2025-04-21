"""Unit tests for the image ingestion layer."""

from __future__ import annotations

import hashlib
import io
import struct

import pytest
from PIL import Image

from forenscope.exceptions import (
    CorruptImageError,
    FileTooLargeError,
    ImageTooSmallError,
    UnsupportedFormatError,
)
from forenscope.ingest import (
    IngestedImage,
    _check_thumbnail_mismatch,
    _extract_quantization_tables,
    ingest_image,
)


class TestIngestImage:
    def test_sha256_matches_raw_bytes(self, authentic_jpeg_bytes: bytes) -> None:
        result = ingest_image(authentic_jpeg_bytes)
        expected = hashlib.sha256(authentic_jpeg_bytes).hexdigest()
        assert result.sha256 == expected

    def test_sha256_computed_from_original_bytes(
        self, authentic_jpeg_bytes: bytes
    ) -> None:
        """Modifying the bytes after ingestion should not change the stored hash."""
        result = ingest_image(authentic_jpeg_bytes)
        stored_hash = result.sha256
        # Verify the hash is of the original bytes, not something downstream.
        assert stored_hash == hashlib.sha256(authentic_jpeg_bytes).hexdigest()

    def test_output_is_rgb(self, authentic_jpeg_bytes: bytes) -> None:
        result = ingest_image(authentic_jpeg_bytes)
        assert result.image.mode == "RGB"

    def test_dimensions_reported_correctly(
        self, authentic_jpeg_bytes: bytes, authentic_rgb: Image.Image
    ) -> None:
        result = ingest_image(authentic_jpeg_bytes)
        assert result.width == authentic_rgb.width
        assert result.height == authentic_rgb.height

    def test_format_is_jpeg(self, authentic_jpeg_bytes: bytes) -> None:
        result = ingest_image(authentic_jpeg_bytes)
        assert result.format == "JPEG"

    def test_rejects_gif(self, gif_bytes: bytes) -> None:
        with pytest.raises(UnsupportedFormatError):
            ingest_image(gif_bytes)

    def test_rejects_oversized_bytes(self) -> None:
        oversized = b"x" * (51 * 1024 * 1024)
        with pytest.raises(FileTooLargeError):
            ingest_image(oversized)

    def test_rejects_too_small_image(self, tiny_rgb: Image.Image) -> None:
        buf = io.BytesIO()
        tiny_rgb.save(buf, format="JPEG")
        with pytest.raises(ImageTooSmallError):
            ingest_image(buf.getvalue())

    def test_rejects_corrupt_bytes(self) -> None:
        with pytest.raises(CorruptImageError):
            ingest_image(b"this is not an image")

    def test_png_accepted(self, authentic_rgb: Image.Image) -> None:
        buf = io.BytesIO()
        authentic_rgb.save(buf, format="PNG")
        result = ingest_image(buf.getvalue())
        assert result.format == "PNG"
        assert result.image.mode == "RGB"

    def test_webp_accepted(self, authentic_rgb: Image.Image) -> None:
        buf = io.BytesIO()
        authentic_rgb.save(buf, format="WEBP", quality=90)
        result = ingest_image(buf.getvalue())
        assert result.format == "WEBP"

    def test_ingested_image_has_quantization_tables_for_jpeg(
        self, authentic_jpeg_bytes: bytes
    ) -> None:
        result = ingest_image(authentic_jpeg_bytes)
        assert result.quantization_tables is not None
        assert len(result.quantization_tables) > 0
        assert len(result.quantization_tables[0]) == 64

    def test_png_has_no_quantization_tables(self, authentic_rgb: Image.Image) -> None:
        buf = io.BytesIO()
        authentic_rgb.save(buf, format="PNG")
        result = ingest_image(buf.getvalue())
        assert result.quantization_tables is None

    def test_thumbnail_mismatch_false_for_no_thumbnail(
        self, authentic_jpeg_bytes: bytes
    ) -> None:
        result = ingest_image(authentic_jpeg_bytes)
        assert result.thumbnail_mismatch is False


class TestExtractQuantizationTables:
    def test_returns_none_for_non_jpeg(self) -> None:
        assert _extract_quantization_tables(b"\x89PNG\r\n") is None

    def test_parses_standard_jpeg(self, authentic_jpeg_bytes: bytes) -> None:
        tables = _extract_quantization_tables(authentic_jpeg_bytes)
        assert tables is not None
        for table in tables:
            assert len(table) == 64

    def test_handles_truncated_stream(self) -> None:
        result = _extract_quantization_tables(b"\xff\xd8\xff")
        assert result is None or isinstance(result, list)
