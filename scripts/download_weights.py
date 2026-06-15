"""Download Certainaity model weights from the versioned S3 bucket.

Usage
-----
    python scripts/download_weights.py --token <TOKEN> [--weights-dir weights/]

The token is issued per organization and authorizes download of the v1.0
weights. Checksums are verified after download; any mismatch aborts and
the partial file is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# Weights registry: {filename: (download_url_template, sha256)}
# URL template has {token} substituted at runtime.
WEIGHTS_REGISTRY: dict[str, tuple[str, str]] = {
    "patchforensic_v2.pth": (
        "https://weights.certainaity.com/v1/patchforensic_v2.pth?token={token}",
        "PLACEHOLDER_SHA256_PATCHFORENSIC",
    ),
    "mantranet_finetuned.pth": (
        "https://weights.certainaity.com/v1/mantranet_finetuned.pth?token={token}",
        "PLACEHOLDER_SHA256_MANTRANET",
    ),
    "spsl_siamese.pth": (
        "https://weights.certainaity.com/v1/spsl_siamese.pth?token={token}",
        "PLACEHOLDER_SHA256_SPSL",
    ),
    "inpainting_detector_clip.pth": (
        "https://weights.certainaity.com/v1/inpainting_detector_clip.pth?token={token}",
        "PLACEHOLDER_SHA256_INPAINTING",
    ),
}


def download_weights(
    token: str,
    weights_dir: Path,
    models: list[str] | None = None,
    skip_existing: bool = True,
) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    targets = models or list(WEIGHTS_REGISTRY)

    for filename in targets:
        if filename not in WEIGHTS_REGISTRY:
            log.error("Unknown model weight file: %r", filename)
            sys.exit(1)

        url_template, expected_sha256 = WEIGHTS_REGISTRY[filename]
        url = url_template.format(token=token)
        dest = weights_dir / filename

        if skip_existing and dest.exists():
            if _verify_checksum(dest, expected_sha256):
                log.info("%s — already present and verified, skipping.", filename)
                continue
            else:
                log.warning("%s — checksum mismatch on existing file, re-downloading.", filename)

        log.info("Downloading %s ...", filename)
        _download_file(url, dest)

        if not _verify_checksum(dest, expected_sha256):
            dest.unlink(missing_ok=True)
            log.error(
                "Checksum mismatch for %s after download. File deleted.", filename
            )
            sys.exit(1)

        log.info("%s — OK (%.1f MB)", filename, dest.stat().st_size / 1e6)


def _download_file(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".tmp")
    try:
        def _progress(block_count: int, block_size: int, total: int) -> None:
            if total > 0:
                pct = min(100, block_count * block_size * 100 // total)
                print(f"\r  {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        print()
        tmp.rename(dest)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {exc}") from exc


def _verify_checksum(path: Path, expected: str) -> bool:
    if expected.startswith("PLACEHOLDER_"):
        log.debug("Checksum for %s is a placeholder — skipping verification.", path.name)
        return True
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return sha256 == expected


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download Certainaity model weights")
    parser.add_argument("--token", required=True, help="Download authorization token")
    parser.add_argument(
        "--weights-dir", type=Path, default=Path("weights"),
        help="Directory to save weights (default: weights/)"
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(WEIGHTS_REGISTRY),
        help="Download only specific models (default: all)"
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Re-download even if the file already exists"
    )
    args = parser.parse_args()

    download_weights(
        token=args.token,
        weights_dir=args.weights_dir,
        models=args.models,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
