#!/usr/bin/env python3
"""Generate an RS256 JWT key pair for development and testing.

Usage
-----
    python scripts/generate_keys.py [--out-dir secrets/] [--key-size 2048]

The private key is written to ``<out-dir>/jwt_private.pem`` (mode 0600)
and the public key to ``<out-dir>/jwt_public.pem`` (mode 0644).

In production, mount the public key as a Docker secret:

    docker secret create jwt_public_key ./secrets/jwt_public.pem

Then reference it in docker-compose.yml:

    secrets:
      jwt_public_key:
        external: true

The API service reads the public key path from
``FORENSCOPE_JWT_PUBLIC_KEY_PATH`` (default: ``secrets/jwt_public.pem``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        print(
            "Error: 'cryptography' package is required.\n"
            "Install it with: pip install cryptography",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Generate an RS256 JWT key pair for ForenScope",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("secrets"),
        help="Directory where key files are written",
    )
    parser.add_argument(
        "--key-size",
        type=int,
        default=2048,
        choices=[2048, 3072, 4096],
        help="RSA modulus size in bits",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=args.key_size,
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_path = args.out_dir / "jwt_private.pem"
    public_path = args.out_dir / "jwt_public.pem"

    private_path.write_bytes(private_pem)
    public_path.write_bytes(public_pem)

    private_path.chmod(0o600)
    public_path.chmod(0o644)

    print(f"Generated {args.key_size}-bit RS256 key pair in {args.out_dir}/")
    print(f"  Private key : {private_path}  (permissions: 0600)")
    print(f"  Public key  : {public_path}  (permissions: 0644)")
    print()
    print("Next steps:")
    print("  1. Add secrets/ to .gitignore (already done — *.pem is ignored).")
    print("  2. Set FORENSCOPE_JWT_PUBLIC_KEY_PATH=secrets/jwt_public.pem in .env")
    print("  3. Use the private key to sign tokens in your auth server.")


if __name__ == "__main__":
    main()
