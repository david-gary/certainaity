"""JWT RS256 bearer-token verification middleware.

The API service mounts the JWT public key as a Docker secret at
``CERTAINAITY_JWT_PUBLIC_KEY_PATH`` (default: ``secrets/jwt_public.pem``).
Tokens must be signed with the corresponding private key (RS256).

Use ``scripts/generate_keys.py`` to create a development key pair.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from certainaity.config import get_settings

log = structlog.get_logger()
_bearer_scheme = HTTPBearer(auto_error=True)


@lru_cache(maxsize=1)
def _load_public_key(key_path: Path) -> str:
    """Read the PEM public key from disk (cached after first load)."""
    try:
        return key_path.read_text()
    except FileNotFoundError as exc:
        log.error("jwt_public_key_not_found", path=str(key_path))
        raise RuntimeError(f"JWT public key not found: {key_path}") from exc


def verify_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict:
    """FastAPI dependency: validate a Bearer JWT and return the decoded payload.

    Raises HTTP 401 if the token is missing, expired, or has an invalid
    signature.  The audience claim is not verified — scope is enforced
    by the issuer (auth server) rather than the resource server.
    """
    from jose import JWTError, jwt

    settings = get_settings()
    try:
        public_key = _load_public_key(settings.jwt_public_key_path)
        payload: dict = jwt.decode(
            credentials.credentials,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        return payload
    except JWTError as exc:
        log.warning("jwt_validation_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
