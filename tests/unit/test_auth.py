"""Unit tests for JWT bearer-token verification."""

from __future__ import annotations

from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
jose = pytest.importorskip("jose")

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import JWTError

from certainaity.api.auth import _load_public_key, verify_jwt

_FAKE_PEM = "-----BEGIN PUBLIC KEY-----\nZmFrZWtleQ==\n-----END PUBLIC KEY-----"


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


@pytest.fixture(autouse=True)
def _clear_key_cache() -> None:
    _load_public_key.cache_clear()
    yield
    _load_public_key.cache_clear()


class TestVerifyJwt:
    def test_valid_token_returns_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CERTAINAITY_JWT_PUBLIC_KEY", _FAKE_PEM)
        expected = {"sub": "user-42", "exp": 9_999_999_999}
        with patch("jose.jwt.decode", return_value=expected):
            result = verify_jwt(_creds("header.payload.sig"))
        assert result["sub"] == "user-42"

    def test_invalid_token_raises_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CERTAINAITY_JWT_PUBLIC_KEY", _FAKE_PEM)
        with patch("jose.jwt.decode", side_effect=JWTError("bad")):
            with pytest.raises(HTTPException) as exc:
                verify_jwt(_creds("bad.token.here"))
        assert exc.value.status_code == 401

    def test_expired_token_raises_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jose.exceptions import ExpiredSignatureError

        monkeypatch.setenv("CERTAINAITY_JWT_PUBLIC_KEY", _FAKE_PEM)
        with patch("jose.jwt.decode", side_effect=ExpiredSignatureError("expired")):
            with pytest.raises(HTTPException) as exc:
                verify_jwt(_creds("expired.token.here"))
        assert exc.value.status_code == 401

    def test_www_authenticate_header_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CERTAINAITY_JWT_PUBLIC_KEY", _FAKE_PEM)
        with patch("jose.jwt.decode", side_effect=JWTError("x")):
            with pytest.raises(HTTPException) as exc:
                verify_jwt(_creds("x"))
        assert "Bearer" in (exc.value.headers or {}).get("WWW-Authenticate", "")

    def test_env_var_key_takes_precedence_over_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CERTAINAITY_JWT_PUBLIC_KEY", _FAKE_PEM)
        payload = {"sub": "env-user"}
        with patch("jose.jwt.decode", return_value=payload) as mock_decode:
            verify_jwt(_creds("tok"))
        # The key passed to decode should be the env-var value (with newline normalised)
        call_args = mock_decode.call_args
        assert "-----BEGIN PUBLIC KEY-----" in call_args[0][1]

    def test_newline_escape_in_env_var_normalised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        escaped = _FAKE_PEM.replace("\n", "\\n")
        monkeypatch.setenv("CERTAINAITY_JWT_PUBLIC_KEY", escaped)
        payload = {"sub": "u"}
        with patch("jose.jwt.decode", return_value=payload) as mock_decode:
            verify_jwt(_creds("tok"))
        used_key = mock_decode.call_args[0][1]
        assert "\\n" not in used_key


class TestLoadPublicKey:
    def test_reads_pem_from_disk(self, tmp_path: Path) -> None:
        from pathlib import Path

        key_file = tmp_path / "jwt_public.pem"
        key_file.write_text(_FAKE_PEM)
        assert _load_public_key(key_file) == _FAKE_PEM

    def test_missing_file_raises_runtime_error(self, tmp_path: Path) -> None:
        from pathlib import Path

        with pytest.raises(RuntimeError, match="JWT public key not found"):
            _load_public_key(tmp_path / "nonexistent.pem")

    def test_result_is_cached(self, tmp_path: Path) -> None:
        from pathlib import Path

        key_file = tmp_path / "key.pem"
        key_file.write_text(_FAKE_PEM)
        r1 = _load_public_key(key_file)
        key_file.write_text("changed")
        r2 = _load_public_key(key_file)
        assert r1 == r2  # cached — second write not reflected
