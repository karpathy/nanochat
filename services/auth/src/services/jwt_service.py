"""RS256 JWT issuance + validation.

Access tokens (1h) are returned to the client for Bearer auth.
Refresh tokens (7d) are stored in an httpOnly cookie.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from ..config import Settings, get_settings


class JWTError(Exception):
    """Raised when a token fails validation."""


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    access_expires_in: int
    refresh_expires_in: int


class JWTService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def issue_access_token(self, *, user_id: str, email: str, name: str | None) -> tuple[str, int]:
        now = self._now()
        exp = now + timedelta(seconds=self._settings.jwt_access_ttl_seconds)
        payload = {
            "sub": user_id,
            "email": email,
            "name": name,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "iss": self._settings.jwt_issuer,
            "type": "access",
        }
        token = jwt.encode(payload, self._settings.jwt_private_key, algorithm="RS256")
        return token, self._settings.jwt_access_ttl_seconds

    def issue_refresh_token(self, *, user_id: str) -> tuple[str, int]:
        now = self._now()
        exp = now + timedelta(seconds=self._settings.jwt_refresh_ttl_seconds)
        payload = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "iss": self._settings.jwt_issuer,
            "type": "refresh",
            "jti": uuid.uuid4().hex,
        }
        token = jwt.encode(payload, self._settings.jwt_private_key, algorithm="RS256")
        return token, self._settings.jwt_refresh_ttl_seconds

    def issue_pair(self, *, user_id: str, email: str, name: str | None) -> TokenPair:
        access, access_ttl = self.issue_access_token(user_id=user_id, email=email, name=name)
        refresh, refresh_ttl = self.issue_refresh_token(user_id=user_id)
        return TokenPair(access, refresh, access_ttl, refresh_ttl)

    def decode(self, token: str, *, expected_type: str | None = None) -> dict[str, Any]:
        try:
            payload = jwt.decode(
                token,
                self._settings.jwt_public_key,
                algorithms=["RS256"],
                issuer=self._settings.jwt_issuer,
                options={"require": ["exp", "iat", "sub", "iss"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise JWTError("token expired") from exc
        except jwt.InvalidTokenError as exc:
            raise JWTError(f"invalid token: {exc}") from exc

        if expected_type and payload.get("type") != expected_type:
            raise JWTError(f"expected {expected_type} token, got {payload.get('type')!r}")
        return payload
