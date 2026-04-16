"""Authentication guard that validates JWTs via the auth service.

Successful validations are cached in an in-memory TTL cache keyed by the raw
JWT string so that a burst of requests from the same user does not fan out to
the auth service on every call.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Annotated, Any

import httpx
from cachetools import TTLCache
from fastapi import Depends, Header, HTTPException, Request, status

from ..config import Settings, get_settings
from ..logging_setup import get_logger, set_user_id

logger = get_logger(__name__)


@dataclass
class AuthenticatedUser:
    id: str
    email: str
    name: str | None
    raw: dict[str, Any]

    @classmethod
    def from_validate_response(cls, payload: dict[str, Any]) -> "AuthenticatedUser":
        user = payload.get("user") or {}
        return cls(
            id=str(user["id"]),
            email=user.get("email", ""),
            name=user.get("name"),
            raw=user,
        )


class AuthCache:
    """Thread-safe TTL cache for validated JWTs."""

    def __init__(self, ttl_seconds: int, max_size: int) -> None:
        self._cache: TTLCache[str, AuthenticatedUser] = TTLCache(
            maxsize=max_size, ttl=ttl_seconds
        )
        self._lock = asyncio.Lock()

    async def get(self, token: str) -> AuthenticatedUser | None:
        async with self._lock:
            return self._cache.get(token)

    async def set(self, token: str, user: AuthenticatedUser) -> None:
        async with self._lock:
            self._cache[token] = user

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()


_auth_cache: AuthCache | None = None


def get_auth_cache() -> AuthCache:
    global _auth_cache
    if _auth_cache is None:
        settings = get_settings()
        _auth_cache = AuthCache(
            ttl_seconds=settings.auth_cache_ttl_seconds,
            max_size=settings.auth_cache_max_size,
        )
    return _auth_cache


def reset_auth_cache() -> None:
    """Testing hook: drop the cached singleton so a fresh one is built."""
    global _auth_cache
    _auth_cache = None


async def _validate_with_auth_service(
    token: str, settings: Settings, http_client: httpx.AsyncClient | None = None
) -> AuthenticatedUser:
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=5.0)
    try:
        response = await client.post(
            f"{settings.auth_service_url.rstrip('/')}/auth/validate",
            headers={"X-Internal-API-Key": settings.internal_api_key},
            json={"token": token},
        )
    except httpx.HTTPError as exc:
        logger.error("auth_service_unreachable", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="auth service unreachable",
        ) from exc
    finally:
        if owns_client:
            await client.aclose()

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    if response.status_code == 403:
        raise HTTPException(status_code=500, detail="internal api key rejected by auth")
    if response.status_code >= 400:
        logger.error(
            "auth_validate_failed",
            status_code=response.status_code,
            body=response.text[:200],
        )
        raise HTTPException(status_code=401, detail="token validation failed")

    data = response.json()
    if not data.get("valid"):
        raise HTTPException(status_code=401, detail=data.get("reason", "invalid token"))

    return AuthenticatedUser.from_validate_response(data)


def _extract_bearer(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="missing authorization header")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(status_code=401, detail="invalid authorization scheme")
    return parts[1].strip()


async def require_user(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> AuthenticatedUser:
    """FastAPI dependency that yields the authenticated user for the request."""
    token = _extract_bearer(authorization)
    cache = get_auth_cache()

    cached = await cache.get(token)
    if cached is not None:
        set_user_id(cached.id)
        request.state.user = cached
        return cached

    http_client: httpx.AsyncClient | None = getattr(
        request.app.state, "auth_http_client", None
    )
    user = await _validate_with_auth_service(token, settings, http_client=http_client)
    await cache.set(token, user)
    set_user_id(user.id)
    request.state.user = user
    return user
