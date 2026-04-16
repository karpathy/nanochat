"""OAuth callback end-to-end with mocked authlib providers."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.services import user_service
from src.services.google_oauth import OAuthProfile


class _MockGoogleClient:
    """Stands in for authlib's StarletteOAuth2App."""

    def __init__(self, userinfo):
        self._userinfo = userinfo

    async def authorize_access_token(self, request):
        return {"access_token": "fake", "userinfo": self._userinfo}

    async def userinfo(self, token=None):
        return self._userinfo


@pytest.mark.asyncio
async def test_google_callback_creates_user_and_sets_refresh_cookie(client, db_session):
    userinfo = {
        "sub": "google-42",
        "email": "new@user.co",
        "name": "New User",
        "picture": "https://img/new.png",
    }
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.google_oauth = SimpleNamespace(google=_MockGoogleClient(userinfo))

    resp = await client.get("/auth/google/callback", follow_redirects=False)
    assert resp.status_code == 302
    assert "access_token=" in resp.headers["location"]

    from src.config import get_settings

    cookie_name = get_settings().refresh_cookie_name
    assert cookie_name in resp.cookies

    # Verify user was persisted.
    from sqlalchemy import select

    from src.models.user import User

    async with db_session.bind._async_engine.connect() if False else db_session as s:  # type: ignore[attr-defined]
        pass
    # Simpler: just query through a fresh session.
    from src.database import get_session_factory

    async with get_session_factory()() as s:
        user = (
            await s.execute(select(User).where(User.provider_id == "google-42"))
        ).scalar_one()
    assert user.email == "new@user.co"


@pytest.mark.asyncio
async def test_google_callback_updates_existing(client, db_session):
    # Seed an existing user.
    profile = OAuthProfile(
        provider="google",
        provider_id="google-99",
        email="old@user.co",
        name="Old",
        avatar_url=None,
    )
    existing = await user_service.upsert_from_oauth(db_session, profile)
    original_login = existing.last_login_at

    userinfo = {
        "sub": "google-99",
        "email": "old@user.co",
        "name": "Updated Name",
        "picture": "https://img/u.png",
    }
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.google_oauth = SimpleNamespace(google=_MockGoogleClient(userinfo))

    resp = await client.get("/auth/google/callback", follow_redirects=False)
    assert resp.status_code == 302

    from sqlalchemy import select

    from src.database import get_session_factory
    from src.models.user import User

    async with get_session_factory()() as s:
        refreshed = (
            await s.execute(select(User).where(User.id == existing.id))
        ).scalar_one()
    assert refreshed.name == "Updated Name"
    assert refreshed.last_login_at >= original_login
