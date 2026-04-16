"""GET /auth/me and PUT /auth/me via Bearer JWT."""
from __future__ import annotations

import pytest

from src.services import user_service
from src.services.google_oauth import OAuthProfile
from src.services.jwt_service import JWTService


@pytest.mark.asyncio
async def test_me_returns_profile(client, db_session):
    profile = OAuthProfile(
        provider="github", provider_id="gh-1", email="me@x.co", name="Me", avatar_url=None
    )
    user = await user_service.upsert_from_oauth(db_session, profile)
    token, _ = JWTService().issue_access_token(
        user_id=str(user.id), email=user.email, name=user.name
    )

    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "me@x.co"
    assert body["provider"] == "github"


@pytest.mark.asyncio
async def test_me_requires_bearer(client):
    resp = await client.get("/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_update_me(client, db_session):
    profile = OAuthProfile(
        provider="google", provider_id="g-9", email="x@y.co", name="Old", avatar_url=None
    )
    user = await user_service.upsert_from_oauth(db_session, profile)
    token, _ = JWTService().issue_access_token(
        user_id=str(user.id), email=user.email, name=user.name
    )

    resp = await client.put(
        "/auth/me",
        json={"name": "New Name", "avatar_url": "https://img/x.png"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "New Name"
    assert body["avatar_url"] == "https://img/x.png"


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/auth/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
