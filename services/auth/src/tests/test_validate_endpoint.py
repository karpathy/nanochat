"""Internal /auth/validate endpoint used by the Chat API service."""
from __future__ import annotations

import pytest

from src.services import user_service
from src.services.google_oauth import OAuthProfile
from src.services.jwt_service import JWTService


@pytest.mark.asyncio
async def test_validate_requires_internal_key(client, db_session):
    profile = OAuthProfile(
        provider="google", provider_id="123", email="v@x.co", name="V", avatar_url=None
    )
    user = await user_service.upsert_from_oauth(db_session, profile)
    token, _ = JWTService().issue_access_token(
        user_id=str(user.id), email=user.email, name=user.name
    )

    missing = await client.post("/auth/validate", json={"token": token})
    assert missing.status_code == 403

    wrong = await client.post(
        "/auth/validate",
        json={"token": token},
        headers={"X-Internal-API-Key": "nope"},
    )
    assert wrong.status_code == 403


@pytest.mark.asyncio
async def test_validate_returns_user_for_valid_token(client, db_session):
    profile = OAuthProfile(
        provider="google", provider_id="456", email="v2@x.co", name="V2", avatar_url=None
    )
    user = await user_service.upsert_from_oauth(db_session, profile)
    token, _ = JWTService().issue_access_token(
        user_id=str(user.id), email=user.email, name=user.name
    )

    resp = await client.post(
        "/auth/validate",
        json={"token": token},
        headers={"X-Internal-API-Key": "test-internal-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["user"]["email"] == "v2@x.co"
    assert body["claims"]["sub"] == str(user.id)


@pytest.mark.asyncio
async def test_validate_rejects_tampered_token(client):
    resp = await client.post(
        "/auth/validate",
        json={"token": "not-a-jwt"},
        headers={"X-Internal-API-Key": "test-internal-key"},
    )
    assert resp.status_code == 401
    assert resp.json()["valid"] is False
