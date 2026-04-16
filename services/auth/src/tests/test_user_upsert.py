"""User upsert flow invoked by OAuth callbacks."""
from __future__ import annotations

import pytest

from src.services import user_service
from src.services.google_oauth import OAuthProfile


@pytest.mark.asyncio
async def test_upsert_creates_new_user(db_session):
    profile = OAuthProfile(
        provider="google",
        provider_id="g-12345",
        email="alice@example.com",
        name="Alice",
        avatar_url="https://img/alice.png",
    )
    user = await user_service.upsert_from_oauth(db_session, profile)
    assert user.id is not None
    assert user.email == "alice@example.com"
    assert user.provider == "google"
    assert user.last_login_at is not None


@pytest.mark.asyncio
async def test_upsert_updates_existing_user(db_session):
    first = OAuthProfile(
        provider="google",
        provider_id="g-12345",
        email="alice@example.com",
        name="Alice",
        avatar_url=None,
    )
    u1 = await user_service.upsert_from_oauth(db_session, first)
    original_login = u1.last_login_at

    # Second login with updated display name + avatar — same provider identity.
    second = OAuthProfile(
        provider="google",
        provider_id="g-12345",
        email="alice@example.com",
        name="Alice Smith",
        avatar_url="https://img/alice2.png",
    )
    u2 = await user_service.upsert_from_oauth(db_session, second)

    assert u2.id == u1.id  # same row
    assert u2.name == "Alice Smith"
    assert u2.avatar_url == "https://img/alice2.png"
    assert u2.last_login_at >= original_login


@pytest.mark.asyncio
async def test_update_profile(db_session):
    profile = OAuthProfile(
        provider="github",
        provider_id="42",
        email="bob@example.com",
        name="Bob",
        avatar_url=None,
    )
    user = await user_service.upsert_from_oauth(db_session, profile)
    updated = await user_service.update_profile(
        db_session, user, name="Robert", avatar_url="https://img/bob.png"
    )
    assert updated.name == "Robert"
    assert updated.avatar_url == "https://img/bob.png"
