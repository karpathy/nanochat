"""User persistence helpers (upsert on OAuth callback, fetch by id)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.user import User
from .google_oauth import OAuthProfile


async def upsert_from_oauth(session: AsyncSession, profile: OAuthProfile) -> User:
    """Insert a new user, or update last_login_at on an existing one."""
    stmt = select(User).where(
        User.provider == profile.provider,
        User.provider_id == profile.provider_id,
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()

    now = datetime.now(timezone.utc)
    if existing is None:
        user = User(
            email=profile.email,
            name=profile.name,
            avatar_url=profile.avatar_url,
            provider=profile.provider,
            provider_id=profile.provider_id,
            created_at=now,
            updated_at=now,
            last_login_at=now,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

    existing.last_login_at = now
    existing.updated_at = now
    if profile.email:
        existing.email = profile.email
    if profile.name is not None:
        existing.name = profile.name
    if profile.avatar_url is not None:
        existing.avatar_url = profile.avatar_url
    await session.commit()
    await session.refresh(existing)
    return existing


async def get_by_id(session: AsyncSession, user_id: str | uuid.UUID) -> User | None:
    if isinstance(user_id, str):
        try:
            user_id = uuid.UUID(user_id)
        except ValueError:
            return None
    return await session.get(User, user_id)


async def update_profile(
    session: AsyncSession,
    user: User,
    *,
    name: str | None,
    avatar_url: str | None,
) -> User:
    if name is not None:
        user.name = name
    if avatar_url is not None:
        user.avatar_url = avatar_url
    user.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(user)
    return user
