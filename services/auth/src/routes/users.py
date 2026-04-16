"""User profile routes (GET /auth/me, PUT /auth/me)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..middleware.auth_middleware import AuthContext, require_user
from ..services import user_service

router = APIRouter(prefix="/auth", tags=["users"])


class UserProfile(BaseModel):
    id: str
    email: str
    name: str | None
    avatar_url: str | None
    provider: str
    provider_id: str
    created_at: str | None
    updated_at: str | None
    last_login_at: str | None


class ProfileUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    avatar_url: str | None = Field(default=None, max_length=2048)


@router.get("/me", response_model=UserProfile)
async def me(ctx: AuthContext = Depends(require_user)) -> UserProfile:
    return UserProfile(**ctx.user.to_dict())


@router.put("/me", response_model=UserProfile)
async def update_me(
    payload: ProfileUpdate,
    ctx: AuthContext = Depends(require_user),
    session: AsyncSession = Depends(get_session),
) -> UserProfile:
    user = await user_service.update_profile(
        session, ctx.user, name=payload.name, avatar_url=payload.avatar_url
    )
    return UserProfile(**user.to_dict())
