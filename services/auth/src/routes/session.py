"""Session/token refresh routes + internal JWT validation."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_session
from ..rate_limit import limiter
from ..services import user_service
from ..services.jwt_service import JWTError, JWTService

router = APIRouter(prefix="/auth", tags=["session"])


class RefreshResponse(BaseModel):
    access_token: str
    expires_in: int


class ValidateRequest(BaseModel):
    token: str


@router.post("/refresh", response_model=RefreshResponse)
@limiter.limit("30/minute")
async def refresh(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    settings = get_settings()
    cookie = request.cookies.get(settings.refresh_cookie_name)
    if not cookie:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing refresh cookie")

    jwt_service = JWTService()
    try:
        payload = jwt_service.decode(cookie, expected_type="refresh")
    except JWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    user = await user_service.get_by_id(session, payload["sub"])
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found")

    access, ttl = jwt_service.issue_access_token(
        user_id=str(user.id), email=user.email, name=user.name
    )
    return RefreshResponse(access_token=access, expires_in=ttl)


@router.post("/validate")
async def validate(
    payload: ValidateRequest,
    session: AsyncSession = Depends(get_session),
    x_internal_api_key: str | None = Header(default=None, alias="X-Internal-API-Key"),
):
    settings = get_settings()
    if not settings.internal_api_key or x_internal_api_key != settings.internal_api_key:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid internal api key")

    jwt_service = JWTService()
    try:
        claims = jwt_service.decode(payload.token, expected_type="access")
    except JWTError as exc:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"valid": False, "reason": str(exc)},
        )

    user = await user_service.get_by_id(session, claims["sub"])
    if user is None:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"valid": False, "reason": "user not found"},
        )

    return {"valid": True, "user": user.to_dict(), "claims": claims}
