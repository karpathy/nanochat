"""OAuth start/callback routes for Google and GitHub."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_session
from ..rate_limit import limiter
from ..services import github_oauth as github_provider
from ..services import google_oauth as google_provider
from ..services import user_service
from ..services.jwt_service import JWTService

router = APIRouter(prefix="/auth", tags=["oauth"])


def _set_refresh_cookie(response: RedirectResponse, token: str, max_age: int) -> None:
    settings = get_settings()
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=token,
        max_age=max_age,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        domain=settings.cookie_domain,
        path="/",
    )


def _google_oauth(request: Request):
    client = getattr(request.app.state, "google_oauth", None)
    if client is None:
        client = google_provider.build_google_client()
        request.app.state.google_oauth = client
    return client.google


def _github_oauth(request: Request):
    client = getattr(request.app.state, "github_oauth", None)
    if client is None:
        client = github_provider.build_github_client()
        request.app.state.github_oauth = client
    return client.github


@router.get("/google")
@limiter.limit("10/minute")
async def google_start(request: Request):
    settings = get_settings()
    redirect_uri = f"{settings.auth_base_url.rstrip('/')}/auth/google/callback"
    return await _google_oauth(request).authorize_redirect(request, redirect_uri)


@router.get("/google/callback")
@limiter.limit("10/minute")
async def google_callback(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    oauth = _google_oauth(request)
    try:
        token = await oauth.authorize_access_token(request)
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"google oauth failed: {exc}") from exc

    userinfo = token.get("userinfo")
    if not userinfo:
        userinfo = await oauth.userinfo(token=token)

    profile = google_provider.profile_from_userinfo(dict(userinfo))
    user = await user_service.upsert_from_oauth(session, profile)

    jwt_service = JWTService()
    pair = jwt_service.issue_pair(user_id=str(user.id), email=user.email, name=user.name)

    settings = get_settings()
    redirect = RedirectResponse(
        url=f"{settings.frontend_url.rstrip('/')}/chat?access_token={pair.access_token}",
        status_code=status.HTTP_302_FOUND,
    )
    _set_refresh_cookie(redirect, pair.refresh_token, pair.refresh_expires_in)
    return redirect


@router.get("/github")
@limiter.limit("10/minute")
async def github_start(request: Request):
    settings = get_settings()
    redirect_uri = f"{settings.auth_base_url.rstrip('/')}/auth/github/callback"
    return await _github_oauth(request).authorize_redirect(request, redirect_uri)


@router.get("/github/callback")
@limiter.limit("10/minute")
async def github_callback(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    oauth = _github_oauth(request)
    try:
        token = await oauth.authorize_access_token(request)
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"github oauth failed: {exc}") from exc

    user_resp = await oauth.get("user", token=token)
    user_resp.raise_for_status()
    userinfo = user_resp.json()

    emails: list[dict] | None = None
    if not userinfo.get("email"):
        emails_resp = await oauth.get("user/emails", token=token)
        if emails_resp.status_code == 200:
            emails = emails_resp.json()

    profile = github_provider.profile_from_userinfo(dict(userinfo), emails)
    user = await user_service.upsert_from_oauth(session, profile)

    jwt_service = JWTService()
    pair = jwt_service.issue_pair(user_id=str(user.id), email=user.email, name=user.name)

    settings = get_settings()
    redirect = RedirectResponse(
        url=f"{settings.frontend_url.rstrip('/')}/chat?access_token={pair.access_token}",
        status_code=status.HTTP_302_FOUND,
    )
    _set_refresh_cookie(redirect, pair.refresh_token, pair.refresh_expires_in)
    return redirect
