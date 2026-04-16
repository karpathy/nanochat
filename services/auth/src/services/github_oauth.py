"""GitHub OAuth provider via authlib.

Authorization URL: https://github.com/login/oauth/authorize
Token URL:         https://github.com/login/oauth/access_token
User API:          https://api.github.com/user
"""
from __future__ import annotations

from typing import Any

from authlib.integrations.starlette_client import OAuth

from ..config import Settings, get_settings
from .google_oauth import OAuthProfile

AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
USERINFO_URL = "https://api.github.com/user"
EMAILS_URL = "https://api.github.com/user/emails"


def build_github_client(settings: Settings | None = None) -> OAuth:
    settings = settings or get_settings()
    oauth = OAuth()
    oauth.register(
        name="github",
        client_id=settings.github_client_id,
        client_secret=settings.github_client_secret,
        access_token_url=TOKEN_URL,
        authorize_url=AUTHORIZE_URL,
        api_base_url="https://api.github.com/",
        client_kwargs={"scope": "read:user user:email"},
    )
    return oauth


def profile_from_userinfo(userinfo: dict[str, Any], emails: list[dict[str, Any]] | None) -> OAuthProfile:
    provider_id = userinfo.get("id")
    if provider_id is None:
        raise ValueError("github userinfo missing id")

    email = userinfo.get("email")
    if not email and emails:
        primary = next((e for e in emails if e.get("primary") and e.get("verified")), None)
        if primary:
            email = primary.get("email")
    if not email:
        raise ValueError("github userinfo missing verified email")

    return OAuthProfile(
        provider="github",
        provider_id=str(provider_id),
        email=str(email),
        name=userinfo.get("name") or userinfo.get("login"),
        avatar_url=userinfo.get("avatar_url"),
    )
