"""Google OAuth provider via authlib.

Discovery URL: https://accounts.google.com/.well-known/openid-configuration
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from authlib.integrations.starlette_client import OAuth

from ..config import Settings, get_settings

DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"


@dataclass
class OAuthProfile:
    provider: str
    provider_id: str
    email: str
    name: str | None
    avatar_url: str | None


def build_google_client(settings: Settings | None = None) -> OAuth:
    settings = settings or get_settings()
    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url=DISCOVERY_URL,
        client_kwargs={"scope": "openid email profile"},
    )
    return oauth


def profile_from_userinfo(userinfo: dict[str, Any]) -> OAuthProfile:
    provider_id = userinfo.get("sub") or userinfo.get("id")
    email = userinfo.get("email")
    if not provider_id or not email:
        raise ValueError("google userinfo missing sub/email")
    return OAuthProfile(
        provider="google",
        provider_id=str(provider_id),
        email=str(email),
        name=userinfo.get("name"),
        avatar_url=userinfo.get("picture"),
    )
