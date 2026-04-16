"""Runtime configuration for the auth service.

All configuration is loaded from environment variables using pydantic-settings.
Private/public keys are PEM-encoded RSA material used for RS256 JWTs.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(default="postgresql+asyncpg://localhost/samosachaat")

    google_client_id: str = Field(default="")
    google_client_secret: str = Field(default="")

    github_client_id: str = Field(default="")
    github_client_secret: str = Field(default="")

    jwt_private_key: str = Field(default="")
    jwt_public_key: str = Field(default="")
    jwt_issuer: str = Field(default="samosachaat-auth")
    jwt_access_ttl_seconds: int = Field(default=3600)
    jwt_refresh_ttl_seconds: int = Field(default=7 * 24 * 3600)

    frontend_url: str = Field(default="http://localhost:3000")
    internal_api_key: str = Field(default="")

    auth_base_url: str = Field(default="http://localhost:8001")
    session_secret: str = Field(default="dev-session-secret-change-me")

    cookie_secure: bool = Field(default=False)
    cookie_domain: str | None = Field(default=None)

    log_level: str = Field(default="INFO")

    @property
    def refresh_cookie_name(self) -> str:
        return "samosachaat_refresh"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
