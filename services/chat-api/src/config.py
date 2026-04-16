"""Runtime configuration for the chat API service."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(default="postgresql+asyncpg://localhost/samosachaat")

    auth_service_url: str = Field(default="http://auth:8001")
    inference_service_url: str = Field(default="http://inference:8000")
    internal_api_key: str = Field(default="")

    max_conversation_history: int = Field(default=50)
    max_token_budget: int = Field(default=6000)

    auth_cache_ttl_seconds: int = Field(default=300)
    auth_cache_max_size: int = Field(default=1024)

    inference_default_temperature: float = Field(default=0.8)
    inference_default_max_tokens: int = Field(default=512)
    inference_default_top_k: int = Field(default=50)

    frontend_url: str = Field(default="http://localhost:3000")

    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
