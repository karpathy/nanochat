from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = "0.0.0.0"
    port: int = 8003
    model_storage_path: Path = Field(default=Path("/models"))
    default_model_tag: str | None = "samosachaat-d12"
    default_step: int | None = None
    hf_token: str | None = None
    hf_repo_owner: str = "manmohan659"
    internal_api_key: str | None = None
    num_workers: int = 1
    device_type: str = ""
    startup_load_enabled: bool = True
    default_temperature: float = 0.8
    default_top_k: int = 50
    default_max_tokens: int = 512

    @property
    def resolved_device_type(self) -> str:
        return self.device_type.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
