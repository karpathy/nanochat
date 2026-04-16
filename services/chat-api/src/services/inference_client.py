"""HTTP client wrapper for the inference service."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx

from ..config import Settings, get_settings


class InferenceClient:
    """Thin async wrapper around the inference service HTTP contract."""

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = http_client
        self._owns_client = http_client is None

    @property
    def base_url(self) -> str:
        return self._settings.inference_service_url.rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        return {"X-Internal-API-Key": self._settings.internal_api_key}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def list_models(self) -> dict:
        client = self._get_client()
        resp = await client.get(f"{self.base_url}/models", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    async def swap_model(self, model_tag: str) -> dict:
        client = self._get_client()
        resp = await client.post(
            f"{self.base_url}/models/swap",
            headers=self.headers,
            json={"model_tag": model_tag},
        )
        resp.raise_for_status()
        return resp.json()

    @asynccontextmanager
    async def stream_generate(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[httpx.Response]:
        temperature = (
            temperature
            if temperature is not None
            else self._settings.inference_default_temperature
        )
        max_tokens = (
            max_tokens
            if max_tokens is not None
            else self._settings.inference_default_max_tokens
        )
        top_k = top_k if top_k is not None else self._settings.inference_default_top_k

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
        }

        client = self._get_client()
        async with client.stream(
            "POST",
            f"{self.base_url}/generate",
            headers=self.headers,
            json=payload,
        ) as response:
            yield response
