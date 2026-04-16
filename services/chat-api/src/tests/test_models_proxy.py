"""Tests for /api/models proxy routes."""
from __future__ import annotations

import uuid

import httpx
import pytest
import respx
import sqlalchemy as sa

from .conftest import stub_auth_validate


def _inference_mock(models_response: dict) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(200, json=models_response)
        if request.url.path == "/models/swap":
            return httpx.Response(200, json={"status": "ok", "current_model": "new-model"})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
@respx.mock
async def test_list_models_proxies_to_inference(app, client, seeded_user):
    stub_auth_validate(respx.mock, seeded_user)
    headers = {"Authorization": "Bearer valid-token"}

    app.state.inference_http_client = httpx.AsyncClient(
        transport=_inference_mock({"current_model": "m1", "models": ["m1", "m2"]})
    )
    try:
        resp = await client.get("/api/models", headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"current_model": "m1", "models": ["m1", "m2"]}
    finally:
        await app.state.inference_http_client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_swap_model_requires_admin(app, client, seeded_user):
    stub_auth_validate(respx.mock, seeded_user)
    headers = {"Authorization": "Bearer valid-token"}

    app.state.inference_http_client = httpx.AsyncClient(transport=_inference_mock({}))
    try:
        resp = await client.post(
            "/api/models/swap",
            json={"model_tag": "new-model"},
            headers=headers,
        )
    finally:
        await app.state.inference_http_client.aclose()
    assert resp.status_code == 403


@pytest.mark.asyncio
@respx.mock
async def test_swap_model_succeeds_for_admin(app, client, session_factory):
    admin_id = str(uuid.uuid4())
    async with session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO users (id, email, name, is_admin) "
                "VALUES (:id, :email, :name, :is_admin)"
            ),
            {"id": admin_id, "email": "root@example.com", "name": "Root", "is_admin": 1},
        )
        await session.commit()

    admin_user = {
        "id": admin_id,
        "email": "root@example.com",
        "name": "Root",
        "is_admin": True,
    }
    stub_auth_validate(respx.mock, admin_user)
    headers = {"Authorization": "Bearer valid-token"}

    app.state.inference_http_client = httpx.AsyncClient(
        transport=_inference_mock({"current_model": "new-model", "models": ["new-model"]})
    )
    try:
        resp = await client.post(
            "/api/models/swap",
            json={"model_tag": "new-model"},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["current_model"] == "new-model"
    finally:
        await app.state.inference_http_client.aclose()
