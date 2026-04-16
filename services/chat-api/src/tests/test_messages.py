"""Tests for the streaming message send + regenerate flows."""
from __future__ import annotations

import json
import uuid

import httpx
import pytest
import respx

from .conftest import stub_auth_validate


def _build_inference_mock(tokens: list[str]) -> httpx.MockTransport:
    """Build an httpx mock transport that streams an SSE response."""
    sse_lines: list[bytes] = []
    for token in tokens:
        sse_lines.append(
            f"data: {json.dumps({'token': token, 'gpu': 0})}\n\n".encode("utf-8")
        )
    sse_lines.append(f"data: {json.dumps({'done': True})}\n\n".encode("utf-8"))

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/generate":
            return httpx.Response(404)

        async def body():
            for chunk in sse_lines:
                yield chunk

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body(),
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
@respx.mock
async def test_send_message_streams_and_persists(app, client, seeded_user):
    stub_auth_validate(respx.mock, seeded_user)
    headers = {"Authorization": "Bearer valid-token"}

    create = await client.post("/api/conversations", json={}, headers=headers)
    convo_id = create.json()["id"]

    app.state.inference_http_client = httpx.AsyncClient(
        transport=_build_inference_mock(["hel", "lo", " world"])
    )
    try:
        resp = await client.post(
            f"/api/conversations/{convo_id}/messages",
            json={"content": "hi there"},
            headers=headers,
        )
        assert resp.status_code == 200
        body = resp.text
        assert '"token": "hel"' in body or '"token":"hel"' in body
        assert '"done": true' in body or '"done":true' in body
    finally:
        await app.state.inference_http_client.aclose()

    fetched = await client.get(
        f"/api/conversations/{convo_id}", headers=headers
    )
    assert fetched.status_code == 200
    payload = fetched.json()
    messages = payload["messages"]

    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant"]
    assert messages[0]["content"] == "hi there"
    assert messages[1]["content"] == "hello world"
    assert messages[1]["token_count"] == 3
    assert messages[1]["inference_time_ms"] >= 0

    # First message should have auto-populated the title
    assert payload["title"] == "hi there"


@pytest.mark.asyncio
@respx.mock
async def test_send_message_rejected_on_foreign_conversation(
    app, client, seeded_user, other_user
):
    stub_auth_validate(respx.mock, seeded_user, token="alice-token")
    stub_auth_validate(respx.mock, other_user, token="bob-token")

    alice_headers = {"Authorization": "Bearer alice-token"}
    bob_headers = {"Authorization": "Bearer bob-token"}

    create = await client.post("/api/conversations", json={}, headers=alice_headers)
    convo_id = create.json()["id"]

    app.state.inference_http_client = httpx.AsyncClient(
        transport=_build_inference_mock(["x"])
    )
    try:
        resp = await client.post(
            f"/api/conversations/{convo_id}/messages",
            json={"content": "steal me"},
            headers=bob_headers,
        )
    finally:
        await app.state.inference_http_client.aclose()
    assert resp.status_code == 404


@pytest.mark.asyncio
@respx.mock
async def test_send_message_returns_404_for_missing_conversation(
    app, client, seeded_user
):
    stub_auth_validate(respx.mock, seeded_user)
    headers = {"Authorization": "Bearer valid-token"}

    missing_id = str(uuid.uuid4())
    resp = await client.post(
        f"/api/conversations/{missing_id}/messages",
        json={"content": "hello"},
        headers=headers,
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
@respx.mock
async def test_regenerate_drops_last_assistant_message(app, client, seeded_user):
    stub_auth_validate(respx.mock, seeded_user)
    headers = {"Authorization": "Bearer valid-token"}

    create = await client.post("/api/conversations", json={}, headers=headers)
    convo_id = create.json()["id"]

    app.state.inference_http_client = httpx.AsyncClient(
        transport=_build_inference_mock(["first"])
    )
    try:
        first = await client.post(
            f"/api/conversations/{convo_id}/messages",
            json={"content": "hi"},
            headers=headers,
        )
        assert first.status_code == 200

        app.state.inference_http_client = httpx.AsyncClient(
            transport=_build_inference_mock(["second", " reply"])
        )
        regen = await client.post(
            f"/api/conversations/{convo_id}/regenerate",
            json={},
            headers=headers,
        )
        assert regen.status_code == 200
    finally:
        await app.state.inference_http_client.aclose()

    fetched = await client.get(
        f"/api/conversations/{convo_id}", headers=headers
    )
    messages = fetched.json()["messages"]
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["content"] == "second reply"
