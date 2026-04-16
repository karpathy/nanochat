"""Unit tests for the inference SSE stream proxy."""
from __future__ import annotations

import json

import httpx
import pytest

from src.services.stream_proxy import StreamResult, proxy_inference_stream


def _make_response(lines: list[str]) -> httpx.Response:
    async def body():
        for line in lines:
            yield f"{line}\n".encode("utf-8")

    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body(),
    )


@pytest.mark.asyncio
async def test_stream_proxy_accumulates_tokens_and_signals_done():
    resp = _make_response(
        [
            "data: " + json.dumps({"token": "hel", "gpu": 0}),
            "",
            "data: " + json.dumps({"token": "lo", "gpu": 0}),
            "",
            "data: " + json.dumps({"done": True}),
            "",
        ]
    )

    captured: dict[str, StreamResult] = {}

    def on_complete(result: StreamResult) -> None:
        captured["result"] = result

    events = []
    async for event in proxy_inference_stream(resp, on_complete=on_complete):
        events.append(event)

    assert [json.loads(e["data"]) for e in events] == [
        {"token": "hel", "gpu": 0},
        {"token": "lo", "gpu": 0},
        {"done": True},
    ]

    result = captured["result"]
    assert result.content == "hello"
    assert result.token_count == 2
    assert result.completed is True
    assert result.inference_time_ms >= 0


@pytest.mark.asyncio
async def test_stream_proxy_surfaces_error_status_codes():
    resp = httpx.Response(502, content=b"upstream down")
    captured: dict[str, StreamResult] = {}

    def on_complete(result: StreamResult) -> None:
        captured["result"] = result

    events = []
    async for event in proxy_inference_stream(resp, on_complete=on_complete):
        events.append(event)

    assert any("error" in e["data"] for e in events)
    assert any('"done": true' in e["data"] or '"done":true' in e["data"] for e in events)
    assert captured["result"].completed is False
