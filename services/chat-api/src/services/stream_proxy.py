"""Proxy the inference SSE stream to the client while accumulating tokens.

The inference service emits lines like `data: {"token": "...", "gpu": 0}` and
terminates with `data: {"done": true}`. We forward each event unchanged to the
client, collect assistant tokens into a buffer, and report the buffer plus
timing info once the stream closes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import httpx

from ..logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class StreamResult:
    content: str
    token_count: int
    inference_time_ms: int
    completed: bool


def _parse_sse_data(raw_line: str) -> dict | None:
    line = raw_line.strip()
    if not line or not line.startswith("data:"):
        return None
    body = line[len("data:"):].strip()
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        logger.warning("stream_proxy_bad_sse_payload", payload=body[:200])
        return None


async def proxy_inference_stream(
    response: httpx.Response,
    *,
    on_complete: Callable[[StreamResult], None] | None = None,
) -> AsyncIterator[dict]:
    """Async generator that yields SSE events as dicts with ``data`` keys.

    Each yielded event is shaped as ``{"data": "<json-string>"}`` so callers can
    pass it straight into sse-starlette's ``EventSourceResponse``.
    """
    started = time.perf_counter()
    buffer: list[str] = []
    token_count = 0
    completed = False

    if response.status_code >= 400:
        body = await response.aread()
        logger.error(
            "inference_error_response",
            status_code=response.status_code,
            body=body.decode("utf-8", errors="replace")[:200],
        )
        error_payload = json.dumps(
            {
                "error": "inference service returned an error",
                "status_code": response.status_code,
            }
        )
        yield {"data": error_payload}
        yield {"data": json.dumps({"done": True})}
        if on_complete is not None:
            on_complete(
                StreamResult(
                    content="",
                    token_count=0,
                    inference_time_ms=int((time.perf_counter() - started) * 1000),
                    completed=False,
                )
            )
        return

    try:
        async for raw_line in response.aiter_lines():
            if not raw_line:
                continue
            parsed = _parse_sse_data(raw_line)
            if parsed is None:
                continue

            if parsed.get("done"):
                completed = True
                yield {"data": json.dumps(parsed)}
                break

            token = parsed.get("token")
            if isinstance(token, str) and token:
                buffer.append(token)
                token_count += 1

            yield {"data": json.dumps(parsed)}
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if on_complete is not None:
            on_complete(
                StreamResult(
                    content="".join(buffer),
                    token_count=token_count,
                    inference_time_ms=elapsed_ms,
                    completed=completed,
                )
            )
