"""Rate limiter applies 10/min to OAuth start routes."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_google_start_rate_limited(client, monkeypatch):
    # Replace the OAuth client with a stub so /auth/google returns immediately.
    async def _stub_redirect(request, redirect_uri):
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url=redirect_uri, status_code=302)

    class _StubProvider:
        authorize_redirect = staticmethod(_stub_redirect)

    class _StubClient:
        google = _StubProvider()

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.google_oauth = _StubClient()

    # First 10 calls allowed, 11th should be rate-limited.
    codes = []
    for _ in range(11):
        resp = await client.get("/auth/google", follow_redirects=False)
        codes.append(resp.status_code)
    assert codes.count(429) >= 1
