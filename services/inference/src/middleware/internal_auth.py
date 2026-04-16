from __future__ import annotations

from fastapi import HTTPException, Request, status


def require_internal_api_key(
    request: Request,
) -> None:
    settings = request.app.state.settings
    expected = settings.internal_api_key
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="INTERNAL_API_KEY is not configured",
        )

    provided = request.headers.get("X-Internal-API-Key") or request.headers.get("INTERNAL_API_KEY")
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API key",
        )
