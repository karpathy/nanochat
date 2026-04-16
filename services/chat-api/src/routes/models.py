"""Proxy routes that forward model management calls to the inference service."""
from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..middleware.auth_guard import AuthenticatedUser, require_user
from ..services.inference_client import InferenceClient

logger = get_logger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


class SwapModelRequest(BaseModel):
    model_tag: str = Field(..., min_length=1, max_length=100)


def _client_for(request: Request, settings: Settings) -> InferenceClient:
    http_client = getattr(request.app.state, "inference_http_client", None)
    return InferenceClient(settings=settings, http_client=http_client)


@router.get("")
async def list_models(
    request: Request,
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
):
    client = _client_for(request, settings)
    try:
        return await client.list_models()
    except httpx.HTTPStatusError as exc:
        logger.error("list_models_proxy_failed", status_code=exc.response.status_code)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="inference service error",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="inference service unreachable",
        ) from exc


@router.post("/swap")
async def swap_model(
    body: SwapModelRequest,
    request: Request,
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
):
    if not user.raw.get("is_admin"):
        raise HTTPException(status_code=403, detail="admin privilege required")

    client = _client_for(request, settings)
    try:
        return await client.swap_model(body.model_tag)
    except httpx.HTTPStatusError as exc:
        logger.error("swap_model_proxy_failed", status_code=exc.response.status_code)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="inference service rejected swap",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="inference service unreachable",
        ) from exc
