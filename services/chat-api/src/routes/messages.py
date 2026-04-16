"""The main chat route: send a message and stream an assistant response."""
from __future__ import annotations

import uuid
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sse_starlette.sse import EventSourceResponse

from ..config import Settings, get_settings
from ..database import get_session_factory
from ..logging_setup import get_logger
from ..middleware.auth_guard import AuthenticatedUser, require_user
from ..services import conversation_service
from ..services.inference_client import InferenceClient
from ..services.stream_proxy import StreamResult, proxy_inference_stream

logger = get_logger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["messages"])


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=8000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    top_k: int | None = Field(default=None, ge=0, le=200)


class RegenerateRequest(BaseModel):
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    top_k: int | None = Field(default=None, ge=0, le=200)


def _parse_uuid(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=404, detail="conversation not found") from exc


async def _ensure_ownership(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
):
    convo = await conversation_service.get_user_conversation(
        session, user_id=user_id, conversation_id=conversation_id
    )
    if convo is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return convo


async def _stream_and_persist(
    *,
    request: Request,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    history: list[dict[str, str]],
    temperature: float | None,
    max_tokens: int | None,
    top_k: int | None,
    model_tag: str,
    first_message: bool,
    first_message_preview: str | None,
    settings: Settings,
) -> AsyncIterator[dict]:
    """Generator that streams inference SSE events to the client and, after the
    stream closes, persists the full assistant message in a fresh DB session.
    """
    http_client = getattr(request.app.state, "inference_http_client", None)
    inference = InferenceClient(settings=settings, http_client=http_client)

    accumulated: dict[str, StreamResult] = {}

    def _capture(result: StreamResult) -> None:
        accumulated["result"] = result

    try:
        async with inference.stream_generate(
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
        ) as response:
            async for event in proxy_inference_stream(response, on_complete=_capture):
                yield event
    except Exception as exc:  # pragma: no cover - defensive path
        logger.error(
            "inference_stream_failed",
            conversation_id=str(conversation_id),
            error=str(exc),
        )
        yield {"data": '{"error":"inference_stream_failed"}'}
        yield {"data": '{"done":true}'}
        return

    result = accumulated.get("result")
    if result is None or not result.completed or not result.content:
        logger.warning(
            "assistant_message_not_persisted",
            conversation_id=str(conversation_id),
            completed=bool(result and result.completed),
            content_len=len(result.content) if result else 0,
        )
        return

    factory: async_sessionmaker[AsyncSession] = get_session_factory()
    try:
        async with factory() as persist_session:
            await conversation_service.append_message(
                persist_session,
                conversation_id=conversation_id,
                role="assistant",
                content=result.content,
                token_count=result.token_count,
                model_tag=model_tag,
                inference_time_ms=result.inference_time_ms,
            )
            if first_message and first_message_preview is not None:
                await conversation_service.update_conversation_title(
                    persist_session,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    title=first_message_preview,
                )
        logger.info(
            "assistant_message_persisted",
            conversation_id=str(conversation_id),
            token_count=result.token_count,
            inference_time_ms=result.inference_time_ms,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.error(
            "assistant_message_persist_failed",
            conversation_id=str(conversation_id),
            error=str(exc),
        )


@router.post("/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    body: SendMessageRequest,
    request: Request,
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
):
    # We own our DB sessions explicitly here so the background task that
    # persists the streamed assistant response can open its own session once
    # the request scope has already closed.
    conv_uuid = _parse_uuid(conversation_id)
    user_uuid = uuid.UUID(user.id)
    session_factory = get_session_factory()

    async with session_factory() as db_session:
        convo = await _ensure_ownership(
            db_session, user_id=user_uuid, conversation_id=conv_uuid
        )
        model_tag = convo.model_tag or "default"

        existing = await conversation_service.get_conversation_messages(
            db_session, conversation_id=conv_uuid, limit=1
        )
        first_message = len(existing) == 0
        first_preview = body.content[:80] if first_message else None

        await conversation_service.append_message(
            db_session,
            conversation_id=conv_uuid,
            role="user",
            content=body.content,
            token_count=None,
            model_tag=model_tag,
        )
        history = await conversation_service.build_history_for_inference(
            db_session, conversation_id=conv_uuid
        )

    logger.info(
        "send_message",
        conversation_id=str(conv_uuid),
        history_len=len(history),
        model_tag=model_tag,
    )

    generator = _stream_and_persist(
        request=request,
        user_id=user_uuid,
        conversation_id=conv_uuid,
        history=history,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        top_k=body.top_k,
        model_tag=model_tag,
        first_message=first_message,
        first_message_preview=first_preview,
        settings=settings,
    )
    return EventSourceResponse(generator, media_type="text/event-stream")


@router.post("/{conversation_id}/regenerate")
async def regenerate(
    conversation_id: str,
    body: RegenerateRequest,
    request: Request,
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
):
    conv_uuid = _parse_uuid(conversation_id)
    user_uuid = uuid.UUID(user.id)
    session_factory = get_session_factory()

    async with session_factory() as db_session:
        convo = await _ensure_ownership(
            db_session, user_id=user_uuid, conversation_id=conv_uuid
        )
        model_tag = convo.model_tag or "default"
        await conversation_service.delete_last_assistant_message(
            db_session, conversation_id=conv_uuid
        )
        history = await conversation_service.build_history_for_inference(
            db_session, conversation_id=conv_uuid
        )

    if not history:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="conversation has no user messages to regenerate from",
        )

    logger.info(
        "regenerate_message",
        conversation_id=str(conv_uuid),
        history_len=len(history),
    )

    generator = _stream_and_persist(
        request=request,
        user_id=user_uuid,
        conversation_id=conv_uuid,
        history=history,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        top_k=body.top_k,
        model_tag=model_tag,
        first_message=False,
        first_message_preview=None,
        settings=settings,
    )
    return EventSourceResponse(generator, media_type="text/event-stream")
