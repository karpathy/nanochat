"""CRUD routes for conversations, scoped to the authenticated user."""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..logging_setup import get_logger
from ..middleware.auth_guard import AuthenticatedUser, require_user
from ..services import conversation_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, max_length=500)
    model_tag: str | None = Field(default=None, max_length=100)


class UpdateConversationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)


def _parse_uuid(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=404, detail="conversation not found") from exc


@router.get("")
async def list_conversations(
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    user_uuid = uuid.UUID(user.id)
    conversations = await conversation_service.list_conversations(
        session, user_id=user_uuid, limit=limit, offset=offset
    )
    grouped = conversation_service.group_by_date(conversations)
    return {
        "items": [c.to_dict() for c in conversations],
        "grouped": grouped,
        "limit": limit,
        "offset": offset,
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: CreateConversationRequest,
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
):
    user_uuid = uuid.UUID(user.id)
    convo = await conversation_service.create_conversation(
        session,
        user_id=user_uuid,
        title=body.title,
        model_tag=body.model_tag,
    )
    logger.info("conversation_created", conversation_id=str(convo.id))
    return convo.to_dict()


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
):
    conv_uuid = _parse_uuid(conversation_id)
    user_uuid = uuid.UUID(user.id)
    convo = await conversation_service.get_user_conversation(
        session, user_id=user_uuid, conversation_id=conv_uuid
    )
    if convo is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    messages = await conversation_service.get_conversation_messages(
        session, conversation_id=conv_uuid
    )
    payload = convo.to_dict()
    payload["messages"] = [m.to_dict() for m in messages]
    return payload


@router.put("/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    body: UpdateConversationRequest,
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
):
    conv_uuid = _parse_uuid(conversation_id)
    user_uuid = uuid.UUID(user.id)
    convo = await conversation_service.update_conversation_title(
        session,
        user_id=user_uuid,
        conversation_id=conv_uuid,
        title=body.title,
    )
    if convo is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return convo.to_dict()


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
):
    conv_uuid = _parse_uuid(conversation_id)
    user_uuid = uuid.UUID(user.id)
    deleted = await conversation_service.delete_conversation(
        session, user_id=user_uuid, conversation_id=conv_uuid
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation not found")
    logger.info("conversation_deleted", conversation_id=str(conv_uuid))
    return None
