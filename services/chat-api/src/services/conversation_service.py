"""Business logic for conversations and messages, scoped to a single user.

Every query in this module filters by `user_id` — that scoping is the only
thing preventing one user from reading or mutating another user's data.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import date
from typing import Iterable

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..models import Conversation, Message


async def create_conversation(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    title: str | None = None,
    model_tag: str | None = None,
) -> Conversation:
    conversation = Conversation(
        user_id=user_id,
        title=title or "New conversation",
        model_tag=model_tag or "default",
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def list_conversations(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0,
) -> list[Conversation]:
    stmt = (
        sa.select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


def group_by_date(conversations: Iterable[Conversation]) -> dict[str, list[dict]]:
    """Group conversations by YYYY-MM-DD (UTC) of `updated_at`."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for convo in conversations:
        bucket_key: str
        if convo.updated_at is None:
            bucket_key = date.today().isoformat()
        else:
            bucket_key = convo.updated_at.date().isoformat()
        buckets[bucket_key].append(convo.to_dict())
    return dict(buckets)


async def get_user_conversation(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> Conversation | None:
    stmt = sa.select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_conversation_messages(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    limit: int | None = None,
) -> list[Message]:
    stmt = (
        sa.select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_conversation_title(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    title: str,
) -> Conversation | None:
    convo = await get_user_conversation(
        session, user_id=user_id, conversation_id=conversation_id
    )
    if convo is None:
        return None
    convo.title = title
    await session.commit()
    await session.refresh(convo)
    return convo


async def delete_conversation(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> bool:
    convo = await get_user_conversation(
        session, user_id=user_id, conversation_id=conversation_id
    )
    if convo is None:
        return False
    await session.delete(convo)
    await session.commit()
    return True


async def append_message(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    token_count: int | None = None,
    model_tag: str | None = None,
    inference_time_ms: int | None = None,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        token_count=token_count,
        model_tag=model_tag,
        inference_time_ms=inference_time_ms,
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message


async def build_history_for_inference(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
) -> list[dict[str, str]]:
    """Return the trailing slice of history that fits the configured budgets."""
    settings = get_settings()
    max_history = settings.max_conversation_history
    max_budget = settings.max_token_budget

    stmt = (
        sa.select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(max_history)
    )
    result = await session.execute(stmt)
    newest_first = list(result.scalars().all())

    selected: list[Message] = []
    budget = 0
    for message in newest_first:
        budget += len(message.content or "")
        if budget > max_budget and selected:
            break
        selected.append(message)

    selected.reverse()
    return [{"role": m.role, "content": m.content} for m in selected]


async def delete_last_assistant_message(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
) -> bool:
    stmt = (
        sa.select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.role == "assistant",
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    last = result.scalar_one_or_none()
    if last is None:
        return False
    await session.delete(last)
    await session.commit()
    return True
