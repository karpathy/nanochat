"""SQLAlchemy 2.0 async model for the `messages` table."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


MESSAGE_ROLES = ("user", "assistant", "system")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="ck_messages_role",
        ),
        sa.Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        sa.ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    content: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    token_count: Mapped[int | None] = mapped_column(sa.Integer(), nullable=True)
    model_tag: Mapped[str | None] = mapped_column(sa.String(100), nullable=True)
    inference_time_ms: Mapped[int | None] = mapped_column(sa.Integer(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )

    conversation: Mapped["Conversation"] = relationship(  # noqa: F821
        "Conversation",
        back_populates="messages",
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "role": self.role,
            "content": self.content,
            "token_count": self.token_count or 0,
            "model_tag": self.model_tag or "",
            "inference_time_ms": self.inference_time_ms or 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
