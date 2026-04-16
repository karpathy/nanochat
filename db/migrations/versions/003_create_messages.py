"""create messages table

Revision ID: 003_create_messages
Revises: 002_create_conversations
Create Date: 2026-04-16

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003_create_messages"
down_revision: Union[str, None] = "002_create_conversations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("model_tag", sa.String(length=100), nullable=True),
        sa.Column("inference_time_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint(
            "role IN ('user','assistant','system')",
            name="ck_messages_role",
        ),
    )
    op.create_index(
        "idx_messages_conversation",
        "messages",
        ["conversation_id", sa.text("created_at ASC")],
    )


def downgrade() -> None:
    op.drop_index("idx_messages_conversation", table_name="messages")
    op.drop_table("messages")
