"""Async SQLAlchemy engine and session factory."""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings


class Base(DeclarativeBase):
    """Shared declarative base for all chat-api ORM models."""


_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> None:
    global _engine, _session_factory
    settings = get_settings()
    _engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


def get_engine():
    if _engine is None:
        _build_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        _build_engine()
    assert _session_factory is not None
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    factory = get_session_factory()
    async with factory() as session:
        yield session


def override_session_factory(factory: async_sessionmaker[AsyncSession]) -> None:
    """Testing hook: swap the session factory for an in-memory engine."""
    global _session_factory
    _session_factory = factory
