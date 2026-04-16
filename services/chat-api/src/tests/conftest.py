"""Shared pytest fixtures for the chat API test suite.

Tests run against a throwaway SQLite database (via aiosqlite) with a stub
`users` table so that the foreign key from `conversations.user_id` validates.
The auth service is mocked with respx; the inference service SSE responses
are mocked with hand-rolled httpx MockTransports.
"""
from __future__ import annotations

import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src import database as db_module
from src.database import Base
from src.middleware import auth_guard
from src.models import Conversation, Message  # noqa: F401  (registers tables)


# Attach a stub `users` table to the shared metadata so the FK on
# `conversations.user_id` can resolve during SQLite-based test setup.
if "users" not in Base.metadata.tables:
    sa.Table(
        "users",
        Base.metadata,
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255)),
        sa.Column("name", sa.String(255)),
        sa.Column("is_admin", sa.Integer(), default=0),
    )


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    # Settings and the auth cache are module-level singletons; wipe them
    # between tests so environment overrides take effect cleanly.
    from src.config import get_settings

    get_settings.cache_clear()
    auth_guard.reset_auth_cache()
    _TOKEN_REGISTRY.clear()
    yield
    get_settings.cache_clear()
    auth_guard.reset_auth_cache()
    _TOKEN_REGISTRY.clear()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("AUTH_SERVICE_URL", "http://auth.test")
    monkeypatch.setenv("INFERENCE_SERVICE_URL", "http://inference.test")
    monkeypatch.setenv("INTERNAL_API_KEY", "test-internal-key")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    from src.config import get_settings

    get_settings.cache_clear()


@pytest_asyncio.fixture
async def engine(_env) -> AsyncIterator:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(engine) -> async_sessionmaker[AsyncSession]:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_module.override_session_factory(factory)
    return factory


@pytest_asyncio.fixture
async def seeded_user(session_factory) -> dict:
    user_id = str(uuid.uuid4())
    async with session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO users (id, email, name, is_admin) "
                "VALUES (:id, :email, :name, :is_admin)"
            ),
            {
                "id": user_id,
                "email": "alice@example.com",
                "name": "Alice",
                "is_admin": 0,
            },
        )
        await session.commit()
    return {"id": user_id, "email": "alice@example.com", "name": "Alice"}


@pytest_asyncio.fixture
async def other_user(session_factory) -> dict:
    user_id = str(uuid.uuid4())
    async with session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO users (id, email, name, is_admin) "
                "VALUES (:id, :email, :name, :is_admin)"
            ),
            {"id": user_id, "email": "bob@example.com", "name": "Bob", "is_admin": 0},
        )
        await session.commit()
    return {"id": user_id, "email": "bob@example.com", "name": "Bob"}


@pytest.fixture
def app():
    from src.main import create_app

    return create_app()


@pytest_asyncio.fixture
async def client(app, session_factory) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


_TOKEN_REGISTRY: dict[str, dict] = {}


def _dispatch_auth_validate(request):
    import json

    import httpx

    payload = json.loads(request.read())
    token = payload.get("token")
    if request.headers.get("X-Internal-API-Key") != "test-internal-key":
        return httpx.Response(403, json={"detail": "invalid internal api key"})
    user = _TOKEN_REGISTRY.get(token)
    if user is None:
        return httpx.Response(401, json={"valid": False, "reason": "invalid token"})
    return httpx.Response(
        200,
        json={"valid": True, "user": user, "claims": {"sub": user["id"]}},
    )


def stub_auth_validate(respx_mock, user: dict, token: str = "valid-token"):
    """Register a respx mock that returns the given user for the given token.

    Multiple calls within a single test accumulate token→user mappings so
    several users can authenticate in the same scenario.
    """
    _TOKEN_REGISTRY[token] = user
    respx_mock.post("http://auth.test/auth/validate").mock(
        side_effect=_dispatch_auth_validate
    )
