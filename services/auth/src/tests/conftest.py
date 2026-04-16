"""Shared pytest fixtures: RSA keys, in-memory DB, FastAPI test client."""
from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def _make_rsa_pair() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private, public


@pytest.fixture(scope="session", autouse=True)
def _test_environment():
    private, public = _make_rsa_pair()
    os.environ["JWT_PRIVATE_KEY"] = private
    os.environ["JWT_PUBLIC_KEY"] = public
    os.environ["INTERNAL_API_KEY"] = "test-internal-key"
    os.environ["FRONTEND_URL"] = "http://localhost:3000"
    os.environ["AUTH_BASE_URL"] = "http://localhost:8001"
    os.environ["SESSION_SECRET"] = "test-session-secret"
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
    os.environ["GOOGLE_CLIENT_ID"] = "g-id"
    os.environ["GOOGLE_CLIENT_SECRET"] = "g-secret"
    os.environ["GITHUB_CLIENT_ID"] = "gh-id"
    os.environ["GITHUB_CLIENT_SECRET"] = "gh-secret"

    from src.config import get_settings

    get_settings.cache_clear()
    yield


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    from src.database import Base, override_session_factory
    from src.models.user import User  # noqa: F401 — register on Base

    engine = create_async_engine(
        f"sqlite+aiosqlite:///file:test_{uuid.uuid4().hex}?mode=memory&cache=shared&uri=true",
        connect_args={"uri": True},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    override_session_factory(factory)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(session_factory) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(session_factory) -> AsyncIterator[AsyncClient]:
    from src.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
