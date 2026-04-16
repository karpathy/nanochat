"""FastAPI entrypoint for the samosaChaat chat API service."""
from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .config import get_settings
from .logging_setup import (
    configure_logging,
    get_logger,
    new_trace_id,
    set_trace_id,
    set_user_id,
)
from .routes import conversations, messages, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.auth_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(5.0, connect=2.0)
    )
    app.state.inference_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=5.0)
    )
    try:
        yield
    finally:
        await app.state.auth_http_client.aclose()
        await app.state.inference_http_client.aclose()


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    logger = get_logger(__name__)

    app = FastAPI(title="samosaChaat Chat API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next) -> Response:
        incoming = request.headers.get("x-trace-id") or request.headers.get("x-request-id")
        trace_id = incoming or new_trace_id()
        set_trace_id(trace_id)
        set_user_id(None)

        logger.info(
            "request_start",
            method=request.method,
            path=request.url.path,
        )
        response = await call_next(request)
        response.headers["x-trace-id"] = trace_id
        logger.info(
            "request_end",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )
        return response

    app.include_router(conversations.router)
    app.include_router(messages.router)
    app.include_router(models.router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "ready": True, "service": "chat-api"}

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


app = create_app()
