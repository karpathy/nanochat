"""FastAPI entrypoint for the samosaChaat auth service."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.sessions import SessionMiddleware

from .config import get_settings
from .rate_limit import limiter
from .routes import oauth, session, users


def _rate_limit_handler(request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="samosaChaat Auth", version="0.1.0")

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_middleware(SlowAPIMiddleware)

    # SessionMiddleware is required by authlib for the OAuth state cookie.
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(oauth.router)
    app.include_router(session.router)
    app.include_router(users.router)

    @app.get("/auth/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
