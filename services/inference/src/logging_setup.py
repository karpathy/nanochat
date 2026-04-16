"""Structured JSON logging for the inference service.

Mirrors the canonical implementation in services/chat-api/src/logging_setup.py
so every service emits the same JSON shape (see contracts/logging-standard.md).
"""
from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

from config import get_settings

_trace_id_ctx: ContextVar[str | None] = ContextVar("trace_id", default=None)
_user_id_ctx: ContextVar[str | None] = ContextVar("user_id", default=None)


def set_trace_id(trace_id: str | None) -> None:
    _trace_id_ctx.set(trace_id)


def set_user_id(user_id: str | None) -> None:
    _user_id_ctx.set(user_id)


def get_trace_id() -> str | None:
    return _trace_id_ctx.get()


def get_user_id() -> str | None:
    return _user_id_ctx.get()


def new_trace_id() -> str:
    return uuid.uuid4().hex


def _inject_context(_logger, _method, event_dict):
    event_dict.setdefault("service", "inference")
    trace_id = _trace_id_ctx.get()
    if trace_id is not None:
        event_dict.setdefault("trace_id", trace_id)
    user_id = _user_id_ctx.get()
    if user_id is not None:
        event_dict.setdefault("user_id", user_id)
    return event_dict


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _inject_context,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)
