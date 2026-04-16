from __future__ import annotations

import asyncio
import json
import random
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from config import Settings, get_settings
from logging_setup import configure_logging, get_logger
from middleware.internal_auth import require_internal_api_key
from services.weight_manager import WeightManager

logger = get_logger(__name__)

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_k: int | None = None


class SwapModelRequest(BaseModel):
    model_tag: str


def validate_generate_request(request: GenerateRequest) -> None:
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request",
        )

    total_length = 0
    for index, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {index} has empty content")
        if len(message.content) > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {index} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message",
            )
        if message.role not in {"user", "assistant"}:
            raise HTTPException(
                status_code=400,
                detail=f"Message {index} has invalid role. Must be 'user' or 'assistant'",
            )
        total_length += len(message.content)

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed",
        )

    if request.temperature is not None and not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(
            status_code=400,
            detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}",
        )

    if request.top_k is not None and not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise HTTPException(status_code=400, detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")

    if request.max_tokens is not None and not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}",
        )


def format_sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def build_conversation_tokens(worker, messages: list[ChatMessage]) -> list[int]:
    bos = worker.tokenizer.get_bos_token_id()
    user_start = worker.tokenizer.encode_special("<|user_start|>")
    user_end = worker.tokenizer.encode_special("<|user_end|>")
    assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

    conversation_tokens = [bos]
    for message in messages:
        if message.role == "user":
            conversation_tokens.append(user_start)
            conversation_tokens.extend(worker.tokenizer.encode(message.content))
            conversation_tokens.append(user_end)
        else:
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(worker.tokenizer.encode(message.content))
            conversation_tokens.append(assistant_end)

    conversation_tokens.append(assistant_start)
    return conversation_tokens


async def generate_stream(worker, request: GenerateRequest, settings: Settings) -> AsyncGenerator[str, None]:
    temperature = request.temperature if request.temperature is not None else settings.default_temperature
    max_new_tokens = request.max_tokens if request.max_tokens is not None else settings.default_max_tokens
    top_k = request.top_k if request.top_k is not None else settings.default_top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()
    accumulated_tokens: list[int] = []
    last_clean_text = ""

    for token_column, _ in worker.engine.generate(
        build_conversation_tokens(worker, request.messages),
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1),
    ):
        token = token_column[0]
        if token == assistant_end or token == bos:
            break

        accumulated_tokens.append(token)
        current_text = worker.tokenizer.decode(accumulated_tokens)
        if current_text.endswith("�"):
            continue

        new_text = current_text[len(last_clean_text):]
        if new_text:
            last_clean_text = current_text
            yield format_sse({"token": new_text, "gpu": worker.gpu_id})

    yield format_sse({"done": True})


class InferenceRuntime:
    def __init__(self, settings: Settings, weight_manager: WeightManager | None = None) -> None:
        self.settings = settings
        self.weight_manager = weight_manager or WeightManager(settings)
        self.worker_pool = None
        self._swap_lock = asyncio.Lock()

    async def startup(self) -> None:
        if not self.settings.startup_load_enabled or not self.settings.default_model_tag:
            return
        try:
            self.worker_pool = await self.weight_manager.build_worker_pool(
                self.settings.default_model_tag,
                step=self.settings.default_step,
            )
        except Exception as exc:  # pragma: no cover - exercised by deployment conditions
            logger.warning("skipping startup model load", error=str(exc))
            self.worker_pool = None

    async def shutdown(self) -> None:
        if self.worker_pool is not None:
            await self.worker_pool.close()

    def health_payload(self) -> dict[str, object]:
        snapshot = self.worker_pool.snapshot() if self.worker_pool is not None else {
            "total_workers": 0,
            "available_workers": 0,
            "busy_workers": 0,
            "draining": False,
            "workers": [],
        }
        return {
            "status": "ok",
            "ready": self.worker_pool is not None and self.weight_manager.current_model is not None,
            "current_model": self.weight_manager.current_model,
            **snapshot,
        }

    def models_payload(self) -> dict[str, object]:
        return {
            "current_model": self.weight_manager.current_model,
            "models": self.weight_manager.list_models(),
        }

    def stats_payload(self) -> dict[str, object]:
        return self.health_payload()

    def require_ready_pool(self):
        if self.worker_pool is None or self.weight_manager.current_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model is currently loaded",
            )
        return self.worker_pool

    async def swap_model(self, model_tag: str) -> dict[str, str]:
        async with self._swap_lock:
            if self.worker_pool is not None and self.weight_manager.current_model == model_tag:
                return {"status": "ok", "current_model": model_tag}

            old_pool = self.worker_pool
            if old_pool is not None:
                await old_pool.drain()

            try:
                new_pool = await self.weight_manager.build_worker_pool(model_tag, step=self.settings.default_step)
            except Exception:
                if old_pool is not None:
                    old_pool.resume_accepting_requests()
                raise

            self.worker_pool = new_pool
            if old_pool is not None:
                await old_pool.close()

            return {"status": "ok", "current_model": model_tag}


def get_runtime(request: Request) -> InferenceRuntime:
    return request.app.state.runtime


def create_app(settings: Settings | None = None, runtime: InferenceRuntime | None = None) -> FastAPI:
    configure_logging()
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_runtime = runtime or InferenceRuntime(resolved_settings)
        app.state.settings = resolved_settings
        app.state.runtime = resolved_runtime
        await resolved_runtime.startup()
        yield
        await resolved_runtime.shutdown()

    app = FastAPI(title="nanochat inference service", version="0.1.0", lifespan=lifespan)

    @app.post("/generate", dependencies=[Depends(require_internal_api_key)])
    async def generate(request_body: GenerateRequest, runtime: InferenceRuntime = Depends(get_runtime)):
        validate_generate_request(request_body)
        worker_pool = runtime.require_ready_pool()

        try:
            worker = await worker_pool.acquire_worker()
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

        async def stream_and_release() -> AsyncGenerator[str, None]:
            try:
                async for chunk in generate_stream(worker, request_body, resolved_settings):
                    yield chunk
            finally:
                await worker_pool.release_worker(worker)

        return StreamingResponse(stream_and_release(), media_type="text/event-stream")

    @app.get("/models", dependencies=[Depends(require_internal_api_key)])
    async def models(runtime: InferenceRuntime = Depends(get_runtime)):
        return runtime.models_payload()

    @app.post("/models/swap", dependencies=[Depends(require_internal_api_key)])
    async def swap_model(request_body: SwapModelRequest, runtime: InferenceRuntime = Depends(get_runtime)):
        payload = await runtime.swap_model(request_body.model_tag)
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=payload)

    @app.get("/health")
    async def health(runtime: InferenceRuntime = Depends(get_runtime)):
        return runtime.health_payload()

    @app.get("/stats", dependencies=[Depends(require_internal_api_key)])
    async def stats(runtime: InferenceRuntime = Depends(get_runtime)):
        return runtime.stats_payload()

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


app = create_app()
