"""Chat API router: all route handlers and streaming generation."""
# pyright: basic

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from nanochat.chat.server.constants import (
    MAX_MAX_TOKENS,
    MAX_MESSAGE_LENGTH,
    MAX_MESSAGES_PER_REQUEST,
    MAX_TEMPERATURE,
    MAX_TOP_K,
    MAX_TOTAL_CONVERSATION_LENGTH,
    MIN_MAX_TOKENS,
    MIN_TEMPERATURE,
    MIN_TOP_K,
)
from nanochat.chat.server.models import ChatRequest
from nanochat.chat.server.worker_pool import Worker

logger = logging.getLogger(__name__)


@dataclass
class GenerationDefaults:
    temperature: float
    top_k: int
    max_tokens: int


def validate_chat_request(request: ChatRequest) -> None:
    """Validate chat request to prevent abuse."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request",
        )

    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")
        if len(message.content) > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message",
            )
        total_length += len(message.content)

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed",
        )

    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(status_code=400, detail=f"Message {i} has invalid role. Must be 'user' or 'assistant'")

    if request.temperature is not None and not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(
            status_code=400, detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
        )

    if request.top_k is not None and not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise HTTPException(status_code=400, detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")

    if request.max_tokens is not None and not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
        raise HTTPException(status_code=400, detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}")


async def generate_stream(
    worker: Worker,
    tokens: list[int],
    defaults: GenerationDefaults,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    top_k: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Generate assistant response tokens as a Server-Sent Events stream."""
    temperature = temperature if temperature is not None else defaults.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else defaults.max_tokens
    top_k = top_k if top_k is not None else defaults.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()
    accumulated_tokens: list[int] = []
    last_clean_text = ""

    for token_column, _ in worker.engine.generate(
        tokens,
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
        if not current_text.endswith("\ufffd"):
            new_text = current_text[len(last_clean_text) :]
            if new_text:
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"


def build_router(defaults: GenerationDefaults) -> APIRouter:
    """Build and return the chat API router.

    Args:
        defaults: Generation defaults used when the request does not override them.
    """
    router = APIRouter()

    @router.get("/")
    async def root():
        """Serve the chat UI."""
        ui_html_path = os.path.join("nanochat", "ui.html")
        with open(ui_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        html_content = html_content.replace(
            "const API_URL = `http://${window.location.hostname}:8000`;", "const API_URL = '';"
        )
        return HTMLResponse(content=html_content)

    @router.get("/logo.svg")
    async def logo():
        """Serve the NanoChat logo."""
        return FileResponse(os.path.join("nanochat", "logo.svg"), media_type="image/svg+xml")

    @router.post("/chat/completions")
    async def chat_completions(request_body: ChatRequest, request: Request):
        """Chat completion endpoint (streaming only)."""
        validate_chat_request(request_body)

        logger.info("=" * 20)
        for message in request_body.messages:
            logger.info(f"[{message.role.upper()}]: {message.content}")
        logger.info("-" * 20)

        worker_pool = request.app.state.worker_pool
        worker = await worker_pool.acquire_worker()

        try:
            bos = worker.tokenizer.get_bos_token_id()
            user_start = worker.tokenizer.encode_special("<|user_start|>")
            user_end = worker.tokenizer.encode_special("<|user_end|>")
            assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
            assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

            conversation_tokens = [bos]
            for message in request_body.messages:
                if message.role == "user":
                    conversation_tokens += [user_start, *worker.tokenizer.encode(message.content), user_end]
                elif message.role == "assistant":
                    conversation_tokens += [assistant_start, *worker.tokenizer.encode(message.content), assistant_end]
            conversation_tokens.append(assistant_start)

            response_tokens: list[str] = []

            async def stream_and_release():
                try:
                    async for chunk in generate_stream(
                        worker,
                        conversation_tokens,
                        defaults=defaults,
                        temperature=request_body.temperature,
                        max_new_tokens=request_body.max_tokens,
                        top_k=request_body.top_k,
                    ):
                        chunk_data = json.loads(chunk.replace("data: ", "").strip())
                        if "token" in chunk_data:
                            response_tokens.append(chunk_data["token"])
                        yield chunk
                finally:
                    logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {''.join(response_tokens)}")
                    logger.info("=" * 20)
                    await worker_pool.release_worker(worker)

            return StreamingResponse(stream_and_release(), media_type="text/event-stream")
        except Exception as e:
            await worker_pool.release_worker(worker)
            raise e

    @router.get("/health")
    async def health(request: Request):
        """Health check with worker pool status."""
        worker_pool = getattr(request.app.state, "worker_pool", None)
        return {
            "status": "ok",
            "ready": worker_pool is not None and bool(worker_pool.workers),
            "num_gpus": worker_pool.num_gpus if worker_pool else 0,
            "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0,
        }

    @router.get("/stats")
    async def stats(request: Request):
        """Worker pool statistics and GPU utilization."""
        worker_pool = request.app.state.worker_pool
        return {
            "total_workers": len(worker_pool.workers),
            "available_workers": worker_pool.available_workers.qsize(),
            "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
            "workers": [{"gpu_id": w.gpu_id, "device": str(w.device)} for w in worker_pool.workers],
        }

    return router
