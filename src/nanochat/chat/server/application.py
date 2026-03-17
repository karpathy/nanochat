"""FastAPI app factory: lifespan, middleware, and router registration."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nanochat.chat.server.router import GenerationDefaults, build_router
from nanochat.chat.server.worker_pool import WorkerPool


def create_app(
    device_type: str,
    base_dir: str,
    num_gpus: int,
    source: str,
    temperature: float,
    top_k: int,
    max_tokens: int,
    model_tag: Optional[str],
    step: Optional[int],
    port: int,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        device_type: Device to run inference on: ``cuda``, ``cpu``, or ``mps``.
        base_dir: Nanochat base directory for checkpoint loading.
        num_gpus: Number of GPU workers to spawn.
        source: Checkpoint source: ``sft`` or ``rl``.
        temperature: Default sampling temperature.
        top_k: Default top-k sampling parameter.
        max_tokens: Default max tokens per response.
        model_tag: Optional model tag to select a specific checkpoint.
        step: Optional step number to load a specific checkpoint.
        port: Port the server is bound to (used in startup log message).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("Loading nanochat models across GPUs...")
        app.state.worker_pool = WorkerPool(device_type, base_dir=base_dir, num_gpus=num_gpus)
        await app.state.worker_pool.initialize(source, model_tag=model_tag, step=step)
        print(f"Server ready at http://localhost:{port}")
        yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    defaults = GenerationDefaults(temperature=temperature, top_k=top_k, max_tokens=max_tokens)
    app.include_router(build_router(defaults))

    return app
