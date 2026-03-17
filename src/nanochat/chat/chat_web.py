# pyright: reportUnusedFunction=false
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.

Launch examples:

- single available GPU (default)
python -m nanochat.scripts.chat_web

- 4 GPUs
python -m nanochat.scripts.chat_web --num-gpus 4

To chat, open the URL printed in the console. (If on cloud box, make sure to use public IP)

Endpoints:
  GET  /           - Chat UI
  POST /chat/completions - Chat API (streaming only)
  GET  /health     - Health check with worker pool status
  GET  /stats      - Worker pool statistics and GPU utilization

Abuse Prevention:
  - Maximum 500 messages per request
  - Maximum 8000 characters per message
  - Maximum 32000 characters total conversation length
  - Temperature clamped to 0.0-2.0
  - Top-k clamped to 0-200 (0 disables top-k filtering, using full vocabulary)
  - Max tokens clamped to 1-4096
"""

import logging
from typing import Optional

from nanochat.config import Config
from nanochat.common import autodetect_device_type, compute_init
from nanochat.chat.server import create_app


def chat_web_server(
    config: Config,
    num_gpus: int,
    source: str,
    temperature: float,
    top_k: int,
    max_tokens: int,
    model_tag: Optional[str],
    step: Optional[int],
    port: int,
    host: str,
) -> None:
    """Start the FastAPI web chat server.

    Loads ``num_gpus`` model replicas from ``config.common.base_dir`` and serves
    a streaming chat API + UI via FastAPI/uvicorn. Requests are distributed across
    the worker pool.

    Args:
        config: Resolved nanochat config. Uses ``config.common.device_type`` and ``config.common.base_dir``.
        num_gpus: Number of GPU workers to spawn (each loads a full model replica).
        source: Checkpoint source to load from: ``sft`` or ``rl``.
        temperature: Default sampling temperature (overridable per request).
        top_k: Default top-k sampling parameter (overridable per request).
        max_tokens: Default max tokens per response (overridable per request).
        model_tag: Optional model tag to select a specific checkpoint.
        step: Optional step number to load a specific checkpoint.
        port: Port to bind the HTTP server to.
        host: Host address to bind the HTTP server to.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    compute_init(device_type)

    app = create_app(
        device_type=device_type,
        base_dir=config.common.base_dir,
        num_gpus=num_gpus,
        source=source,
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens,
        model_tag=model_tag,
        step=step,
        port=port,
    )

    import uvicorn
    print("Starting NanoChat Web Server")
    print(f"Temperature: {temperature}, Top-k: {top_k}, Max tokens: {max_tokens}")
    uvicorn.run(app, host=host, port=port)
