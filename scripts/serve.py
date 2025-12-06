#!/usr/bin/env python3
"""
Production-ready Inference Server for NanoChat.
Implements an OpenAI-compatible API using FastAPI.

Endpoints:
  POST /v1/chat/completions - Standard chat completion (streaming & non-streaming)
  GET  /v1/models           - List available models
  GET  /health              - Health check
  GET  /                    - Web UI

Usage:
  python -m scripts.serve [args]
"""

import argparse
import json
import os
import time
import uuid
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import List, Optional, Union, Dict, Any, AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
from contextlib import nullcontext

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("nanochat.serve")

# -----------------------------------------------------------------------------
# Configuration & Args
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='NanoChat OpenAI-compatible API Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type: cuda|cpu|mps (default: auto)')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
parser.add_argument('--mock', action='store_true', help='Run in mock mode without loading real models')

# We parse args only if running as main, otherwise we use defaults or env vars
if __name__ == "__main__":
    args = parser.parse_args()
else:
    # Dummy args for when imported
    args = argparse.Namespace(
        num_gpus=int(os.getenv("NANOCHAT_NUM_GPUS", "1")),
        source=os.getenv("NANOCHAT_SOURCE", "sft"),
        model_tag=os.getenv("NANOCHAT_MODEL_TAG", None),
        step=int(os.getenv("NANOCHAT_STEP")) if os.getenv("NANOCHAT_STEP") else None,
        port=int(os.getenv("NANOCHAT_PORT", "8000")),
        host=os.getenv("NANOCHAT_HOST", "0.0.0.0"),
        device_type=os.getenv("NANOCHAT_DEVICE_TYPE", ""),
        dtype=os.getenv("NANOCHAT_DTYPE", "bfloat16"),
        mock=os.getenv("NANOCHAT_MOCK", "0") == "1"
    )

# -----------------------------------------------------------------------------
# OpenAI API Models
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "nanochat"
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = Field(default=50, ge=1, le=200) # Added top_k as it's useful
    user: Optional[str] = None

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "nanochat"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

# -----------------------------------------------------------------------------
# Worker & Pool (Multi-GPU support)
# -----------------------------------------------------------------------------
class Worker:
    def __init__(self, gpu_id: int, device: torch.device, engine: Engine, tokenizer: Any, autocast_ctx: Any):
        self.gpu_id = gpu_id
        self.device = device
        self.engine = engine
        self.tokenizer = tokenizer
        self.autocast_ctx = autocast_ctx

class WorkerPool:
    def __init__(self, num_gpus: int, device_type: str):
        self.num_gpus = num_gpus
        if num_gpus > 1 and device_type != "cuda":
             logger.warning("Multiple GPUs requested but device is not CUDA. Forcing num_gpus=1.")
             self.num_gpus = 1

        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()
        self.ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        if args.mock:
            logger.info("Initializing worker pool in MOCK mode...")
            from unittest.mock import MagicMock
            # Create a dummy worker
            mock_engine = MagicMock()

            # Helper to simulate generation
            def mock_generate(*args, **kwargs):
                # Yield "Hello from Mock!" tokens
                # We need to yield (token_column, token_masks)
                # Let's assume some dummy tokens
                dummy_tokens = [100, 101, 102] # dummy
                assistant_end = 99999
                bos = 50256

                for t in dummy_tokens:
                    yield ([t], [1])
                    asyncio.sleep(0.01) # simulate compute

                # End
                yield ([assistant_end], [1])

            mock_engine.generate = mock_generate

            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.encode_special.return_value = 99999
            mock_tokenizer.get_bos_token_id.return_value = 50256

            # Make decode return incremental strings for the dummy tokens
            # We used [100, 101, 102]
            decode_map = {
                100: "Hello",
                101: " from",
                102: " Mock!",
                99999: ""
            }
            # Simpler decode: just join what we have
            def mock_decode(tokens):
                return "".join([decode_map.get(t, "") for t in tokens])

            mock_tokenizer.decode.side_effect = mock_decode

            worker = Worker(0, torch.device("cpu"), mock_engine, mock_tokenizer, nullcontext())
            self.workers.append(worker)
            await self.available_workers.put(worker)
            logger.info("Mock worker initialized.")
            return

        logger.info(f"Initializing worker pool with {self.num_gpus} workers...")

        # Determine global device type
        dev_type = autodetect_device_type() if args.device_type == "" else args.device_type

        for gpu_id in range(self.num_gpus):
            if dev_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                logger.info(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(dev_type)
                logger.info(f"Loading model on {dev_type}...")

            # Load model
            # Note: load_model handles DDP logic internally if needed, but here we manually place models
            # We assume single-process, multi-device (not DDP) for the inference server for now,
            # or simple data parallelism via this pool.
            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            autocast_ctx = torch.amp.autocast(device_type=dev_type, dtype=self.ptdtype) if dev_type == "cuda" else nullcontext()

            worker = Worker(gpu_id, device, engine, tokenizer, autocast_ctx)
            self.workers.append(worker)
            await self.available_workers.put(worker)

        logger.info("All workers initialized.")

    async def acquire(self) -> Worker:
        return await self.available_workers.get()

    async def release(self, worker: Worker):
        await self.available_workers.put(worker)

# -----------------------------------------------------------------------------
# Application Setup
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus, device_type=args.device_type)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    yield
    # Shutdown (if needed)

app = FastAPI(title="NanoChat API", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def format_prompt(tokenizer, messages: List[ChatMessage]) -> List[int]:
    """Encodes the conversation into tokens using special tokens."""
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    tokens = [bos]
    for msg in messages:
        if msg.role == "user":
            tokens.append(user_start)
            tokens.extend(tokenizer.encode(msg.content))
            tokens.append(user_end)
        elif msg.role == "assistant":
            tokens.append(assistant_start)
            tokens.extend(tokenizer.encode(msg.content))
            tokens.append(assistant_end)
        elif msg.role == "system":
            # For now treat system as user or ignore? NanoChat usually treats system as initial instruction.
            # Let's just append it as text if needed, or mapped to user logic.
            # Simple fallback:
            tokens.append(user_start)
            tokens.extend(tokenizer.encode(msg.content))
            tokens.append(user_end)

    tokens.append(assistant_start)
    return tokens

def create_chunk(id: str, content: str, model: str, finish_reason: Optional[str] = None):
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason
            }
        ]
    }

async def stream_generator(worker: Worker, tokens: List[int], request: ChatCompletionRequest, req_id: str):
    max_tokens = request.max_tokens if request.max_tokens else 1024
    temp = request.temperature
    top_k = request.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    accumulated_tokens = []
    last_clean_text = ""

    try:
        with worker.autocast_ctx:
            # We assume batch size 1 for now
            for token_column, _ in worker.engine.generate(
                tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temp,
                top_k=top_k,
                seed=random.randint(0, 2**31 - 1)
            ):
                token = token_column[0]

                if token == assistant_end or token == bos:
                    yield f"data: {json.dumps(create_chunk(req_id, '', request.model, 'stop'))}\n\n"
                    break

                accumulated_tokens.append(token)
                current_text = worker.tokenizer.decode(accumulated_tokens)

                # Check for replacement character indicating incomplete UTF-8
                if not current_text.endswith('\ufffd'):
                    new_text = current_text[len(last_clean_text):]
                    if new_text:
                        yield f"data: {json.dumps(create_chunk(req_id, new_text, request.model))}\n\n"
                        last_clean_text = current_text

            # If loop finishes without stop token (max tokens reached)
            else:
                 yield f"data: {json.dumps(create_chunk(req_id, '', request.model, 'length'))}\n\n"

        yield "data: [DONE]\n\n"

    finally:
        # We must release the worker!
        await app.state.worker_pool.release(worker)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelCard(id="nanochat")])

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire()

    req_id = f"chatcmpl-{uuid.uuid4()}"

    try:
        prompt_tokens = format_prompt(worker.tokenizer, request.messages)

        if request.stream:
            return StreamingResponse(
                stream_generator(worker, prompt_tokens, request, req_id),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming
            max_tokens = request.max_tokens if request.max_tokens else 1024
            generated_tokens = []
            finish_reason = "length"

            assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
            bos = worker.tokenizer.get_bos_token_id()

            with worker.autocast_ctx:
                for token_column, _ in worker.engine.generate(
                    prompt_tokens,
                    num_samples=1,
                    max_tokens=max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    seed=random.randint(0, 2**31 - 1)
                ):
                    token = token_column[0]
                    if token == assistant_end or token == bos:
                        finish_reason = "stop"
                        break
                    generated_tokens.append(token)

            full_text = worker.tokenizer.decode(generated_tokens)

            # Release worker immediately
            await worker_pool.release(worker)

            return {
                "id": req_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_text
                        },
                        "finish_reason": finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": len(prompt_tokens) + len(generated_tokens)
                }
            }

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        await worker_pool.release(worker)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    pool = app.state.worker_pool
    return {
        "status": "ok",
        "workers_total": len(pool.workers),
        "workers_available": pool.available_workers.qsize()
    }

# -----------------------------------------------------------------------------
# Static Files / UI
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    ui_path = os.path.join("nanochat", "ui.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Inject API URL if needed, but relative paths work best
            # content = content.replace("API_URL = ''", f"API_URL = 'http://{args.host}:{args.port}'")
        return HTMLResponse(content)
    return HTMLResponse("NanoChat API Server Running. UI not found.", status_code=200)

@app.get("/logo.svg")
async def logo():
    path = os.path.join("nanochat", "logo.svg")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/svg+xml")
    return JSONResponse({"error": "not found"}, status_code=404)

if __name__ == "__main__":
    print(f"Starting NanoChat API Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
