#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.
Run with: python web_chat.py
Then open http://localhost:8000 in your browser.
"""

import argparse
import json
import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator

from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

parser = argparse.ArgumentParser(description='NanoChat Web Server')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
args = parser.parse_args()

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    print("Loading nanochat model...")
    app.state.model, app.state.tokenizer, _ = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    app.state.engine = Engine(app.state.model, app.state.tokenizer)
    print(f"Server ready at http://localhost:{args.port}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r") as f:
        html_content = f.read()
    # Replace the API_URL to use the same origin
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """Serve the NanoChat logo for favicon and header."""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    engine,
    tokenizer,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """Generate assistant response with streaming."""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()

    with autocast_ctx:
        for token_column, token_masks in engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        ):
            token = token_column[0]

            if token == assistant_end or token == bos:
                break

            token_text = tokenizer.decode([token])
            yield f"data: {json.dumps({'token': token_text})}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint with streaming."""
    engine = app.state.engine
    tokenizer = app.state.tokenizer

    # Build conversation tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    conversation_tokens = [bos]
    for message in request.messages:
        if message.role == "user":
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(message.content))
            conversation_tokens.append(user_end)
        elif message.role == "assistant":
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(tokenizer.encode(message.content))
            conversation_tokens.append(assistant_end)

    conversation_tokens.append(assistant_start)

    if request.stream:
        return StreamingResponse(
            generate_stream(
                engine,
                tokenizer,
                conversation_tokens,
                temperature=request.temperature,
                max_new_tokens=request.max_tokens,
                top_k=request.top_k
            ),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        temperature = request.temperature if request.temperature is not None else args.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else args.max_tokens
        top_k = request.top_k if request.top_k is not None else args.top_k

        with autocast_ctx:
            result_tokens, masks = engine.generate_batch(
                conversation_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )[0]

        response_tokens = result_tokens[len(conversation_tokens):]
        response_text = tokenizer.decode(response_tokens)
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }]
        }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "ready": hasattr(app.state, 'model') and app.state.model is not None
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Web Server")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
