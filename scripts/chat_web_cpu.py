#!/usr/bin/env python3
"""
CPU-compatible web chat server - serves both UI and API from a single FastAPI instance.
Run with: python chat_web_cpu.py --model-dir /path/to/model
Then open http://localhost:8000 in your browser.
"""

import argparse
import json
import os
import glob
import pickle
import math
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Minimal GPT implementation (copied from generate_cpu.py)

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        return logits


# -----------------------------------------------------------------------------
# Simple tokenizer wrapper

class SimpleTokenizer:
    def __init__(self, enc):
        self.enc = enc
        try:
            self.bos_token_id = enc.encode_single_token("<|bos|>")
        except:
            try:
                self.bos_token_id = enc.encode_single_token("<|endoftext|>")
            except:
                self.bos_token_id = 0
        
        # Get special tokens
        try:
            self.user_start = enc.encode_single_token("<|user_start|>")
            self.user_end = enc.encode_single_token("<|user_end|>")
            self.assistant_start = enc.encode_single_token("<|assistant_start|>")
            self.assistant_end = enc.encode_single_token("<|assistant_end|>")
        except:
            # Fallback if special tokens don't exist
            self.user_start = 0
            self.user_end = 0
            self.assistant_start = 0
            self.assistant_end = 0
    
    def get_bos_token_id(self):
        return self.bos_token_id
    
    def encode_special(self, token):
        try:
            return self.enc.encode_single_token(token)
        except:
            return 0
    
    def encode(self, text):
        return self.enc.encode_ordinary(text)
    
    def decode(self, tokens):
        return self.enc.decode(tokens)


# -----------------------------------------------------------------------------
# Simple generator (no Engine class needed)

def generate_tokens(model, input_tokens, max_tokens=512, temperature=0.8, top_k=50, device='cpu'):
    """Generate tokens one at a time."""
    x = torch.tensor([input_tokens], dtype=torch.long, device=device)
    generated = []
    
    with torch.inference_mode():
        for _ in range(max_tokens):
            logits = model(x)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token.item())
            x = torch.cat([x, next_token], dim=1)
            
            yield next_token.item()


# -----------------------------------------------------------------------------
# FastAPI app

parser = argparse.ArgumentParser(description='NanoChat Web Server (CPU)')
parser.add_argument('--model-dir', type=str, required=True, help='Path to model directory containing model_*.pt, meta_*.json, and tokenizer.pkl')
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
args = parser.parse_args()

device = torch.device("cpu")

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
    print(f"Loading model from {args.model_dir}...")
    
    # Find model and meta files
    model_files = glob.glob(os.path.join(args.model_dir, "model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {args.model_dir}")
    model_file = model_files[0]
    
    meta_files = glob.glob(os.path.join(args.model_dir, "meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No meta files found in {args.model_dir}")
    meta_file = meta_files[0]
    
    # Load metadata
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    
    model_config_kwargs = meta["model_config"]
    print(f"Model config: {model_config_kwargs}")
    
    # Build the model
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    
    # Load model weights
    print("Loading model weights...")
    model_data = torch.load(model_file, map_location=device, weights_only=False)
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}
    
    # Convert bfloat16 to float32 for CPU
    print("Converting model to float32 for CPU...")
    model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.pkl")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    with open(tokenizer_path, "rb") as f:
        import tiktoken
        enc = pickle.load(f)
    
    tokenizer = SimpleTokenizer(enc)
    
    app.state.model = model
    app.state.tokenizer = tokenizer
    
    print(f"✓ Model loaded successfully!")
    print(f"✓ Server ready at http://localhost:{args.port}")
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
    model,
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

    for token in generate_tokens(model, tokens, max_new_tokens, temperature, top_k, device):
        if token == assistant_end or token == bos:
            break

        token_text = tokenizer.decode([token])
        yield f"data: {json.dumps({'token': token_text})}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint with streaming."""
    model = app.state.model
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
                model,
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

        generated_tokens = list(generate_tokens(model, conversation_tokens, max_tokens, temperature, top_k, device))
        response_text = tokenizer.decode(generated_tokens)
        
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
    print(f"Starting NanoChat Web Server (CPU mode)")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)

