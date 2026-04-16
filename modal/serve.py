"""
samosaChaat — Modal GPU inference endpoint.

Downloads nanochat model weights from HuggingFace into a Modal Volume,
loads them on a GPU, and exposes an SSE streaming endpoint compatible
with the samosaChaat chat-api service.

Deploy:  modal deploy modal/serve.py
Dev:     modal serve modal/serve.py
"""
from __future__ import annotations

import json
import os
import time

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_REPO = "nanochat-students/base-d20"   # 1 GB, native nanochat format
MODEL_PT   = "model_021400.pt"
META_JSON  = "meta_021400.json"
MODEL_TAG  = "d20"
GPU_TYPE   = "T4"                           # cheapest, 16 GB VRAM — plenty for 1 GB model
VOLUME_NAME = "samosachaat-weights"

# ---------------------------------------------------------------------------
# Modal app + image
# ---------------------------------------------------------------------------
app = modal.App("samosachaat-inference")

# Build the container image with all dependencies
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "tiktoken>=0.11.0",
        "tokenizers>=0.22.0",
        "huggingface_hub>=0.25.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_file("modal/_model.py", "/root/_model.py")
    .add_local_file("modal/_tokenizer.py", "/root/_tokenizer.py")
)

# Persistent volume for model weights
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Download weights into the volume (runs once)
# ---------------------------------------------------------------------------
@app.function(
    image=inference_image,
    volumes={"/weights": volume},
    timeout=600,
)
def download_weights():
    """Download model weights from HuggingFace into the Modal volume."""
    from huggingface_hub import hf_hub_download

    model_dir = f"/weights/{MODEL_TAG}"
    os.makedirs(model_dir, exist_ok=True)

    for filename in [MODEL_PT, META_JSON, "token_bytes.pt", "tokenizer.pkl"]:
        dest = os.path.join(model_dir, filename)
        if os.path.exists(dest):
            print(f"  Already exists: {dest}")
            continue
        print(f"  Downloading {filename} from {MODEL_REPO}...")
        path = hf_hub_download(MODEL_REPO, filename)
        # Copy to volume
        import shutil
        shutil.copy2(path, dest)
        print(f"  Saved to {dest}")

    volume.commit()
    print("Weights downloaded and committed to volume.")


# ---------------------------------------------------------------------------
# Inference class — GPU singleton
# ---------------------------------------------------------------------------
@app.cls(
    image=inference_image,
    volumes={"/weights": volume},
    gpu=GPU_TYPE,
    scaledown_window=300,            # keep warm for 5 min after last request
    # concurrency handled by @modal.concurrent below
    timeout=120,
)
class Inference:
    model: object
    tokenizer: object
    engine: object
    device: object

    @modal.enter()
    def load_model(self):
        """Called once when the container starts — loads model onto GPU."""
        import torch
        import sys

        # Add the nanochat engine code path
        # We inline the minimal loading logic here to avoid importing the full
        # nanochat package (which has heavy deps we don't need on Modal).
        print("Loading model...")
        t0 = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        model_dir = f"/weights/{MODEL_TAG}"
        meta_path = os.path.join(model_dir, META_JSON)
        model_path = os.path.join(model_dir, MODEL_PT)

        # Load meta
        with open(meta_path) as f:
            meta = json.load(f)
        model_config = meta if "model_config" not in meta else meta["model_config"]

        # Patch missing config keys
        model_config.setdefault("window_pattern", "L")

        print(f"  Config: {model_config}")

        # Build model
        # We need the GPT class — download it from the repo itself
        # For simplicity, we define a minimal inline version that matches nanochat
        from _model import GPT, GPTConfig

        config = GPTConfig(**model_config)
        model_data = torch.load(model_path, map_location=device, weights_only=False)

        # Fix torch compile prefix
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

        # Patch missing keys
        n_layer = config.n_layer
        if "resid_lambdas" not in model_data:
            model_data["resid_lambdas"] = torch.ones(n_layer)
        if "x0_lambdas" not in model_data:
            model_data["x0_lambdas"] = torch.zeros(n_layer)

        # Auto-detect architecture from checkpoint
        # Convert bfloat16 weights to float32 for compatibility
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }

        # Auto-detect architecture from checkpoint
        model = GPT.from_state_dict(config, model_data)
        model.to(device)
        model.init_weights()
        model.load_state_dict(model_data, strict=True, assign=True)
        model.eval()

        self.model = model
        self.config = config

        # Load tokenizer
        from _tokenizer import get_tokenizer
        self.tokenizer = get_tokenizer(model_dir)

        dt = time.time() - t0
        print(f"Model loaded in {dt:.1f}s on {device}")

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def generate(self, request: dict):
        """
        Streaming chat endpoint — SSE compatible with samosaChaat format.

        Input: {"messages": [{"role": "user", "content": "..."}], "temperature": 0.8, "max_tokens": 512, "top_k": 50}
        Output: SSE stream of data: {"token": "...", "gpu": 0} then data: {"done": true}
        """
        import torch
        from fastapi.responses import StreamingResponse

        messages = request.get("messages", [])
        temperature = min(max(request.get("temperature", 0.8), 0.0), 2.0)
        max_tokens = min(max(request.get("max_tokens", 512), 1), 2048)
        top_k = min(max(request.get("top_k", 50), 0), 200)

        # Build token sequence from messages
        tokens = []
        bos = self.tokenizer.encode_special("<|bos|>")
        user_start = self.tokenizer.encode_special("<|user_start|>")
        user_end = self.tokenizer.encode_special("<|user_end|>")
        assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        tokens.extend(bos)
        for msg in messages:
            if msg["role"] == "user":
                tokens.extend(user_start)
                tokens.extend(self.tokenizer.encode(msg["content"]))
                tokens.extend(user_end)
            elif msg["role"] == "assistant":
                tokens.extend(assistant_start)
                tokens.extend(self.tokenizer.encode(msg["content"]))
                tokens.extend(assistant_end)
        # Prompt the model to generate an assistant response
        tokens.extend(assistant_start)

        # Truncate to fit context
        max_context = self.config.sequence_len - max_tokens
        if len(tokens) > max_context:
            tokens = tokens[-max_context:]

        async def stream():
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

            with torch.no_grad():
                generated = []
                for _ in range(max_tokens):
                    # Forward pass
                    logits = self.model(input_ids)
                    next_logits = logits[:, -1, :]

                    # Temperature
                    if temperature > 0:
                        next_logits = next_logits / temperature

                    # Top-k filtering
                    if top_k > 0:
                        v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                        next_logits[next_logits < v[:, [-1]]] = float('-inf')

                    # Sample
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    token_id = next_token.item()

                    # Check for stop tokens
                    if token_id in [t[0] for t in [assistant_end, bos]]:
                        break

                    # Decode and yield
                    token_text = self.tokenizer.decode([token_id])
                    yield f"data: {json.dumps({'token': token_text, 'gpu': 0})}\n\n"

                    # Append for next iteration
                    input_ids = torch.cat([input_ids, next_token], dim=1)

                    # Truncate if exceeding sequence length
                    if input_ids.size(1) > self.config.sequence_len:
                        input_ids = input_ids[:, -self.config.sequence_len:]

            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @modal.fastapi_endpoint(method="GET", docs=True)
    def health(self):
        return {
            "status": "ok",
            "model": MODEL_TAG,
            "gpu": GPU_TYPE,
            "ready": self.model is not None,
        }
