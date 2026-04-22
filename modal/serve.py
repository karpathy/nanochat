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
MODEL_REPO = "ManmohanSharma/nanochat-d24"
MODEL_PT   = "chatsft_checkpoints/d24-sft-r6/model_000754.pt"
META_JSON  = "chatsft_checkpoints/d24-sft-r6/meta_000754.json"
TOKENIZER_PKL = "tokenizer/tokenizer.pkl"
TOKEN_BYTES   = "tokenizer/token_bytes.pt"
MODEL_TAG  = "d24-sft-r6"
GPU_TYPE   = "L4"                           # 24 GB VRAM — fits 4 GB bf16 model loaded as fp32
VOLUME_NAME = "samosachaat-weights"
HF_SECRET_NAME = "huggingface"              # Modal secret containing HF_TOKEN
TAVILY_SECRET_NAME = "tavily"                # Modal secret containing TAVILY_API_KEY

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
        "requests>=2.31.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_file("modal/_model.py", "/root/_model.py")
    .add_local_file("modal/_tokenizer.py", "/root/_tokenizer.py")
    .add_local_file("modal/_tools.py", "/root/_tools.py")
    .add_local_file("modal/_query_classifier.py", "/root/_query_classifier.py")
)

# Persistent volume for model weights
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Download weights into the volume (runs once)
# ---------------------------------------------------------------------------
@app.function(
    image=inference_image,
    volumes={"/weights": volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=1800,
)
def download_weights():
    """Download model weights from HuggingFace into the Modal volume."""
    import shutil
    from huggingface_hub import hf_hub_download

    model_dir = f"/weights/{MODEL_TAG}"
    os.makedirs(model_dir, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    # (HF source path, local filename in volume)
    files = [
        (MODEL_PT,       "model.pt"),
        (META_JSON,      "meta.json"),
        (TOKENIZER_PKL,  "tokenizer.pkl"),
        (TOKEN_BYTES,    "token_bytes.pt"),
    ]

    for src, local_name in files:
        dest = os.path.join(model_dir, local_name)
        if os.path.exists(dest):
            print(f"  Already exists: {dest}")
            continue
        print(f"  Downloading {src} from {MODEL_REPO}...")
        path = hf_hub_download(MODEL_REPO, src, token=token)
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
    secrets=[modal.Secret.from_name(TAVILY_SECRET_NAME)],
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
        meta_path = os.path.join(model_dir, "meta.json")
        model_path = os.path.join(model_dir, "model.pt")

        # Load meta
        with open(meta_path) as f:
            meta = json.load(f)
        model_config = meta if "model_config" not in meta else meta["model_config"]

        # Normalize config key names (HF format → nanochat format)
        # Map HF config keys → nanochat GPTConfig keys
        seq_len = model_config.pop("n_positions", None) or model_config.pop("n_ctx", None)
        if seq_len and "sequence_len" not in model_config:
            model_config["sequence_len"] = seq_len
        # Also remove n_ctx if sequence_len was already set
        model_config.pop("n_ctx", None)
        model_config.pop("n_positions", None)
        # Remove HF-specific keys that GPTConfig doesn't accept
        for k in ["architectures", "model_type", "rotary", "rotary_base", "tie_word_embeddings"]:
            model_config.pop(k, None)

        # Patch missing config keys
        model_config.setdefault("window_pattern", "L")

        print(f"  Config: {model_config}")

        # Build model
        # We need the GPT class — download it from the repo itself
        # For simplicity, we define a minimal inline version that matches nanochat
        from _model import GPT, GPTConfig

        config = GPTConfig(**model_config)
        model_data = torch.load(model_path, map_location=device, weights_only=False)

        # Strip torch.compile prefix
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

        # Convert bfloat16 weights to float32 for compatibility on non-Hopper GPUs
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }

        model = GPT.from_state_dict(config, model_data)
        model.load_state_dict(model_data, strict=True, assign=True)
        model.to(device)
        model.init_rotary(device=device, dtype=torch.float32)
        model.eval()

        self.model = model
        self.config = config

        # Load tokenizer
        from _tokenizer import get_tokenizer, SPECIAL_TOKENS
        self.tokenizer = get_tokenizer(model_dir)

        # Resolve actual special-token IDs (nanochat appends specials at end of vocab)
        self.special_token_ids = set()
        for name in SPECIAL_TOKENS:
            ids = self.tokenizer.encode_special(name)
            self.special_token_ids.update(ids)
        self.assistant_end_id = self.tokenizer.encode_special("<|assistant_end|>")[0]
        print(f"  Special token IDs: {sorted(self.special_token_ids)}")

        # Initialize tool registry (Tavily web_search + calculator)
        import sys as _sys
        if '/root' not in _sys.path: _sys.path.insert(0, '/root')
        from _tools import build_default_tool_registry, parse_tool_call_payload
        from _query_classifier import needs_web_search
        self.tool_registry = build_default_tool_registry()
        self._parse_tool_call = parse_tool_call_payload
        self._needs_web_search = needs_web_search
        # Marker tokens for tool state machine
        self.python_start_id = self.tokenizer.encode_special("<|python_start|>")[0]
        self.python_end_id = self.tokenizer.encode_special("<|python_end|>")[0]
        self.output_start_id = self.tokenizer.encode_special("<|output_start|>")[0]
        self.output_end_id = self.tokenizer.encode_special("<|output_end|>")[0]
        # Stop tokens (exclude tool markers so generation continues through tool calls)
        self._stop_token_ids = {self.assistant_end_id, self.tokenizer.get_bos_token_id() if hasattr(self.tokenizer, "get_bos_token_id") else self.tokenizer.encode_special("<|bos|>")[0]}

        dt = time.time() - t0
        print(f"Model loaded in {dt:.1f}s on {device} | tools: {[t for t in self.tool_registry._tools.keys()] if hasattr(self.tool_registry, '_tools') else 'registered'}")

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
        force_web_search = bool(request.get("force_web_search", False))

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

        # --- Forced tool use ---
        # The model's SFT training doesn't always trigger web_search even when
        # a question clearly needs current info (e.g. "present president" vs
        # "current president"). We classify the last user message and, if it
        # matches tool-worthy patterns, pre-seed the assistant turn with a real
        # tool call + Tavily result. The model then just writes the final
        # grounded answer instead of hallucinating from stale memory.
        forced_prefix_text = ""
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        # chat-api inlines the system prompt into the first user message as
        # "{SYS_PROMPT}\n\n{actual_user_question}". Strip the sys prefix for
        # the classifier so Tavily gets a clean query.
        query_for_classify = last_user
        if "\n\n" in query_for_classify:
            query_for_classify = query_for_classify.rsplit("\n\n", 1)[-1].strip()
        try:
            needs_search, rewritten = self._needs_web_search(query_for_classify)
        except Exception:
            needs_search, rewritten = False, ""
        # Explicit user toggle wins — always force when force_web_search is True
        if force_web_search and query_for_classify:
            needs_search = True
            if not rewritten:
                # if classifier didn't rewrite, do a minimal cleanup
                rewritten = query_for_classify.strip().rstrip("?.!") + " 2026"
        if needs_search and rewritten:
            preface = "I'll look that up for you. "
            tool_call_json = json.dumps(
                {"arguments": {"query": rewritten, "top_k": 1}, "tool": "web_search"},
                separators=(",", ":"),
            )
            try:
                invocation = self._parse_tool_call(tool_call_json)
                tool_result = self.tool_registry.execute(invocation.tool_name, invocation.arguments)
                result_text = tool_result.to_payload()[:4096]
            except Exception as exc:
                result_text = json.dumps({"error": str(exc)[:500]})
            forced_prefix_text = (
                preface
                + "<|python_start|>" + tool_call_json + "<|python_end|>"
                + "<|output_start|>" + result_text + "<|output_end|>\n"
            )
            tokens.extend(self.tokenizer.encode(forced_prefix_text))

        # Truncate to fit context
        max_context = self.config.sequence_len - max_tokens
        if len(tokens) > max_context:
            tokens = tokens[-max_context:]

        # The SFT loader tokenizes assistant content with .encode() (not .encode_special()),
        # so these markers are emitted as multi-token byte sequences, and BPE has
        # multiple valid tokenizations of the same string — so matching on a single
        # expected id sequence is unreliable. Instead we decode the tail of the
        # generated token stream and search for the marker TEXT.
        tool_start_str = "<|python_start|>"
        tool_end_str = "<|python_end|>"
        out_start_str = "<|output_start|>"
        out_end_str = "<|output_end|>"

        async def stream():
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            gen_ids: list[int] = []           # everything the MODEL sampled this turn
            tool_injected = bool(forced_prefix_text)   # forced prefix counts as an injection
            pre_injection_len = 0             # len(gen_ids) right before we start injection
            post_injection_start = 0          # index in gen_ids AFTER injection finished

            # If we pre-seeded a forced tool call + result, stream it to the client
            # now so the UI can render the tool-call / tool-result cards.
            if forced_prefix_text:
                yield "data: " + json.dumps({"token": forced_prefix_text, "gpu": 0}) + "\n\n"

            def _append_token(tid):
                nonlocal input_ids
                nt = torch.tensor([[tid]], dtype=torch.long, device=self.device)
                input_ids = torch.cat([input_ids, nt], dim=1)
                if input_ids.size(1) > self.config.sequence_len:
                    input_ids = input_ids[:, -self.config.sequence_len:]

            def _decode_tail_text(last_n: int = 40) -> str:
                if not gen_ids:
                    return ""
                try:
                    return self.tokenizer.decode(gen_ids[-last_n:])
                except Exception:
                    return ""

            with torch.no_grad():
                num_generated = 0
                while num_generated < max_tokens:
                    logits = self.model(input_ids)
                    next_logits = logits[:, -1, :]
                    if temperature > 0:
                        next_logits = next_logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                        next_logits[next_logits < v[:, [-1]]] = float('-inf')
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    token_id = next_token.item()

                    if token_id in self._stop_token_ids:
                        break

                    # Commit to context + sequence
                    _append_token(token_id)
                    gen_ids.append(token_id)
                    num_generated += 1

                    # Stream raw decoded text (may be empty for partial BPE bytes)
                    try:
                        token_text = self.tokenizer.decode([token_id])
                    except Exception:
                        token_text = ""
                    if token_text:
                        yield "data: " + json.dumps({"token": token_text, "gpu": 0}) + "\n\n"

                    # --- tool-call detection on decoded-tail text ---
                    if not tool_injected:
                        # Decode the whole turn-so-far and look for markers
                        try:
                            full_text = self.tokenizer.decode(gen_ids)
                        except Exception:
                            full_text = ""
                        if full_text:
                            ps = full_text.rfind(tool_start_str)
                            if ps >= 0:
                                pe = full_text.find(tool_end_str, ps + len(tool_start_str))
                                if pe >= 0:
                                    payload_text = full_text[ps + len(tool_start_str):pe]
                                    try:
                                        invocation = self._parse_tool_call(payload_text)
                                        result = self.tool_registry.execute(invocation.tool_name, invocation.arguments)
                                        result_text = result.to_payload()[:4096]
                                    except Exception as exc:
                                        result_text = json.dumps({"error": str(exc)[:500]})

                                    pre_injection_len = len(gen_ids)
                                    wrapped = out_start_str + result_text + out_end_str
                                    for rid in self.tokenizer.encode(wrapped):
                                        try:
                                            rt = self.tokenizer.decode([rid])
                                        except Exception:
                                            rt = ""
                                        if rt:
                                            yield "data: " + json.dumps({"token": rt, "gpu": 0}) + "\n\n"
                                        _append_token(rid)
                                        gen_ids.append(rid)
                                        num_generated += 1
                                        if num_generated >= max_tokens:
                                            break
                                    tool_injected = True
                                    post_injection_start = len(gen_ids)  # ← scan only what the model generates AFTER our injection

                    # After injection (forced OR runtime): the model often loops and
                    # emits another fake <|output_start|>…<|output_end|> / <|python_start|>…
                    # block. Scan only the model's POST-injection tokens — not our own.
                    elif tool_injected and len(gen_ids) > post_injection_start + 6:
                        try:
                            post_text = self.tokenizer.decode(gen_ids[post_injection_start:])
                        except Exception:
                            post_text = ""
                        for bad in (out_start_str, out_end_str, tool_start_str, tool_end_str):
                            if bad in post_text:
                                break_now = True
                                break
                        else:
                            break_now = False
                        if break_now:
                            break

            yield "data: " + json.dumps({"done": True}) + "\n\n"

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
