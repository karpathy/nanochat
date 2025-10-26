"""
Serve the d32 model from HuggingFace using Modal.

This script is specifically designed to work with the uploaded d32 model
which has its own tokenizer separate from your trained d20 model.

Usage:
    modal serve modal_d32_serve.py
"""

import os
from pathlib import Path
import modal

APP_NAME = "nanochat-d32-serve"
VOLUME_NAME = "nanochat-data"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Get the local directory path
LOCAL_DIR = Path(__file__).parent

# Build image with nanochat code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "unzip")
    .add_local_dir("dev", "/nanochat/dev", copy=True)
    .add_local_dir("nanochat", "/nanochat/nanochat", copy=True)
    .add_local_dir("rustbpe", "/nanochat/rustbpe", copy=True)
    .add_local_dir("scripts", "/nanochat/scripts", copy=True)
    .add_local_dir("tasks", "/nanochat/tasks", copy=True)
    .add_local_dir("tests", "/nanochat/tests", copy=True)
    .add_local_file("pyproject.toml", "/nanochat/pyproject.toml", copy=True)
    .add_local_file(".python-version", "/nanochat/.python-version", copy=True)
    .add_local_file("README.md", "/nanochat/README.md", copy=True)
    .add_local_file("LICENSE", "/nanochat/LICENSE", copy=True)
    .workdir("/nanochat")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    )
    .env({"PATH": "/root/.cargo/bin:/root/.local/bin:$PATH"})
    .uv_sync(extras=["gpu"])
    .run_commands(
        "uv run maturin develop --release --manifest-path rustbpe/Cargo.toml",
    )
)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100:1",  # Single H100 for serving
    timeout=60 * 60,
    container_idle_timeout=300,
)
def chat_d32(prompt: str, temperature: float = 0.6, top_k: int = 50) -> str:
    """
    Chat with the d32 model.

    Args:
        prompt: User prompt/question
        temperature: Sampling temperature (default: 0.6)
        top_k: Top-k sampling parameter (default: 50)

    Returns:
        Model's response as a string
    """
    import sys
    import torch
    from contextlib import nullcontext

    # Add nanochat to path
    sys.path.insert(0, '/nanochat')

    # Import after adding to path
    from nanochat.common import get_base_dir, autodetect_device_type, compute_init
    from nanochat.checkpoint_manager import build_model
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.engine import Engine

    # Setup environment to point to d32's tokenizer
    DATA = Path("/data")
    BASE_DIR = DATA / ".cache" / "nanochat"

    # CRITICAL: Override the base dir so it uses our volume
    os.environ["NANOCHAT_BASE_DIR"] = str(BASE_DIR)

    # Setup device
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # Load the d32 model
    checkpoint_dir = BASE_DIR / "chatsft_checkpoints" / "d32"
    step = 650  # The uploaded checkpoint is at step 650

    print(f"Loading d32 model from {checkpoint_dir} at step {step}")
    model, _, meta = build_model(str(checkpoint_dir), step, device, phase="eval")

    # Load the d32-specific tokenizer
    tokenizer_dir = BASE_DIR / "tokenizer_d32"
    print(f"Loading d32 tokenizer from {tokenizer_dir}")
    tokenizer = RustBPETokenizer.from_directory(str(tokenizer_dir))

    # Verify vocab size matches
    assert tokenizer.get_vocab_size() == model.config.vocab_size, \
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} != model vocab size {model.config.vocab_size}"

    # Create engine
    engine = Engine(model, tokenizer)

    # Special tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    # Build conversation tokens
    conversation_tokens = [bos]
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(prompt))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)

    # Generate response
    conversation_tokens = torch.tensor(conversation_tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad(), autocast_ctx:
        generated, _ = engine.generate(
            conversation_tokens,
            max_new_tokens=2048,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=[assistant_end],
        )

    # Decode response (skip the prompt)
    response_tokens = generated[0, conversation_tokens.size(1):].tolist()
    response = tokenizer.decode(response_tokens)

    return response


@app.local_entrypoint()
def main(prompt: str = "What is the capital of France?"):
    """
    Test the d32 model with a prompt.

    Args:
        prompt: The prompt to send to the model

    Examples:
        modal run modal_d32_serve.py
        modal run modal_d32_serve.py --prompt "Explain quantum computing"
    """
    print(f"\n{'='*80}")
    print(f"🤖 NanoChat d32 Model")
    print(f"{'='*80}\n")
    print(f"Prompt: {prompt}\n")

    response = chat_d32.remote(prompt)

    print(f"Response: {response}")
    print(f"\n{'='*80}\n")
