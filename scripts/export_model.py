"""
Script to export nanochat checkpoints to other formats.
Supports:
- HuggingFace Safetensors
- GGUF (for llama.cpp)

Usage:
python scripts/export_model.py --checkpoint_dir=path/to/checkpoint --format=safetensors
python scripts/export_model.py --checkpoint_dir=path/to/checkpoint --format=gguf
"""

import os
import argparse
import json
import torch
import numpy as np
from safetensors.torch import save_file
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.common import get_base_dir, autodetect_device_type

def export_to_safetensors(model_state, output_path):
    print(f"Exporting to safetensors: {output_path}")
    # Safetensors expects a flat dict of tensors
    # Check for any non-tensor data (which shouldn't happen in model_state usually)
    tensors = {}
    for k, v in model_state.items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v.contiguous()
        else:
            print(f"Warning: Skipping non-tensor key {k}")
    save_file(tensors, output_path)
    print("Done.")

def export_to_gguf(model_state, model_config, output_path):
    try:
        import gguf
    except ImportError:
        print("Error: gguf package not found. Please install it.")
        return

    print(f"Exporting to GGUF: {output_path}")

    # Initialize GGUF writer
    gguf_writer = gguf.GGUFWriter(output_path, "nanochat")

    # Architecture
    # For now, we map to GPT-2 architecture as it's the closest standard one supported by llama.cpp
    # or we define a custom one. Let's try to stick to gpt2 keys if possible, or generic.
    # Actually, nanochat is GPT-2 like but with Rotary Embeddings (RoPE) and other mods.
    # Standard GPT-2 doesn't have RoPE.
    # Llama architecture has RoPE.
    # Let's try to map to "llama" architecture if possible, or "gpt2" with extensions.
    # However, modded-nanogpt has specific layers (RMSNorm, GLU variants etc).
    #
    # Current best bet for custom models in llama.cpp is to use the generic architecture definition
    # if supported, or map to a known one.
    #
    # Given the complexity of "modded-nanogpt" (RMSNorm, RoPE, etc.), it closely resembles Llama
    # but with different naming and potentially different block structure.
    #
    # Let's write the raw tensors and minimal metadata. Users might need a custom llama.cpp build
    # or specific conversion script logic in llama.cpp to read this if it doesn't match standard archs.
    #
    # But wait, the task says "efficient inference on Strix Halo NPU (via llama.cpp)".
    # This implies there is a path.
    #
    # Let's set some basic metadata.

    n_layer = model_config.get("n_layer", 12)
    n_head = model_config.get("n_head", 12)
    n_embd = model_config.get("n_embd", 768)
    # n_ctx = model_config.get("sequence_len", 1024)
    # vocab_size = model_config.get("vocab_size", 50257)

    # We will try to map to 'gpt2' for now, but note the differences.
    # Actually, if we want it to work with standard llama.cpp, we might need to be more careful.
    # For now, I will dump the weights with the original names.
    # Users can use `llama-convert-py` or similar if they need standard remapping,
    # but GGUF allows custom keys.

    gguf_writer.add_architecture() # Custom architecture name is implied by the constructor or just generic?
    # GGUFWriter's add_architecture method doesn't take an argument in recent versions.
    # The architecture is usually inferred or set via specific keys.
    # However, to be compliant with llama.cpp, we typically need to set 'general.architecture'

    # Manually set the architecture key
    # Note: Standard llama.cpp builds will likely not recognize "nanochat" architecture
    # without modifications to the C++ code to support the specific tensor naming and block structure.
    # This export is primarily for experimental use or with custom llama.cpp forks.
    gguf_writer.add_string("general.architecture", "nanochat")

    gguf_writer.add_uint32("nanochat.context_length", model_config.get("sequence_len", 1024))
    gguf_writer.add_uint32("nanochat.embedding_length", n_embd)
    gguf_writer.add_uint32("nanochat.block_count", n_layer)
    gguf_writer.add_uint32("nanochat.feed_forward_length", 4 * n_embd) # Approximation
    gguf_writer.add_uint32("nanochat.attention.head_count", n_head)

    # Add tensor data
    for k, v in model_state.items():
        if isinstance(v, torch.Tensor):
            # GGUF expects numpy arrays (fp32 or fp16)
            # nanochat uses bfloat16 often. numpy doesn't fully support bfloat16.
            # We should convert to float32 or float16.
            data = v.detach().cpu().float().numpy()
            gguf_writer.add_tensor(k, data)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Export Nanochat Model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory (e.g., base_checkpoints/d12)")
    parser.add_argument("--step", type=int, default=-1, help="Step to load (-1 for latest)")
    parser.add_argument("--format", type=str, choices=["safetensors", "gguf"], required=True, help="Export format")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load model on")

    args = parser.parse_args()

    # Determine paths
    if not os.path.exists(args.checkpoint_dir):
        # try relative to base dir
        base_dir = get_base_dir()
        candidate = os.path.join(base_dir, args.checkpoint_dir)
        if os.path.exists(candidate):
            args.checkpoint_dir = candidate
        else:
             # try relative to repo root if user passed e.g. "base_checkpoints/d12"
             # but get_base_dir returns ~/.cache/nanochat
             pass

    print(f"Loading checkpoint from {args.checkpoint_dir}...")

    # Load checkpoint
    # We only need model_state and config
    try:
        model_state, _, meta_data = load_checkpoint(args.checkpoint_dir, args.step, args.device, load_optimizer=False)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    model_config = meta_data.get("model_config", {})

    # Determine output path
    if args.output is None:
        filename = f"model_{meta_data['step']}.{args.format}"
        args.output = os.path.join(args.checkpoint_dir, filename)

    if args.format == "safetensors":
        export_to_safetensors(model_state, args.output)
    elif args.format == "gguf":
        export_to_gguf(model_state, model_config, args.output)

if __name__ == "__main__":
    main()
