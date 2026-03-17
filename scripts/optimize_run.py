"""
Unified script to run an optimized training/inference based on the PC's capacity.
"""

import argparse
import torch
from nanochat.common import recommend_config, estimate_model_vram, print0, autodetect_device_type
from nanochat.gpt import GPT, GPTConfig
from nanochat.quantization import convert_to_int8
from nanochat.pruning import prune_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vram", type=float, default=6.0, help="Available VRAM in GB")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization")
    parser.add_argument("--prune", type=float, default=0.0, help="Pruning amount (0.0 to 1.0)")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    print0(f"Device: {device_type}")
    
    cfg_params = recommend_config(args.vram, training=(args.mode=="train"), device_type=device_type)
    print0(f"Recommended config for {args.vram}GB VRAM: {cfg_params}")
    
    config = GPTConfig(n_layer=cfg_params["depth"], n_embd=cfg_params["n_embd"], n_head=8, n_kv_head=8)
    model = GPT(config)
    
    # Initialize weights
    model.init_weights()
    
    # Move to device
    device = torch.device("cuda" if device_type == "cuda" else "cpu")
    model.to(device)
    
    if args.prune > 0:
        print0(f"Applying pruning (amount={args.prune})...")
        prune_model(model, amount=args.prune)
        
    if args.quantize:
        print0("Applying INT8 weight-only quantization...")
        convert_to_int8(model)
        
    print0("Model is ready.")
    
    # Final VRAM estimate
    # Note: after quantization, parameters take 1 byte instead of 2/4.
    # Our estimator doesn't handle quantization yet, but we can print a manual check.
    param_count = sum(p.numel() for p in model.parameters())
    print0(f"Final parameter count: {param_count/1e6:.2f}M")
    
    # For a real run, you would now start your training or inference loop.
    print0("Optimization check complete.")

if __name__ == "__main__":
    main()
