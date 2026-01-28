"""
Create a dummy UNet checkpoint with random weights for testing.

This is useful for testing inference engine without waiting for training to complete.

Usage:
python -m scripts.create_dummy_checkpoint --depth 6 12 --model-dim 768 1536
python -m scripts.create_dummy_checkpoint --depth 4 8 --model-dim 512 1024 --pool-factor 4
"""

import os
import argparse
import torch

from nanochat.unet import UNet, UNetConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.common import get_base_dir, print0


def parse_args():
    parser = argparse.ArgumentParser(description='Create dummy UNet checkpoint')
    
    # Model architecture
    parser.add_argument('--depth', type=int, nargs='+', default=[6, 12],
                       help='Number of layers per stage (e.g., "6 12" for 2 stages)')
    parser.add_argument('--model-dim', type=int, nargs='+', default=[768, 1536],
                       help='Hidden dimensions per stage (e.g., "768 1536")')
    parser.add_argument('--head-dim', type=int, default=128,
                       help='Head dimension for attention')
    parser.add_argument('--max-seq-len', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--pool-factor', type=int, default=2,
                       help='Pooling factor between stages (2 = halve sequence, 4 = quarter sequence)')
    
    # Output
    parser.add_argument('--model-tag', type=str, default=None,
                       help='Model tag for checkpoint directory (auto-generated if not provided)')
    parser.add_argument('--step', type=int, default=0,
                       help='Step number for checkpoint filename')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print0("=" * 80)
    print0("Creating Dummy UNet Checkpoint")
    print0("=" * 80)
    
    # Get tokenizer for vocab size
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    
    # Validate inputs
    if len(args.depth) != len(args.model_dim):
        print0(f"ERROR: depth and model-dim must have same length")
        print0(f"  depth: {args.depth} (length {len(args.depth)})")
        print0(f"  model-dim: {args.model_dim} (length {len(args.model_dim)})")
        return 1
    
    n_stages = len(args.depth)
    print0(f"Number of stages: {n_stages}")
    print0(f"Layers per stage: {args.depth}")
    print0(f"Model dimensions: {args.model_dim}")
    print0(f"Pooling factor: {args.pool_factor}")
    
    # Calculate number of heads (ensure divisibility)
    num_heads = tuple(d // args.head_dim for d in args.model_dim)
    num_kv_heads = num_heads  # 1:1 GQA ratio
    
    print0(f"Number of heads: {num_heads}")
    print0(f"Head dimension: {args.head_dim}")
    
    # Create model config
    if args.pool_factor != 2:
        print0(f"\nWARNING: Custom pooling factor {args.pool_factor} is not yet implemented in UNet.")
        print0(f"The current UNet implementation always uses pooling factor of 2.")
        print0(f"Proceeding with default pooling factor 2...\n")
    
    model_config_kwargs = dict(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=tuple(args.depth),
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=tuple(args.model_dim),
    )
    
    print0("\nCreating model with random weights...")
    
    # Create model on CPU (faster for initialization)
    device = torch.device("cpu")
    with torch.device("meta"):
        model_config = UNetConfig(**model_config_kwargs)
        model = UNet(model_config)
    
    model.to_empty(device=device)
    model.init_weights()
    
    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Model created: {num_params:,} parameters")
    
    # Generate model tag if not provided
    if args.model_tag is None:
        # Use simpler naming without brackets to avoid glob issues
        depth_str = "_".join(map(str, args.depth))
        dim_str = "_".join(map(str, args.model_dim))
        args.model_tag = f"d{depth_str}-h{dim_str}"
    
    print0(f"Model tag: {args.model_tag}")
    
    # Create checkpoint directory
    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save checkpoint
    print0(f"\nSaving checkpoint to: {checkpoint_dir}")
    
    metadata = {
        "step": args.step,
        "val_bpb": None,  # No validation data for dummy checkpoint
        "model_config": model_config_kwargs,
        "user_config": {
            "depth": args.depth,
            "model_dim": args.model_dim,
            "head_dim": args.head_dim,
            "max_seq_len": args.max_seq_len,
            "note": "Dummy checkpoint created with random weights for testing",
        },
        "device_batch_size": 32,  # dummy value
        "max_seq_len": args.max_seq_len,
        "dataloader_state_dict": None,
        "loop_state": {
            "min_val_bpb": float("inf"),
            "smooth_train_loss": 0.0,
            "total_training_time": 0.0,
        },
    }
    
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=args.step,
        model_data=model.state_dict(),
        optimizer_data=None,  # No optimizer state for dummy checkpoint
        meta_data=metadata,
        rank=0,
    )
    
    model_path = os.path.join(checkpoint_dir, f"model_{args.step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{args.step:06d}.json")
    print0(f"✓ Model saved: {model_path}")
    print0(f"✓ Metadata saved: {meta_path}")
    
    print0("\n" + "=" * 80)
    print0("Checkpoint Creation Complete!")
    print0("=" * 80)
    print0("\nYou can now use this checkpoint for testing:")
    print0(f"  python -m inference.unet_engine")
    print0(f"  python -m scripts.benchmark_unet --model-tag {args.model_tag} --step {args.step}")
    print0("\nNote: This is a randomly initialized model and will produce gibberish.")
    print0("      Use it for testing inference speed, not model quality!")
    
    return 0


if __name__ == "__main__":
    exit(main())
