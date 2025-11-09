#!/usr/bin/env python3
"""
Model Size & Cost Calculator

A simple tool to calculate model parameters, memory requirements, and training costs.
Perfect for understanding model scaling and planning training runs.

Usage:
    python tools/model_calculator.py --preset gpt2-small
    python tools/model_calculator.py --depth 12 --hidden-dim 768 --vocab-size 50257
"""

import argparse
from typing import Dict


class ModelCalculator:
    """Calculate model size, memory, and training costs."""

    # Common model presets for quick calculations
    PRESETS = {
        'gpt2-small': {'depth': 12, 'hidden_dim': 768, 'vocab_size': 50257, 'heads': 12},
        'gpt2-medium': {'depth': 24, 'hidden_dim': 1024, 'vocab_size': 50257, 'heads': 16},
        'gpt2-large': {'depth': 36, 'hidden_dim': 1280, 'vocab_size': 50257, 'heads': 20},
        'gpt2-xl': {'depth': 48, 'hidden_dim': 1600, 'vocab_size': 50257, 'heads': 25},
        'nanochat-tiny': {'depth': 6, 'hidden_dim': 384, 'vocab_size': 32000, 'heads': 6},
        'nanochat-small': {'depth': 12, 'hidden_dim': 768, 'vocab_size': 32000, 'heads': 12},
    }

    def __init__(self, depth: int, hidden_dim: int, vocab_size: int, heads: int = None):
        """
        Initialize the calculator with model configuration.

        Args:
            depth: Number of transformer layers
            hidden_dim: Hidden dimension size (d_model)
            vocab_size: Vocabulary size
            heads: Number of attention heads (optional, for display only)
        """
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.heads = heads or (hidden_dim // 64)  # Common default

    def calculate_parameters(self) -> Dict[str, int]:
        """
        Calculate the number of parameters in each component.

        Returns:
            Dictionary with parameter counts for each component
        """
        d = self.hidden_dim
        vocab = self.vocab_size

        # Token embeddings: vocab_size × hidden_dim
        token_embeddings = vocab * d

        # Per-layer parameters
        # Attention: Q, K, V projections + output projection
        # For MQA (Multi-Query Attention), K and V are smaller
        # But for simplicity, we'll calculate standard multi-head attention
        attention_qkv = 3 * d * d  # Q, K, V projections
        attention_out = d * d       # Output projection
        attention_total = attention_qkv + attention_out

        # MLP: Two linear layers with 4x expansion
        mlp_up = d * (4 * d)        # First layer: d -> 4d
        mlp_down = (4 * d) * d      # Second layer: 4d -> d
        mlp_total = mlp_up + mlp_down

        # Layer norms: 2 per layer (pre-attention and pre-MLP)
        # Each layer norm has 2 * d parameters (scale and shift)
        layernorm_total = 2 * d * 2

        # Total per layer
        params_per_layer = attention_total + mlp_total + layernorm_total

        # All layers
        all_layers = params_per_layer * self.depth

        # Final layer norm
        final_ln = 2 * d

        # Output head (language modeling head)
        # Often tied with token embeddings, but we'll count separately
        lm_head = vocab * d

        # Total parameters
        total = token_embeddings + all_layers + final_ln + lm_head

        return {
            'token_embeddings': token_embeddings,
            'attention_per_layer': attention_total,
            'mlp_per_layer': mlp_total,
            'layernorm_per_layer': layernorm_total,
            'params_per_layer': params_per_layer,
            'all_layers': all_layers,
            'final_layernorm': final_ln,
            'lm_head': lm_head,
            'total': total,
            'total_millions': total / 1_000_000,
            'total_billions': total / 1_000_000_000,
        }

    def calculate_memory(self, params: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate memory requirements for training and inference.

        Args:
            params: Parameter counts from calculate_parameters()

        Returns:
            Dictionary with memory estimates in different units
        """
        total_params = params['total']

        # Model weights in different precisions
        # fp32: 4 bytes per parameter
        # fp16/bf16: 2 bytes per parameter
        model_fp32_bytes = total_params * 4
        model_fp16_bytes = total_params * 2

        # Training memory (rough estimates):
        # - Model weights (fp32): 4 bytes
        # - Gradients (fp32): 4 bytes
        # - Optimizer states (AdamW has 2 states): 8 bytes
        # Total: 16 bytes per parameter
        training_bytes = total_params * 16

        # Inference memory (fp16):
        # Just the model weights in fp16
        inference_bytes = model_fp16_bytes

        # Convert to readable units
        def to_mb(bytes_val):
            return bytes_val / (1024 ** 2)

        def to_gb(bytes_val):
            return bytes_val / (1024 ** 3)

        return {
            'model_fp32_mb': to_mb(model_fp32_bytes),
            'model_fp32_gb': to_gb(model_fp32_bytes),
            'model_fp16_mb': to_mb(model_fp16_bytes),
            'model_fp16_gb': to_gb(model_fp16_bytes),
            'training_mb': to_mb(training_bytes),
            'training_gb': to_gb(training_bytes),
            'inference_mb': to_mb(inference_bytes),
            'inference_gb': to_gb(inference_bytes),
        }

    def estimate_training_cost(self,
                              params: Dict[str, int],
                              batch_size: int = 64,
                              sequence_length: int = 1024,
                              total_tokens: int = 10_000_000_000,
                              tokens_per_sec: int = 100_000) -> Dict:
        """
        Estimate training time and computational cost.

        Args:
            params: Parameter counts from calculate_parameters()
            batch_size: Training batch size
            sequence_length: Sequence length (context window)
            total_tokens: Total tokens to train on
            tokens_per_sec: Estimated throughput (tokens/second)

        Returns:
            Dictionary with training estimates
        """
        total_params = params['total']

        # FLOPs calculation (rough approximation)
        # Forward pass: ~6 * params * tokens
        # Backward pass: ~2 * forward pass
        # Total: ~6 * params * tokens * 3 = 18 * params * tokens
        flops_per_token = 6 * total_params
        total_flops = flops_per_token * total_tokens

        # Training time
        total_seconds = total_tokens / tokens_per_sec
        hours = total_seconds / 3600
        days = hours / 24

        # Number of steps
        tokens_per_batch = batch_size * sequence_length
        total_steps = total_tokens / tokens_per_batch

        return {
            'total_tokens': total_tokens,
            'total_tokens_billions': total_tokens / 1_000_000_000,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'tokens_per_batch': tokens_per_batch,
            'total_steps': int(total_steps),
            'tokens_per_sec': tokens_per_sec,
            'training_seconds': int(total_seconds),
            'training_hours': hours,
            'training_days': days,
            'flops_per_token': flops_per_token,
            'total_flops': total_flops,
            'total_petaflops': total_flops / 1e15,
        }

    def print_report(self, batch_size: int = 64, sequence_length: int = 1024,
                    total_tokens: int = 10_000_000_000, tokens_per_sec: int = 100_000):
        """Print a comprehensive report of model size and costs."""

        print("=" * 70)
        print("MODEL SIZE & COST CALCULATOR")
        print("=" * 70)

        print("\n📊 MODEL CONFIGURATION")
        print("-" * 70)
        print(f"  Layers (depth):        {self.depth}")
        print(f"  Hidden dimension:      {self.hidden_dim}")
        print(f"  Vocabulary size:       {self.vocab_size:,}")
        print(f"  Attention heads:       {self.heads}")

        # Calculate parameters
        params = self.calculate_parameters()

        print("\n🔢 PARAMETER BREAKDOWN")
        print("-" * 70)
        print(f"  Token embeddings:      {params['token_embeddings']:>15,} params")
        print(f"  Per-layer breakdown:")
        print(f"    - Attention:         {params['attention_per_layer']:>15,} params")
        print(f"    - MLP:               {params['mlp_per_layer']:>15,} params")
        print(f"    - LayerNorm:         {params['layernorm_per_layer']:>15,} params")
        print(f"    - Total per layer:   {params['params_per_layer']:>15,} params")
        print(f"  All {self.depth} layers:        {params['all_layers']:>15,} params")
        print(f"  Final LayerNorm:       {params['final_layernorm']:>15,} params")
        print(f"  LM head:               {params['lm_head']:>15,} params")
        print(f"  {'─' * 60}")
        print(f"  TOTAL PARAMETERS:      {params['total']:>15,} params")
        print(f"                         {params['total_millions']:>15.2f} M")
        if params['total_billions'] >= 0.1:
            print(f"                         {params['total_billions']:>15.2f} B")

        # Calculate memory
        memory = self.calculate_memory(params)

        print("\n💾 MEMORY REQUIREMENTS")
        print("-" * 70)
        print(f"  Model weights (fp32):  {memory['model_fp32_gb']:>10.2f} GB")
        print(f"  Model weights (fp16):  {memory['model_fp16_gb']:>10.2f} GB")
        print(f"  Training (fp32+opt):   {memory['training_gb']:>10.2f} GB")
        print(f"  Inference (fp16):      {memory['inference_gb']:>10.2f} GB")

        # Calculate training cost
        cost = self.estimate_training_cost(
            params, batch_size, sequence_length, total_tokens, tokens_per_sec
        )

        print("\n⏱️  TRAINING ESTIMATES")
        print("-" * 70)
        print(f"  Training tokens:       {cost['total_tokens_billions']:.1f}B tokens")
        print(f"  Batch size:            {cost['batch_size']} sequences")
        print(f"  Sequence length:       {cost['sequence_length']} tokens")
        print(f"  Tokens per batch:      {cost['tokens_per_batch']:,}")
        print(f"  Total steps:           {cost['total_steps']:,}")
        print(f"  Throughput:            {cost['tokens_per_sec']:,} tokens/sec")
        print(f"  Training time:         {cost['training_hours']:.1f} hours ({cost['training_days']:.1f} days)")
        print(f"  Total FLOPs:           {cost['total_petaflops']:.1f} PetaFLOPs")

        print("\n💡 LEARNING INSIGHTS")
        print("-" * 70)
        # Calculate percentage breakdowns
        attn_pct = (params['attention_per_layer'] * self.depth / params['total']) * 100
        mlp_pct = (params['mlp_per_layer'] * self.depth / params['total']) * 100
        embed_pct = (params['token_embeddings'] / params['total']) * 100

        print(f"  • Embeddings use ~{embed_pct:.1f}% of parameters")
        print(f"  • Attention layers use ~{attn_pct:.1f}% of parameters")
        print(f"  • MLP layers use ~{mlp_pct:.1f}% of parameters")
        print(f"  • Training needs ~4x more memory than inference")
        print(f"  • Each parameter sees {total_tokens / params['total']:.0f} tokens during training")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate model size, memory, and training costs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a preset configuration
  python tools/model_calculator.py --preset gpt2-small
  python tools/model_calculator.py --preset nanochat-tiny

  # Custom configuration
  python tools/model_calculator.py --depth 12 --hidden-dim 768 --vocab-size 50257

  # Custom training parameters
  python tools/model_calculator.py --preset gpt2-small --batch-size 32 --total-tokens 20000000000

Available presets: gpt2-small, gpt2-medium, gpt2-large, gpt2-xl, nanochat-tiny, nanochat-small
        """
    )

    parser.add_argument('--preset', type=str, choices=list(ModelCalculator.PRESETS.keys()),
                       help='Use a preset model configuration')
    parser.add_argument('--depth', type=int, help='Number of transformer layers')
    parser.add_argument('--hidden-dim', type=int, help='Hidden dimension size')
    parser.add_argument('--vocab-size', type=int, help='Vocabulary size')
    parser.add_argument('--heads', type=int, help='Number of attention heads (optional)')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--sequence-length', type=int, default=1024,
                       help='Sequence length (default: 1024)')
    parser.add_argument('--total-tokens', type=int, default=10_000_000_000,
                       help='Total tokens to train on (default: 10B)')
    parser.add_argument('--tokens-per-sec', type=int, default=100_000,
                       help='Estimated throughput in tokens/sec (default: 100k)')

    args = parser.parse_args()

    # Get configuration from preset or custom args
    if args.preset:
        config = ModelCalculator.PRESETS[args.preset].copy()
    elif args.depth and args.hidden_dim and args.vocab_size:
        config = {
            'depth': args.depth,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
            'heads': args.heads,
        }
    else:
        parser.error("Either --preset or all of (--depth, --hidden-dim, --vocab-size) required")

    # Create calculator and print report
    calc = ModelCalculator(
        depth=config['depth'],
        hidden_dim=config['hidden_dim'],
        vocab_size=config['vocab_size'],
        heads=config.get('heads'),
    )

    calc.print_report(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        total_tokens=args.total_tokens,
        tokens_per_sec=args.tokens_per_sec,
    )


if __name__ == '__main__':
    main()
