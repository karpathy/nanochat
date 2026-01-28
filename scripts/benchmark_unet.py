"""
Benchmark UNet model inference performance.

This script provides comprehensive benchmarking for UNet models:
- Compare naive vs cached generation (correctness + speedup)
- Measure prefill latency and decode throughput
- Profile memory usage
- Sweep across different sequence lengths and batch sizes
- Optional PyTorch profiler integration

Examples:

# Basic comparison (naive vs engine)
python -m scripts.benchmark_unet --compare

# Full benchmark suite
python -m scripts.benchmark_unet --all

# Profile with torch.profiler
python -m scripts.benchmark_unet --profile --profile-output trace.json

# Sequence length sweep
python -m scripts.benchmark_unet --sweep-lengths --lengths 64 128 256 512

# Batch size sweep
python -m scripts.benchmark_unet --sweep-batch --batch-sizes 1 2 4 8

# Run on specific model checkpoint
python -m scripts.benchmark_unet --model-tag d[6,12]-h[768,1536] --step 10000
"""

import argparse
import json
import os
import sys
import torch
from contextlib import nullcontext

from nanochat.common import compute_init, autodetect_device_type, print0, get_base_dir
from nanochat.checkpoint_manager import load_model
from inference.unet_engine import UNetEngine
from inference.benchmark import InferenceBenchmark
from inference.profiler import TorchProfiler, SimpleTimer, MemoryProfiler, profile_model_inference


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark UNet model inference')
    
    # Model selection
    parser.add_argument('--model-tag', type=str, default=None, 
                       help='Model tag for checkpoint directory (e.g., d[6,12]-h[768,1536])')
    parser.add_argument('--step', type=int, default=None,
                       help='Checkpoint step to load (default: latest)')
    parser.add_argument('--device-type', type=str, default='',
                       help='Device type: cuda|cpu|mps (empty = autodetect)')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'],
                       help='Data type for inference')
    
    # Benchmark modes
    parser.add_argument('--compare', action='store_true',
                       help='Compare naive vs engine generation')
    parser.add_argument('--sweep-lengths', action='store_true',
                       help='Sweep across different prompt lengths')
    parser.add_argument('--sweep-batch', action='store_true',
                       help='Sweep across different batch sizes')
    parser.add_argument('--profile', action='store_true',
                       help='Run detailed profiling with torch.profiler')
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks')
    
    # Benchmark parameters
    parser.add_argument('--prompt', type=str, default='The capital of France is',
                       help='Text prompt to use for benchmarking')
    parser.add_argument('--max-tokens', type=int, default=64,
                       help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (0 = greedy)')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Top-k sampling parameter')
    
    # Sweep parameters
    parser.add_argument('--lengths', type=int, nargs='+', default=[32, 64, 128, 256, 512],
                       help='Prompt lengths to test in sweep')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='Batch sizes to test in sweep')
    
    # Profiler parameters
    parser.add_argument('--profile-output', type=str, default=None,
                       help='Path to save profiler chrome trace (e.g., trace.json)')
    parser.add_argument('--profile-warmup', type=int, default=3,
                       help='Warmup iterations for profiling')
    parser.add_argument('--profile-iters', type=int, default=10,
                       help='Profiling iterations')
    parser.add_argument('--profile-engine', action='store_true',
                       help='Profile engine (with KV cache) instead of naive generation')
    
    # Output
    parser.add_argument('--output-json', type=str, default=None,
                       help='Save results to JSON file')
    
    return parser.parse_args()


def check_unet_model(model):
    """Verify that the loaded model is a UNet architecture."""
    if not hasattr(model, 'encoder') or not hasattr(model, 'decoder'):
        print0("=" * 80)
        print0("ERROR: Loaded model is not a UNet architecture!")
        print0("=" * 80)
        print0("This benchmark script is designed for UNet models.")
        print0("Please ensure you have a trained UNet checkpoint.")
        print0("\nTo train a UNet model, run:")
        print0("  python -m scripts.unet.base_train")
        print0("\nOr specify a valid UNet checkpoint with --model-tag and --step")
        print0("=" * 80)
        return False
    return True


def main():
    args = parse_args()
    
    # Enable all benchmarks if --all is specified
    if args.all:
        args.compare = True
        args.sweep_lengths = True
        args.sweep_batch = True
        args.profile = True
    
    # If no benchmark mode specified, default to comparison
    if not (args.compare or args.sweep_lengths or args.sweep_batch or args.profile):
        args.compare = True
    
    print0("=" * 80)
    print0("UNet Inference Benchmark")
    print0("=" * 80)
    
    # Setup device
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    print0(f"Device: {device_type}")
    
    # Setup autocast
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    # Load model
    print0("\nLoading UNet model...")
    try:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        print0(f"Loaded model: step {meta.get('step', 'unknown')}")
        
        if 'model_config' in meta:
            config = meta['model_config']
            print0(f"Config: n_layer={config.get('n_layer')}, n_embd={config.get('n_embd')}")
    
    except Exception as e:
        print0(f"ERROR: Could not load model: {e}")
        print0("\nMake sure you have a trained UNet checkpoint.")
        print0("Train one with: python -m scripts.unet.base_train")
        sys.exit(1)
    
    # Verify it's a UNet model
    if not check_unet_model(model):
        sys.exit(1)
    
    # Create engine
    print0("\nInitializing UNet engine...")
    engine = UNetEngine(model, tokenizer)
    
    # Prepare prompt
    bos = tokenizer.get_bos_token_id()
    prompt_tokens = tokenizer.encode(args.prompt, prepend=bos)
    print0(f"Prompt: '{args.prompt}'")
    print0(f"Prompt tokens: {len(prompt_tokens)}")
    
    # Initialize benchmark utilities
    benchmark = InferenceBenchmark(model, tokenizer, device, autocast_ctx)
    
    # Store results for JSON output
    results = {
        'model_tag': args.model_tag,
        'step': meta.get('step'),
        'device': device_type,
        'dtype': args.dtype,
        'prompt': args.prompt,
        'prompt_length': len(prompt_tokens),
        'max_tokens': args.max_tokens,
    }
    
    # -------------------------------------------------------------------------
    # Benchmark 1: Compare naive vs engine
    # -------------------------------------------------------------------------
    if args.compare:
        print0("\n" + "=" * 80)
        print0("Benchmark 1: Naive vs Engine Comparison")
        print0("=" * 80)
        
        comparison = benchmark.compare_methods(
            prompt_tokens=prompt_tokens,
            max_tokens=args.max_tokens,
            engine=engine,
            temperature=args.temperature,
            top_k=args.top_k,
            model_tag=args.model_tag,
        )
        
        print0(str(comparison))
        
        results['comparison'] = {
            'naive': comparison.naive_result.to_dict(),
            'engine': comparison.engine_result.to_dict(),
            'outputs_match': comparison.outputs_match,
            'speedup': comparison.speedup,
            'memory_overhead_mb': comparison.memory_overhead_mb,
        }
    
    # -------------------------------------------------------------------------
    # Benchmark 2: Sequence length sweep
    # -------------------------------------------------------------------------
    if args.sweep_lengths:
        print0("\n" + "=" * 80)
        print0("Benchmark 2: Sequence Length Sweep")
        print0("=" * 80)
        print0(f"Testing lengths: {args.lengths}")
        
        length_results = benchmark.sweep_sequence_lengths(
            engine=engine,
            base_prompt=args.prompt,
            lengths=args.lengths,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        print0("\nResults:")
        print0(f"{'Length':>8} {'Prefill (ms)':>15} {'Decode (tok/s)':>15} {'Memory (MB)':>12}")
        print0("-" * 60)
        for result in length_results:
            print0(f"{result.prompt_length:>8} {result.prefill_time*1000:>15.2f} "
                  f"{result.decode_tokens_per_sec:>15.1f} {result.peak_memory_mb:>12.1f}")
        
        results['length_sweep'] = [r.to_dict() for r in length_results]
    
    # -------------------------------------------------------------------------
    # Benchmark 3: Batch size sweep
    # -------------------------------------------------------------------------
    if args.sweep_batch:
        print0("\n" + "=" * 80)
        print0("Benchmark 3: Batch Size Sweep")
        print0("=" * 80)
        print0(f"Testing batch sizes: {args.batch_sizes}")
        
        batch_results = benchmark.sweep_batch_sizes(
            engine=engine,
            prompt_tokens=prompt_tokens,
            batch_sizes=args.batch_sizes,
            max_tokens=args.max_tokens,
            temperature=1.0,  # Use temperature > 0 for batch diversity
        )
        
        print0("\nResults:")
        print0(f"{'Batch':>8} {'Total Time (s)':>15} {'Throughput (tok/s)':>20} {'Memory (MB)':>12}")
        print0("-" * 65)
        for result in batch_results:
            print0(f"{result.batch_size:>8} {result.total_time:>15.2f} "
                  f"{result.decode_tokens_per_sec:>20.1f} {result.peak_memory_mb:>12.1f}")
        
        results['batch_sweep'] = [r.to_dict() for r in batch_results]
    
    # -------------------------------------------------------------------------
    # Benchmark 4: Detailed profiling
    # -------------------------------------------------------------------------
    if args.profile:
        print0("\n" + "=" * 80)
        print0("Benchmark 4: Detailed Profiling")
        print0("=" * 80)
        
        if device_type != "cuda":
            print0("WARNING: Detailed profiling is only available on CUDA devices")
        else:
            method = "Engine (with KV cache)" if args.profile_engine else "Naive (no cache)"
            print0(f"Method: {method}")
            print0(f"Warmup iterations: {args.profile_warmup}")
            print0(f"Profile iterations: {args.profile_iters}")
            
            # If no explicit output path, auto-generate will be used
            auto_save = args.profile_output is None
            
            profile_result = profile_model_inference(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_tokens=args.max_tokens,
                batch_size=1,
                temperature=args.temperature,
                warmup_iters=args.profile_warmup,
                profile_iters=args.profile_iters,
                output_path=args.profile_output,
                auto_save=auto_save,
                model_tag=args.model_tag,
                device_type=device_type,
                use_engine=args.profile_engine,
                engine=engine if args.profile_engine else None,
            )
            
            print0(str(profile_result))
            
            results['profile'] = {
                'method': method,
                'total_time_ms': profile_result.total_time_ms,
                'gpu_time_ms': profile_result.gpu_time_ms,
                'cpu_time_ms': profile_result.cpu_time_ms,
                'memory_mb': profile_result.memory_mb,
                'trace_path': profile_result.trace_path,
            }
            
            if profile_result.trace_path:
                print0(f"\n✓ Chrome trace saved to: {profile_result.trace_path}")
                print0("  Open in chrome://tracing to visualize")
    
    # -------------------------------------------------------------------------
    # Save results to JSON
    # -------------------------------------------------------------------------
    if args.output_json:
        output_path = args.output_json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print0(f"\nResults saved to: {output_path}")
    
    print0("\n" + "=" * 80)
    print0("Benchmark Complete!")
    print0("=" * 80)


if __name__ == "__main__":
    main()
