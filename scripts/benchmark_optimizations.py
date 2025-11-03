"""
Benchmark script for measuring inference performance (tokens/second) of model generation.
Enables before/after comparison of KV-cache optimizations.

Example usage:
    python -m scripts.benchmark_optimizations --output v1_baseline --model-source sft --model-tag d20 --step 650
    python -m scripts.benchmark_optimizations --output v2_kvcache_fixed --model-source sft
"""
import argparse
import time
import torch
import numpy as np
from nanochat.common import compute_init
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Benchmark model inference performance')
parser.add_argument('--output', type=str, required=True, help='Version label (e.g., "v1_baseline", "v2_kvcache_fixed")')
parser.add_argument('--model-source', type=str, required=True, choices=['sft', 'mid', 'rl', 'base'], help='Model type: sft, mid, rl, or base')
parser.add_argument('--model-tag', type=str, default=None, help='Model variant (e.g., "d20") - optional')
parser.add_argument('--step', type=int, default=None, help='Checkpoint step number - optional')
parser.add_argument('--num-iterations', type=int, default=5, help='Number of generation iterations for statistical stability (default: 5)')
parser.add_argument('--max-tokens', type=int, default=150, help='Number of tokens to generate per iteration (default: 150)')
parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for generation (default: 0.6)')
parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling parameter (default: 50)')
args = parser.parse_args()

def main():
    print("=" * 80)
    print(f"BENCHMARK: {args.output}")
    print("=" * 80)
    
    try:
        # Initialize device
        print("\n[1/6] Initializing device...")
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
        print(f"  ✓ Device: {device}")
        print(f"  ✓ DDP: {ddp} (rank {ddp_rank}/{ddp_world_size})")
        
        # Setup autocast context
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        
        # Load model
        print(f"\n[2/6] Loading model...")
        print(f"  - Source: {args.model_source}")
        print(f"  - Model Tag: {args.model_tag if args.model_tag else 'auto-detect'}")
        print(f"  - Step: {args.step if args.step else 'latest'}")
        
        model, tokenizer, meta = load_model(
            args.model_source, 
            device, 
            phase="eval", 
            model_tag=args.model_tag, 
            step=args.step
        )
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Config: {meta.get('model_config', {})}")
        
        # Create Engine for efficient generation
        engine = Engine(model, tokenizer)
        
        # Define test prompt
        print(f"\n[3/6] Preparing test prompt...")
        test_prompt = (
            "Write a detailed explanation of how neural networks learn through backpropagation. "
            "Include the key concepts of forward pass, loss calculation, and gradient descent."
        )
        
        # Tokenize the prompt
        bos = tokenizer.get_bos_token_id()
        prompt_tokens = [bos] + tokenizer.encode(test_prompt)
        print(f"  ✓ Test prompt: \"{test_prompt[:80]}...\"")
        print(f"  ✓ Prompt length: {len(prompt_tokens)} tokens")
        
        # Warmup run (not counted in statistics)
        print(f"\n[4/6] Running warmup iteration...")
        torch.cuda.reset_peak_memory_stats(device)
        with autocast_ctx:
            warmup_tokens = []
            for token_column, token_masks in engine.generate(
                prompt_tokens, 
                num_samples=1,
                max_tokens=50,  # Short warmup
                temperature=args.temperature,
                top_k=args.top_k
            ):
                warmup_tokens.append(token_column[0])
        print(f"  ✓ Warmup complete ({len(warmup_tokens)} tokens generated)")
        
        # Performance measurement
        print(f"\n[5/6] Running benchmark ({args.num_iterations} iterations, {args.max_tokens} tokens each)...")
        iteration_times = []
        iteration_tokens_per_sec = []
        sample_output = None
        
        for i in range(args.num_iterations):
            # Reset memory stats for this iteration
            torch.cuda.reset_peak_memory_stats(device)
            
            # Start timing
            start_time = time.perf_counter()
            
            generated_tokens = []
            with autocast_ctx:
                for token_column, token_masks in engine.generate(
                    prompt_tokens, 
                    num_samples=1,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=42 + i  # Different seed per iteration
                ):
                    token = token_column[0]  # Extract from batch dimension
                    generated_tokens.append(token)
            
            # End timing
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            # Calculate tokens per second
            num_tokens = len(generated_tokens)
            tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
            
            iteration_times.append(elapsed_time)
            iteration_tokens_per_sec.append(tokens_per_sec)
            
            print(f"  Iteration {i+1}/{args.num_iterations}: {num_tokens} tokens in {elapsed_time:.3f}s = {tokens_per_sec:.2f} tok/s")
            
            # Save first iteration output for coherence check
            if i == 0:
                sample_output = tokenizer.decode(generated_tokens)
        
        # Measure peak GPU memory (after all iterations)
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_gb = peak_memory_bytes / (1024 ** 3)
        
        # Calculate statistics
        mean_time = np.mean(iteration_times)
        std_time = np.std(iteration_times)
        mean_tokens_per_sec = np.mean(iteration_tokens_per_sec)
        std_tokens_per_sec = np.std(iteration_tokens_per_sec)
        
        # Print results
        print(f"\n[6/6] Results Summary")
        print("=" * 80)
        print(f"Version:                  {args.output}")
        print(f"Model Source:             {args.model_source}")
        print(f"Model Tag:                {meta.get('model_tag', args.model_tag)}")
        print(f"Model Step:               {meta.get('step', args.step)}")
        print("-" * 80)
        print(f"Performance Metrics:")
        print(f"  Average Tokens/Second:  {mean_tokens_per_sec:.2f} ± {std_tokens_per_sec:.2f}")
        print(f"  Average Time/Iteration: {mean_time:.3f}s ± {std_time:.3f}s")
        print(f"  Peak GPU Memory:        {peak_memory_gb:.3f} GB")
        print("-" * 80)
        print(f"Individual Iteration Results:")
        for i, (t, tps) in enumerate(zip(iteration_times, iteration_tokens_per_sec)):
            print(f"  Iteration {i+1}: {t:.3f}s, {tps:.2f} tok/s")
        print("-" * 80)
        print(f"Sample Output (first 200 chars):")
        print(f"  \"{sample_output[:200]}...\"")
        print("=" * 80)
        
        # Success message
        print(f"\n✓ Benchmark completed successfully!")
        print(f"  Version: {args.output}")
        print(f"  Performance: {mean_tokens_per_sec:.2f} ± {std_tokens_per_sec:.2f} tok/s")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Model checkpoint not found")
        print(f"  {e}")
        print(f"  Please check that the model exists and NANOCHAT_BASE_DIR is set correctly.")
        return 1
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n✗ Error: GPU out of memory")
        print(f"  {e}")
        print(f"  Try reducing --max-tokens or use a smaller model.")
        return 1
        
    except Exception as e:
        print(f"\n✗ Error: Benchmark failed")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
