"""
Automatic batch size discovery module for maximizing GPU utilization.

This module implements an intelligent batch size search algorithm that:
1. Uses exponential search to quickly find an upper bound
2. Refines with binary search for optimal size
3. Applies safety margin to prevent edge-case OOMs
4. Supports DDP multi-GPU coordination
5. Caches results for faster subsequent runs
"""

import os
import json
import time
import hashlib
import torch

from nanochat.common import print0, get_base_dir, get_dist_info


def find_optimal_device_batch_size(
    model,
    max_seq_len,
    grad_accum_steps,
    data_sample_fn,
    device,
    override=None,
    enable_cache=True,
    safety_margin=0.85,
):
    """
    Main entry point for automatic batch size discovery.
    
    Args:
        model: PyTorch model to test
        max_seq_len: Maximum sequence length
        grad_accum_steps: Number of gradient accumulation steps
        data_sample_fn: Callable(batch_size, max_seq_len) -> (inputs, targets)
        device: Device to run tests on
        override: If set, skip discovery and return this value
        enable_cache: Whether to use caching
        safety_margin: Fraction of optimal batch size to use (default 0.85)
    
    Returns:
        optimal_batch_size: Optimal device batch size for this GPU
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    # Handle manual override
    if override is not None:
        print0(f"Using manual batch_size override: {override}")
        return override
    
    optimal_batch_size = None
    
    # Only rank 0 performs discovery
    if ddp_rank == 0:
        start_time = time.time()
        print0(f"\n{'='*60}")
        print0(f"Starting automatic batch size discovery...")
        print0(f"Parameters: max_seq_len={max_seq_len}, grad_accum_steps={grad_accum_steps}")
        print0(f"Safety margin: {safety_margin:.2%}")
        print0(f"{'='*60}\n")
        
        # Check cache
        cache_key = None
        if enable_cache:
            cache_key = _get_cache_key(model, max_seq_len)
            cached_batch_size = _load_from_cache(cache_key)
            if cached_batch_size is not None:
                print0(f"✓ Cache hit! Using cached batch_size: {cached_batch_size}")
                optimal_batch_size = cached_batch_size
        
        # Run discovery if no cache hit
        if optimal_batch_size is None:
            try:
                # Warmup CUDA
                _warmup_cuda(device)
                
                # Run the search algorithm
                optimal_batch_size = _find_batch_size_internal(
                    model=model,
                    max_seq_len=max_seq_len,
                    grad_accum_steps=grad_accum_steps,
                    data_sample_fn=data_sample_fn,
                    device=device,
                    safety_margin=safety_margin,
                )
                
                # Save to cache
                if enable_cache and cache_key is not None and optimal_batch_size is not None:
                    _save_to_cache(cache_key, optimal_batch_size)
                
                elapsed = time.time() - start_time
                print0(f"\n{'='*60}")
                print0(f"✓ Found optimal batch_size={optimal_batch_size} in {elapsed:.1f} seconds")
                print0(f"{'='*60}\n")
                
            except Exception as e:
                print0(f"⚠ Warning: Batch size discovery failed with error: {e}")
                optimal_batch_size = None
        
        # Fallback to conservative defaults if discovery failed
        if optimal_batch_size is None:
            print0(f"⚠ Warning: Using conservative fallback batch_size=8")
            optimal_batch_size = 8
    
    # DDP: Broadcast result from rank 0 to all ranks
    if ddp_world_size > 1:
        try:
            import torch.distributed as dist
            tensor = torch.tensor([optimal_batch_size if optimal_batch_size is not None else 8], 
                                dtype=torch.long, device=device)
            dist.broadcast(tensor, src=0)
            optimal_batch_size = tensor.item()
        except Exception as e:
            print0(f"⚠ Warning: DDP broadcast failed: {e}")
            if optimal_batch_size is None:
                optimal_batch_size = 8
    
    return optimal_batch_size


def _find_batch_size_internal(model, max_seq_len, grad_accum_steps, data_sample_fn, device, safety_margin):
    """
    Core algorithm implementing exponential search followed by binary search.
    
    Returns:
        optimal_batch_size: The largest batch size that fits in memory (with safety margin)
    """
    # Phase 1: Exponential search to find upper bound
    print0("Phase 1: Exponential search to find upper bound...")
    batch_size = 1
    last_successful = None
    
    while True:
        print0(f"  Testing batch_size={batch_size}...", end=" ")
        success = _test_batch_size(
            model=model,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            grad_accum_steps=grad_accum_steps,
            data_sample_fn=data_sample_fn,
            device=device,
        )
        
        if success:
            print0("✓ Success")
            last_successful = batch_size
            batch_size *= 2
        else:
            print0("✗ OOM")
            break
    
    # If even batch_size=1 failed, return None
    if last_successful is None:
        print0("✗ Even batch_size=1 caused OOM!")
        return None
    
    # Phase 2: Binary search refinement
    print0(f"\nPhase 2: Binary search refinement between {last_successful} and {batch_size}...")
    lower = last_successful
    upper = batch_size
    
    while upper - lower > 1:
        mid = (lower + upper) // 2
        print0(f"  Testing batch_size={mid}...", end=" ")
        success = _test_batch_size(
            model=model,
            batch_size=mid,
            max_seq_len=max_seq_len,
            grad_accum_steps=grad_accum_steps,
            data_sample_fn=data_sample_fn,
            device=device,
        )
        
        if success:
            print0("✓ Success")
            lower = mid
        else:
            print0("✗ OOM")
            upper = mid
    
    # Phase 3: Apply safety margin
    optimal_batch_size = int(lower * safety_margin)
    print0(f"\nApplying safety margin: {lower} × {safety_margin:.2%} = {optimal_batch_size}")
    
    return optimal_batch_size


def _test_batch_size(model, batch_size, max_seq_len, grad_accum_steps, data_sample_fn, device):
    """
    Test if a specific batch size fits in memory by simulating training loop.
    
    Returns:
        bool: True if batch size fits, False if OOM
    """
    try:
        # Clear CUDA cache before test
        torch.cuda.empty_cache()
        
        # Set model to training mode
        model.train()
        
        # Zero gradients
        model.zero_grad(set_to_none=True)
        
        # Simulate gradient accumulation steps
        for _ in range(grad_accum_steps):
            # Generate test batch
            inputs, targets = data_sample_fn(batch_size, max_seq_len)
            
            # Forward pass with bfloat16 autocast
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                # Compute loss (cross entropy)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            
            # Backward pass
            loss.backward()
        
        # Synchronize CUDA to ensure all operations complete
        torch.cuda.synchronize()
        
        # Clear cache after test
        torch.cuda.empty_cache()
        
        return True
        
    except torch.cuda.OutOfMemoryError:
        # Clear cache and return False on OOM
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        # Handle other exceptions
        print0(f"\n⚠ Warning: Test failed with unexpected error: {e}")
        torch.cuda.empty_cache()
        return False


def _warmup_cuda(device):
    """Warmup CUDA by allocating and freeing a small tensor."""
    try:
        x = torch.zeros(1, device=device)
        del x
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as e:
        print0(f"⚠ Warning: CUDA warmup failed: {e}")


def _get_cache_key(model, max_seq_len):
    """
    Generate cache key from model config hash, GPU model, and max_seq_len.
    
    Returns:
        str: Hash string to use as cache key
    """
    try:
        # Get model config attributes
        config = model.config if hasattr(model, 'config') else None
        if config is None:
            # Try to get from original model (in case of compiled model)
            config = model._orig_mod.config if hasattr(model, '_orig_mod') else None
        
        if config is None:
            return None
        
        # Build config string
        config_parts = [
            f"vocab_size={config.vocab_size}",
            f"n_layer={config.n_layer}",
            f"n_embd={config.n_embd}",
            f"n_head={config.n_head}",
            f"n_kv_head={config.n_kv_head}",
        ]
        config_str = "|".join(config_parts)
        
        # Get GPU model name
        gpu_name = torch.cuda.get_device_name(0)
        
        # Combine all components
        key_str = f"{config_str}|gpu={gpu_name}|seq_len={max_seq_len}"
        
        # Hash to create a short key
        cache_key = hashlib.md5(key_str.encode()).hexdigest()
        
        return cache_key
        
    except Exception as e:
        print0(f"⚠ Warning: Failed to generate cache key: {e}")
        return None


def _load_from_cache(cache_key):
    """
    Load cached batch size from JSON file.
    
    Returns:
        int or None: Cached batch size, or None if not found
    """
    if cache_key is None:
        return None
    
    try:
        base_dir = get_base_dir()
        cache_dir = os.path.join(base_dir, "auto_batch_cache")
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return data.get('batch_size')
            
    except Exception as e:
        print0(f"⚠ Warning: Failed to load from cache: {e}")
        return None


def _save_to_cache(cache_key, batch_size):
    """Save batch size to JSON cache file."""
    if cache_key is None or batch_size is None:
        return
    
    try:
        base_dir = get_base_dir()
        cache_dir = os.path.join(base_dir, "auto_batch_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        
        data = {
            'batch_size': batch_size,
            'timestamp': time.time(),
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print0(f"✓ Saved batch_size={batch_size} to cache")
        
    except Exception as e:
        print0(f"⚠ Warning: Failed to save to cache: {e}")
