"""
Auto-discovery module for finding optimal batch sizes.

This is a minimal stub implementation to enable testing.
The full implementation should be added as part of Task 41 (Auto Batch Size Module).
"""

import os
import json
import hashlib
import torch
import torch.distributed as dist
from typing import Optional, Callable, Dict, Any
from nanochat.common import print0, get_base_dir


def discover_batch_size(
    model: torch.nn.Module,
    max_seq_len: int,
    device: torch.device,
    safety_margin: float = 0.85,
    min_batch_size: int = 1,
    max_batch_size: int = 128,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    use_cache: bool = False,
    cache_key_components: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Discover the optimal batch size for a model.
    
    Args:
        model: The model to test
        max_seq_len: Maximum sequence length
        device: Device to run on
        safety_margin: Safety factor (e.g., 0.85 = use 85% of max)
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try
        ddp_rank: Rank in distributed setting
        ddp_world_size: World size in distributed setting
        use_cache: Whether to use cache
        cache_key_components: Components for cache key
    
    Returns:
        Discovered batch size
    """
    # Only rank 0 performs discovery in DDP
    if ddp_rank == 0:
        print0("Running auto-discovery on rank 0")
        
        # Check cache first
        if use_cache and cache_key_components:
            cached_size = _load_from_cache(cache_key_components)
            if cached_size is not None:
                print0(f"Cache hit! Using batch_size={cached_size}")
                discovered_size = cached_size
            else:
                print0("Cache miss, performing discovery")
                discovered_size = _perform_discovery(
                    model, max_seq_len, device, safety_margin, 
                    min_batch_size, max_batch_size
                )
                if cache_key_components:
                    _save_to_cache(cache_key_components, discovered_size)
        else:
            discovered_size = _perform_discovery(
                model, max_seq_len, device, safety_margin,
                min_batch_size, max_batch_size
            )
        
        print0(f"Auto-discovery found device_batch_size={discovered_size}")
    else:
        discovered_size = 0  # Will be broadcast from rank 0
    
    # Broadcast to all ranks in DDP
    if ddp_world_size > 1:
        discovered_tensor = torch.tensor(discovered_size, dtype=torch.int32, device=device)
        dist.broadcast(discovered_tensor, src=0)
        discovered_size = discovered_tensor.item()
        if ddp_rank != 0:
            print0(f"Received batch size from rank 0: {discovered_size}")
    
    return discovered_size


def _perform_discovery(
    model: torch.nn.Module,
    max_seq_len: int,
    device: torch.device,
    safety_margin: float,
    min_batch_size: int,
    max_batch_size: int,
) -> int:
    """
    Perform the actual discovery using exponential + binary search.
    
    This is a stub implementation that returns a fixed value.
    The real implementation should:
    1. Exponential search to find upper bound
    2. Binary search to refine
    3. Apply safety margin
    """
    # Stub: return a fixed reasonable value
    # Real implementation would perform exponential + binary search
    batch_size = min(32, max_batch_size)
    return max(int(batch_size * safety_margin), min_batch_size)


def _test_batch_size(
    model: torch.nn.Module,
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
) -> bool:
    """
    Test if a given batch size fits in memory.
    
    Returns:
        True if batch size works, False if OOM
    """
    try:
        # Create dummy inputs
        inputs = torch.randint(0, 50000, (batch_size, max_seq_len), device=device, dtype=torch.int32)
        targets = torch.randint(0, 50000, (batch_size, max_seq_len), device=device, dtype=torch.int64)
        
        # Forward + backward pass
        model.train()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(inputs, targets)
        loss.backward()
        model.zero_grad(set_to_none=True)
        
        # Clean up
        del inputs, targets, loss
        torch.cuda.empty_cache()
        
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        print0(f"Error testing batch size {batch_size}: {e}")
        torch.cuda.empty_cache()
        return False


def _get_cache_key(components: Dict[str, Any]) -> str:
    """Generate cache key from components."""
    key_str = json.dumps(components, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_from_cache(components: Dict[str, Any]) -> Optional[int]:
    """Load batch size from cache if available."""
    try:
        base_dir = get_base_dir()
        cache_dir = os.path.join(base_dir, "auto_batch_cache")
        cache_key = _get_cache_key(components)
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return data.get('batch_size')
    except Exception as e:
        print0(f"Cache load error: {e}")
    return None


def _save_to_cache(components: Dict[str, Any], batch_size: int) -> None:
    """Save batch size to cache."""
    try:
        base_dir = get_base_dir()
        cache_dir = os.path.join(base_dir, "auto_batch_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_key = _get_cache_key(components)
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        
        with open(cache_file, 'w') as f:
            json.dump({
                'batch_size': batch_size,
                'components': components,
            }, f, indent=2)
    except Exception as e:
        print0(f"Cache save error: {e}")
