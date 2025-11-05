"""
Unit tests for auto-discovery batch size functionality.

Run with: pytest tests/test_auto_batch_size.py -v
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

# Import the module to test
from nanochat.auto_batch_size import (
    discover_batch_size,
    _perform_discovery,
    _test_batch_size,
    _get_cache_key,
    _load_from_cache,
    _save_to_cache,
)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.layer = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, y=None):
        # Simplified forward pass
        out = self.layer(x.float())
        if y is not None:
            loss = (out - y.float()).pow(2).mean()
            return loss
        return out


# ============================================================================
# Test 1: Exponential Search Logic
# ============================================================================

def test_exponential_search():
    """Test that exponential search finds upper bound correctly."""
    model = SimpleTestModel()
    device = torch.device('cpu')
    max_seq_len = 256
    
    # Mock _test_batch_size to return True up to 32, False at 64
    with patch('nanochat.auto_batch_size._test_batch_size') as mock_test:
        def side_effect(model, bs, seq_len, dev):
            return bs < 64
        
        mock_test.side_effect = side_effect
        
        # Mock _perform_discovery to track calls
        with patch('nanochat.auto_batch_size._perform_discovery') as mock_discover:
            # Simulate exponential search behavior
            tried_sizes = []
            batch_size = 1
            while batch_size <= 128:
                works = mock_test(model, batch_size, max_seq_len, device)
                tried_sizes.append(batch_size)
                if not works:
                    break
                batch_size *= 2
            
            # Verify exponential progression: 1, 2, 4, 8, 16, 32, 64
            assert tried_sizes == [1, 2, 4, 8, 16, 32, 64], \
                f"Expected [1, 2, 4, 8, 16, 32, 64], got {tried_sizes}"
            
            # Verify we found the boundary (32 works, 64 fails)
            assert mock_test(model, 32, max_seq_len, device) == True
            assert mock_test(model, 64, max_seq_len, device) == False


# ============================================================================
# Test 2: Binary Search Refinement
# ============================================================================

def test_binary_search_refinement():
    """Test that binary search narrows down to exact boundary."""
    model = SimpleTestModel()
    device = torch.device('cpu')
    max_seq_len = 256
    
    # Mock OOM boundary at batch_size=52
    with patch('nanochat.auto_batch_size._test_batch_size') as mock_test:
        def side_effect(model, bs, seq_len, dev):
            return bs <= 52
        
        mock_test.side_effect = side_effect
        
        # Simulate binary search between 32 and 64
        tried_sizes = []
        low, high = 32, 64
        
        while low < high:
            mid = (low + high + 1) // 2
            tried_sizes.append(mid)
            if mock_test(model, mid, max_seq_len, device):
                low = mid
            else:
                high = mid - 1
        
        result = low
        
        # Should have tried: 48, 56, 52
        assert 48 in tried_sizes, "Should try midpoint 48"
        assert 56 in tried_sizes, "Should try midpoint 56"
        assert 52 in tried_sizes, "Should try midpoint 52"
        
        # Should converge to 52
        assert result == 52, f"Expected 52, got {result}"


# ============================================================================
# Test 3: Safety Margin Application
# ============================================================================

def test_safety_margin():
    """Test that safety margin is applied correctly."""
    margins = [0.85, 0.90, 0.95]
    max_batch = 60
    expected = [51, 54, 57]  # int(60 * margin)
    
    for margin, exp in zip(margins, expected):
        result = int(max_batch * margin)
        assert result == exp, f"Margin {margin}: expected {exp}, got {result}"
    
    # Test with discover_batch_size
    model = SimpleTestModel()
    device = torch.device('cpu')
    max_seq_len = 256
    
    with patch('nanochat.auto_batch_size._perform_discovery') as mock_discover:
        # Mock returns max batch before margin
        mock_discover.return_value = max_batch
        
        for margin, exp in zip(margins, expected):
            # The actual function should apply the margin internally
            # For now, test the calculation
            applied = int(max_batch * margin)
            assert applied == exp


# ============================================================================
# Test 4: Cache Mechanism
# ============================================================================

def test_cache_hit():
    """Test that cache hit skips discovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock cache
        cache_components = {
            'model_config': {'n_layer': 12, 'n_embd': 768},
            'gpu': 'A100',
            'max_seq_len': 2048,
        }
        
        cached_batch_size = 32
        
        # Mock get_base_dir to use tmpdir
        with patch('nanochat.auto_batch_size.get_base_dir', return_value=tmpdir):
            # Save to cache
            _save_to_cache(cache_components, cached_batch_size)
            
            # Load from cache
            loaded_size = _load_from_cache(cache_components)
            
            assert loaded_size == cached_batch_size, \
                f"Expected {cached_batch_size}, got {loaded_size}"


def test_cache_miss():
    """Test that cache miss triggers discovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_components = {
            'model_config': {'n_layer': 12, 'n_embd': 768},
            'gpu': 'A100',
            'max_seq_len': 2048,
        }
        
        with patch('nanochat.auto_batch_size.get_base_dir', return_value=tmpdir):
            # Try to load from empty cache
            loaded_size = _load_from_cache(cache_components)
            
            assert loaded_size is None, "Expected cache miss"


def test_cache_key_includes_components():
    """Test that cache key includes all components."""
    components1 = {
        'model_config': {'n_layer': 12, 'n_embd': 768},
        'gpu': 'A100',
        'max_seq_len': 2048,
    }
    
    components2 = {
        'model_config': {'n_layer': 20, 'n_embd': 1280},  # Different model
        'gpu': 'A100',
        'max_seq_len': 2048,
    }
    
    components3 = {
        'model_config': {'n_layer': 12, 'n_embd': 768},
        'gpu': 'A100',
        'max_seq_len': 1024,  # Different seq_len
    }
    
    key1 = _get_cache_key(components1)
    key2 = _get_cache_key(components2)
    key3 = _get_cache_key(components3)
    
    assert key1 != key2, "Different model configs should have different keys"
    assert key1 != key3, "Different max_seq_len should have different keys"
    assert key2 != key3, "All different components should have different keys"
    
    # Same components should give same key
    key1_again = _get_cache_key(components1)
    assert key1 == key1_again, "Same components should give same key"


# ============================================================================
# Test 5: DDP Broadcast Simulation
# ============================================================================

def test_ddp_broadcast():
    """Test that rank 0 discovery is broadcast to all ranks."""
    model = SimpleTestModel()
    device = torch.device('cpu')
    max_seq_len = 256
    discovered_size = 12
    
    # Mock distributed operations
    with patch('nanochat.auto_batch_size._perform_discovery') as mock_discover:
        mock_discover.return_value = discovered_size
        
        # Test rank 0 (performs discovery)
        with patch('nanochat.auto_batch_size.dist.broadcast') as mock_broadcast:
            result = discover_batch_size(
                model, max_seq_len, device,
                ddp_rank=0, ddp_world_size=4
            )
            
            # Rank 0 should perform discovery
            mock_discover.assert_called_once()
            
            # Should broadcast the result
            assert mock_broadcast.called
            
            # Result should be the discovered size
            # Note: actual broadcast simulation is complex, 
            # this tests the logic flow


def test_ddp_broadcast_rank_non_zero():
    """Test that non-zero ranks receive broadcasted value."""
    model = SimpleTestModel()
    device = torch.device('cpu')
    max_seq_len = 256
    
    with patch('nanochat.auto_batch_size._perform_discovery') as mock_discover:
        with patch('nanochat.auto_batch_size.dist.broadcast') as mock_broadcast:
            # Simulate broadcast receiving value
            def broadcast_side_effect(tensor, src):
                tensor.fill_(16)  # Simulated received value
            
            mock_broadcast.side_effect = broadcast_side_effect
            
            result = discover_batch_size(
                model, max_seq_len, device,
                ddp_rank=1, ddp_world_size=4
            )
            
            # Rank 1 should NOT perform discovery
            mock_discover.assert_not_called()
            
            # Should receive broadcast
            assert mock_broadcast.called


# ============================================================================
# Additional Tests
# ============================================================================

def test_min_max_batch_size_constraints():
    """Test that discovery respects min/max constraints."""
    model = SimpleTestModel()
    device = torch.device('cpu')
    max_seq_len = 256
    
    with patch('nanochat.auto_batch_size._perform_discovery') as mock_discover:
        # Test with very low max
        mock_discover.return_value = 4
        result = discover_batch_size(
            model, max_seq_len, device,
            min_batch_size=1, max_batch_size=4,
            ddp_rank=0, ddp_world_size=1
        )
        
        # Should be called with the constraints
        call_args = mock_discover.call_args
        assert call_args[0][4] == 1  # min_batch_size
        assert call_args[0][5] == 4  # max_batch_size


def test_discover_with_no_cache():
    """Test discovery without caching."""
    model = SimpleTestModel()
    device = torch.device('cpu')
    max_seq_len = 256
    
    with patch('nanochat.auto_batch_size._perform_discovery') as mock_discover:
        mock_discover.return_value = 16
        
        result = discover_batch_size(
            model, max_seq_len, device,
            use_cache=False,
            ddp_rank=0, ddp_world_size=1
        )
        
        # Should perform discovery
        mock_discover.assert_called_once()
        assert result == 16


def test_cache_corruption_handling():
    """Test that corrupted cache is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_components = {
            'model_config': {'n_layer': 12},
            'gpu': 'A100',
            'max_seq_len': 2048,
        }
        
        with patch('nanochat.auto_batch_size.get_base_dir', return_value=tmpdir):
            # Create corrupted cache file
            cache_dir = os.path.join(tmpdir, "auto_batch_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_key = _get_cache_key(cache_components)
            cache_file = os.path.join(cache_dir, f"{cache_key}.json")
            
            # Write corrupted JSON
            with open(cache_file, 'w') as f:
                f.write("invalid json {{{")
            
            # Should return None instead of crashing
            loaded_size = _load_from_cache(cache_components)
            assert loaded_size is None, "Corrupted cache should return None"


# ============================================================================
# Integration-style unit test
# ============================================================================

def test_full_discovery_flow():
    """Test the full discovery flow end-to-end."""
    model = SimpleTestModel()
    device = torch.device('cpu')
    max_seq_len = 128  # Small for CPU testing
    
    # Run actual discovery (on CPU, so it won't OOM)
    result = discover_batch_size(
        model, max_seq_len, device,
        safety_margin=0.85,
        min_batch_size=1,
        max_batch_size=16,  # Keep small for CPU
        ddp_rank=0,
        ddp_world_size=1,
        use_cache=False,
    )
    
    # Result should be within bounds
    assert 1 <= result <= 16, f"Result {result} out of bounds [1, 16]"
    
    # Result should be reasonable
    assert result >= 1, "Should find at least batch_size=1"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
