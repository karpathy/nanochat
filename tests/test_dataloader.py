"""
Tests for data loading functionality.

Note: The actual dataloader requires CUDA and parquet files, so these tests
are simplified to test the core concepts.

Run with:
python -m pytest tests/test_dataloader.py -v -s
"""

import torch
import pytest


def test_batch_creation():
    """Test creating batches from token sequences."""
    # Simulate what the dataloader does internally
    tokens = list(range(100))
    batch_size = 4
    seq_len = 10
    
    # Need batch_size * seq_len + 1 tokens for inputs and targets
    needed = batch_size * seq_len + 1
    assert len(tokens) >= needed
    
    # Create inputs and targets
    inputs = torch.tensor(tokens[:batch_size * seq_len]).view(batch_size, seq_len)
    targets = torch.tensor(tokens[1:batch_size * seq_len + 1]).view(batch_size, seq_len)
    
    # Check shapes
    assert inputs.shape == (batch_size, seq_len)
    assert targets.shape == (batch_size, seq_len)
    
    # Check that targets are shifted by 1
    assert targets[0, 0] == inputs[0, 1]


def test_token_buffer_simulation():
    """Test token buffering logic."""
    from collections import deque
    
    token_buffer = deque()
    
    # Simulate adding tokens
    for i in range(100):
        token_buffer.append(i)
    
    assert len(token_buffer) == 100
    
    # Simulate consuming tokens
    needed = 50
    consumed = []
    for _ in range(needed):
        consumed.append(token_buffer.popleft())
    
    assert len(consumed) == needed
    assert len(token_buffer) == 50
    assert consumed[0] == 0
    assert consumed[-1] == 49


def test_distributed_rank_sharding():
    """Test how data is distributed across ranks."""
    total_shards = 8
    world_size = 4
    
    # Each rank gets every world_size'th shard
    for rank in range(world_size):
        shards = list(range(rank, total_shards, world_size))
        assert len(shards) == total_shards // world_size


def test_sequence_packing():
    """Test packing tokens into sequences."""
    # Simulate the reshape operation in dataloader
    batch_size = 2
    seq_len = 4
    
    # Flat token sequence
    tokens = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    
    # Pack into batch
    batch = tokens.view(batch_size, seq_len)
    
    assert batch.shape == (batch_size, seq_len)
    assert batch[0, 0] == 0
    assert batch[0, -1] == 3
    assert batch[1, 0] == 4
    assert batch[1, -1] == 7


def test_input_target_alignment():
    """Test that inputs and targets are properly aligned."""
    seq_len = 10
    tokens = list(range(20))
    
    # Inputs: tokens[:-1]
    # Targets: tokens[1:]
    inputs = tokens[:seq_len]
    targets = tokens[1:seq_len + 1]
    
    # Each target should be the next token after corresponding input
    for i in range(seq_len):
        assert targets[i] == inputs[i] + 1


def test_bos_token_prepending():
    """Test BOS token prepending logic."""
    # Simulate what tokenizer does with prepend
    bos_token = 255
    text_tokens = [10, 20, 30, 40]
    
    # With prepend
    tokens_with_bos = [bos_token] + text_tokens
    
    assert tokens_with_bos[0] == bos_token
    assert len(tokens_with_bos) == len(text_tokens) + 1
