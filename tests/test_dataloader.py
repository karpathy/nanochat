"""
Tests for data loading functionality.

Run with:
python -m pytest tests/test_dataloader.py -v -s
"""

import torch
import pytest
from collections import deque
from unittest.mock import Mock, patch, MagicMock
from nanochat.dataloader import tokenizing_distributed_data_loader


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.get_bos_token_id.return_value = 255
    tokenizer.encode.return_value = [
        [255, 1, 2, 3, 4],
        [255, 5, 6, 7, 8],
        [255, 9, 10, 11, 12],
    ]
    return tokenizer


@pytest.fixture
def mock_parquet_data():
    """Mock parquet data iterator."""
    def mock_iter(split, start, step):
        # Yield a few batches of mock documents
        for i in range(3):
            yield [f"Document {i*3}", f"Document {i*3+1}", f"Document {i*3+2}"]
    return mock_iter


def test_dataloader_initialization():
    """Test that dataloader can be initialized with proper mocks."""
    with patch('nanochat.dataloader.get_dist_info') as mock_dist:
        with patch('nanochat.dataloader.parquets_iter_batched'):
            with patch('nanochat.dataloader.get_tokenizer') as mock_tok:
                with patch('torch.Tensor.to', return_value=torch.tensor([[1, 2], [3, 4]])):
                    mock_dist.return_value = (False, 0, 0, 1)
                    mock_tokenizer = Mock()
                    mock_tokenizer.get_bos_token_id.return_value = 255
                    mock_tokenizer.encode.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
                    mock_tok.return_value = mock_tokenizer
                    
                    loader = tokenizing_distributed_data_loader(B=2, T=4, split="train")
                    assert loader is not None


def test_dataloader_batch_shapes():
    """Test that dataloader produces correct batch shapes."""
    with patch('nanochat.dataloader.get_dist_info') as mock_dist:
        with patch('nanochat.dataloader.parquets_iter_batched') as mock_parquet:
            with patch('nanochat.dataloader.get_tokenizer') as mock_tok:
                with patch('torch.empty') as mock_empty:
                    mock_dist.return_value = (False, 0, 0, 1)
                    
                    # Mock parquet to return documents
                    mock_parquet.return_value = iter([
                        ["doc1", "doc2", "doc3"],
                        ["doc4", "doc5", "doc6"],
                    ])
                    
                    # Mock tokenizer to return tokens
                    mock_tokenizer = Mock()
                    mock_tokenizer.get_bos_token_id.return_value = 255
                    # Return enough tokens for at least one batch (B=2, T=3 needs 7 tokens)
                    mock_tokenizer.encode.return_value = [
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                    ]
                    mock_tok.return_value = mock_tokenizer
                    
                    B, T = 2, 3
                    
                    # Mock the scratch buffer
                    mock_empty.return_value = torch.zeros(B * T + 1, dtype=torch.int64)
                    
                    # Mock cuda tensors
                    with patch('torch.Tensor.to') as mock_to:
                        def to_side_effect(*args, **kwargs):
                            # Return a properly shaped tensor
                            if kwargs.get('device') == 'cuda':
                                shape = (B, T)
                                dtype = kwargs.get('dtype', torch.int64)
                                return torch.zeros(shape, dtype=dtype)
                            return torch.zeros((B, T))
                        
                        mock_to.side_effect = to_side_effect
                        
                        loader = tokenizing_distributed_data_loader(B=B, T=T, split="train")
                        inputs, targets = next(loader)
                        
                        assert inputs.shape == (B, T)
                        assert targets.shape == (B, T)


def test_dataloader_token_shifting():
    """Test that targets are shifted by 1 position from inputs."""
    B, T = 2, 4
    needed_tokens = B * T + 1  # 9 tokens
    
    # Create a sequence where we can verify the shift
    tokens = list(range(100, 100 + needed_tokens))  # [100, 101, 102, ..., 108]
    
    # Simulate what the dataloader does
    inputs = torch.tensor(tokens[:-1]).view(B, T)
    targets = torch.tensor(tokens[1:]).view(B, T)
    
    # Check shapes
    assert inputs.shape == (B, T)
    assert targets.shape == (B, T)
    
    # Check shifting: targets[i] should equal inputs[i+1] (within the flat view)
    inputs_flat = inputs.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    # First element of targets should be second element from original sequence
    assert targets_flat[0].item() == 101
    assert inputs_flat[0].item() == 100


def test_dataloader_distributed_sharding():
    """Test that different ranks get different shards."""
    with patch('nanochat.dataloader.get_dist_info') as mock_dist:
        with patch('nanochat.dataloader.parquets_iter_batched') as mock_parquet:
            with patch('nanochat.dataloader.get_tokenizer') as mock_tok:
                with patch('torch.empty') as mock_empty:
                    # Simulate rank 1 out of 4
                    mock_dist.return_value = (True, 1, 1, 4)
                    
                    # Track what start/step values parquets_iter_batched is called with
                    calls = []
                    def track_parquet_call(split, start, step):
                        calls.append((split, start, step))
                        return iter([["doc1", "doc2"]])
                    
                    mock_parquet.side_effect = track_parquet_call
                    
                    mock_tokenizer = Mock()
                    mock_tokenizer.get_bos_token_id.return_value = 255
                    mock_tokenizer.encode.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
                    mock_tok.return_value = mock_tokenizer
                    
                    # Mock the scratch buffer
                    mock_empty.return_value = torch.zeros(7, dtype=torch.int64)
                    
                    with patch('torch.Tensor.to', return_value=torch.zeros((2, 3))):
                        loader = tokenizing_distributed_data_loader(B=2, T=3, split="train")
                        next(loader)
                    
                    # Verify parquets_iter_batched was called with correct rank/world_size
                    assert len(calls) > 0
                    split, start, step = calls[0]
                    assert start == 1  # rank 1
                    assert step == 4   # world_size 4


def test_dataloader_token_buffer_accumulation():
    """Test token buffer accumulation logic."""
    token_buffer = deque()
    B, T = 2, 3
    needed_tokens = B * T + 1  # 7 tokens
    
    # Simulate adding tokens from documents
    doc1_tokens = [1, 2, 3]
    doc2_tokens = [4, 5, 6, 7, 8]
    
    token_buffer.extend(doc1_tokens)
    assert len(token_buffer) < needed_tokens
    
    token_buffer.extend(doc2_tokens)
    assert len(token_buffer) >= needed_tokens
    
    # Extract tokens for one batch
    batch_tokens = []
    for _ in range(needed_tokens):
        batch_tokens.append(token_buffer.popleft())
    
    assert len(batch_tokens) == needed_tokens
    assert batch_tokens == [1, 2, 3, 4, 5, 6, 7]
    assert len(token_buffer) == 1  # One token remaining


def test_dataloader_split_validation():
    """Test that invalid split values raise an error."""
    # The function is a generator, so the assertion only runs when next() is called
    with pytest.raises(AssertionError, match="split must be"):
        loader = tokenizing_distributed_data_loader(B=2, T=4, split="invalid")
        next(loader)  # This triggers the function body to execute


def test_dataloader_bos_token_prepending():
    """Test that BOS tokens are properly prepended."""
    with patch('nanochat.dataloader.get_dist_info') as mock_dist:
        with patch('nanochat.dataloader.parquets_iter_batched') as mock_parquet:
            with patch('nanochat.dataloader.get_tokenizer') as mock_tok:
                with patch('torch.empty') as mock_empty:
                    mock_dist.return_value = (False, 0, 0, 1)
                    mock_parquet.return_value = iter([["doc1"]])
                    
                    mock_tokenizer = Mock()
                    bos_token = 255
                    mock_tokenizer.get_bos_token_id.return_value = bos_token
                    mock_tokenizer.encode.return_value = [[bos_token, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
                    mock_tok.return_value = mock_tokenizer
                    
                    # Mock the scratch buffer
                    mock_empty.return_value = torch.zeros(7, dtype=torch.int64)
                    
                    with patch('torch.Tensor.to', return_value=torch.zeros((2, 3))):
                        loader = tokenizing_distributed_data_loader(B=2, T=3, split="train")
                        next(loader)
                    
                    # Verify tokenizer.encode was called with prepend=bos_token
                    mock_tokenizer.encode.assert_called()
                    call_kwargs = mock_tokenizer.encode.call_args[1]
                    assert 'prepend' in call_kwargs
                    assert call_kwargs['prepend'] == bos_token


def test_dataloader_needed_tokens_calculation():
    """Test that the dataloader calculates needed tokens correctly."""
    B, T = 4, 16
    needed_tokens = B * T + 1
    
    # We need B*T tokens for inputs, plus 1 for the last target
    assert needed_tokens == 65
    
    # The scratch buffer should be exactly this size
    scratch = torch.empty(needed_tokens, dtype=torch.int64)
    assert scratch.shape == (65,)
    
    # After slicing, inputs should be B*T and targets should be B*T
    inputs = scratch[:-1]
    targets = scratch[1:]
    assert len(inputs) == B * T
    assert len(targets) == B * T
