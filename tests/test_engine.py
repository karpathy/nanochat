"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import os
import json
import torch
import pytest
from dataclasses import dataclass

from huggingface_hub import snapshot_download
from nanochat.engine import KVCache, Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.checkpoint_manager import find_last_step

# -----------------------------------------------------------------------------
# Ensure deterministic behavior for reproducible tests
# See: https://docs.pytorch.org/docs/stable/notes/randomness.html

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for CUDA >= 10.2 determinism
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Mock classes for testing Engine without loading a real model

@dataclass
class MockConfig:
    """Minimal config for Engine tests."""
    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128


class MockModel:
    """
    Mock model that returns uniform logits over the vocab.
    This ensures that with temperature > 0, different samples should
    (with very high probability) produce different tokens.
    """
    def __init__(self, vocab_size=262):  # 256 bytes + 6 special tokens
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = "cpu"

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None, attention_mask=None):
        """Return uniform logits so sampling is spread across vocab."""
        B, T = ids.shape
        # Simulate what a real transformer does: insert k,v into the cache for each layer
        if kv_cache is not None:
            head_dim = self.config.n_embd // self.config.n_head
            for layer_idx in range(self.config.n_layer):
                k = torch.zeros(B, self.config.n_kv_head, T, head_dim)
                v = torch.zeros(B, self.config.n_kv_head, T, head_dim)
                kv_cache.insert_kv(layer_idx, k, v)
        # Uniform logits -> equal probability for all tokens
        logits = torch.zeros(B, T, self.vocab_size)
        return logits


class ByteTokenizer:
    """
    Simple byte-level tokenizer for testing.
    Tokens 0-255 are raw bytes, 256+ are special tokens.
    """
    def __init__(self):
        # Special tokens start at 256
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))  # bytes 0-255
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        # Filter out special tokens before decoding
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")


def get_model_and_tokenizer(use_pretrained=False):
    """
    Get a model and tokenizer for testing. Requires CUDA.

    Args:
        use_pretrained: If True, download and load the pretrained nanochat-d34 model.
                       If False, create a small randomly initialized model.

    Returns:
        (model, tokenizer) tuple
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these tests")
    device = torch.device("cuda")

    if use_pretrained:
        # Download the checkpoint
        cache_dir = snapshot_download(repo_id="karpathy/nanochat-d34")

        # Find the last step
        step = find_last_step(cache_dir)

        # Load model data
        model_path = os.path.join(cache_dir, f"model_{step:06d}.pt")
        model_data = torch.load(model_path, map_location=device)

        # Fix torch compile key prefix
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

        # Convert all tensors to bfloat16 for consistent dtypes (checkpoint has mixed bfloat16/float32)
        model_data = {
            k: v.bfloat16() if v.is_floating_point() else v
            for k, v in model_data.items()
        }

        # Load metadata
        meta_path = os.path.join(cache_dir, f"meta_{step:06d}.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # Build model
        model_config = GPTConfig(**meta_data["model_config"])
        with torch.device("meta"):
            model = GPT(model_config)
        model.to_empty(device=device)
        model.init_weights()
        model.load_state_dict(model_data, strict=True, assign=True)
        model.eval()

        # Load tokenizer from the checkpoint directory
        tokenizer = RustBPETokenizer.from_directory(cache_dir)
    else:
        # Small model for fast testing
        config = GPTConfig(
            sequence_len=256,
            vocab_size=262,  # 256 bytes + 6 special tokens
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
        )
        model = GPT(config)
        model.init_weights()
        model = model.to(device)
        model.eval()
        tokenizer = ByteTokenizer()

    return model, tokenizer


# -----------------------------------------------------------------------------
# KVCache tests

def test_kv_cache_resize():
    """
    The KV cache was not resized correctly, more information here:
    https://github.com/karpathy/nanochat/pull/186
    This test reproduces the issue and will be merged alongside the fix.
    """

    batch_size = 2
    num_heads = 3
    seq_len = 4
    head_dim = 5
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers
    )

    # Insert a single token with a distinct fill value to all layers
    def insert_token(token_idx):
        for layer_idx in range(num_layers):
            k = torch.full((batch_size, num_heads, 1, head_dim), fill_value=float(token_idx), dtype=torch.float32)
            v = torch.full((batch_size, num_heads, 1, head_dim), fill_value=float(token_idx * 100), dtype=torch.float32)
            kv_cache.insert_kv(layer_idx, k, v)

    # Insert 4 tokens (fills the initial seq_len=4)
    for i in range(4):
        insert_token(i)

    # Record the original state of the cache
    original_cache = kv_cache.kv_cache.clone()
    original_seq_len = original_cache.shape[4]

    # Insert the 5th token, which will trigger a resize
    insert_token(4)
    # Verify that the cache actually resized
    new_seq_len = kv_cache.kv_cache.shape[4]
    assert new_seq_len > original_seq_len, f"Cache did not resize: original seq_len={original_seq_len}, new seq_len={new_seq_len}"

    # Verify that the original 4 tokens are still intact after resize
    for layer_idx in range(num_layers):
        for token_idx in range(4):
            # Check that resized cache matches expected values
            expected_k = float(token_idx)
            expected_v = float(token_idx * 100)
            actual_k = kv_cache.kv_cache[layer_idx, 0, :, :, token_idx, :]
            actual_v = kv_cache.kv_cache[layer_idx, 1, :, :, token_idx, :]
            assert (actual_k == expected_k).all(), f"Layer {layer_idx}, token {token_idx}: key corrupted, expected {expected_k}"
            assert (actual_v == expected_v).all(), f"Layer {layer_idx}, token {token_idx}: value corrupted, expected {expected_v}"
            # And that the original cache matches resized cache
            original_k = original_cache[layer_idx, 0, :, :, token_idx, :]
            original_v = original_cache[layer_idx, 1, :, :, token_idx, :]
            assert (actual_k == original_k).all(), f"Layer {layer_idx}, token {token_idx}: key doesn't match original"
            assert (actual_v == original_v).all(), f"Layer {layer_idx}, token {token_idx}: value doesn't match original"


def test_multi_sample_first_token_diversity():
    """
    Test that when generating multiple samples, each sample gets an independently
    sampled first token (not a broadcast of the same token to all rows).

    Previously, the first token after prefill was sampled once and broadcast to all
    rows, causing all samples to start identically. The fix expands the prefill logits
    to num_samples and samples independently for each row.

    With uniform logits over 262 tokens and 16 samples, the probability that all
    samples independently pick the same token is (1/262)^15 ≈ 10^-36. So if they're
    all identical, it indicates tokens are being broadcast instead of independently sampled.
    """
    model = MockModel(vocab_size=262)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    # Generate 16 samples with temperature=1.0 (stochastic sampling)
    prompt_tokens = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    num_samples = 16

    # Collect the first generated token from each sample
    first_tokens = []
    gen = engine.generate(
        prompt_tokens,
        num_samples=num_samples,
        max_tokens=1,  # We only need the first token
        temperature=1.0,
        seed=42,
    )
    for token_column, token_masks in gen:
        first_tokens = token_column  # This is the first (and only) yield

    # With uniform distribution and 16 samples, they should NOT all be identical
    # If they are all identical, the bug exists (broadcasting instead of sampling)
    unique_tokens = set(first_tokens)
    assert len(unique_tokens) > 1, (
        f"All {num_samples} samples got the same first token ({first_tokens[0]}). "
        f"With uniform logits, this is statistically impossible (~10^-36 probability) "
        f"unless tokens are being broadcast instead of independently sampled."
    )


# -----------------------------------------------------------------------------
# Batched generation tests

@pytest.mark.parametrize("use_pretrained", [False, True])
def test_batched_generation_consistency(use_pretrained):
    """
    Test that batched generation produces the same results as individual generation.

    This test:
    1. Generates from each prompt individually (existing single-prompt behavior)
    2. Generates from all prompts together in a batch (new batched behavior)
    3. Asserts that the results match exactly

    Uses temperature=0.0 for deterministic outputs.
    """
    try:
        model, tokenizer = get_model_and_tokenizer(use_pretrained=use_pretrained)
    except Exception as e:
        if use_pretrained:
            pytest.skip(f"Could not load pretrained model: {e}")
        raise

    engine = Engine(model, tokenizer)

    # Define test prompts with different lengths
    bos = tokenizer.get_bos_token_id()
    prompts = [
        tokenizer.encode("hi", prepend=bos),
        tokenizer.encode("the capital of France is", prepend=bos),
        tokenizer.encode("hello, I'm a", prepend=bos),
    ]

    num_samples = 2
    # Deterministic decoding
    generation_kwargs = dict(max_tokens=10, temperature=0.0, seed=0)

    # 1) Generate individually for each prompt
    individual_results = []
    individual_masks = []
    for prompt in prompts:
        results, masks = engine.generate_batch(prompt, num_samples=num_samples, **generation_kwargs)
        individual_results.append(results)  # results is list[list[int]] of shape (num_samples, seq_len)
        individual_masks.append(masks)  # masks is list[list[int]] of shape (num_samples, seq_len)

    # 2) Generate batched (all prompts together)
    batched_results, batched_masks = engine.generate_batch(prompts, num_samples=num_samples, **generation_kwargs)

    # 3) Assert results match
    assert len(individual_results) == len(batched_results), \
        f"Prompt count mismatch: {len(individual_results)} vs {len(batched_results)}"

    for prompt_idx, (ind_samples, batch_samples, ind_masks, batch_masks) in enumerate(
            zip(individual_results, batched_results, individual_masks, batched_masks)):
        assert len(ind_samples) == len(batch_samples), f"Sample count mismatch for prompt {prompt_idx}"
        for sample_idx, (ind_result, batch_result, ind_mask, batch_mask) in enumerate(
                zip(ind_samples, batch_samples, ind_masks, batch_masks)):
            assert ind_result == batch_result, (
                f"Mismatch for prompt {prompt_idx}, sample {sample_idx}:\n"
                f"  Individual: {ind_result}\n"
                f"  Batched:    {batch_result}"
            )
            assert ind_mask == batch_mask, (
                f"Mask mismatch for prompt {prompt_idx}, sample {sample_idx}:\n"
                f"  Individual: {ind_mask}\n"
                f"  Batched:    {batch_mask}"
            )


def test_batched_generation_single_prompt():
    """
    Test that batched generation with a single prompt in the batch
    produces the same result as non-batched single prompt generation.
    """
    model, tokenizer = get_model_and_tokenizer(use_pretrained=False)
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompt = tokenizer.encode("the capital of France is", prepend=bos)
    num_samples = 3
    generation_kwargs = dict(max_tokens=8, temperature=0.0, seed=0)

    # Generate non-batched: returns shape (num_samples, seq_len)
    single_results, single_masks = engine.generate_batch(prompt, num_samples=num_samples, **generation_kwargs)

    # Generate batched with single prompt: returns shape (1, num_samples, seq_len)
    batched_results, batched_masks = engine.generate_batch([prompt], num_samples=num_samples, **generation_kwargs)

    assert single_results == batched_results[0], (
        f"Single vs batched single-prompt mismatch:\n"
        f"  Single:  {single_results}\n"
        f"  Batched: {batched_results[0]}"
    )
    assert single_masks == batched_masks[0], (
        f"Single vs batched single-prompt mask mismatch:\n"
        f"  Single:  {single_masks}\n"
        f"  Batched: {batched_masks[0]}"
    )


def test_batched_generation_stochastic():
    """
    Test that batched generation with temperature > 0 produces diverse outputs.
    """
    model, tokenizer = get_model_and_tokenizer(use_pretrained=False)
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompts = [
        tokenizer.encode("hi", prepend=bos),
        tokenizer.encode("the capital of France is", prepend=bos),
    ]

    num_samples = 4
    generation_kwargs = dict(max_tokens=64, temperature=1.0, seed=0)

    # Generate batched: returns shape (num_prompts, num_samples, seq_len)
    results, _ = engine.generate_batch(prompts, num_samples=num_samples, **generation_kwargs)

    # Check structure
    assert len(results) == len(prompts)

    # Check that samples within each prompt are diverse (not all identical)
    for prompt_idx, samples in enumerate(results):
        assert len(samples) == num_samples
        unique_samples = set(tuple(s) for s in samples)
        assert len(unique_samples) > 1, (
            f"All {num_samples} samples for prompt {prompt_idx} are identical. "
            f"With temperature=1.0, samples should differ."
        )
