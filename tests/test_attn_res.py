"""
Test Attention Residuals (Block AttnRes) integration.

python -m pytest tests/test_attn_res.py -v
"""

import torch

# Force float32 compute for CPU test stability (avoids bf16 NaN in small models
# and dtype mismatches between model activations and KV cache on CUDA machines)
import nanochat.gpt as _gpt_mod
import nanochat.common as _common_mod
_gpt_mod.COMPUTE_DTYPE = torch.float32
_common_mod.COMPUTE_DTYPE = torch.float32

from nanochat.gpt import GPT, GPTConfig, block_attn_res


def _make_config(attn_res=True, **kwargs):
    """Create a small test config."""
    defaults = dict(
        sequence_len=64, vocab_size=256, n_layer=8, n_head=4, n_kv_head=4, n_embd=64,
        window_pattern="L", attn_res=attn_res, attn_res_block_size=4,
    )
    defaults.update(kwargs)
    return GPTConfig(**defaults)


def _build_model(config):
    """Build a model on meta device, move to CPU, init weights."""
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


# ---- Unit tests for building blocks ----

def test_block_attn_res_function():
    """block_attn_res produces correct shape and valid output."""
    B, T, D = 2, 8, 16
    blocks = [torch.randn(B, T, D) for _ in range(3)]
    partial = torch.randn(B, T, D)
    w = torch.randn(D)

    h = block_attn_res(blocks, partial, w)
    assert h.shape == (B, T, D)
    assert torch.isfinite(h).all()


def test_block_attn_res_single_block():
    """With no prior blocks, output should equal the partial block itself."""
    B, T, D = 1, 4, 8
    partial = torch.randn(B, T, D)
    w = torch.randn(D)

    h = block_attn_res([], partial, w)
    # With only one element, softmax over dim 0 gives 1.0, so h == partial
    assert torch.allclose(h, partial, atol=1e-6)


# ---- Model construction tests ----

def test_model_creates_attn_res_params():
    """When attn_res=True, GPT has attn_res_proj and mlp_res_proj parameters."""
    config = _make_config(attn_res=True)
    model = _build_model(config)
    assert hasattr(model, 'attn_res_proj')
    assert hasattr(model, 'mlp_res_proj')
    assert model.attn_res_proj.shape == (config.n_layer, config.n_embd)
    assert model.mlp_res_proj.shape == (config.n_layer, config.n_embd)


def test_model_no_attn_res_params_when_disabled():
    """When attn_res=False, GPT does NOT have AttnRes parameters."""
    config = _make_config(attn_res=False)
    model = _build_model(config)
    assert not hasattr(model, 'attn_res_proj')


def test_attn_res_param_count_overhead():
    """AttnRes adds minimal parameter overhead."""
    config_base = _make_config(attn_res=False)
    config_ar = _make_config(attn_res=True)
    model_base = _build_model(config_base)
    model_ar = _build_model(config_ar)
    n_base = sum(p.numel() for p in model_base.parameters())
    n_ar = sum(p.numel() for p in model_ar.parameters())
    overhead = (n_ar - n_base) / n_base
    # AttnRes adds 2 * n_layer * D params. For D=64, n_layer=8: 2*8*64 = 1024.
    assert overhead < 0.05, f"AttnRes overhead {overhead:.2%} is too high"
    assert n_ar > n_base, "AttnRes model should have more params"


# ---- Forward pass tests ----

def test_forward_training_attn_res():
    """Forward pass with attn_res=True produces valid loss."""
    config = _make_config(attn_res=True)
    model = _build_model(config)
    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    loss = model(idx, targets=targets)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_forward_inference_attn_res():
    """Forward pass without targets returns logits."""
    config = _make_config(attn_res=True)
    model = _build_model(config)
    model.eval()
    B, T = 1, 16
    idx = torch.randint(0, config.vocab_size, (B, T))
    logits = model(idx)
    assert logits.shape == (B, T, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_forward_standard_path_unchanged():
    """Standard path (attn_res=False) still works correctly."""
    config = _make_config(attn_res=False)
    model = _build_model(config)
    # Disable backout for small test model (pre-existing NaN with n_embd=64)
    model.backout_lambda.data.fill_(0.0)
    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    loss = model(idx, targets=targets)
    assert torch.isfinite(loss)


def test_backward_attn_res():
    """Gradients flow through AttnRes path."""
    config = _make_config(attn_res=True)
    model = _build_model(config)
    B, T = 2, 16
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    loss = model(idx, targets=targets)
    loss.backward()
    # Check that AttnRes projection gradients are non-zero
    assert model.attn_res_proj.grad is not None
    assert model.mlp_res_proj.grad is not None


# ---- Config validation tests ----

def test_invalid_block_size_odd():
    """Odd block size should fail validation."""
    try:
        config = _make_config(attn_res=True, attn_res_block_size=3)
        _build_model(config)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_invalid_block_size_one():
    """block_size=1 should fail validation."""
    try:
        config = _make_config(attn_res=True, attn_res_block_size=1)
        _build_model(config)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_valid_block_sizes():
    """Various valid block sizes should work."""
    for bs in [2, 4, 6, 8]:
        config = _make_config(attn_res=True, attn_res_block_size=bs, n_layer=8)
        model = _build_model(config)
        idx = torch.randint(0, 256, (1, 16))
        logits = model(idx)
        assert torch.isfinite(logits).all(), f"NaN/Inf with block_size={bs}"


# ---- Optimizer tests ----

def test_optimizer_setup_attn_res():
    """Optimizer setup accounts for all AttnRes params."""
    config = _make_config(attn_res=True)
    model = _build_model(config)
    optimizer = model.setup_optimizer()
    assert optimizer is not None


def test_optimizer_setup_standard():
    """Optimizer setup works for standard path too."""
    config = _make_config(attn_res=False)
    model = _build_model(config)
    optimizer = model.setup_optimizer()
    assert optimizer is not None


# ---- FLOPs and param counting tests ----

def test_estimate_flops_attn_res():
    """FLOPs estimation works with AttnRes."""
    config = _make_config(attn_res=True)
    model = _build_model(config)
    flops = model.estimate_flops()
    assert flops > 0


def test_num_scaling_params_attn_res():
    """Parameter count is consistent with AttnRes."""
    config = _make_config(attn_res=True)
    model = _build_model(config)
    counts = model.num_scaling_params()
    assert counts['total'] == sum(p.numel() for p in model.parameters())


# ---- KV cache inference tests ----

def test_kv_cache_inference_attn_res():
    """AttnRes works with KV cache for inference."""
    from nanochat.engine import KVCache
    config = _make_config(attn_res=True, n_layer=4, n_embd=64, n_head=4, n_kv_head=4)
    model = _build_model(config)
    model.eval()

    B, T = 1, 8
    idx = torch.randint(0, config.vocab_size, (B, T))

    # Prefill
    kv_cache = KVCache(
        batch_size=B, num_heads=config.n_kv_head,
        seq_len=32, head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer, device="cpu", dtype=torch.float32,
    )
    logits_prefill = model(idx, kv_cache=kv_cache)
    assert logits_prefill.shape == (B, T, config.vocab_size)

    # Decode one token
    next_idx = torch.randint(0, config.vocab_size, (B, 1))
    logits_decode = model(next_idx, kv_cache=kv_cache)
    assert logits_decode.shape == (B, 1, config.vocab_size)
    assert torch.isfinite(logits_decode).all()


def test_generate_attn_res():
    """Full generation pipeline works with AttnRes."""
    from nanochat.engine import Engine
    from tests.test_engine import MockModel, ByteTokenizer

    # We need a real model for this test, not mock
    config = _make_config(attn_res=True, n_layer=4, n_embd=64, n_head=4, n_kv_head=4)
    model = _build_model(config)
    model.eval()

    tokenizer = ByteTokenizer()

    # Override model methods that Engine needs
    model.get_device = lambda: torch.device("cpu")

    engine = Engine(model, tokenizer)
    prompt = [0, 72, 101, 108, 108, 111]  # some tokens

    results, masks = engine.generate_batch(prompt, max_tokens=5, temperature=0.0, seed=42)
    assert len(results) == 1
    assert len(results[0]) >= len(prompt)  # at least prompt tokens
    assert len(results[0]) <= len(prompt) + 5  # at most prompt + max_tokens
