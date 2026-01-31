"""
Test merge_model.py using real GPT models initialized like init_random_model.py.

Uses depth=2 (tiny model) to keep tests fast while exercising real model state dicts.

We create 3 checkpoints with different random seeds, then verify:
1. Simple average: manually compute (sd0 + sd1 + sd2) / 3 and compare
2. EMA: manually compute the recurrence and compare
3. WMA: manually compute the weighted sum and compare
4. Merged state dict loads correctly into a GPT model
"""

import os
import shutil
import tempfile

import torch

from nanochat.gpt import GPT, GPTConfig
from scripts.merge_model import (
    get_sorted_checkpoints,
    select_checkpoints,
    merge_simple_average,
    merge_ema,
    merge_wma,
    load_state_dict,
)


# Small model config for testing
DEPTH = 2
MODEL_CONFIG_KWARGS = dict(
    sequence_len=128,
    vocab_size=512,
    n_layer=DEPTH,
    n_head=max(1, (DEPTH * 64 + 127) // 128),
    n_kv_head=max(1, (DEPTH * 64 + 127) // 128),
    n_embd=DEPTH * 64,
)


def create_random_model(seed: int) -> dict:
    """Initialize a real GPT model with a given seed and return its state dict."""
    torch.manual_seed(seed)
    with torch.device("meta"):
        config = GPTConfig(**MODEL_CONFIG_KWARGS)
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    sd = model.state_dict()
    return sd


def setup_real_model_dir(seeds: list[int]) -> str:
    """Create a temp dir with real model checkpoints, one per seed."""
    tmpdir = tempfile.mkdtemp()
    for step, seed in enumerate(seeds):
        sd = create_random_model(seed)
        path = os.path.join(tmpdir, f"model_{step:06d}.pt")
        torch.save(sd, path)
    return tmpdir


def max_abs_diff(sd1: dict, sd2: dict) -> float:
    """Compute the maximum absolute difference across all parameters."""
    max_diff = 0.0
    for key in sd1:
        diff = (sd1[key].float() - sd2[key].float()).abs().max().item()
        if diff > max_diff:
            max_diff = diff
    return max_diff


# =============================================================================
# Tests
# =============================================================================

def test_simple_average_real_model():
    """
    Create 3 real models, merge with simple average, and verify against
    manually computed (sd0 + sd1 + sd2) / 3.
    """
    print("=" * 60)
    print("TEST: Simple average with real GPT models")
    print("=" * 60)

    tmpdir = setup_real_model_dir(seeds=[42, 123, 999])
    try:
        # Load all 3 state dicts manually
        sd0 = load_state_dict(os.path.join(tmpdir, "model_000000.pt"), "cpu")
        sd1 = load_state_dict(os.path.join(tmpdir, "model_000001.pt"), "cpu")
        sd2 = load_state_dict(os.path.join(tmpdir, "model_000002.pt"), "cpu")

        # Compute expected: (sd0 + sd1 + sd2) / 3
        expected = {}
        for key in sd0:
            expected[key] = (sd0[key].float() + sd1[key].float() + sd2[key].float()) / 3.0

        # Run merge (newest first: sd2, sd1, sd0)
        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]
        merged = merge_simple_average(paths, device="cpu")

        diff = max_abs_diff(merged, expected)
        assert diff < 1e-5, f"Max abs diff = {diff}, expected < 1e-5"
        print(f"  Max abs diff from manual computation: {diff:.2e}")
        print("[PASS] Simple average matches manual computation")
    finally:
        shutil.rmtree(tmpdir)


def test_ema_real_model():
    """
    Create 3 real models, merge with EMA (alpha=0.6), and verify against
    manually computed recurrence.

    Checkpoints newest first: [sd2, sd1, sd0]
    Reversed for EMA (oldest to newest): [sd0, sd1, sd2]
      m_avg = sd0
      m_avg = 0.6 * sd1 + 0.4 * sd0
      m_avg = 0.6 * sd2 + 0.4 * (0.6 * sd1 + 0.4 * sd0)
    """
    print("\n" + "=" * 60)
    print("TEST: EMA with real GPT models")
    print("=" * 60)

    alpha = 0.6
    tmpdir = setup_real_model_dir(seeds=[42, 123, 999])
    try:
        sd0 = load_state_dict(os.path.join(tmpdir, "model_000000.pt"), "cpu")
        sd1 = load_state_dict(os.path.join(tmpdir, "model_000001.pt"), "cpu")
        sd2 = load_state_dict(os.path.join(tmpdir, "model_000002.pt"), "cpu")

        # Manual EMA computation (oldest to newest: sd0, sd1, sd2)
        expected = {}
        for key in sd0:
            m = sd0[key].float()
            m = alpha * sd1[key].float() + (1 - alpha) * m
            m = alpha * sd2[key].float() + (1 - alpha) * m
            expected[key] = m

        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]
        merged = merge_ema(paths, alpha=alpha, device="cpu")

        diff = max_abs_diff(merged, expected)
        assert diff < 1e-5, f"Max abs diff = {diff}, expected < 1e-5"
        print(f"  Max abs diff from manual computation: {diff:.2e}")
        print("[PASS] EMA matches manual computation")
    finally:
        shutil.rmtree(tmpdir)


def test_wma_real_model():
    """
    Create 3 real models, merge with WMA (alpha=0.5), and verify against
    manually computed weighted sum.

    Oldest to newest: [sd0, sd1, sd2], n=3
      w_1 = 0.5 * 0.5^2 = 0.125
      w_2 = 0.5 * 0.5^1 = 0.25
      w_3 = 0.5 * 0.5^0 = 0.5
      total = 0.875
      normalized: [0.142857, 0.285714, 0.571429]
    """
    print("\n" + "=" * 60)
    print("TEST: WMA with real GPT models")
    print("=" * 60)

    alpha = 0.5
    tmpdir = setup_real_model_dir(seeds=[42, 123, 999])
    try:
        sd0 = load_state_dict(os.path.join(tmpdir, "model_000000.pt"), "cpu")
        sd1 = load_state_dict(os.path.join(tmpdir, "model_000001.pt"), "cpu")
        sd2 = load_state_dict(os.path.join(tmpdir, "model_000002.pt"), "cpu")

        # Compute weights (oldest=index1, newest=index3)
        n = 3
        raw_weights = [alpha * ((1 - alpha) ** (n - i)) for i in range(1, n + 1)]
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights]
        print(f"  Normalized weights: {[f'{w:.6f}' for w in weights]}")

        # Manual weighted sum
        sds = [sd0, sd1, sd2]  # oldest to newest
        expected = {}
        for key in sd0:
            val = torch.zeros_like(sd0[key].float())
            for sd, w in zip(sds, weights):
                val += w * sd[key].float()
            expected[key] = val

        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]
        merged = merge_wma(paths, alpha=alpha, device="cpu")

        diff = max_abs_diff(merged, expected)
        assert diff < 1e-5, f"Max abs diff = {diff}, expected < 1e-5"
        print(f"  Max abs diff from manual computation: {diff:.2e}")
        print("[PASS] WMA matches manual computation")
    finally:
        shutil.rmtree(tmpdir)


def test_merged_model_loads():
    """
    Verify that a merged state dict can be loaded back into a real GPT model
    and produce a forward pass without errors.
    """
    print("\n" + "=" * 60)
    print("TEST: Merged model loads into GPT and runs forward pass")
    print("=" * 60)

    tmpdir = setup_real_model_dir(seeds=[42, 123, 999])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]
        merged_sd = merge_simple_average(paths, device="cpu")

        # Build a fresh model and load merged weights
        with torch.device("meta"):
            config = GPTConfig(**MODEL_CONFIG_KWARGS)
            model = GPT(config)
        model.to_empty(device="cpu")
        model.init_weights()
        model.load_state_dict(merged_sd, strict=True, assign=True)
        model.eval()

        # Run a forward pass
        seq_len = MODEL_CONFIG_KWARGS["sequence_len"]
        dummy_input = torch.randint(0, MODEL_CONFIG_KWARGS["vocab_size"], (1, seq_len))
        dummy_target = torch.randint(0, MODEL_CONFIG_KWARGS["vocab_size"], (1, seq_len))

        with torch.no_grad():
            loss = model(dummy_input, dummy_target)

        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
        print(f"  Forward pass loss: {loss.item():.4f}")
        print("[PASS] Merged model loads and runs forward pass successfully")
    finally:
        shutil.rmtree(tmpdir)


def test_different_seeds_produce_different_models():
    """Sanity check: models with different seeds should have different weights."""
    print("\n" + "=" * 60)
    print("TEST: Different seeds produce different models")
    print("=" * 60)

    sd1 = create_random_model(seed=42)
    sd2 = create_random_model(seed=123)

    diff = max_abs_diff(sd1, sd2)
    assert diff > 0.01, f"Models are too similar (max diff = {diff})"
    print(f"  Max abs diff between seed=42 and seed=123: {diff:.4f}")
    print("[PASS] Different seeds produce meaningfully different weights")


def test_same_seed_produces_same_model():
    """Sanity check: same seed should produce identical weights."""
    print("\n" + "=" * 60)
    print("TEST: Same seed produces identical model")
    print("=" * 60)

    sd1 = create_random_model(seed=42)
    sd2 = create_random_model(seed=42)

    diff = max_abs_diff(sd1, sd2)
    assert diff == 0.0, f"Models differ (max diff = {diff})"
    print(f"  Max abs diff between two seed=42 models: {diff:.2e}")
    print("[PASS] Same seed is deterministic")


if __name__ == "__main__":
    # Sanity checks
    test_different_seeds_produce_different_models()
    test_same_seed_produces_same_model()

    # Merge method correctness
    test_simple_average_real_model()
    test_ema_real_model()
    test_wma_real_model()

    # Integration: load merged model and run forward pass
    test_merged_model_loads()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
