"""
Test merge_model.py by creating checkpoints with known constant weights
and verifying the merge math produces expected results.

Setup: 3 checkpoints where all parameters are set to constant values:
  - model_000000.pt: all weights = 1.0
  - model_000001.pt: all weights = 2.0
  - model_000002.pt: all weights = 3.0
"""

import os
import shutil
import tempfile
import torch

from scripts.merge_model import (
    get_sorted_checkpoints,
    select_checkpoints,
    merge_simple_average,
    merge_ema,
    merge_wma,
    load_state_dict,
    InsufficientCheckpointsError,
)


def create_constant_state_dict(value: float) -> dict:
    """Create a small fake state dict where every tensor is filled with `value`."""
    return {
        "transformer.wte.weight": torch.full((100, 64), value),
        "transformer.h.0.attn.c_q.weight": torch.full((64, 64), value),
        "transformer.h.0.attn.c_k.weight": torch.full((64, 64), value),
        "transformer.h.0.mlp.c_fc.weight": torch.full((256, 64), value),
        "lm_head.weight": torch.full((100, 64), value),
    }


def setup_test_dir(values: list[float]) -> str:
    """Create a temp dir with checkpoints, one per value, at steps 0, 1, 2, ..."""
    tmpdir = tempfile.mkdtemp()
    for step, val in enumerate(values):
        sd = create_constant_state_dict(val)
        path = os.path.join(tmpdir, f"model_{step:06d}.pt")
        torch.save(sd, path)
    return tmpdir


def setup_test_dir_with_steps(step_value_pairs: list[tuple[int, float]]) -> str:
    """Create a temp dir with checkpoints at specific step numbers."""
    tmpdir = tempfile.mkdtemp()
    for step, val in step_value_pairs:
        sd = create_constant_state_dict(val)
        path = os.path.join(tmpdir, f"model_{step:06d}.pt")
        torch.save(sd, path)
    return tmpdir


def assert_all_values_close(sd: dict, expected: float, tol: float = 1e-5):
    """Assert every value in every tensor in the state dict is close to `expected`."""
    for key, tensor in sd.items():
        actual = tensor.float().mean().item()
        assert abs(actual - expected) < tol, (
            f"Key '{key}': expected {expected}, got {actual}"
        )


# =============================================================================
# Checkpoint Discovery Tests
# =============================================================================

def test_get_sorted_checkpoints():
    tmpdir = setup_test_dir([1.0, 2.0, 3.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)

        assert len(checkpoints) == 3
        # Should be sorted descending by step
        assert checkpoints[0][1] == 2  # step 2 (newest)
        assert checkpoints[1][1] == 1  # step 1
        assert checkpoints[2][1] == 0  # step 0 (oldest)
        print("[PASS] get_sorted_checkpoints: correct ordering")
    finally:
        shutil.rmtree(tmpdir)


def test_select_checkpoints():
    tmpdir = setup_test_dir([1.0, 2.0, 3.0, 4.0, 5.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)

        # Select 3 with step_size=1 -> steps 4, 3, 2
        selected = select_checkpoints(checkpoints, num_models=3, step_size=1)
        assert len(selected) == 3
        print("[PASS] select_checkpoints: step_size=1")

        # Select 2 with step_size=2 -> steps 4, 2
        selected = select_checkpoints(checkpoints, num_models=2, step_size=2)
        assert len(selected) == 2
        steps = [int(os.path.basename(p).replace("model_", "").replace(".pt", "")) for p in selected]
        assert steps == [4, 2], f"Expected [4, 2], got {steps}"
        print("[PASS] select_checkpoints: step_size=2")
    finally:
        shutil.rmtree(tmpdir)


# =============================================================================
# Simple Average Tests
# =============================================================================

def test_simple_average():
    """Average of [1, 2, 3] = 2.0"""
    tmpdir = setup_test_dir([1.0, 2.0, 3.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]  # [step2=3.0, step1=2.0, step0=1.0]

        merged = merge_simple_average(paths, device="cpu")
        assert_all_values_close(merged, expected=2.0)
        print("[PASS] simple_average: avg(1, 2, 3) = 2.0")
    finally:
        shutil.rmtree(tmpdir)


def test_simple_average_identical():
    """Average of [5, 5, 5] = 5.0"""
    tmpdir = setup_test_dir([5.0, 5.0, 5.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]

        merged = merge_simple_average(paths, device="cpu")
        assert_all_values_close(merged, expected=5.0)
        print("[PASS] simple_average: avg(5, 5, 5) = 5.0")
    finally:
        shutil.rmtree(tmpdir)


# =============================================================================
# EMA Tests
# =============================================================================

def test_ema():
    """
    EMA with alpha=0.5 on values [1, 2, 3] (oldest to newest):
      Checkpoints sorted newest first: [3.0 (step2), 2.0 (step1), 1.0 (step0)]
      Reversed for EMA (oldest to newest): [1.0, 2.0, 3.0]

      m_avg = 1.0                                (init with oldest)
      m_avg = 0.5 * 2.0 + 0.5 * 1.0 = 1.5       (blend in step1)
      m_avg = 0.5 * 3.0 + 0.5 * 1.5 = 2.25      (blend in step2)
    """
    tmpdir = setup_test_dir([1.0, 2.0, 3.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]  # [step2=3.0, step1=2.0, step0=1.0]

        merged = merge_ema(paths, alpha=0.5, device="cpu")
        assert_all_values_close(merged, expected=2.25)
        print("[PASS] ema(alpha=0.5): [1, 2, 3] -> 2.25")
    finally:
        shutil.rmtree(tmpdir)


def test_ema_alpha_1():
    """
    EMA with alpha=1.0 should just return the newest checkpoint.
      Reversed: [1.0, 2.0, 3.0]
      m_avg = 1.0
      m_avg = 1.0 * 2.0 + 0.0 * 1.0 = 2.0
      m_avg = 1.0 * 3.0 + 0.0 * 2.0 = 3.0
    """
    tmpdir = setup_test_dir([1.0, 2.0, 3.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]

        merged = merge_ema(paths, alpha=1.0, device="cpu")
        assert_all_values_close(merged, expected=3.0)
        print("[PASS] ema(alpha=1.0): returns newest = 3.0")
    finally:
        shutil.rmtree(tmpdir)


def test_ema_alpha_low():
    """
    EMA with very low alpha should heavily favor the oldest checkpoint.
      alpha=0.1, reversed: [1.0, 2.0, 3.0]
      m_avg = 1.0
      m_avg = 0.1 * 2.0 + 0.9 * 1.0 = 1.1
      m_avg = 0.1 * 3.0 + 0.9 * 1.1 = 0.3 + 0.99 = 1.29
    """
    tmpdir = setup_test_dir([1.0, 2.0, 3.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]

        merged = merge_ema(paths, alpha=0.1, device="cpu")
        assert_all_values_close(merged, expected=1.29)
        print("[PASS] ema(alpha=0.1): [1, 2, 3] -> 1.29")
    finally:
        shutil.rmtree(tmpdir)


# =============================================================================
# WMA Tests
# =============================================================================

def test_wma():
    """
    WMA with alpha=0.5 on values [1, 2, 3] (oldest to newest):
      Checkpoints sorted newest first: [3.0 (step2), 2.0 (step1), 1.0 (step0)]
      Reversed: [1.0, 2.0, 3.0], n=3

      Weights (i=1,2,3):
        w_1 = 0.5 * (0.5)^2 = 0.125  (oldest)
        w_2 = 0.5 * (0.5)^1 = 0.25
        w_3 = 0.5 * (0.5)^0 = 0.5    (newest)
      Total = 0.875
      Normalized: [0.125/0.875, 0.25/0.875, 0.5/0.875]
                = [0.142857, 0.285714, 0.571429]

      Result = 0.142857*1.0 + 0.285714*2.0 + 0.571429*3.0
             = 0.142857 + 0.571429 + 1.714286
             = 2.428571
    """
    tmpdir = setup_test_dir([1.0, 2.0, 3.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]

        merged = merge_wma(paths, alpha=0.5, device="cpu")
        assert_all_values_close(merged, expected=2.428571, tol=1e-4)
        print("[PASS] wma(alpha=0.5): [1, 2, 3] -> 2.4286")
    finally:
        shutil.rmtree(tmpdir)


def test_wma_alpha_1():
    """
    WMA with alpha=1.0: all weight on newest.
      w_i = 1.0 * 0.0^(n-i)
      Only w_n = 1.0 * 0^0 = 1.0, all others = 0.
      Result = newest = 3.0
    """
    tmpdir = setup_test_dir([1.0, 2.0, 3.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]

        merged = merge_wma(paths, alpha=1.0, device="cpu")
        assert_all_values_close(merged, expected=3.0)
        print("[PASS] wma(alpha=1.0): returns newest = 3.0")
    finally:
        shutil.rmtree(tmpdir)


def test_wma_four_models():
    """
    WMA with alpha=0.5 on [1, 2, 3, 4] (oldest to newest), n=4:
      w_1 = 0.5 * 0.5^3 = 0.0625
      w_2 = 0.5 * 0.5^2 = 0.125
      w_3 = 0.5 * 0.5^1 = 0.25
      w_4 = 0.5 * 0.5^0 = 0.5
      Total = 0.9375
      Normalized: [0.06667, 0.13333, 0.26667, 0.53333]

      Result = 0.06667*1 + 0.13333*2 + 0.26667*3 + 0.53333*4
             = 0.06667 + 0.26667 + 0.80000 + 2.13333
             = 3.26667
    """
    tmpdir = setup_test_dir([1.0, 2.0, 3.0, 4.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]

        merged = merge_wma(paths, alpha=0.5, device="cpu")
        assert_all_values_close(merged, expected=3.26667, tol=1e-4)
        print("[PASS] wma(alpha=0.5): [1, 2, 3, 4] -> 3.2667")
    finally:
        shutil.rmtree(tmpdir)


# =============================================================================
# Edge Case Tests
# =============================================================================

def test_two_models_simple():
    """Average of [10, 20] = 15.0"""
    tmpdir = setup_test_dir([10.0, 20.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        paths = [c[0] for c in checkpoints]

        merged = merge_simple_average(paths, device="cpu")
        assert_all_values_close(merged, expected=15.0)
        print("[PASS] simple_average: avg(10, 20) = 15.0")
    finally:
        shutil.rmtree(tmpdir)


def test_step_size_selection():
    """With 5 checkpoints [1,2,3,4,5] and step_size=2, selecting 3 gives steps 4,2,0 -> values 5,3,1."""
    tmpdir = setup_test_dir([1.0, 2.0, 3.0, 4.0, 5.0])
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        selected = select_checkpoints(checkpoints, num_models=3, step_size=2)

        # Load and verify we got values 5, 3, 1
        sd_newest = load_state_dict(selected[0], "cpu")
        sd_middle = load_state_dict(selected[1], "cpu")
        sd_oldest = load_state_dict(selected[2], "cpu")

        key = "lm_head.weight"
        assert_all_values_close({"k": sd_newest[key]}, expected=5.0)
        assert_all_values_close({"k": sd_middle[key]}, expected=3.0)
        assert_all_values_close({"k": sd_oldest[key]}, expected=1.0)
        print("[PASS] step_size=2: selects correct checkpoints (5, 3, 1)")

        # Simple average of [5, 3, 1] = 3.0
        merged = merge_simple_average(selected, device="cpu")
        assert_all_values_close(merged, expected=3.0)
        print("[PASS] simple_average with step_size=2: avg(5, 3, 1) = 3.0")
    finally:
        shutil.rmtree(tmpdir)


def test_step_size_not_multiple_of_interval():
    """step_size=75 with checkpoint interval=50 should raise InsufficientCheckpointsError."""
    # Checkpoints every 50 steps: 0, 50, 100, 150, 200, 250, 300
    pairs = [(s, float(s)) for s in range(0, 350, 50)]
    tmpdir = setup_test_dir_with_steps(pairs)
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)
        try:
            select_checkpoints(checkpoints, num_models=3, step_size=75)
            assert False, "Should have raised InsufficientCheckpointsError"
        except InsufficientCheckpointsError:
            pass
        print("[PASS] step_size not multiple of interval raises error")
    finally:
        shutil.rmtree(tmpdir)


def test_checkpoint_interval_50_merge_every_100():
    """Checkpoints saved every 50 steps, merge every 100 steps."""
    # Checkpoints at steps 0, 50, 100, 150, 200, 250, 300
    # Values increase so we can verify which ones were picked
    pairs = [(s, float(s)) for s in range(0, 350, 50)]
    tmpdir = setup_test_dir_with_steps(pairs)
    try:
        checkpoints = get_sorted_checkpoints(tmpdir)

        # Select 3 models with step_size=100 -> steps 300, 200, 100
        selected = select_checkpoints(checkpoints, num_models=3, step_size=100)
        assert len(selected) == 3

        key = "lm_head.weight"
        sd0 = load_state_dict(selected[0], "cpu")
        sd1 = load_state_dict(selected[1], "cpu")
        sd2 = load_state_dict(selected[2], "cpu")
        assert_all_values_close({"k": sd0[key]}, expected=300.0)
        assert_all_values_close({"k": sd1[key]}, expected=200.0)
        assert_all_values_close({"k": sd2[key]}, expected=100.0)
        print("[PASS] checkpoint interval=50, step_size=100: selects steps 300, 200, 100")

        # Simple average of [300, 200, 100] = 200.0
        merged = merge_simple_average(selected, device="cpu")
        assert_all_values_close(merged, expected=200.0)
        print("[PASS] simple_average with interval=50, step_size=100: avg = 200.0")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    # Discovery and selection
    test_get_sorted_checkpoints()
    test_select_checkpoints()

    # Simple average
    test_simple_average()
    test_simple_average_identical()

    # EMA
    test_ema()
    test_ema_alpha_1()
    test_ema_alpha_low()

    # WMA
    test_wma()
    test_wma_alpha_1()
    test_wma_four_models()

    # Edge cases
    test_two_models_simple()
    test_step_size_selection()

    # Step interval validation
    test_step_size_not_multiple_of_interval()
    test_checkpoint_interval_50_merge_every_100()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
