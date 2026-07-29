"""
Test that FLOPs accounting is invariant to FP8 conversion.

convert_to_float8_training() replaces nn.Linear submodules with Float8Linear.
Because Float8Linear subclasses nn.Linear (and not nanochat's own Linear), an
isinstance() check against the narrower class silently stops matching after
conversion, zeroing the matmul term of estimate_flops().

Runs on CPU: the conversion is pure module surgery, no FP8 kernel is launched.

python -m pytest tests/test_flops_fp8.py -v
"""

import torch.nn as nn

from nanochat.gpt import GPT, GPTConfig
from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training


def _build_small_model():
    config = GPTConfig(
        sequence_len=128, vocab_size=256, n_layer=2,
        n_head=2, n_kv_head=2, n_embd=256, window_pattern="L",
    )
    model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def _fp8_filter(mod, fqn):
    """Mirrors the filter in scripts/base_train.py."""
    if not isinstance(mod, nn.Linear):
        return False
    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
        return False
    return min(mod.in_features, mod.out_features) >= 128


def test_flops_invariant_to_fp8_conversion():
    model = _build_small_model()
    params_before = model.num_matmul_params()
    flops_before = model.estimate_flops()

    convert_to_float8_training(
        model,
        config=Float8LinearConfig.from_recipe_name("tensorwise"),
        module_filter_fn=_fp8_filter,
    )
    assert any("Float8" in type(m).__name__ for m in model.modules()), "no modules were converted"

    assert model.num_matmul_params() == params_before, (
        f"matmul params changed: {params_before:,} -> {model.num_matmul_params():,}")
    assert model.estimate_flops() == flops_before
