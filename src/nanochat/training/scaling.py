"""Scaling law computations for training hyperparameter derivation."""

import math
from dataclasses import dataclass

import torch

B_REF = 2**19  # optimal batch size at d12 ~= 524,288 tokens (measured empirically)


def get_scaling_params(model: torch.nn.Module) -> int:
    """Return the scaling-law parameter count (transformer matrices + lm_head)."""
    counts = model.num_scaling_params()
    return counts["transformer_matrices"] + counts["lm_head"]


@dataclass
class TrainingHyperparams:
    """Derived hyperparameters from scaling law computations."""

    num_scaling_params: int
    target_tokens: int
    total_batch_size: int
    batch_lr_scale: float
    weight_decay_scaled: float


def compute_training_hyperparams(
    target_param_data_ratio: float,
    total_batch_size_override: int,
    weight_decay: float,
    num_scaling_params: int,
    d12_scaling_params: int,
) -> TrainingHyperparams:
    """Derive optimal batch size, LR scale, and weight decay from scaling laws.

    Steps:
      1. target_tokens = target_param_data_ratio * num_scaling_params
      2. total_batch_size: use override if != -1, else compute via Power Lines (D^0.383)
      3. batch_lr_scale: sqrt scaling η ∝ √(B/B_ref)
      4. weight_decay_scaled: T_epoch framework λ ∝ √(B/B_ref) * (D_ref/D)
    """
    D_REF = target_param_data_ratio * d12_scaling_params
    target_tokens = int(target_param_data_ratio * num_scaling_params)

    # 2) Optimal batch size
    if total_batch_size_override != -1:
        total_batch_size = total_batch_size_override
    else:
        batch_size_ratio = target_tokens / D_REF
        predicted_batch_size = B_REF * batch_size_ratio**0.383
        total_batch_size = 2 ** round(math.log2(predicted_batch_size))

    # 3) LR scale: η ∝ √(B/B_ref)
    batch_ratio = total_batch_size / B_REF
    batch_lr_scale = batch_ratio**0.5

    # 4) Weight decay: λ = λ_ref · √(B/B_ref) · (D_ref/D)
    weight_decay_scaled = weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)

    return TrainingHyperparams(
        num_scaling_params=num_scaling_params,
        target_tokens=target_tokens,
        total_batch_size=total_batch_size,
        batch_lr_scale=batch_lr_scale,
        weight_decay_scaled=weight_decay_scaled,
    )
