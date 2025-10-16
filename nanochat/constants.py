"""
Constants used throughout the nanochat codebase.
Centralizes magic numbers for better maintainability.
"""

# Model Architecture Constants
MODEL_ASPECT_RATIO = 64  # model_dim = depth * aspect_ratio
HEAD_DIM_TARGET = 128  # Target dimension per attention head
ROTARY_CACHE_MULTIPLIER = 10  # Precompute rotary embeddings for N times sequence length
LOGIT_SOFTCAP = 15  # Soft capping value for logits using tanh (must be > 0)

# Memory Management
KV_CACHE_GROWTH_CHUNK = 1024  # Grow KV cache in chunks of this size (must be power of 2 for efficient bitwise rounding)

# Training Constants
DEFAULT_WARMUP_RATIO = 0.0  # Fraction of training for learning rate warmup
DEFAULT_WARMDOWN_RATIO = 0.2  # Fraction of training for learning rate warmdown
DEFAULT_FINAL_LR_FRAC = 0.0  # Final LR as fraction of initial LR
MUON_MOMENTUM_RAMPUP_STEPS = 300  # Steps to ramp up Muon momentum
MUON_MOMENTUM_START = 0.85  # Starting momentum for Muon optimizer
MUON_MOMENTUM_END = 0.95  # Final momentum for Muon optimizer

# Evaluation Constants
LOSS_EMA_BETA = 0.9  # Exponential moving average decay for training loss
WARMUP_IGNORE_STEPS = 10  # Ignore first N steps when calculating training time

# Calculator Tool Constants
CALCULATOR_TIMEOUT_SECONDS = 3  # Maximum time for calculator evaluation
