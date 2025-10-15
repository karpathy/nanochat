# Pure Mamba configuration for d20 model (561M parameters)
# This replaces all transformer blocks with Mamba SSM blocks
# Expected benefits: faster training/inference, lower memory, better long-range modeling

# Model architecture
depth = 20
block_pattern = ["M"] * 20  # All Mamba blocks

# Mamba-specific parameters
mamba_d_state = 16      # Conservative state dimension for 12GB GPUs
mamba_d_conv = 4         # Standard convolution kernel
mamba_expand = 2         # Standard expansion factor
mamba_use_mlp = False    # Mamba has built-in gating, MLP often redundant

# Training (same as base_train.py defaults)
max_seq_len = 2048
device_batch_size = 32   # Can potentially use more since no attention overhead
total_batch_size = 524288
target_param_data_ratio = 20  # Chinchilla ratio

# Optimization
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0
grad_clip = 1.0

# For 12GB GPUs, use:
# device_batch_size = 4
# max_seq_len = 1024

