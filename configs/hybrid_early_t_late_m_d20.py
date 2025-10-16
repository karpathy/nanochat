# Hybrid configuration: Early Transformer, Late Mamba (d20 model)
# Strategy: Use transformers for token-level patterns, Mamba for long-range dependencies
# Expected: Best balance of attention power and SSM efficiency

# Model architecture
depth = 20
block_pattern = ["T"] * 12 + ["M"] * 8  # 60% transformer, 40% Mamba

# Mamba-specific parameters
mamba_d_state = 16
mamba_d_conv = 4
mamba_expand = 2
mamba_use_mlp = False

# Training (same as base_train.py defaults)
max_seq_len = 2048
device_batch_size = 32
total_batch_size = 524288
target_param_data_ratio = 20

# Optimization
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0
grad_clip = 1.0

# For 12GB GPUs, use:
# device_batch_size = 4
# max_seq_len = 1024

