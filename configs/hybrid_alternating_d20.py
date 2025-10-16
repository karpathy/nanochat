# Hybrid configuration: Alternating Transformer and Mamba (d20 model)
# Strategy: Interleave attention and SSM for balanced local/global processing
# Expected: Good general-purpose hybrid model

# Model architecture
depth = 20
block_pattern = ["T", "M"] * 10  # 50-50 split, alternating

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

