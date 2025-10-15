# Pure Transformer configuration for d20 model (BASELINE)
# This is the default nanochat architecture
# Use this as baseline for comparing hybrid/Mamba models

# Model architecture
depth = 20
block_pattern = None  # None = all transformer (backward compatible)
# Or explicitly: block_pattern = ["T"] * 20

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

