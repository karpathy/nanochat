# RTX 3070 (12GB) optimized configuration
# Smaller model (d16, 390M params) with hybrid architecture
# Tuned for consumer GPU memory constraints

# Model architecture
depth = 16
block_pattern = ["T"] * 10 + ["M"] * 6  # Early transformer, late Mamba

# Mamba-specific parameters
mamba_d_state = 16      # Conservative for memory
mamba_d_conv = 4
mamba_expand = 2
mamba_use_mlp = False

# Training - optimized for 12GB VRAM
max_seq_len = 1024       # Reduced from 2048
device_batch_size = 4    # Safe for 12GB
total_batch_size = 524288
target_param_data_ratio = 20

# Optimization
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0
grad_clip = 1.0

# Notes:
# - This should fit comfortably on 12GB
# - If still OOM, reduce device_batch_size to 2
# - Can increase to d20 if you reduce device_batch_size to 2

