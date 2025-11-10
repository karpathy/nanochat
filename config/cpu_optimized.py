# CPU-optimized configuration for i5 processors
# This config prioritizes training stability over speed

# Model sizing - smaller architecture for CPU
d_model = 256          # Reduced from typical 512/768
n_layers = 6           # Fewer layers for CPU efficiency
n_heads = 4            # Fewer attention heads
n_kv_heads = 2         # Reduced KV heads for memory efficiency
vocab_size = 50257     # Keep original vocabulary

# Training parameters optimized for CPU
device_batch_size = 1  # Single sample per batch to reduce memory
target_examples_per_step = 4  # Smaller effective batch size
grad_accum_steps = 4   # Fewer accumulation steps

# Sequence length - shorter for CPU
max_seq_len = 512       # Reduced sequence length
rotary_seq_len = 1024   # Match max sequence length

# Training duration - realistic for CPU
num_iterations = 100  # Much shorter training run
num_epochs = 1        # Single epoch for testing
eval_every = 20        # More frequent evaluation
eval_steps = 10        # Shorter evaluation

# Learning rate - more conservative for CPU stability
learning_rate = 1e-4    # Lower learning rate
init_lr_frac = 0.1      # Gradual warmup
weight_decay = 0.01     # Standard regularization

# Memory optimization
use_flash_attention = False  # Not available on CPU
use_gradient_checkpointing = True  # Trade compute for memory

# Consciousness features - optional on CPU
enable_consciousness = True  # Keep core feature
consciousness_weight = 0.1   # Reduce consciousness loss weight

# Data mixture - smaller datasets for CPU
train_data_mixture = [
    "SimpleSpelling(size=50)",     # Very small spelling tasks
    "SpellingBee(size=50)",        # Small spelling bee
    "SmolTalk(stop=100)",         # Limited conversation data
]  # Total ~200 samples for quick CPU testing

# Validation - minimal for CPU
val_dataset = "SmolTalk(split='test', stop=50)"  # Just 50 validation samples

# Checkpointing - frequent saves for CPU
save_every = 50         # Save every 50 steps
keep_checkpoints = 3    # Keep only recent checkpoints

# Logging - reduced for CPU
wandb_project = "nanochat-cpu"  # Separate CPU project
log_every = 10          # Log every 10 steps
print_every = 5         # Print every 5 steps

# Performance monitoring
monitor_memory = True   # Track memory usage
max_memory_gb = 4       # Stay under 4GB RAM usage

# Thermal protection
cpu_throttle_temp = 80  # Pause if CPU hits 80°C
pause_on_high_temp = True