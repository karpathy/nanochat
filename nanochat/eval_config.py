"""
Configuration for evaluation memory management and performance tuning.

These settings control memory cleanup intervals and other evaluation parameters
to prevent memory leaks and progressive slowdown during long-running evaluations.
"""

# Memory Management Settings
# ---------------------------

# Periodic cache cleanup interval (in examples processed)
# After processing this many examples, trigger torch.cuda.empty_cache() and gc.collect()
# to prevent memory fragmentation and progressive slowdown.
#
# Rationale for 256:
# - Balances cleanup overhead (~10-50ms per cleanup) vs memory accumulation
# - Power of 2 (efficient modulo operation)
# - For HellaSwag (10,000 examples): 39 cleanups total (~2s overhead)
# - For MMLU (100-1000 examples): 0-4 cleanups total (negligible overhead)
#
# Lower values (e.g., 100): More frequent cleanup, less fragmentation, higher overhead
# Higher values (e.g., 512): Less overhead, more fragmentation risk
CACHE_CLEANUP_INTERVAL = 256

# Enable periodic cache cleanup during evaluation
# Set to False to disable all periodic cleanup (not recommended for long evaluations)
ENABLE_PERIODIC_CLEANUP = True

# Enable final cleanup after task completes
# Set to False to skip final cleanup (saves ~50ms but leaves memory cached)
ENABLE_FINAL_CLEANUP = True
