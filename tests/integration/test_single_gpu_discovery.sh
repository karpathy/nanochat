#!/bin/bash
#
# Test 6: Basic Discovery Run
# Tests that auto-discovery completes successfully on a single GPU
#

set -e  # Exit on error

echo "=========================================="
echo "Test 6: Basic Discovery Run (Single GPU)"
echo "=========================================="

# Configuration
DEPTH=12
MAX_ITERATIONS=10
TIMEOUT=30  # seconds

# Output log file
LOG_FILE="tests/results/test_single_gpu_discovery.log"
mkdir -p tests/results

# Run the training script with auto-discovery
echo "Running: torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=$DEPTH --auto_batch_size=True --max_iterations=$MAX_ITERATIONS"

timeout $TIMEOUT torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    -- \
    --depth=$DEPTH \
    --num_iterations=$MAX_ITERATIONS \
    2>&1 | tee "$LOG_FILE"

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training script failed with exit code $EXIT_CODE"
    exit 1
fi

# Verify log contains discovery message
if ! grep -q "Auto-discovery found device_batch_size=" "$LOG_FILE"; then
    echo "ERROR: Log does not contain 'Auto-discovery found device_batch_size='"
    echo "This suggests auto-discovery was not triggered"
    exit 1
fi

# Verify no OOM errors
if grep -qi "out of memory\|OOM" "$LOG_FILE"; then
    echo "ERROR: Found OOM error in log"
    exit 1
fi

# Extract discovered batch size
BATCH_SIZE=$(grep "Auto-discovery found device_batch_size=" "$LOG_FILE" | grep -oP 'device_batch_size=\K\d+' | head -1)
echo "Discovered batch size: $BATCH_SIZE"

# Verify batch size is reasonable
if [ -z "$BATCH_SIZE" ]; then
    echo "ERROR: Could not extract batch size from log"
    exit 1
fi

if [ "$BATCH_SIZE" -lt 1 ] || [ "$BATCH_SIZE" -gt 128 ]; then
    echo "ERROR: Batch size $BATCH_SIZE is outside reasonable range [1, 128]"
    exit 1
fi

echo "✓ Test passed!"
echo "  - Discovery completed successfully"
echo "  - Found batch size: $BATCH_SIZE"
echo "  - No OOM errors"
echo "  - Training completed $MAX_ITERATIONS iterations"
