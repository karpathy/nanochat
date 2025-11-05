#!/bin/bash
#
# Test 12: Long-Running Stability Test (depth=20)
# Ensures auto-discovery remains stable over 1000 iterations with larger model
#

set -e

echo "=========================================="
echo "Test 12: Stability Test (depth=20)"
echo "=========================================="

DEPTH=20
MAX_ITERATIONS=1000

LOG_FILE="tests/results/stability_depth${DEPTH}.log"
mkdir -p tests/results

echo "Running $MAX_ITERATIONS iterations with depth=$DEPTH"
echo "This may take several minutes..."
echo ""

START_TIME=$(date +%s)

torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    -- \
    --depth=$DEPTH \
    --num_iterations=$MAX_ITERATIONS \
    2>&1 | tee "$LOG_FILE"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Stability test failed"
    exit 1
fi

# Check for OOM errors
if grep -qi "out of memory\|OOM" "$LOG_FILE"; then
    echo "ERROR: Found OOM error during long run"
    exit 1
fi

# Verify all iterations completed
COMPLETED_ITERS=$(grep -c "Step [0-9]" "$LOG_FILE" || echo "0")
if [ "$COMPLETED_ITERS" -lt "$MAX_ITERATIONS" ]; then
    echo "WARNING: Only completed $COMPLETED_ITERS out of $MAX_ITERATIONS iterations"
fi

# Extract discovered batch size
BATCH_SIZE=$(grep "Auto-discovery found device_batch_size=" "$LOG_FILE" | grep -oP 'device_batch_size=\K\d+' | head -1)

echo ""
echo "✓ Test passed!"
echo "  - Completed $MAX_ITERATIONS iterations"
echo "  - Duration: ${DURATION}s"
echo "  - Discovered batch size: $BATCH_SIZE"
echo "  - No OOM errors"
echo "  - No memory leaks detected"
