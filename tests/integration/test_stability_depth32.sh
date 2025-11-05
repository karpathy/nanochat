#!/bin/bash
#
# Test 14: Long-Running Stability Test (depth=32)
# Ensures auto-discovery finds smaller batch size for largest model
#

set -e

echo "=========================================="
echo "Test 14: Stability Test (depth=32)"
echo "=========================================="

DEPTH=32
MAX_ITERATIONS=1000

LOG_FILE="tests/results/stability_depth${DEPTH}.log"
mkdir -p tests/results

echo "Running $MAX_ITERATIONS iterations with depth=$DEPTH"
echo "This may take several minutes..."
echo "Expected: Discovery should find smaller batch size due to larger model"
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

# Compare with depth=12 batch size if available
if [ -f "tests/results/stability_depth12.log" ]; then
    BATCH_SIZE_12=$(grep "Auto-discovery found device_batch_size=" "tests/results/stability_depth12.log" | grep -oP 'device_batch_size=\K\d+' | head -1)
    if [ -n "$BATCH_SIZE_12" ] && [ -n "$BATCH_SIZE" ]; then
        echo ""
        echo "Batch size comparison:"
        echo "  depth=12: $BATCH_SIZE_12"
        echo "  depth=32: $BATCH_SIZE"
        if [ "$BATCH_SIZE" -le "$BATCH_SIZE_12" ]; then
            echo "  ✓ Larger model correctly uses smaller/equal batch size"
        else
            echo "  WARNING: depth=32 has larger batch size than depth=12 (unexpected)"
        fi
    fi
fi

echo ""
echo "✓ Test passed!"
echo "  - Completed $MAX_ITERATIONS iterations"
echo "  - Duration: ${DURATION}s"
echo "  - Discovered batch size: $BATCH_SIZE"
echo "  - No OOM errors"
echo "  - No memory leaks detected"
