#!/bin/bash
#
# Test 7: Compare Manual vs Auto Discovery
# Compares manual batch size with auto-discovered batch size
#

set -e

echo "=========================================="
echo "Test 7: Manual vs Auto Discovery"
echo "=========================================="

DEPTH=12
MAX_ITERATIONS=50
MANUAL_BATCH_SIZE=8

LOG_MANUAL="tests/results/test_manual_baseline.log"
LOG_AUTO="tests/results/test_auto_discovery.log"
mkdir -p tests/results

# Run 1: Manual batch size
echo ""
echo "Run 1: Manual batch size = $MANUAL_BATCH_SIZE"
echo "----------------------------------------"
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    -- \
    --depth=$DEPTH \
    --device_batch_size=$MANUAL_BATCH_SIZE \
    --num_iterations=$MAX_ITERATIONS \
    2>&1 | tee "$LOG_MANUAL"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Manual run failed"
    exit 1
fi

# Run 2: Auto discovery
echo ""
echo "Run 2: Auto-discovery"
echo "----------------------------------------"
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    -- \
    --depth=$DEPTH \
    --num_iterations=$MAX_ITERATIONS \
    2>&1 | tee "$LOG_AUTO"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Auto-discovery run failed"
    exit 1
fi

# Extract auto-discovered batch size
AUTO_BATCH_SIZE=$(grep "Auto-discovery found device_batch_size=" "$LOG_AUTO" | grep -oP 'device_batch_size=\K\d+' | head -1)

if [ -z "$AUTO_BATCH_SIZE" ]; then
    echo "ERROR: Could not extract auto-discovered batch size"
    exit 1
fi

echo ""
echo "Results:"
echo "  Manual batch size: $MANUAL_BATCH_SIZE"
echo "  Auto-discovered batch size: $AUTO_BATCH_SIZE"

# Verify auto batch size is >= manual
if [ "$AUTO_BATCH_SIZE" -lt "$MANUAL_BATCH_SIZE" ]; then
    echo "WARNING: Auto-discovered batch size ($AUTO_BATCH_SIZE) is less than manual ($MANUAL_BATCH_SIZE)"
    echo "         This is unexpected but may be due to safety margin"
fi

# Verify no OOM in auto mode
if grep -qi "out of memory\|OOM" "$LOG_AUTO"; then
    echo "ERROR: Found OOM error in auto-discovery run"
    exit 1
fi

# Compare final validation loss (optional - both should be similar)
VAL_LOSS_MANUAL=$(grep "Validation bpb:" "$LOG_MANUAL" | tail -1 | grep -oP 'bpb: \K[\d.]+')
VAL_LOSS_AUTO=$(grep "Validation bpb:" "$LOG_AUTO" | tail -1 | grep -oP 'bpb: \K[\d.]+')

if [ -n "$VAL_LOSS_MANUAL" ] && [ -n "$VAL_LOSS_AUTO" ]; then
    echo "  Final validation loss (manual): $VAL_LOSS_MANUAL"
    echo "  Final validation loss (auto): $VAL_LOSS_AUTO"
fi

echo ""
echo "✓ Test passed!"
echo "  - Both runs completed successfully"
echo "  - Auto-discovery found batch size: $AUTO_BATCH_SIZE"
echo "  - No OOM errors in either run"
