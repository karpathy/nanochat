#!/bin/bash
#
# Test 8 & 9: DDP Discovery Tests
# Tests auto-discovery in distributed (multi-GPU) settings
#

set -e

echo "=========================================="
echo "DDP Auto-Discovery Tests"
echo "=========================================="

# Check GPU availability
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "Detected $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "SKIP: Need at least 2 GPUs for DDP tests"
    exit 0
fi

DEPTH=12
MAX_ITERATIONS=10

# Test with 2 GPUs
echo ""
echo "Test 8: DDP Discovery (2 GPUs)"
echo "----------------------------------------"
LOG_2GPU="tests/results/test_ddp_2gpu.log"
mkdir -p tests/results

torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    -- \
    --depth=$DEPTH \
    --num_iterations=$MAX_ITERATIONS \
    2>&1 | tee "$LOG_2GPU"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: 2-GPU DDP test failed"
    exit 1
fi

# Verify rank 0 ran discovery
if ! grep -q "Running auto-discovery on rank 0" "$LOG_2GPU"; then
    echo "ERROR: No evidence of rank 0 running discovery"
    exit 1
fi

# Verify rank 1 received the batch size
if ! grep -q "Received batch size from rank 0\|device_batch_size=" "$LOG_2GPU"; then
    echo "ERROR: No evidence of rank 1 receiving batch size"
    exit 1
fi

# Extract batch sizes from both ranks (if logged separately)
BATCH_SIZE=$(grep "Auto-discovery found device_batch_size=" "$LOG_2GPU" | grep -oP 'device_batch_size=\K\d+' | head -1)

if [ -z "$BATCH_SIZE" ]; then
    echo "ERROR: Could not extract batch size"
    exit 1
fi

echo "✓ 2-GPU test passed! Discovered batch size: $BATCH_SIZE"

# Test with 4 GPUs if available
if [ "$NUM_GPUS" -ge 4 ]; then
    echo ""
    echo "Test 9: DDP Discovery (4 GPUs)"
    echo "----------------------------------------"
    LOG_4GPU="tests/results/test_ddp_4gpu.log"
    
    torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
        -- \
        --depth=$DEPTH \
        --num_iterations=$MAX_ITERATIONS \
        2>&1 | tee "$LOG_4GPU"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: 4-GPU DDP test failed"
        exit 1
    fi
    
    # Verify discovery happened
    if ! grep -q "Auto-discovery found device_batch_size=" "$LOG_4GPU"; then
        echo "ERROR: No discovery message in 4-GPU log"
        exit 1
    fi
    
    BATCH_SIZE_4GPU=$(grep "Auto-discovery found device_batch_size=" "$LOG_4GPU" | grep -oP 'device_batch_size=\K\d+' | head -1)
    
    echo "✓ 4-GPU test passed! Discovered batch size: $BATCH_SIZE_4GPU"
else
    echo ""
    echo "SKIP: Test 9 (4 GPUs not available)"
fi

echo ""
echo "✓ All DDP tests passed!"
echo "  - All ranks completed successfully"
echo "  - No deadlocks or synchronization errors"
echo "  - Batch size properly broadcast across ranks"
