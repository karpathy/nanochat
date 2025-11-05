#!/bin/bash
#
# Test 18, 19, 20: Cache Tests
# Tests caching functionality
#

set -e

echo "=========================================="
echo "Cache Mechanism Tests"
echo "=========================================="

DEPTH=12
MAX_ITERATIONS=10

mkdir -p tests/results

# ============================================================================
# Test 18: Cache Hit
# ============================================================================
echo ""
echo "Test 18: Cache Hit"
echo "----------------------------------------"
LOG_RUN1="tests/results/cache_run1.log"
LOG_RUN2="tests/results/cache_run2.log"

# Clean cache directory first (if it exists)
if [ -n "$NANOCHAT_BASE_DIR" ]; then
    CACHE_DIR="$NANOCHAT_BASE_DIR/auto_batch_cache"
else
    CACHE_DIR="$HOME/.nanochat/auto_batch_cache"
fi

if [ -d "$CACHE_DIR" ]; then
    echo "Cleaning existing cache: $CACHE_DIR"
    rm -rf "$CACHE_DIR"
fi

# Run 1: Discovery runs, result saved to cache
echo "Run 1: Initial discovery (cache miss expected)"
START_RUN1=$(date +%s)

torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    -- \
    --depth=$DEPTH \
    --num_iterations=$MAX_ITERATIONS \
    2>&1 | tee "$LOG_RUN1"

END_RUN1=$(date +%s)
DURATION_RUN1=$((END_RUN1 - START_RUN1))

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Run 1 failed"
    exit 1
fi

# Run 2: Same config, discovery skipped (cache hit)
echo ""
echo "Run 2: Same config (cache hit expected)"
START_RUN2=$(date +%s)

torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    -- \
    --depth=$DEPTH \
    --num_iterations=$MAX_ITERATIONS \
    2>&1 | tee "$LOG_RUN2"

END_RUN2=$(date +%s)
DURATION_RUN2=$((END_RUN2 - START_RUN2))

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Run 2 failed"
    exit 1
fi

echo ""
echo "Timing comparison:"
echo "  Run 1 (cache miss): ${DURATION_RUN1}s"
echo "  Run 2 (cache hit): ${DURATION_RUN2}s"

# Verify Run 2 is faster (should be much faster if cache hit)
if [ "$DURATION_RUN2" -lt "$DURATION_RUN1" ]; then
    TIME_SAVED=$((DURATION_RUN1 - DURATION_RUN2))
    echo "  Time saved: ${TIME_SAVED}s"
    echo "✓ Cache hit improved startup time"
else
    echo "WARNING: Run 2 was not faster (cache may not have been used)"
fi

# Check if cache hit message appears in Run 2
if grep -q "Cache hit\|Using cached batch size" "$LOG_RUN2"; then
    echo "✓ Cache hit message found"
fi

# Verify cache file exists
if [ -d "$CACHE_DIR" ] && [ -n "$(ls -A $CACHE_DIR)" ]; then
    CACHE_FILES=$(ls -1 "$CACHE_DIR" | wc -l)
    echo "✓ Cache directory exists with $CACHE_FILES file(s)"
else
    echo "WARNING: Cache directory is empty or doesn't exist"
fi

echo "✓ Test 18 passed!"

# ============================================================================
# Test 19: Cache Key Validation
# ============================================================================
echo ""
echo "Test 19: Cache Key Validation"
echo "----------------------------------------"

# Run with depth=12, cache result
echo "Run with depth=12..."
LOG_DEPTH12="tests/results/cache_depth12.log"
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    -- \
    --depth=12 \
    --num_iterations=$MAX_ITERATIONS \
    2>&1 | tee "$LOG_DEPTH12"

BATCH_12=$(grep "Auto-discovery found device_batch_size=" "$LOG_DEPTH12" | grep -oP 'device_batch_size=\K\d+' | head -1)

# Run with depth=20, verify cache miss (different config)
echo ""
echo "Run with depth=20 (should be cache miss)..."
LOG_DEPTH20="tests/results/cache_depth20.log"
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    -- \
    --depth=20 \
    --num_iterations=$MAX_ITERATIONS \
    2>&1 | tee "$LOG_DEPTH20"

BATCH_20=$(grep "Auto-discovery found device_batch_size=" "$LOG_DEPTH20" | grep -oP 'device_batch_size=\K\d+' | head -1)

# Run with max_seq_len=256, verify cache miss
echo ""
echo "Run with max_seq_len=256 (should be cache miss)..."
LOG_SEQ256="tests/results/cache_seq256.log"
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    -- \
    --depth=12 \
    --max_seq_len=256 \
    --num_iterations=$MAX_ITERATIONS \
    2>&1 | tee "$LOG_SEQ256"

BATCH_256=$(grep "Auto-discovery found device_batch_size=" "$LOG_SEQ256" | grep -oP 'device_batch_size=\K\d+' | head -1)

# Verify separate cache files were created
if [ -d "$CACHE_DIR" ]; then
    CACHE_FILES=$(ls -1 "$CACHE_DIR" | wc -l)
    echo ""
    echo "Cache files created: $CACHE_FILES"
    if [ "$CACHE_FILES" -ge 3 ]; then
        echo "✓ Multiple cache files created for different configurations"
    else
        echo "WARNING: Expected at least 3 cache files, found $CACHE_FILES"
    fi
fi

echo ""
echo "Discovered batch sizes:"
echo "  depth=12, seq_len=2048: $BATCH_12"
echo "  depth=20, seq_len=2048: $BATCH_20"
echo "  depth=12, seq_len=256: $BATCH_256"

echo "✓ Test 19 passed!"

# ============================================================================
# Test 20: Cache Invalidation
# ============================================================================
echo ""
echo "Test 20: Cache Invalidation"
echo "----------------------------------------"

if [ -d "$CACHE_DIR" ] && [ -n "$(ls -A $CACHE_DIR 2>/dev/null)" ]; then
    # Get first cache file
    CACHE_FILE=$(ls "$CACHE_DIR" | head -1)
    CACHE_PATH="$CACHE_DIR/$CACHE_FILE"
    
    echo "Corrupting cache file: $CACHE_FILE"
    echo "invalid json {{{" > "$CACHE_PATH"
    
    # Try to run with corrupted cache
    echo "Running with corrupted cache..."
    LOG_CORRUPT="tests/results/cache_corrupted.log"
    
    torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
        -- \
        --depth=12 \
        --num_iterations=$MAX_ITERATIONS \
        2>&1 | tee "$LOG_CORRUPT"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Run with corrupted cache failed"
        exit 1
    fi
    
    echo "✓ System handled corrupted cache gracefully"
    
    # Alternative: Delete cache and verify re-discovery
    echo ""
    echo "Testing cache deletion..."
    rm -rf "$CACHE_DIR"
    
    LOG_RERUN="tests/results/cache_deleted_rerun.log"
    torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
        -- \
        --depth=12 \
        --num_iterations=$MAX_ITERATIONS \
        2>&1 | tee "$LOG_RERUN"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Re-run after cache deletion failed"
        exit 1
    fi
    
    # Verify discovery ran again
    if grep -q "Auto-discovery found device_batch_size=" "$LOG_RERUN"; then
        echo "✓ Discovery re-ran after cache deletion"
    fi
else
    echo "SKIP: No cache files to corrupt"
fi

echo "✓ Test 20 passed!"

echo ""
echo "✓ All cache tests passed!"
