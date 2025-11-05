#!/bin/bash
#
# Run all integration tests for auto-discovery functionality
# These tests require GPU access and may take considerable time
#

set -e

echo "=========================================="
echo "Running Integration Tests"
echo "=========================================="
echo ""
echo "Note: These tests require GPU access"
echo "Some tests may take several minutes to complete"
echo ""

# Track test results
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Function to run a test script
run_test() {
    local test_script=$1
    local test_name=$(basename "$test_script" .sh)
    
    echo ""
    echo "=========================================="
    echo "Running: $test_name"
    echo "=========================================="
    
    TESTS_RUN=$((TESTS_RUN + 1))
    
    if bash "$test_script"; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo "✓ $test_name PASSED"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            # Exit code 0 but test indicated skip
            TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
            echo "○ $test_name SKIPPED"
        else
            TESTS_FAILED=$((TESTS_FAILED + 1))
            echo "✗ $test_name FAILED"
        fi
    fi
}

# ============================================================================
# Single GPU Tests
# ============================================================================
echo ""
echo "========================================"
echo "Single GPU Tests"
echo "========================================"

run_test "tests/integration/test_single_gpu_discovery.sh"
run_test "tests/integration/test_manual_vs_auto.sh"

# ============================================================================
# Multi-GPU DDP Tests
# ============================================================================
echo ""
echo "========================================"
echo "Multi-GPU Tests"
echo "========================================"

NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo "0")
echo "Detected $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -ge 2 ]; then
    run_test "tests/integration/test_ddp_discovery.sh"
else
    echo "SKIP: DDP tests require at least 2 GPUs"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
fi

# ============================================================================
# Throughput Tests
# ============================================================================
echo ""
echo "========================================"
echo "Throughput Tests"
echo "========================================"

run_test "tests/integration/test_throughput_comparison.sh"

# ============================================================================
# Stability Tests
# ============================================================================
echo ""
echo "========================================"
echo "Stability Tests"
echo "========================================"
echo "Note: These tests run 1000 iterations and may take 10+ minutes each"
echo ""

# Ask user if they want to run long tests (or check environment variable)
if [ "${RUN_LONG_TESTS:-}" = "1" ]; then
    echo "Running long stability tests (RUN_LONG_TESTS=1)..."
    run_test "tests/integration/test_stability_depth12.sh"
    run_test "tests/integration/test_stability_depth20.sh"
    run_test "tests/integration/test_stability_depth26.sh"
    run_test "tests/integration/test_stability_depth32.sh"
else
    echo "SKIP: Long stability tests (set RUN_LONG_TESTS=1 to enable)"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 4))
fi

# ============================================================================
# Override Tests
# ============================================================================
echo ""
echo "========================================"
echo "Override Tests"
echo "========================================"

run_test "tests/integration/test_overrides.sh"

# ============================================================================
# Cache Tests
# ============================================================================
echo ""
echo "========================================"
echo "Cache Tests"
echo "========================================"

run_test "tests/integration/test_cache_mechanism.sh"

# ============================================================================
# Failure Handling Tests
# ============================================================================
echo ""
echo "========================================"
echo "Failure Handling Tests"
echo "========================================"

run_test "tests/integration/test_failure_handling.sh"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Tests run:     $TESTS_RUN"
echo "Tests passed:  $TESTS_PASSED"
echo "Tests failed:  $TESTS_FAILED"
echo "Tests skipped: $TESTS_SKIPPED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
