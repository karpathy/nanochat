# Auto-Discovery Testing Suite

Comprehensive tests for the auto-discovery batch size functionality in NanoChat.

## Overview

This testing suite validates the auto-discovery system across different scenarios:
- **Unit Tests**: Isolated testing of core algorithms (exponential search, binary search, caching)
- **Integration Tests**: End-to-end testing with actual training scripts
- **Stability Tests**: Long-running tests to detect memory leaks and OOM issues
- **Performance Tests**: Throughput comparisons between manual and auto-discovered batch sizes

## Quick Start

### Run All Tests

```bash
# Run unit tests only (fast, ~10 seconds)
bash tests/run_unit_tests.sh

# Run integration tests (requires GPU, 10-30 minutes)
bash tests/run_integration_tests.sh

# Run integration tests including long stability tests (1+ hours)
RUN_LONG_TESTS=1 bash tests/run_integration_tests.sh
```

### Run Individual Tests

```bash
# Unit tests
pytest tests/test_auto_batch_size.py -v

# Specific integration test
bash tests/integration/test_single_gpu_discovery.sh
bash tests/integration/test_ddp_discovery.sh
bash tests/integration/test_throughput_comparison.sh
```

## Test Categories

### Unit Tests (`test_auto_batch_size.py`)

Tests the core discovery algorithms in isolation using mocks:

- **Test 1**: Exponential search finds upper bound (1, 2, 4, 8, 16, 32, 64)
- **Test 2**: Binary search refines to exact boundary
- **Test 3**: Safety margin application (0.85, 0.90, 0.95)
- **Test 4**: Cache hit/miss behavior
- **Test 5**: DDP broadcast simulation

**Run with:**
```bash
pytest tests/test_auto_batch_size.py -v --tb=short
```

### Integration Tests

#### Single GPU Tests

- **Test 6**: Basic discovery run (`test_single_gpu_discovery.sh`)
  - Verifies discovery completes in < 30 seconds
  - Checks for proper log messages
  - Validates no OOM errors

- **Test 7**: Manual vs Auto comparison (`test_manual_vs_auto.sh`)
  - Compares manual batch_size=8 with auto-discovery
  - Validates auto batch size ≥ manual
  - Ensures both runs complete successfully

#### Multi-GPU Tests

- **Test 8**: 2-GPU DDP discovery (`test_ddp_discovery.sh`)
  - Verifies rank 0 performs discovery
  - Checks broadcast to rank 1
  - Validates synchronization

- **Test 9**: 4-GPU DDP discovery (if available)
  - Same as Test 8 with 4 GPUs
  - Skipped if fewer than 4 GPUs available

#### Throughput Tests

- **Test 10**: Throughput comparison (`test_throughput_comparison.sh`)
  - Measures iterations/second for manual vs auto
  - Calculates speedup ratio
  - Target: ≥ 1.3x speedup (allows for discovery overhead)
  - Saves results to `tests/results/throughput_comparison.json`

#### Stability Tests

Long-running tests (1000 iterations each):

- **Test 11**: Depth=12 (`test_stability_depth12.sh`)
- **Test 12**: Depth=20 (`test_stability_depth20.sh`)
- **Test 13**: Depth=26 (`test_stability_depth26.sh`)
- **Test 14**: Depth=32 (`test_stability_depth32.sh`)
  - Verifies larger models use smaller batch sizes
  - Monitors for memory leaks
  - Ensures no OOM during long runs

**Run with:**
```bash
RUN_LONG_TESTS=1 bash tests/run_integration_tests.sh
```

#### Override Tests

- **Test 15**: Manual override (`test_overrides.sh`)
  - Verifies `--device_batch_size=16` skips auto-discovery
  - Checks for manual batch size usage message

- **Test 16**: Disable auto-discovery
  - Tests with auto-discovery disabled
  - Verifies fallback to default batch_size=8

- **Test 17**: Custom safety margin
  - Tests `--batch_size_margin=0.85` vs `0.90`
  - Verifies higher margin gives larger batch size

#### Cache Tests

- **Test 18**: Cache hit (`test_cache_mechanism.sh`)
  - First run: discovery + cache save
  - Second run: cache hit (< 5 seconds)
  - Verifies cache file creation

- **Test 19**: Cache key validation
  - Different depth → different cache key
  - Different max_seq_len → different cache key
  - Verifies multiple cache files created

- **Test 20**: Cache invalidation
  - Corrupts cache file
  - Verifies graceful fallback to re-discovery
  - Tests cache deletion and re-run

#### Failure Handling Tests

- **Test 21**: Artificial memory constraint (`test_failure_handling.sh`)
  - Tests with very large model (depth=40)
  - Verifies fallback to defaults
  - Checks for warning messages

- **Test 22**: Mid-training override warning
  - Tests mid_train.py with larger batch size than pretrain
  - Verifies "FOOTGUN WARNING" appears
  - Ensures training continues despite warning

## Test Results

Results are saved to `tests/results/`:

```
tests/results/
├── test_single_gpu_discovery.log
├── test_manual_baseline.log
├── test_auto_discovery.log
├── throughput_comparison.json
├── stability_depth12.log
├── stability_depth20.log
├── cache_run1.log
├── cache_run2.log
└── ...
```

### Throughput Results Format

`tests/results/throughput_comparison.json`:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "depth": 12,
  "max_iterations": 100,
  "manual": {
    "batch_size": 8,
    "duration_seconds": 120,
    "throughput_iter_per_sec": 0.833
  },
  "auto": {
    "batch_size": 32,
    "duration_seconds": 60,
    "throughput_iter_per_sec": 1.667
  },
  "speedup_ratio": 2.0
}
```

## Requirements

### Unit Tests
- Python 3.8+
- PyTorch
- pytest
- No GPU required (runs on CPU)

### Integration Tests
- CUDA-capable GPU (≥ 24GB VRAM recommended)
- Multiple GPUs for DDP tests (optional)
- Environment variables:
  - `NANOCHAT_BASE_DIR`: Base directory for checkpoints/cache (optional)
  - `RUN_LONG_TESTS=1`: Enable 1000-iteration stability tests (optional)

## CI/CD Integration

For automated testing in CI:

```bash
# Quick validation (unit tests + fast integration tests)
bash tests/run_unit_tests.sh
bash tests/run_integration_tests.sh  # ~15 minutes

# Full validation (includes long tests)
RUN_LONG_TESTS=1 bash tests/run_integration_tests.sh  # ~1 hour
```

### GitHub Actions Example

```yaml
name: Auto-Discovery Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: bash tests/run_unit_tests.sh
      - name: Run integration tests
        run: bash tests/run_integration_tests.sh
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: tests/results/
```

## Troubleshooting

### Common Issues

1. **"SKIP: Need at least 2 GPUs for DDP tests"**
   - Expected if you have only 1 GPU
   - DDP tests will be skipped automatically

2. **"Cache directory is empty or doesn't exist"**
   - Cache may be disabled or path issue
   - Check `NANOCHAT_BASE_DIR` environment variable

3. **"Discovery takes longer than 30 seconds"**
   - May indicate large model or slow GPU
   - Increase timeout in test script if needed

4. **"Speedup ratio below threshold"**
   - Discovery overhead may be high for short runs
   - Try longer runs (increase `MAX_ITERATIONS`)

### Debug Mode

Run tests with verbose output:

```bash
# Unit tests with full traceback
pytest tests/test_auto_batch_size.py -vv --tb=long

# Integration tests with set -x
bash -x tests/integration/test_single_gpu_discovery.sh
```

## Success Criteria

### Unit Tests
- ✓ All 5 unit tests pass
- ✓ Tests complete in < 10 seconds
- ✓ Code coverage ≥ 80% for `nanochat/auto_batch_size.py`

### Integration Tests
- ✓ Single GPU discovery completes in < 30 seconds
- ✓ No OOM errors during 1000+ iteration stability tests
- ✓ Throughput improvement ≥ 1.3x compared to manual baseline
- ✓ DDP tests show identical batch size across all ranks
- ✓ Override tests correctly skip discovery or use manual values
- ✓ Cache tests show < 5 second cache hit time vs 15-30 second discovery

### Failure Handling
- ✓ Artificial memory constraints trigger fallback to defaults
- ✓ Warning messages appear in logs for fallback scenarios
- ✓ No crashes or exceptions, only graceful degradation

## Contributing

When adding new tests:

1. Add unit tests to `tests/test_auto_batch_size.py`
2. Add integration tests as new `.sh` scripts in `tests/integration/`
3. Update `tests/run_integration_tests.sh` to include new tests
4. Update this README with test descriptions
5. Ensure tests clean up after themselves (delete temp files, clear cache)

## License

Same as NanoChat project.
