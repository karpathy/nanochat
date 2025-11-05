# Implementation Notes for Auto-Discovery Testing

## Overview

This document describes the implementation of the comprehensive testing suite for the auto-discovery batch size functionality in NanoChat.

## Current Status

### What Has Been Implemented

1. **Stub Auto-Discovery Module** (`nanochat/auto_batch_size.py`)
   - Minimal working implementation with expected interface
   - Supports the full API required by tests
   - Includes caching, DDP broadcast, and safety margin features
   - Ready for full implementation to replace the stub logic

2. **Unit Tests** (`tests/test_auto_batch_size.py`)
   - 11 comprehensive unit tests covering all core algorithms
   - Tests for exponential search, binary search, safety margins
   - Cache mechanism validation (hit/miss, key generation)
   - DDP broadcast simulation
   - Mock-based testing for isolation
   - All tests runnable on CPU without GPU

3. **Integration Test Scripts** (`tests/integration/*.sh`)
   - 17 bash-based integration tests (Tests 6-22)
   - Single GPU discovery validation
   - Multi-GPU DDP testing with auto-detection
   - Throughput comparison with JSON output
   - Stability tests for depths 12, 20, 26, 32
   - Override and cache mechanism tests
   - Failure handling and graceful degradation tests

4. **Test Infrastructure**
   - `tests/run_unit_tests.sh` - Unit test runner
   - `tests/run_integration_tests.sh` - Integration test orchestrator
   - `tests/results/` - Output directory for logs and results
   - Comprehensive documentation (README, TEST_PLAN)

### What Still Needs to Be Done

The tests are **ready to run** once the full auto-discovery implementation is complete. The current stub implementation allows the test framework to be validated, but for the tests to be meaningful, the following need to be implemented in `nanochat/auto_batch_size.py`:

1. **Real Exponential Search Algorithm**
   - Currently returns a fixed value
   - Needs to implement doubling strategy (1, 2, 4, 8, 16, ...)
   - Must detect OOM boundary

2. **Real Binary Search Refinement**
   - Currently not implemented in stub
   - Should narrow down from exponential search bounds
   - Must find exact maximum batch size that fits

3. **OOM Detection in `_test_batch_size()`**
   - Currently has basic try-catch for OOM
   - May need more robust handling
   - Should properly clean up GPU memory

4. **Integration with Training Scripts**
   - Scripts need to call `discover_batch_size()` when appropriate
   - Need to add command-line flags:
     - `--auto_batch_size=True/False`
     - `--batch_size_margin=0.85` (optional)
     - `--batch_size_cache=True/False` (optional)
   - Need to add logic to skip discovery if manual batch size provided
   - Need to add logging messages that tests expect

5. **GPU Info for Cache Keys**
   - Currently uses placeholder GPU name
   - Should detect actual GPU model for cache keys

## Integration Points

### Training Scripts That Need Updates

1. **`scripts/base_train.py`**
   ```python
   # Add near top after imports
   from nanochat.auto_batch_size import discover_batch_size
   
   # Add to config section
   auto_batch_size = False  # Enable auto-discovery
   batch_size_margin = 0.85  # Safety margin
   batch_size_cache = True  # Enable caching
   
   # Add after compute_init() and before model creation
   if auto_batch_size and device_batch_size is None:
       device_batch_size = discover_batch_size(
           model=temp_model,  # or create temp model just for discovery
           max_seq_len=max_seq_len,
           device=device,
           safety_margin=batch_size_margin,
           ddp_rank=ddp_rank,
           ddp_world_size=ddp_world_size,
           use_cache=batch_size_cache,
           cache_key_components={
               'model_config': model_config_kwargs,
               'gpu': torch.cuda.get_device_name(),
               'max_seq_len': max_seq_len,
           }
       )
   ```

2. **`scripts/mid_train.py`**
   - Similar integration as base_train
   - Add warning if device_batch_size > pretrain batch size

3. **`scripts/chat_sft.py`**
   - Similar integration
   - Default batch size is 4, so auto-discovery should help significantly

## Test Validation

### To Verify Tests Are Working

1. **Run unit tests** (should work now with stub):
   ```bash
   bash tests/run_unit_tests.sh
   ```
   Expected: All tests pass (some may be skipped due to stub limitations)

2. **Make scripts executable**:
   ```bash
   bash tests/make_executable.sh
   ```

3. **Try a quick integration test** (requires GPU):
   ```bash
   bash tests/integration/test_single_gpu_discovery.sh
   ```
   Expected: Will fail with current stub, but should run without errors

4. **Once full implementation is done**:
   ```bash
   bash tests/run_integration_tests.sh
   ```
   Expected: Most tests should pass

## Expected Test Behavior

### With Current Stub Implementation

- **Unit tests**: Most pass, some may have limitations due to stub
- **Integration tests**: Will run but may not find meaningful batch sizes
- **Cache tests**: Should work (caching logic is implemented)
- **DDP tests**: Broadcast should work, discovery logic is stubbed

### With Full Implementation

- **Unit tests**: All should pass
- **Single GPU tests**: Should discover reasonable batch sizes (16-64 range)
- **DDP tests**: Should show proper rank 0 discovery and broadcast
- **Throughput tests**: Should show 1.5-3x speedup
- **Stability tests**: Should complete 1000 iterations without OOM
- **Cache tests**: Should show significant startup time improvement

## Troubleshooting Guide

### Common Issues and Solutions

1. **"Auto-discovery found device_batch_size=" not in log**
   - Training script not calling `discover_batch_size()`
   - Check integration in training script
   - Verify `--auto_batch_size=True` is being passed

2. **Tests fail with "Command not found"**
   - Scripts may not be executable
   - Run: `bash tests/make_executable.sh`

3. **Cache tests fail**
   - Check `NANOCHAT_BASE_DIR` environment variable
   - Verify write permissions to cache directory
   - Try: `mkdir -p ~/.nanochat/auto_batch_cache`

4. **DDP tests skipped**
   - Expected if fewer than 2 GPUs
   - Tests auto-detect GPU count

5. **OOM during stability tests**
   - Discovery may not be working correctly
   - Check safety margin (should be 0.85 or lower)
   - Verify model size vs GPU memory

## Performance Expectations

### Discovery Time
- Initial discovery: 15-30 seconds
- Cache hit: < 5 seconds
- Overhead per training run: 15-30 seconds (first run only)

### Batch Size Improvements
Based on A100 80GB GPU:
- depth=12: 8 (manual) → 64-96 (auto) = 8-12x larger
- depth=20: 8 (manual) → 32-48 (auto) = 4-6x larger
- depth=26: 8 (manual) → 16-32 (auto) = 2-4x larger
- depth=32: 8 (manual) → 8-16 (auto) = 1-2x larger

### Throughput Improvements
- Expected speedup: 1.5-3.0x
- Measured after discovery overhead
- Varies by model size and GPU

## Next Steps for Full Implementation

1. **Implement core discovery algorithms** in `nanochat/auto_batch_size.py`:
   - Replace stub `_perform_discovery()` with real search
   - Implement exponential + binary search
   - Improve OOM detection

2. **Integrate into training scripts**:
   - Add command-line flags
   - Add discovery calls
   - Add appropriate logging

3. **Validate with tests**:
   - Run unit tests to verify algorithms
   - Run integration tests to verify end-to-end
   - Run stability tests for production validation

4. **Optimize and tune**:
   - Adjust safety margins if needed
   - Tune cache key components
   - Add more robust error handling

## Files Created

### Core Implementation
- `nanochat/auto_batch_size.py` (stub with full interface)

### Tests
- `tests/test_auto_batch_size.py` (unit tests)
- `tests/integration/test_single_gpu_discovery.sh`
- `tests/integration/test_manual_vs_auto.sh`
- `tests/integration/test_ddp_discovery.sh`
- `tests/integration/test_throughput_comparison.sh`
- `tests/integration/test_stability_depth{12,20,26,32}.sh`
- `tests/integration/test_overrides.sh`
- `tests/integration/test_cache_mechanism.sh`
- `tests/integration/test_failure_handling.sh`

### Infrastructure
- `tests/run_unit_tests.sh`
- `tests/run_integration_tests.sh`
- `tests/make_executable.sh`

### Documentation
- `tests/README.md` (user guide)
- `tests/TEST_PLAN.md` (test specifications)
- `tests/IMPLEMENTATION_NOTES.md` (this file)

### Results Directory
- `tests/results/.gitkeep`
- Updated `.gitignore` to exclude test logs

## Conclusion

The testing infrastructure is **complete and ready to use**. The stub implementation allows the test framework to be validated and demonstrates the expected interface. Once the full auto-discovery implementation is complete, these tests will provide comprehensive validation of correctness, performance, and stability.

The tests are designed to be:
- **Comprehensive**: Cover all major functionality and edge cases
- **Maintainable**: Clear structure, good documentation
- **CI-ready**: Can run unattended with clear pass/fail
- **Fast**: Unit tests in seconds, full suite in ~30 minutes
- **Reliable**: Auto-skip tests when requirements not met (e.g., multiple GPUs)

For questions or issues, refer to:
- `tests/README.md` for usage instructions
- `tests/TEST_PLAN.md` for test specifications
- Test logs in `tests/results/` for debugging
