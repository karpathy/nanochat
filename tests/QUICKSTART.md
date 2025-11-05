# Quick Start Guide - Auto-Discovery Tests

## TL;DR

```bash
# Make scripts executable
bash tests/make_executable.sh

# Run unit tests (10 seconds, no GPU)
bash tests/run_unit_tests.sh

# Run integration tests (30 minutes, requires GPU)
bash tests/run_integration_tests.sh
```

## First Time Setup

1. **Make test scripts executable**:
   ```bash
   bash tests/make_executable.sh
   ```

2. **Verify environment**:
   ```bash
   # Check Python/PyTorch
   python -c "import torch; print(torch.__version__)"
   
   # Check GPU (if available)
   nvidia-smi
   ```

3. **Install test dependencies** (if not already installed):
   ```bash
   pip install pytest
   ```

## Running Tests

### Unit Tests (Recommended First)

Fast tests that don't require GPU:

```bash
bash tests/run_unit_tests.sh
```

Expected output:
```
==========================================
Running Unit Tests
==========================================

tests/test_auto_batch_size.py::test_exponential_search PASSED
tests/test_auto_batch_size.py::test_binary_search_refinement PASSED
tests/test_auto_batch_size.py::test_safety_margin PASSED
tests/test_auto_batch_size.py::test_cache_hit PASSED
tests/test_auto_batch_size.py::test_cache_miss PASSED
...

✓ All unit tests passed!
```

### Integration Tests (Requires GPU)

```bash
# Standard suite (~30 minutes)
bash tests/run_integration_tests.sh

# Full suite with long stability tests (~2 hours)
RUN_LONG_TESTS=1 bash tests/run_integration_tests.sh
```

### Individual Tests

Run specific integration tests:

```bash
# Test basic discovery
bash tests/integration/test_single_gpu_discovery.sh

# Test manual vs auto comparison
bash tests/integration/test_manual_vs_auto.sh

# Test DDP (requires 2+ GPUs)
bash tests/integration/test_ddp_discovery.sh

# Test throughput improvement
bash tests/integration/test_throughput_comparison.sh

# Test caching
bash tests/integration/test_cache_mechanism.sh
```

## Expected Results

### Unit Tests
- ✓ All 11 tests pass
- ✓ Completes in < 10 seconds
- ✓ No GPU required

### Integration Tests (with full implementation)
- ✓ Discovery completes in < 30 seconds
- ✓ Auto batch size > manual batch size
- ✓ No OOM errors
- ✓ Throughput improvement ≥ 1.3x
- ✓ Cache reduces startup time to < 5 seconds

## Viewing Results

Test outputs are saved to `tests/results/`:

```bash
# View latest discovery log
cat tests/results/test_single_gpu_discovery.log

# View throughput comparison
cat tests/results/throughput_comparison.json

# List all results
ls -lh tests/results/
```

## Common Issues

### "pytest: command not found"
```bash
pip install pytest
```

### "Permission denied" when running scripts
```bash
bash tests/make_executable.sh
```

### "CUDA out of memory"
- Reduce model size in test scripts
- Or skip long stability tests (they're optional)

### "SKIP: DDP tests require at least 2 GPUs"
- Normal if you have only 1 GPU
- Tests will automatically skip

## Next Steps

1. **Read the docs**:
   - `tests/README.md` - Full documentation
   - `tests/TEST_PLAN.md` - Detailed test specifications
   - `tests/IMPLEMENTATION_NOTES.md` - Implementation details

2. **Check implementation status**:
   - Unit tests should pass with stub implementation
   - Integration tests need full implementation

3. **Contribute**:
   - Add new tests to `tests/test_auto_batch_size.py`
   - Create new integration scripts in `tests/integration/`
   - Update documentation

## Questions?

- Check `tests/README.md` for detailed documentation
- Look at test logs in `tests/results/`
- Review `tests/IMPLEMENTATION_NOTES.md` for troubleshooting

## Summary of Test Coverage

| Category | Count | Time | GPU |
|----------|-------|------|-----|
| Unit Tests | 11 | 10s | No |
| Single GPU Tests | 6 | 15min | 1 GPU |
| Multi-GPU Tests | 2 | 5min | 2+ GPUs |
| Performance Tests | 1 | 10min | 1 GPU |
| Stability Tests | 4 | 1-2hr | 1 GPU |
| Override Tests | 3 | 10min | 1 GPU |
| Cache Tests | 3 | 10min | 1 GPU |
| Failure Tests | 2 | 10min | 1 GPU |

**Total**: 22 tests covering all aspects of auto-discovery functionality.
