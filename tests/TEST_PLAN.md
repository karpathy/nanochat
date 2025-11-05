# Auto-Discovery Test Plan

## Test Coverage Matrix

| Test # | Name | Type | Duration | GPU Required | Status |
|--------|------|------|----------|--------------|--------|
| 1 | Exponential Search Logic | Unit | < 1s | No | ✓ Implemented |
| 2 | Binary Search Refinement | Unit | < 1s | No | ✓ Implemented |
| 3 | Safety Margin Application | Unit | < 1s | No | ✓ Implemented |
| 4 | Cache Mechanism | Unit | < 1s | No | ✓ Implemented |
| 5 | DDP Broadcast Simulation | Unit | < 1s | No | ✓ Implemented |
| 6 | Basic Discovery Run | Integration | 30s | 1 GPU | ✓ Implemented |
| 7 | Manual vs Auto Comparison | Integration | 2-3 min | 1 GPU | ✓ Implemented |
| 8 | DDP Discovery (2 GPUs) | Integration | 1-2 min | 2 GPUs | ✓ Implemented |
| 9 | DDP Discovery (4 GPUs) | Integration | 1-2 min | 4 GPUs | ✓ Implemented |
| 10 | Throughput Comparison | Integration | 5-10 min | 1 GPU | ✓ Implemented |
| 11 | Stability (depth=12) | Integration | 10-15 min | 1 GPU | ✓ Implemented |
| 12 | Stability (depth=20) | Integration | 15-20 min | 1 GPU | ✓ Implemented |
| 13 | Stability (depth=26) | Integration | 20-25 min | 1 GPU | ✓ Implemented |
| 14 | Stability (depth=32) | Integration | 25-30 min | 1 GPU | ✓ Implemented |
| 15 | Manual Override | Integration | 1-2 min | 1 GPU | ✓ Implemented |
| 16 | Disable Auto-Discovery | Integration | 1-2 min | 1 GPU | ✓ Implemented |
| 17 | Custom Safety Margin | Integration | 2-3 min | 1 GPU | ✓ Implemented |
| 18 | Cache Hit | Integration | 2-3 min | 1 GPU | ✓ Implemented |
| 19 | Cache Key Validation | Integration | 3-4 min | 1 GPU | ✓ Implemented |
| 20 | Cache Invalidation | Integration | 2-3 min | 1 GPU | ✓ Implemented |
| 21 | Artificial Memory Constraint | Integration | 2-3 min | 1 GPU | ✓ Implemented |
| 22 | Mid-Training Override Warning | Integration | 2-3 min | 1 GPU | ✓ Implemented |

## Test Execution Time Estimates

### Fast Suite (Unit Tests Only)
- **Duration**: ~10 seconds
- **GPU**: Not required
- **Command**: `bash tests/run_unit_tests.sh`

### Standard Suite (Unit + Short Integration)
- **Duration**: ~15-30 minutes
- **GPU**: 1 GPU required
- **Command**: `bash tests/run_integration_tests.sh`

### Full Suite (Including Long Stability Tests)
- **Duration**: ~1-2 hours
- **GPU**: 1 GPU required
- **Command**: `RUN_LONG_TESTS=1 bash tests/run_integration_tests.sh`

### Multi-GPU Suite
- **Duration**: ~20-40 minutes
- **GPU**: 2-4 GPUs required
- **Command**: `bash tests/run_integration_tests.sh` (auto-detects GPUs)

## Success Criteria

### Unit Tests
- [ ] All 5 unit tests pass
- [ ] Tests complete in < 10 seconds total
- [ ] Code coverage ≥ 80% for `nanochat/auto_batch_size.py`

### Integration Tests - Basic
- [ ] Single GPU discovery completes in < 30 seconds
- [ ] Auto-discovered batch size ≥ manual baseline (8)
- [ ] No OOM errors in any test
- [ ] All logs contain expected messages

### Integration Tests - DDP
- [ ] Rank 0 performs discovery, other ranks receive broadcast
- [ ] All ranks use identical batch size
- [ ] No deadlocks or synchronization errors
- [ ] Tests complete successfully on 2 and 4 GPUs

### Integration Tests - Performance
- [ ] Throughput improvement ≥ 1.3x compared to manual baseline
- [ ] Speedup ratio calculated and logged
- [ ] Results saved to JSON for analysis

### Integration Tests - Stability
- [ ] All 1000 iterations complete without errors
- [ ] No OOM errors during long runs
- [ ] No memory leaks detected
- [ ] Larger models (depth=32) use smaller batch sizes than smaller models (depth=12)

### Integration Tests - Overrides
- [ ] Manual `--device_batch_size` skips auto-discovery
- [ ] Custom safety margins produce expected batch sizes
- [ ] Disabled auto-discovery uses default values

### Integration Tests - Cache
- [ ] Cache hit reduces startup time from 15-30s to < 5s
- [ ] Different configurations create different cache keys
- [ ] Corrupted cache handled gracefully (fallback to re-discovery)
- [ ] Cache files created in correct directory

### Integration Tests - Failure Handling
- [ ] Artificial memory constraints trigger fallback
- [ ] Warning messages logged appropriately
- [ ] Mid-training override warning appears
- [ ] No crashes or exceptions, only graceful degradation

## Known Limitations

1. **Cache Tests**: Require write access to cache directory (usually `~/.nanochat/auto_batch_cache/`)
2. **DDP Tests**: Automatically skipped if fewer than 2 GPUs available
3. **Long Tests**: Disabled by default, require `RUN_LONG_TESTS=1` environment variable
4. **Memory Constraint Tests**: Difficult to reliably simulate on all systems
5. **Mid-Training Tests**: Require existing checkpoint from base_train

## Test Maintenance

### Adding New Tests

1. **Unit Tests**: Add to `tests/test_auto_batch_size.py`
   ```python
   def test_new_feature():
       # Test implementation
       assert result == expected
   ```

2. **Integration Tests**: Create new script in `tests/integration/`
   ```bash
   #!/bin/bash
   # tests/integration/test_new_feature.sh
   set -e
   # Test implementation
   ```

3. Update `tests/run_integration_tests.sh` to include new test
4. Update this test plan document

### Debugging Failed Tests

1. **Check logs**: All test output saved to `tests/results/*.log`
2. **Run individually**: Execute specific test script in isolation
3. **Increase verbosity**: Use `-x` flag for bash scripts, `-vv` for pytest
4. **Check GPU state**: Run `nvidia-smi` before and after tests
5. **Clear cache**: Remove `~/.nanochat/auto_batch_cache/` if cache issues suspected

## CI/CD Integration

### Recommended CI Pipeline

```yaml
stages:
  - test-unit
  - test-integration-fast
  - test-integration-full

test-unit:
  script:
    - bash tests/run_unit_tests.sh
  duration: 1 minute

test-integration-fast:
  script:
    - bash tests/run_integration_tests.sh
  duration: 30 minutes
  requires: [test-unit]

test-integration-full:
  script:
    - RUN_LONG_TESTS=1 bash tests/run_integration_tests.sh
  duration: 2 hours
  requires: [test-integration-fast]
  when: manual  # Only run on-demand
```

### Pre-commit Hooks

```bash
#!/bin/bash
# .git/hooks/pre-commit
bash tests/run_unit_tests.sh
```

## Test Data

### Expected Batch Sizes (A100 80GB GPU)
- depth=12: ~64-96
- depth=20: ~32-48
- depth=26: ~16-32
- depth=32: ~8-16

**Note**: Actual values depend on GPU memory, safety margin, and max_seq_len.

### Expected Speedups
- Baseline: device_batch_size=8
- Auto-discovered: device_batch_size=32-64
- Expected speedup: 1.5-3.0x (target: ≥1.3x after overhead)

## Appendix: Test File Structure

```
tests/
├── README.md                          # User-facing documentation
├── TEST_PLAN.md                       # This file
├── test_auto_batch_size.py           # Unit tests
├── run_unit_tests.sh                 # Unit test runner
├── run_integration_tests.sh          # Integration test runner
├── make_executable.sh                # Helper to chmod +x scripts
├── integration/                      # Integration test scripts
│   ├── test_single_gpu_discovery.sh
│   ├── test_manual_vs_auto.sh
│   ├── test_ddp_discovery.sh
│   ├── test_throughput_comparison.sh
│   ├── test_stability_depth12.sh
│   ├── test_stability_depth20.sh
│   ├── test_stability_depth26.sh
│   ├── test_stability_depth32.sh
│   ├── test_overrides.sh
│   ├── test_cache_mechanism.sh
│   └── test_failure_handling.sh
└── results/                          # Test output (gitignored)
    ├── .gitkeep
    ├── *.log
    └── throughput_comparison.json
```

## Version History

- **v1.0** (2024-01): Initial test suite implementation
  - 5 unit tests
  - 17 integration tests (Tests 6-22)
  - Unit and integration test runners
  - Comprehensive documentation
