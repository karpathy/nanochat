# Local State — nanochat (karpathy fork)

Documented 2026-04-09 before machine teardown.

## Branch: fa3-flex-sdpa (current)
- Tracking: `fork/fa3-flex-sdpa` (ademeure/nanochat) — pushed and up to date
- 1 commit ahead of upstream master: `3d0dec5 FA3/FlexAttention/SDPA attention + PyTorch 2.11/CUDA 13.0`

## Branch: pytorch-2.11-cu130
- Tracking: `fork/pytorch-2.11-cu130` — pushed and up to date
- 2 commits ahead of master

## Branch: pytorch-2.11-cu128-test
- **Local-only, no upstream** — but 0 commits ahead of master, just a branch pointer. No unique content.

## Uncommitted changes (being committed now)

### scripts/base_train.py
- Added env-var-controlled profiling hooks (`NANOCHAT_PROFILE_START`, `NANOCHAT_PROFILE_STOP`, `NANOCHAT_PROFILE_EXIT`, `NANOCHAT_TORCH_PROFILE_DIR`)
- CUDA profiler start/stop integration around training steps
- PyTorch profiler with tensorboard trace output
- Early exit after profiling completes
- This is a work-in-progress profiling integration — functional but may need further tuning

### scripts/profile_step.py (new file)
- Standalone profiling script for a single training step (fwd/bwd/opt)
- Supports nsys and ncu profiling with NVTX ranges
- Usage: `nsys profile -o out python -m scripts.profile_step --depth 6`
- Supports `--phase {all,fwd,bwd,opt}` for targeted kernel analysis

### profiles/ (NOT committed — binary nsys artifacts)
- `nsys_d32_full.nsys-rep` (1.6M) — nsys trace, depth=32
- `nsys_d32_full.sqlite` (2.4M) — exported sqlite
- `nsys_d32_minimal.nsys-rep` (1.5M) — minimal nsys trace
- These are reproducible output artifacts, not committed to git
