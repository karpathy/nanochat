---
title: "Refactor common.py into common/ Package"
summary: "Split monolithic common.py (280+ lines) into 7 focused modules under common/ package, absorbed paths.py. Backward-compatible via __init__.py re-exports."
status: archived
archived_date: "2026-03-15"
archived_reason: "All sub-tasks complete. Package split done, all consumers working, no new pyright errors."
---

# Refactor `common.py` into `common/` Package ✅ COMPLETED

**Goal**: Split the monolithic `common.py` into focused modules and absorb `paths.py` into the package.

## Package Layout

```
src/nanochat/common/
├── __init__.py       # Re-exports all public names (backward-compatible)
├── dtype.py          # _detect_compute_dtype, get_compute_dtype, get_compute_dtype_reason
├── logging.py        # ColoredFormatter, setup_default_logging
├── distributed.py    # is_ddp_requested, is_ddp_initialized, get_dist_info, compute_init, compute_cleanup, autodetect_device_type
├── io.py             # get_base_dir, download_file_with_lock, print0, print_banner
├── hardware.py       # get_peak_flops
├── wandb.py          # DummyWandb
└── paths.py          # Absorbed from top-level paths.py
```

## Key Decisions

- `__init__.py` re-exports all public names so the 16 consumers importing `from nanochat.common import ...` required zero changes
- Only 5 files needed import updates: those importing from `nanochat.paths` → `nanochat.common.paths` (dataset.py, checkpoint.py, tok_train.py, tokenizer.py, test_paths.py)
- `paths.py` imports `get_base_dir` from `common.io` instead of the old circular `common` → `paths` dependency

## Files Created
- `src/nanochat/common/__init__.py`
- `src/nanochat/common/dtype.py`
- `src/nanochat/common/logging.py`
- `src/nanochat/common/distributed.py`
- `src/nanochat/common/io.py`
- `src/nanochat/common/hardware.py`
- `src/nanochat/common/wandb.py`
- `src/nanochat/common/paths.py`

## Files Deleted
- `src/nanochat/common.py`
- `src/nanochat/paths.py`

## Files Modified (import updates)
- `src/nanochat/data/dataset.py`
- `src/nanochat/data/tokenizer.py`
- `src/nanochat/training/checkpoint.py`
- `src/nanochat/scripts/tok_train.py`
- `tests/test_paths.py`

## Sub-tasks

- [x] Create `common/` package with `__init__.py` re-exporting all public names
- [x] Split `common.py` into focused modules (dtype, logging, distributed, io, hardware, wandb)
- [x] Move `paths.py` → `common/paths.py`, update its internal import from `common.io.get_base_dir`
- [x] Update 5 consumer files importing from `nanochat.paths`
- [x] Update `tests/test_paths.py` import path
- [x] Verify: 56 tests pass, zero ruff errors, no new pyright errors (128 unchanged)

## Tests

- 56 passed, 10 skipped (unchanged)
- Zero ruff errors in src/
- 128 pyright errors (unchanged — no regressions)
