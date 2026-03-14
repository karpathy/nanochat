---
title: "Phase 1.5.0 — Data Layout & Configuration System"
summary: "Completed data layout documentation, configuration system, paths centralization, and hierarchical directory structure."
status: archived
archived_date: "2026-03-15"
archived_reason: "All sub-tasks complete. Paths centralized, config system wired, Python upgraded to 3.13."
---

# Phase 1.5.0 — Data Layout & Configuration System ✅ COMPLETED

**Goal**: Establish data layout, configuration system, and centralized path management as foundation for compression-based optimization.

## Key Features Delivered

### Configuration System
- `TrainingConfig.from_args()` wired into `base_train.py`
- `--config` flag for TOML config loading (`tomllib` read, `tomli-w` write)
- `--base-dir` flag overriding `NANOCHAT_BASE_DIR` env var
- Auto-save config to checkpoint dir via `TrainingConfig.save()`
- Compression fields added to `TrainingConfig` with matching CLI defaults

### Centralized Path Management
- `src/nanochat/paths.py` — single source of truth for all directory paths
- Hierarchical `NANOCHAT_BASE_DIR` layout: `data/`, `checkpoints/{base,sft,rl}/`, `eval/`
- 8 consumer files updated to use paths module
- 8 tests in `test_paths.py`

### Side Effects Deferral
- `COMPUTE_DTYPE` lazy via `get_compute_dtype()` / `get_compute_dtype_reason()`
- Logging setup deferred to first `compute_init()` call (idempotent)
- `DATA_DIR` in `dataset.py` lazy via function
- Redundant `setup_default_logging()` removed from `checkpoint.py`

### Documentation
- `docs/data-layout.md` — `NANOCHAT_BASE_DIR` structure, resolution order, model tags

### Python Upgrade
- Python 3.10 → 3.13, dropped `tomli` dependency (stdlib `tomllib`)
- Updated `.python-version` and `pyproject.toml`

## Sub-tasks

- [x] Wire `TrainingConfig.from_args()` into `base_train.py`
- [x] Add `--config` flag to load config from TOML file
- [x] Use TOML for config format
- [x] Add `--base-dir` flag
- [x] Auto-save config to checkpoint dir
- [x] Add compression fields to `TrainingConfig`
- [x] Document data directory layout
- [x] Upgrade Python to 3.13

## Tests

- 8 path tests (`test_paths.py`)
- 3 config tests (TOML round-trip, from_args mapping, base_dir default)
- 56 passed, 10 skipped total
