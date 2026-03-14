---
title: "nanochat Roadmap"
summary: "Current and future implementation phases for nanochat."
read_when:
  - Planning nanochat development
  - Deciding what to implement next
  - Understanding scope and sequencing
status: draft
last_updated: "2026-03-15"
---

# nanochat Roadmap

## Current Status

- **Phase 0**: Architecture optimizations — ✅ Complete
- **Phase 0.5**: Refactor to modular architecture — ✅ Complete (2026-03-13)
- **Phase 1**: Architecture experiments — ❌ Attempted, negative results
- **Phase 1.5**: Compression-based optimization — 🔜 Code done, validation pending

## Completed Phases

| Phase | Completed | Summary |
|-------|-----------|--------|
| [Phase 0 — Optimizations](archive/phase-0-optimizations.md) | 2026-03-05 | FA3, FP8, ClimbMix (27% speedup), Muon, sliding window, auto batch |
| [Phase 0.5 — Refactor](archive/phase-0.5-refactor.md) | 2026-03-13 | Modular src/nanochat/ layout, 44 files, 30 tests, CI/CD |
| [Phase 1 — Architecture Experiments](archive/phase-1-architecture-experiments.md) | 2026-02-19 | SwiGLU, MoE, multi-token prediction — all negative results |

## Active Phase

### Phase 1.5 — Compression-Based Optimization

**Goal**: Validate compression-based optimization on current hardware before investing in scaling infrastructure.

**Sub-phases**:
- **1.5.0**: Data layout & configuration system — ✅ Complete (2026-03-14)
  - [x] Wire `TrainingConfig.from_args()` into `base_train.py` — replace raw `args` usage
  - [x] Add `--config` flag to load config from TOML file
  - [x] Use TOML for config format (`tomllib` read, `tomli-w` write) — supports comments for documenting hyperparameter choices
  - [x] Add `--base-dir` flag (overrides `NANOCHAT_BASE_DIR` env var)
  - [x] Auto-save config to checkpoint dir via `TrainingConfig.save()`
  - [x] Add compression fields to `TrainingConfig` (`track_compression`, `compression_log_every`, `track_layer_compression`, `compression_early_stop`) — defaults must match CLI defaults
  - [x] Document data directory layout (`NANOCHAT_BASE_DIR`, data/checkpoints/tokenizer structure) — see [data-layout.md](data-layout.md)
  - [x] Upgrade Python base version (currently 3.10 → target 3.13): torch 2.9.1 supports up to 3.14, 3.13 is locally installed — drop `tomli` dep (stdlib `tomllib` available), update `.python-version` and `pyproject.toml` `requires-python`
    - Note: 3.14 blocked by `tiktoken` (no pre-built wheel yet, requires Rust compiler to build from source)
- **1.5.0.1**: Script entry-point refactor — ✅ Complete
- **1.5.1**: Compression metrics integration — ✅ Code complete, 🔜 validation pending — [Validation Checklist](phase-1.5.1-validation-checklist.md)
- **1.5.2**: Dataset quality via compression
- **1.5.3**: Compression-aware optimization

#### Phase 1.5.0.1 — Script Entry-Point Refactor

**Goal**: Wrap top-level script code behind `main()` so scripts are importable without side effects.

Each script currently runs argparse, model init, optimizer setup, and dataloader creation at module level (~200+ lines). This prevents importing any function from these modules without triggering the full setup.

**Sub-tasks**:
- [x] `base_train.py` (~214 lines of top-level setup, 899 total) — wrap argparse, model build, optimizer, dataloader, and training loop dispatch into `main()`. Also fixed `config` shadowing bug in `build_model_meta()`, added missing `Path` import, converted `train_base_model` from 30-param function to closure.
- [x] `chat_sft.py` (~238 lines of top-level setup, 593 total) — wrap into `main()`, extract `build_parser()`, convert globals to nonlocal in data generator
- [x] `chat_rl.py` (~103 lines of top-level setup, 370 total) — wrap into `main()`, extract `build_parser()`, nested closures for get_batch/run_gsm8k_eval
- [x] Verify all three scripts still work identically via `python -m` and `torchrun`
- [x] Update tests if any import from these scripts directly — no tests import from scripts directly

**Sequencing**: Can be done independently of 1.5.1 validation (no GPU required). Recommended before 1.5.2 since dataset quality work will want to import training utilities without side effects.

**Exit criteria**:
- [ ] Compression ratio correlates with val loss (R² > 0.7)
- [ ] Compression-aware optimizer shows 10%+ faster convergence
- [ ] Overall 15%+ total improvement → proceed to Phase 2

**Decision point** (after Phase 1.5.3):
- **>15% improvement** → Phase 2 (infrastructure), then scale to 7B
- **5–15% improvement** → refine compression approach, iterate at small scale
- **<5% improvement** → skip to Phase 6 (SP-Transformer hybrid research)

## Future Phases

Sequencing depends on Phase 1.5 decision point. If compression validates (>15%), follow Phase 2→3→4→5. If it doesn't (<5%), pivot to Phase 6.

### Phase 2 — Training Infrastructure
Scale beyond single node, enable 10B+ parameter training.

### Phase 3 — Data Pipeline
Match frontier pre-training data quality through compression-aware selection. See [detailed plan](phase-3-data-pipeline.md).

### Phase 4 — Post-Training Alignment
Turn base model into capable assistant with stable learning.

### Phase 5 — Capabilities
Long context, tool use, transparency, multimodal. See [tool use & transparency plan](phase-5-tools-transparency.md).

### Phase 6 — SP-Transformer Hybrid (Research)
Combine transformer efficiency with SP Theory advantages. Also the fallback path if compression approach shows <5% improvement. See [detailed plan](phase-6-hybrid-architecture.md).

## Improvements

### Defer Module-Level Side Effects

Completed:
- [x] Make `DATA_DIR` in `dataset.py` lazy (use `_get_data_dir()` function instead of module-level constant)
- [x] Defer `COMPUTE_DTYPE` detection to first use via `get_compute_dtype()`
- [x] Defer logging setup to first `compute_init()` call (idempotent `setup_default_logging()`)
- [x] Remove redundant `setup_default_logging()` from `checkpoint.py`

Remaining work moved to Phase 1.5.0.1.

### Apple Silicon (MPS) Documentation
The [M3 Max guide](m3-max-guide.md) contains raw notes on running experiments on Apple Silicon. Needs cleanup into proper project documentation.

- [ ] Document MPS backend setup and limitations
- [ ] Add hardware-specific batch size and memory guidelines
- [ ] Integrate MPS device support into training scripts

## Deferred Phases

None currently.
