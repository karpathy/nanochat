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
| [Phase 1.5.0 — Data Layout & Config](archive/phase-1.5.0-data-layout-config.md) | 2026-03-14 | Config system, centralized paths, hierarchical dirs, Python 3.13 |
| [Phase 1.5.0.1 — Script Entry-Points](archive/phase-1.5.0.1-script-entry-points.md) | 2026-03-15 | Wrapped all scripts in main(), importable without side effects |

## Active Phase

### Phase 1.5 — Compression-Based Optimization

**Goal**: Validate compression-based optimization on current hardware before investing in scaling infrastructure.

**Sub-phases**:
- **1.5.0**: Data layout & configuration system — ✅ [Archived](archive/phase-1.5.0-data-layout-config.md)
- **1.5.0.1**: Script entry-point refactor — ✅ [Archived](archive/phase-1.5.0.1-script-entry-points.md)
- **1.5.0.2**: Code review & quality cleanup — 🔜 Next
- **1.5.1**: Compression metrics integration — ✅ Code complete, 🔜 validation pending — [Validation Checklist](phase-1.5.1-validation-checklist.md)
- **1.5.2**: Dataset quality via compression
- **1.5.3**: Compression-aware optimization

#### Phase 1.5.0.2 — Code Review & Quality Cleanup

**Goal**: Full code quality review and address findings before moving to GPU-dependent validation work.

**Sub-tasks**:
- [x] Full code review of `src/nanochat/` (structure, quality, patterns)
- [ ] Full code review of `tests/` (coverage gaps, test quality)
- [ ] Fix syntax error: extra `)` in `dataset.py` `__main__` block
- [ ] Wrap remaining 5 scripts in `main()`: `tok_train.py`, `tok_eval.py`, `chat_cli.py`, `chat_web.py`, `chat_eval.py` — console scripts in `pyproject.toml` reference `:main` that doesn't exist
- [ ] Wire `checkpoint.py` through `paths` module — currently bypasses `paths.checkpoints_dir()` with manual `get_base_dir()` + `os.path.join()`
- [ ] Defer `USE_FA3` in `flash_attention.py` — currently computed at import time, triggers dtype detection
- [ ] Fix bare `except:` in `report.py` — `run_command()` and `extract_timestamp()` catch KeyboardInterrupt
- [ ] Fix typo: "Addapted" → "Adapted" in `optimizer.py` docstring
- [ ] Address any findings from `tests/` review

**Sequencing**: No GPU required. Clean up codebase before compression validation (1.5.1+).

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

### Apple Silicon (MPS) Documentation
The [M3 Max guide](m3-max-guide.md) contains raw notes on running experiments on Apple Silicon. Needs cleanup into proper project documentation.

- [ ] Document MPS backend setup and limitations
- [ ] Add hardware-specific batch size and memory guidelines
- [ ] Integrate MPS device support into training scripts

## Deferred Phases

None currently.
