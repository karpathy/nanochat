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
| [Phase 1.5.0.2 — Code Review & Cleanup](archive/phase-1.5.0.2-code-review-cleanup.md) | 2026-03-15 | Full code review, 6 fixes, lazy USE_FA3, paths wiring, all 9 scripts wrapped |
| [Refactor common.py into common/ Package](archive/refactor-common-package.md) | 2026-03-15 | Split monolithic common.py into 7 focused modules, absorbed paths.py, backward-compatible |
| [Type Annotations & Pyright Compliance](archive/type-annotations-pyright-compliance.md) | 2026-03-15 | Pyright strict mode, 17 suppression rules, 323→0 reportMissingParameterType across 29 files |
| [Apple Silicon (MPS) Documentation](archive/apple-silicon-mps-documentation.md) | 2026-03-15 | Rewrote MPS guide with accurate device detection, dtype, limitations, and batch size guidelines |

## Active Phase

### Phase 1.5 — Compression-Based Optimization

**Goal**: Validate compression-based optimization on current hardware before investing in scaling infrastructure.

**Sub-phases**:
- **1.5.0**: Data layout & configuration system — ✅ [Archived](archive/phase-1.5.0-data-layout-config.md)
- **1.5.0.1**: Script entry-point refactor — ✅ [Archived](archive/phase-1.5.0.1-script-entry-points.md)
- **1.5.0.2**: Code review & quality cleanup — ✅ [Archived](archive/phase-1.5.0.2-code-review-cleanup.md)
- **1.5.1**: Compression metrics integration — ✅ Code complete, 🔜 validation pending — [Validation Checklist](phase-1.5.1-validation-checklist.md)
- **1.5.2**: Dataset quality via compression
- **1.5.3**: Compression-aware optimization

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

### Type Annotations & Pyright Compliance — ✅ [Archived](archive/type-annotations-pyright-compliance.md)

### Refactor `common.py` into `common/` Package — ✅ [Archived](archive/refactor-common-package.md)

### Remaining Pyright Errors (128)
After completing the `reportMissingParameterType` retrofit, 128 errors remain across these categories:

| Category | Count | Notes |
|---|---|---|
| `reportUnusedVariable` | ~42 | Tuple unpacking (`ddp`, `ddp_local_rank`, etc.), loop vars (`i`, `B`, `T`) |
| `reportReturnType` | ~14 | `dict` invariance (need `Mapping`), numpy `bool_`/`floating` vs Python types |
| `reportIndexIssue` | ~14 | `object` typed params lacking `__getitem__` (meta dicts, dataset rows) |
| `reportPossiblyUnboundVariable` | ~12 | Conditional branches (`optimizer_data`, `meta_data`, `val_bpb`, `content_len`) |
| `reportMissingImports` | ~7 | `tasks.*` imports unresolvable (runtime `sys.path` manipulation) |
| `reportUnusedFunction` | ~5 | FastAPI route handlers registered via decorators |
| `reportGeneralTypeIssues` | ~4 | `Module` not iterable, `object` not iterable |
| `reportUnnecessaryComparison` | ~3 | Defensive guards on `None` / `Literal` types |
| `reportMissingTypeArgument` | ~3 | Generic `dict`, `set`, `Queue` without type args |
| `reportUnnecessaryIsInstance` | ~2 | Redundant `isinstance` checks in tokenizer |
| Other | ~6 | `reportOptionalSubscript`, `reportOptionalCall`, `reportAssignmentType` |

- [ ] Fix `reportUnusedVariable` (underscore prefixes or `_` for unpacking)
- [ ] Fix `reportReturnType` (use `Mapping` for covariant dict returns, cast numpy types)
- [ ] Fix `reportIndexIssue` (narrow `object` params to typed dicts)
- [ ] Fix `reportPossiblyUnboundVariable` (initialize before conditional branches)
- [ ] Suppress or fix remaining minor categories

### MPS Backend Improvements
PyTorch 2.9.1 on M3 Max supports float16/bfloat16 autocast and MPS-specific APIs that the codebase doesn't use. Three changes to improve MPS training performance and observability.

**Measured on M3 Max (PyTorch 2.9.1)**:
- fp16/bf16 matmuls ~10-30% faster than fp32
- SDPA fp16 ~25% faster than fp32
- `torch.mps.synchronize()` and `torch.mps.empty_cache()` both available

**Changes**:
- [ ] **dtype.py**: Detect MPS and return `torch.float16` instead of `torch.float32` — halves memory, enables GradScaler (already wired for fp16), ~10-30% speed gain
- [ ] **base_train.py / chat_sft.py**: Use `torch.mps.synchronize()` for accurate step timing and `torch.mps.current_allocated_memory()` for memory reporting (currently no-ops on MPS)
- [ ] **base_train.py / chat_sft.py**: Call `torch.mps.empty_cache()` after eval steps to reclaim memory during long runs

**Files**: `common/dtype.py`, `scripts/base_train.py`, `scripts/chat_sft.py`, `docs/m3-max-guide.md`

### Apple Silicon (MPS) Documentation — ✅ [Archived](archive/apple-silicon-mps-documentation.md)

## Deferred Phases

None currently.
