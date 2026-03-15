---
title: "nanochat Roadmap"
summary: "Current and future implementation phases for nanochat."
read_when:
  - Planning nanochat development
  - Deciding what to implement next
  - Understanding scope and sequencing
status: draft
last_updated: "2026-03-16"
---

# nanochat Roadmap

## Current Status

- **Phase 0**: Architecture optimizations — ✅ Complete
- **Phase 0.5**: Refactor to modular architecture — ✅ Complete (2026-03-13)
- **Phase 1**: Architecture experiments — ❌ Attempted, negative results
- **Phase 1.5**: Compression-based optimization — 🔜 Code done, validation pending

## Completed Phases

| Phase                                                                                   | Completed  | Summary                                                                                         |
| --------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------- |
| [Phase 0 — Optimizations](archive/phase-0-optimizations.md)                             | 2026-03-05 | FA3, FP8, ClimbMix (27% speedup), Muon, sliding window, auto batch                              |
| [Phase 0.5 — Refactor](archive/phase-0.5-refactor.md)                                   | 2026-03-13 | Modular src/nanochat/ layout, 44 files, 30 tests, CI/CD                                         |
| [Phase 1 — Architecture Experiments](archive/phase-1-architecture-experiments.md)       | 2026-02-19 | SwiGLU, MoE, multi-token prediction — all negative results                                      |
| [Phase 1.5.0 — Data Layout & Config](archive/phase-1.5.0-data-layout-config.md)         | 2026-03-14 | Config system, centralized paths, hierarchical dirs, Python 3.13                                |
| [Phase 1.5.0.1 — Script Entry-Points](archive/phase-1.5.0.1-script-entry-points.md)     | 2026-03-15 | Wrapped all scripts in main(), importable without side effects                                  |
| [Phase 1.5.0.2 — Code Review & Cleanup](archive/phase-1.5.0.2-code-review-cleanup.md)   | 2026-03-15 | Full code review, 6 fixes, lazy USE_FA3, paths wiring, all 9 scripts wrapped                    |
| [Refactor common.py into common/ Package](archive/refactor-common-package.md)           | 2026-03-15 | Split monolithic common.py into 7 focused modules, absorbed paths.py, backward-compatible       |
| [Type Annotations & Pyright Compliance](archive/type-annotations-pyright-compliance.md) | 2026-03-15 | Pyright strict mode, 17 suppression rules, 323→0 reportMissingParameterType across 29 files     |
| [Apple Silicon (MPS) Documentation](archive/apple-silicon-mps-documentation.md)         | 2026-03-15 | Rewrote MPS guide with accurate device detection, dtype, limitations, and batch size guidelines |
| [MPS Backend Improvements](archive/mps-backend-improvements.md)                         | 2026-03-15 | fp16 dtype, torch.mps.synchronize/memory, empty_cache, get_device_sync refactor                 |
| [Pyright Strict Compliance](archive/pyright-strict-compliance.md)                       | 2026-03-16 | 128→0 pyright errors across 10 categories under strict mode                                     |
| [Phase 1.5.1 Bugfixes & Tooling](archive/phase-1.5.1-bugfixes-tooling.md)               | 2026-03-15 | --config/--base-dir plumbing, argparse SUPPRESS fix, compression console output, LocalWandb     |

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

## Deferred Phases

None currently.

## Open Questions

### MPS fp16 gradient stability

`get_compute_dtype()` returns `torch.float16` on MPS (MPS has no bf16 support). `GradScaler` is CUDA-only and silently disables itself on MPS, producing a warning:

```
UserWarning: torch.cuda.amp.GradScaler is enabled, but CUDA is not available. Disabling.
```

This means fp16 training on MPS runs **without gradient scaling**. Whether this is safe is unknown:

- **If MPS handles fp16 gradient stability internally** → gate scaler on `device_type == "cuda"`, suppress the warning, no further action needed
- **If MPS fp16 is unstable without scaling** → need an alternative: either force bf16 on MPS (if a future PyTorch version adds support), use fp32, or implement a device-agnostic scaler

**Required experiment**: run two d8 training runs on MPS to completion and compare loss curves:
1. Current behavior (no scaler, fp16)
2. Forced fp32 (`--device-type` override or `COMPUTE_DTYPE=fp32`) as a stable baseline

If curves match → fp16 on MPS is stable, fix is just gating the scaler on CUDA. If fp16 diverges → escalate to a proper fix before any long MPS runs.

## Improvements

### Unified CLI, sectioned TOML config, and wandb consolidation

See [unified-cli.md](unified-cli.md) for the full design. Summary:

- Sectioned TOML (`[common]`, `[training]`, `[sft]`, ...) with matching dataclasses (`CommonConfig`, `TrainingConfig(CommonConfig)`, `SFTConfig(CommonConfig)`)
- `CommonConfig.wandb: online | local | disabled` replaces `--run="dummy"` magic and `WANDB_MODE` env var
- Shared `load_config()` and `init_wandb()` helpers eliminate duplication across all entry points
- Single `nanochat` CLI with subcommands; global `--config` / `--base-dir` on all entry points
- CLI args always override TOML values

### Full CLI flag audit across all entry points

A full review of all flags across all 10 entry points to identify inconsistencies, missing flags, and opportunities for shared config. Current state:

| Flag | `base_train` | `chat_sft` | `chat_rl` | `base_eval` | `chat_eval` | `tok_train` | `tok_eval` | `chat_cli` | `chat_web` | `data download` |
|---|---|---|---|---|---|---|---|---|---|---|
| `--config` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--base-dir` | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| `--device-type` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--run` (wandb) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--model-tag` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |
| `--step` | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |
| `--device-batch-size` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--eval-every` | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--save-every` | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| short flags (`-g`, `-s`, `-t`) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |

**Key issues**:
- `--base-dir` and `--config` missing from 7+ entry points
- `--device-type` missing from inference/eval scripts
- `--save-every` inconsistent between `chat_sft` and `chat_rl`
- Short flags (`-g`, `-s`, `-t`) only on interactive scripts — inconsistent style
- `--source` (sft|rl) used in eval/chat scripts but named differently from `--model-tag`

This audit should be done as part of the unified CLI work.
