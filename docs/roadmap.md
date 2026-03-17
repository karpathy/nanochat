---
title: "nanochat Roadmap"
summary: "Development phases, active work, and future direction for nanochat."
read_when:
  - Planning nanochat development
  - Deciding what to implement next
  - Understanding scope and sequencing
status: draft
last_updated: "2026-06-14"
---

# nanochat Roadmap

## Completed

| Phase | Completed | Summary |
|---|---|---|
| [Phase 0 — Optimizations](archive/phase-0-optimizations.md) | 2026-03-05 | FA3, FP8, ClimbMix (27% speedup), Muon, sliding window, auto batch |
| Phase 0.5 — Modular Refactor | 2026-03-13 | src/nanochat/ layout, unified CLI, pyright strict, CI/CD |
| [Phase 1 — Architecture Experiments](archive/phase-1-architecture-experiments.md) | 2026-02-19 | SwiGLU, MoE, MTP — all negative results |
| [Phase 1.5.0 — Data Layout & Config](archive/phase-1.5.0-data-layout-config.md) | 2026-03-14 | Config system, centralized paths, hierarchical dirs, Python 3.13 |
| [Phase 1.5.1 — Bugfixes & Tooling](archive/phase-1.5.1-bugfixes-tooling.md) | 2026-03-15 | argparse SUPPRESS fix, compression console output, LocalWandb |

## Active — Phase 1.5: Compression-Based Optimization

**Goal**: Validate whether compression-based optimization improves training efficiency before investing in scaling infrastructure.

| Sub-phase | Status | Notes |
|---|---|---|
| 1.5.0 — Data layout & config | ✅ Done | |
| 1.5.1 — Compression metrics | ✅ Code done | [Validation checklist](phase-1.5.1-validation-checklist.md) |
| 1.5.2 — Dataset quality via compression | 🔜 Pending | |
| 1.5.3 — Compression-aware optimization | 🔜 Pending | |

**Exit criteria**:
- [ ] Compression ratio correlates with val loss (R² > 0.7)
- [ ] Compression-aware optimizer shows 10%+ faster convergence
- [ ] Overall 15%+ improvement → proceed to Phase 2

**Decision point** (after 1.5.3):
- **>15%** → Phase 2 (infrastructure), then scale to 7B
- **5–15%** → refine and iterate at small scale
- **<5%** → skip to Phase 6 (SP-Transformer hybrid)

## Future Phases

Sequencing depends on the Phase 1.5 outcome.

| Phase | Description | Plan |
|---|---|---|
| Phase 2 — Training Infrastructure | Scale beyond single node, enable 10B+ training | — |
| Phase 3 — Data Pipeline | Compression-aware data quality and selection | [plan](phase-3-data-pipeline.md) |
| Phase 4 — Post-Training Alignment | SFT, RLHF, stable assistant behavior | — |
| Phase 5 — Capabilities | Long context, tool use, multimodal | [plan](phase-5-tools-transparency.md) |
| Phase 6 — SP-Transformer Hybrid | Combine transformer efficiency with SP Theory | [plan](phase-6-hybrid-architecture.md) |

## Deferred

- **TrainingState refactor** — extract mutable training loop state into a dataclass, eliminate the closure in `train_base`. See [plan](training-state-refactor.md).
- **MPS fp16 gradient stability** — `GradScaler` silently disables on MPS; unknown whether fp16 is stable without it. Needs a d8 comparison run: fp16 vs fp32 loss curves.
