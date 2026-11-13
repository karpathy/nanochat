# Fix-forward Mac Phase 1.5 — sparse enrichments

> **Overview:** Ship the four “cheap” enrichments deferred from [2026-11-13-fix-forward-mac.md](2026-11-13-fix-forward-mac.md) Phase 1.5 candidates — knob diff, health strip, trend crumb, one tip — without breaking the frozen echo contract.

**Status:** IMPLEMENTED  
<!-- transition: PROPOSED → IMPLEMENTED 2026-11-13 (execute-slice Phase 1 starting) -->  
**slice_id:** `fix-forward-mac-phase15`  
**Depends on:** Phase 1 + Phase 2 (merged)  
**Branch:** `feat/fix-forward-mac-phase15` (streams: `stream/p15-{shell,trend,tests,docs}`)

## Todos

- [x] Knob diff vs last run (`last_run.json`)
- [x] Health strip at stage boundaries (mem + ckpt size + shards) — packed with `purpose:` on one START line
- [x] Trend crumb after val_bpb / CORE — fail-soft `emit_metric_trend`
- [x] Educational tip once per script run (`nanochat/run_tip.py`)
- [x] RUNNING.md glossary rows + tests

## Acceptance criteria

1. **Knob diff:** On script start (after pipeline map), if `$NANOCHAT_BASE_DIR/last_run.json` exists, print exactly one enrichment line comparing current profile knobs to the stored run, e.g. `changed: device_batch_size 32→8; same: depth=6,profile=runcpu`. If missing, skip silently (first run). After a successful full script (`ALL DONE`), write/update `last_run.json` with at least: `profile`, `depth`, `device_batch_size`, `max_seq_len`, `git_sha`, `ts`.
2. **Health strip:** At each stage `START` (one enrichment after `purpose:`), print `health: <mem> · ckpt=<size or n/a> · shards=<n/needed or n/a>` using best-effort discovery. Never fail the run if discovery fails.
3. **Trend crumb:** When `base_train` / `base_eval` prints a new `val_bpb` or CORE metric and a prior value exists in `$NANOCHAT_BASE_DIR/last_metrics.json`, print one line e.g. `trend: val_bpb 7.47 → better than prior 8.1`. Update `last_metrics.json` on successful metric write. No trend line when no prior.
4. **Tip once:** Each of `runcpu.sh` / `speedrun.sh` prints exactly one `tip:` line per invocation (script start, after knob diff or pipeline), not per stage. Tip text may be fixed or rotated from a tiny static list; must mention a real knob (e.g. `--device-batch-size` / grad accumulation).
5. **Docs:** `docs/RUNNING.md` enrichment glossary documents `changed:`/`same:`, `health:`, `trend:`, `tip:`.
6. **Discipline:** Still ≤ one enrichment line besides the required contract line at each event. No emoji spam. Frozen START/DONE/SKIP/FAIL/auto-fit shapes unchanged.
7. **Tests:** Unit tests cover (a) last_run diff formatting when keys change/same/missing file, (b) trend crumb when prior exists vs absent, (c) tip emitted once in a dry helper if extracted.

## Scope

**In**
- `runs/_stage_lib.sh`, `runs/runcpu.sh`, `runs/speedrun.sh`
- Small helper(s) under `nanochat/` if JSON diff/metrics need Python (prefer shell for last_run if trivial; Python OK for metrics)
- `scripts/base_train.py`, `scripts/base_eval.py` (trend only; minimal touch)
- `docs/RUNNING.md`, this plan / TRAIL / PROGRESS / GROWTH
- `tests/test_phase15_enrichments.py` (or split files)

**Out**
- Heartbeat / ETA train-loop work
- Cloud $ estimates, thermal APIs, webhooks
- Changing Phase 1 echo contract formats
- Miniseries / scaling_laws refactor

## Work streams

| Stream | Owner files | Depends |
|--------|-------------|---------|
| shell-enrich | `_stage_lib.sh`, runcpu, speedrun — last_run, health, tip | — |
| trend-python | base_train / base_eval + `last_metrics.json` helper | — |
| tests | `tests/test_phase15_enrichments.py` (+ any helper module) | after shell+trend interfaces exist (can stub against pure helpers) |
| docs | RUNNING.md glossary | format strings frozen in this plan |

## Derived smoke tests [DERIVED]

| AC | Smoke |
|----|-------|
| 1 | Write a fake `last_run.json`, invoke helper/diff; assert one `changed:`/`same:` line |
| 2 | Call health formatter with temp dirs; assert `health:` prefix and no throw |
| 3 | Call trend helper with/without prior; assert line presence rules |
| 4 | Grep runcpu/speedrun for a single tip emission site at script start |
| 5 | Grep RUNNING.md for the four new glossary terms |

## Reviewers

| Reviewer | Applicable? |
|----------|-------------|
| nw-software-crafter-reviewer | YES — shell + Python logic |
| nw-acceptance-designer-reviewer | YES — echo/CLI UX contract |
| nw-solution-architect-reviewer | YES — enrichment vs frozen contract boundary |
| nw-ddd-architect-reviewer | NOT_APPLICABLE — no domain/schema model |
