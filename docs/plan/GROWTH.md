# Abstraction Views Delta (GROWTH)

## fix-forward-mac-phase1

**Added**
- `runs/_stage_lib.sh` — shared stage orchestration (log, skip gates, FORCE_STAGES, pipeline map, DONE enrichments)
- `docs/RUNNING.md` — L2 progressive runbook (echo legend, artifacts, Mac/NaN troubleshooting)
- `tests/test_norm.py`, `tests/test_checkpoint_errors.py` — Mac-compat and checkpoint UX coverage

**Changed**
- `nanochat/gpt.py` — import-time `norm` bind (native `F.rms_norm` vs manual rsqrt)
- `runs/runcpu.sh` / `runs/speedrun.sh` — fail-fast idempotent stage wrappers
- `nanochat/checkpoint_manager.py` — actionable missing-dir errors
- `scripts/{base_train,base_eval,tok_eval,chat_eval}.py` — NaN abort / eval markers / startup echo

**Deferred (Phase 2)**
- `nanochat/defaults.py` + emitted `runs/profiles.sh` SSOT
- `nanochat/memory.py` auto-fit + OOM re-halve
