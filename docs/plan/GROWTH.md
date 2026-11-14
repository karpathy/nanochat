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

**Deferred (Phase 2)** — now shipped; see `fix-forward-mac-phase2` below
- `nanochat/defaults.py` + emitted `runs/profiles.sh` SSOT
- `nanochat/memory.py` auto-fit + OOM re-halve

## fix-forward-mac-phase2

**Added**
- `nanochat/defaults.py` — SSOT for profiles, vocab/softcap, memory margin/fudge; `emit-shell` → `runs/profiles.sh`
- `nanochat/memory.py` — peak estimate, `fit_device_batch_size`, auto-fit echoes, OOM rehalve helpers (eager fallback after rehalve)
- `tests/test_defaults.py`, extended `tests/test_memory.py` — SSOT sync + fit/echo/OOM coverage

**Changed**
- `scripts/{base_train,base_eval,chat_sft}.py` — `--auto-fit-batch` (mps on / cuda off); preflight fit; train OOM retry once then eager
- `runs/runcpu.sh` / `runs/speedrun.sh` — `source profiles.sh`; knobs from `$RUNCPU_*` / `$SPEEDRUN_*` (runcpu batch 8)
- `docs/RUNNING.md` / README — Phase 2 memory/auto-fit + defaults reference

## fix-forward-mac-phase15

**Added**
- `nanochat/last_run.py` — knob diff + `last_run.json` write (CLI for shell)
- `nanochat/metrics_trend.py` — trend crumb + fail-soft `emit_metric_trend`
- `nanochat/run_tip.py` — tip text SSOT
- `tests/test_phase15_enrichments.py` — knob/trend/tip/fail-soft coverage

**Changed**
- `runs/_stage_lib.sh` / `runcpu.sh` / `speedrun.sh` — script-start tip+knob; packed START `purpose · health`; write last_run on ALL DONE
- `scripts/{base_train,base_eval}.py` — trend after val_bpb / CORE
- `docs/RUNNING.md` — enrichment glossary for changed/health/trend/tip

**Deferred**
- Heartbeat / train-loop ETA (still planned)
- Tip rotation / knob-diff vs last run UI beyond one line

## fix-mps-skip-inductor-compile

**Added**
- `nanochat.common.inductor_supports_device` / `maybe_torch_compile` — skip Inductor on MPS (echo lives in the helper)
- `tests/test_compile_device.py` — eager-on-mps + AdamW/Muon dispatch + foreach-copy path
- `tests/test_stage_completion_markers.sh` — `.train_ok` required for train SKIP
- `tests/test_rotary_cache.py` — initial `sequence_len*10` cache + grow-on-demand

**Changed**
- `scripts/{base_train,chat_sft}.py` — compile via helper; step-0-only forward announce; write `.train_ok`
- `nanochat/optim.py` — AdamW/Muon MPS float-scalar paths; no empty-params IndexError; foreach bypass
- `nanochat/loss_eval.py` — val-bpb progress on non-CUDA only (CUDA stays quiet)
- `nanochat/gpt.py` — grow RoPE cos/sin when `T0+T` exceeds cache (`sequence_len*10`, 2× buffer)
- `runs/speedrun.sh` — hard `exit 1` without CUDA
- `runs/_stage_lib.sh` — train SKIP via `.train_ok` (not any `model_*.pt`)
- `docs/RUNNING.md` — `[compile]` legend row; CUDA/speedrun + marker docs

**Deferred (parked)**
- Softcap / logits memory (upstreamable gather-at-needed-positions vs fork `PROFILE_RUNCPU` knobs) — discussed in playgroup softcap/eval chat; not scheduled until a design is picked

## cli-learn-color-ux

**Added**
- `nanochat/glossary.py` — term catalog (9 terms), `STAGE_TERMS`, `format_learn_line`, `emit_define`; CLI `learn --stage`
- `nanochat/term_style.py` — `enabled()` (TTY and `NO_COLOR`), `paint(role, text)` for direct print/shell output
- `nanochat/stage_art.py` — ASCII plaque renderer and CLI; `NANOCHAT_PLAIN=1` suppresses art
- `tests/test_glossary.py`, `tests/test_term_style.py`, `tests/test_stage_art.py` — GWT-named unit tests; §6b parse test keeps RUNNING in sync with the glossary

**Changed**
- `runs/_stage_lib.sh` — colour on START/DONE/SKIP/FAIL; plaques on START/DONE/FAIL; `learn:` line after `purpose:`; per-command `NANOCHAT_STAGE_BRIEFED`
- `runs/modal_run.py` — `NO_COLOR=1` in `_modal_env()`
- `scripts/base_train.py` — `emit_define` for BPB/CORE/MFU/Muon; done-colour on metric lines and trend
- `docs/RUNNING.md` — §6b Terms table; §5 emission order; §6 two START enrichment lines; §9 hook failure DONE→FAIL; duplicate top blurb removed
- `README.md` — one bullet under CPU/MPS pointing to RUNNING §5–§6b

**Deferred**
- `scripts/tok_train.py`, `tok_eval.py`, `chat_sft.py`, `chat_eval.py` colour and define lines (Could)
- `scripts/chat_cli.py` welcome plaque (Could)
