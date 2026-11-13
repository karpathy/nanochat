# Fix-forward Mac (merged, reviewer-fixed)

> **Overview:** Merged fix-forward Mac plan with a rich but disciplined echo UX: original→adjusted, timing/ETA, pipeline map, skip-savings, plain-language health — Phase 1 then Phase 2.

**Supersedes:** [2026-11-13-echoes-idempotent-resume.md](2026-11-13-echoes-idempotent-resume.md), [2026-11-13-mps-memory-auto-fit.md](2026-11-13-mps-memory-auto-fit.md).

Incorporates reviews from nw-documentarist, nw-solution-architect, and critic.

**Ship order:** Phase 1 lands first; Phase 2 depends on Phase 1 (echo contract frozen; Phase 2 only adds auto-fit lines).

```mermaid
flowchart LR
  p1["Phase1 echoes skip rms_norm"]
  p2["Phase2 defaults memory auto-fit"]
  p1 --> p2
```

## Todos

### Phase 1
- [x] Fail-fast, stage echoes+timing+ETA, pipeline map, skip-savings, markers
- [x] rms_norm fallback + compile smoke + norm echo
- [x] Stage purpose one-liner, next-action on DONE/FAIL, mem snapshot, disk artifact size
- [x] nan warn/abort-after-5; RUNNING.md echo legend + enrichment glossary

**Status:** PASSED (Phase 1) — 2026-11-13 — PR https://github.com/neomatrix369/nanochat/pull/4  
**slice_id:** `fix-forward-mac-phase1`

### Phase 2
- [x] defaults.py canonical; emit profiles.sh; test_defaults sync
- [x] memory auto-fit echoes + OOM re-halve; plain-language mem fit line

**Status:** VERIFIED — 2026-11-13 — PR https://github.com/neomatrix369/nanochat/pull/6  
**slice_id:** `fix-forward-mac-phase2`

## Reviewer fixes baked in (checklist)

- Echo **format table** + **emission order** (no guessing)
- **Doc mapping** L1/L2/L3 → exact files
- **SSOT:** Python `nanochat/defaults.py` canonical; shell obtained from Python (no dual truth)
- **rms_norm:** import-time bind + **torch.compile smoke test** on torch 2.2.2
- **Markers:** named writers for `.eval_ok` / `.tok_eval_ok`
- **NaN:** warn at first nan step; **abort after 5 consecutive nan steps**
- **Memory:** named fudge **1.5**, margin **0.6** (override `NANOCHAT_MEMORY_FIT_MARGIN`); **one OOM re-halve retry**
- **Profile:** CLI default batch stays 32; `PROFILE_RUNCPU` batch **8**; echo full precedence chain
- **FORCE_STAGES:** `all` or comma-list; never ambiguous `=1`
- Concrete tests: `tests/test_norm.py`, `tests/test_memory.py`, `tests/test_defaults.py`

---

## Echo contract (frozen in Phase 1; Phase 2 adds auto-fit only)

### Emission order (every train/eval stage)

1. Shell `START` — timestamp, stage, profile, key knobs
2. Python startup — torch version, device, paths
3. Python auto-fit / fallback — only if Phase 2 or rms_norm path (see formats)
4. Unchanged knobs one-liner (optional, once)
5. Shell `DONE` | `SKIP` | `FAIL` | `FORCE` — elapsed + cumulative; ETA rule

### Format table

| Kind | Exact shape |
|------|-------------|
| START | `[HH:MM:SS] START stage=base_train profile=runcpu depth=6 device_batch_size=8 ETA: not yet knowable` |
| DONE | `[HH:MM:SS] DONE stage=base_train elapsed=71m cumulative=2h45m` |
| SKIP | `[HH:MM:SS] SKIP stage=tok_train reason=artifact exists path=... (FORCE_STAGES=tok_train to redo)` |
| FORCE | `[HH:MM:SS] FORCE stage=tok_train reason=FORCE_STAGES=tok_train` |
| FAIL | `[HH:MM:SS] FAIL stage=base_train elapsed=… hint=…` |
| auto-fit | `[auto-fit] device_batch_size: original/default=8 → adjusted=2 (reason: est_peak=12.4GiB > margin*avail=0.6*18.0GiB)` |
| unchanged | `[auto-fit] device_batch_size=8 (unchanged; estimate fits)` |
| rms_norm | `[norm] original=F.rms_norm → adjusted=manual_rsqrt (reason: missing on torch 2.2.2)` |
| ETA known | `ETA: ~Xm remaining (step i/N, avg dt=…)` |
| ETA unknown | `ETA: not yet knowable` |
| ETA skip | `ETA: n/a (skipped)` |

Never invent ETAs. Never silently change a knob.

### Creative / intuitive enrichments (keep sparse — one idea per moment)

Goal: the log should feel like a **friendly flight attendant**, not a firehose. Prefer analogies and “what should I do next?” over raw dumps.

**In scope (Phase 1 unless noted)**

| Moment | Echo idea | Example |
|--------|-----------|---------|
| Script start | **Pipeline map** — where you are in the journey | `pipeline: [tok✓] [base▶] [eval○] [sft○] [chat_eval○]` |
| Each START | **Stage purpose** (one plain sentence) | `purpose: teach raw next-token prediction on web text` |
| Each DONE | **Next action** | `next: base_eval (or SKIP if .eval_ok)` |
| Each SKIP | **Time saved** (if we have a prior elapsed for that stage, else unknown) | `skip-saved: ~34m (last run) · cumulative_saved=~1h10m` |
| FAIL | **Root vs cascade** + single next fix | `root_stage=base_train · cascade_suppressed=yes · try: see RUNNING.md#rms-norm` |
| DONE checkpoint | **Artifact on disk** | `wrote: ~/.cache/nanochat/base_checkpoints/d6 (1.2GiB)` |
| Python startup | **Human device line** | `device: Apple MPS · dtype=float32 · torch=2.2.2` |
| After mem fit (P2) | **Plain-language fit** | `memory: aiming to use ≤60% of ~18GiB; batch 8→2 so logits fit` |
| Train progress (existing step line +) | **Heartbeat metaphor** (optional, rank0, every N steps) | `heartbeat: loss falling · 12% of steps · ETA ~40m` |
| Script end success | **How to play** | `try: python -m scripts.chat_cli -p "Why is the sky blue?"` |

**Strong candidates — Phase 1.5 / Phase 2 if cheap**

- **Knob diff vs last run** — store `~/.cache/nanochat/last_run.json` (profile, depth, batch, git sha); on start echo `changed: device_batch_size 32→8; same: depth=6`
- **Health strip** at stage boundaries: free/approx mem, checkpoint dir size, shard count `shards=171/170 needed`
- **Trend crumb** after val bpb / CORE: `val_bpb 7.47 → better than first eval 8.1` (needs prior metric in marker/json)
- **Educational tip** once per run (not per stage): `tip: smaller --device-batch-size uses grad accumulation to keep the same total batch`

**Out of scope (noise / hard / wrong layer)**

- Per-microbatch dumps, emoji spam, fake confidence scores
- Cloud $ estimates unless user opts in later
- Thermal/Fan APIs, Slack/webhook notifications
- LLM-generated narration of logs

**Discipline:** at most **one enrichment line** besides the required contract line at each event. Enrichment never replaces `original→adjusted` or ETA rules.

---

## Doc mapping (progressive disclosure)

| Layer | File | Content from this plan |
|-------|------|------------------------|
| L1 | [README.md](../../README.md) Running on CPU/MPS (~5–8 lines) | Use `runcpu.sh`; fail-fast + skip on re-run; link RUNNING.md; one line auto-fit + defaults |
| L2 | [docs/RUNNING.md](../RUNNING.md) | Pick script; fix-forward loop; artifacts; **Echo legend** (table above); Memory/OOM; Defaults/profiles; NaN troubleshooting; ETA timing note (“after first train step”) |
| L3 | [runs/runcpu.sh](../../runs/runcpu.sh) / [runs/speedrun.sh](../../runs/speedrun.sh) headers | Audience + link RUNNING.md |
| Plan SSOT in-repo | [docs/plan/](./) | This document |

---

## Phase 1 — Echoes, idempotency, rms_norm

### 1.1 Fail-fast + stage echoes + skip

Files: [runs/speedrun.sh](../../runs/speedrun.sh), [runs/runcpu.sh](../../runs/runcpu.sh)

- `set -euo pipefail`; `log()` like miniseries
- Skip gates + **marker ownership**:
  - `tok_train` — skip if `tokenizer/tokenizer.pkl`; no marker file needed
  - `tok_eval` — skip if `tokenizer/.tok_eval_ok`; **written by tok_eval success path or shell after successful exit**
  - `base_train` — skip if any `base_checkpoints/**/model_*.pt`
  - `base_eval` — skip if `base_checkpoints/.eval_ok`; **written at end of successful [scripts/base_eval.py](../../scripts/base_eval.py) main** (last action before clean exit)
  - `chat_sft` — skip if any `chatsft_checkpoints/**/model_*.pt`
  - `chat_eval` — skip if `chatsft_checkpoints/.eval_ok`; **written at end of successful [scripts/chat_eval.py](../../scripts/chat_eval.py)**
- `FORCE_STAGES=all` or `FORCE_STAGES=tok_train,base_train` (comma-list). Do not use bare `=1`.
- Soft CUDA warning on speedrun when no CUDA

### 1.2 rms_norm + compile smoke

File: [nanochat/gpt.py](../../nanochat/gpt.py)

- Import-time bind of `norm` (native vs manual rsqrt); echo `[norm] original→adjusted`
- Test [tests/test_norm.py](../../tests/test_norm.py): eager + **`torch.compile` forward must not raise `InternalTorchDynamoError`** on this machine’s torch

### 1.3 Python boundary echoes + NaN

- Startup paths/device; checkpoint missing errors clarified in [nanochat/checkpoint_manager.py](../../nanochat/checkpoint_manager.py)
- Train loop: on first `isnan(loss)` warn (not memory); **after 5 consecutive nan steps → exit nonzero** with RUNNING.md pointer

### 1.4 Phase 1 verify

- Skip dry-run with existing tokenizer cache
- `pytest tests/test_norm.py`
- Friendly missing-checkpoint message

---

## Phase 2 — Defaults SSOT + MPS auto-fit (depends on Phase 1)

### 2.0 SSOT authority

- **Canonical:** [nanochat/defaults.py](../../nanochat/defaults.py) (`VOCAB_SIZE`, `LOGIT_SOFTCAP`, `MEMORY_FIT_MARGIN`, `PROFILE_RUNCPU`, `PROFILE_SPEEDRUN`, argparse defaults)
- **Shell:** [runs/profiles.sh](../../runs/profiles.sh) is **generated or emitted** from Python (e.g. `python -m nanochat.defaults emit-shell > runs/profiles.sh` checked in, or `source <(python -m nanochat.defaults emit-shell)` at script start). Python wins on conflict.
- Test [tests/test_defaults.py](../../tests/test_defaults.py): parse emitted shell vars == `PROFILE_*`

### 2.1 Memory helper

[nanochat/memory.py](../../nanochat/memory.py):

- Estimate with **documented fudge 1.5** on (params + optim + logits `B*T*V*4` + 1.5× logits activations)
- Margin default **0.6**; env `NANOCHAT_MEMORY_FIT_MARGIN`
- `fit_device_batch_size` halves to ≥1
- On train OOM after fit: **empty_cache, halve once more, retry once**; if still OOM at 1, fail with clear echo
- MPS: `empty_cache` after device known, **before** model load in eval
- Test [tests/test_memory.py](../../tests/test_memory.py): fit shrinks; estimate monotonic in B

### 2.2 Wire auto-fit

- `--auto-fit-batch`: default **True on mps**, **False on cuda**
- Echo precedence: `CLI_default → profile → fitted` when they differ
- `PROFILE_RUNCPU.device_batch_size=8`; global argparse default remains **32**

### 2.3 Phase 2 verify

- `pytest tests/test_defaults.py tests/test_memory.py`
- Dry-run fitted B print for runcpu profile

---

## Out of scope

- Full miniseries/scaling_laws profile refactor
- Every optimizer LR extracted
- Root-cause nan beyond warn/abort
- Chunked cross-entropy
- Perfect MPS fragmentation model
