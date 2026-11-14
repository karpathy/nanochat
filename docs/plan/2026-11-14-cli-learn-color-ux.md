# CLI stage marking, glossary, and color UX

> **Overview:** Make pipeline stages easy to spot in console and log output (color plus a small ASCII plaque at each stage start, done, or fail), teach each stage's jargon once with a definition and link, keep the frozen echo contract intact, and align RUNNING.md and README.md with a progressive-disclosure ladder (README teaser, then RUNNING legend and jargon, then deep links).

## Plan home

Canonical copy: [`docs/plan/2026-11-14-cli-learn-color-ux.md`](2026-11-14-cli-learn-color-ux.md) (this file, git-tracked SSOT).

**Status:** PROPOSED. Branch `feat/cli-learn-color-ux` and worktree `.worktrees/cli-learn-color-ux` already exist at `master` with no commits yet. Nothing is implemented until this plan is approved.
**slice_id:** `cli-learn-color-ux`

## Current code this plan builds on

Verified against the tree at `master` (edfb3d3, post PR #18 and #20).

| Concern | Where it lives today | How this plan uses it |
|---------|----------------------|-----------------------|
| Stage contract lines `START` / `DONE` / `SKIP` / `FAIL` | `runs/_stage_lib.sh` `run_stage()` (uses `log`, which prefixes `[HH:MM:SS]`) | Wrap in color only; do not change tokens |
| Stage purpose text | `runs/_stage_lib.sh` `stage_purpose()` | **Single owner.** Plaque subtitle reuses it. Glossary does not repeat it |
| `purpose:` + `health:` line | `run_stage()` before the train command | `learn:` line goes directly after it |
| `pipeline:` map | `pipeline_map()`, called from `runs/runcpu.sh:48`, `runs/speedrun.sh:66` | Color prefix only |
| Tip line | `print_run_tip()` → `python3 -m nanochat.run_tip` | Color prefix only; pattern for new CLIs |
| Banner | `nanochat/common.py` `print_banner()`, called at import in `scripts/base_train.py:42` | **Stays the only banner.** No new shell banner (see decisions) |
| Logging color | `nanochat/common.py` `ColoredFormatter` (line 34), `setup_default_logging()` (line 60) | Separate path. Not reused; see decisions |
| Metric trend | `nanochat/metrics_trend.py` `format_trend()`, `emit_metric_trend(print_fn=...)` | Color applied at the call site through `print_fn`. Module unchanged |
| Run tee | `runs/runmodal.sh:79` pipes to `tee run.log` | Stdout is not a TTY there, so color stays off and the archived log stays plain |
| Run record spans | `run_stage()` calls `record_span` on success, failure, and hook failure; writes `$NANOCHAT_RUN_DIR/events.jsonl` via `nanochat/run_record.py` | Must keep every `record_span` call and its order. Plaques and `learn:` print text only and never touch the record |
| Post-stage hook | `NANOCHAT_POST_STAGE_HOOK` runs after DONE (Modal volume commit) and can turn DONE into FAIL | DONE plaque prints with the DONE line. If the hook fails, print a FAIL plaque, same as the existing FAIL line |
| Modal docs | `docs/MODAL.md` stage table and marker text; links to RUNNING for stage detail | No edit needed. Plaques show in `run.log`, which MODAL.md does not describe |
| Chat stages | `runs/speedrun.sh` runs `chat_eval` via `scripts/chat_eval.py`; `chat_cli.py` is the interactive chat, not a stage | `chat_eval` has a STAGE_TERMS entry (ChatCORE), so `scripts/chat_eval.py` joins the Could list |
| Tests to extend | `tests/test_phase15_enrichments.py`, `tests/test_stage_completion_markers.sh`, `tests/test_run_record.py` | Extend existing files for shell and enrichment checks. New test file only for `glossary` / `stage_art` / `term_style` |
| Ledger | `docs/plan/GROWTH.md` (Added / Changed / Deferred sections), `docs/plan/PROGRESS.md` (table: date, PR, summary) | One entry each |

## Decisions (locked)

- **Definitions once per stage.** The shell `learn:` line at `START` is the briefing. `run_stage()` exports `NANOCHAT_STAGE_BRIEFED=1` before it runs the stage command. `emit_define` is a no-op when that variable is set. A Python script run alone (no shell briefing) prints `define:` once per process.
- **Banner prints once.** `base_train.py:42` already prints the banner. The shell adds no separate script-start banner. This avoids a double banner under `runcpu.sh` and `speedrun.sh`.
- **Plaques only on START, DONE, FAIL.** SKIP gets color only. Re-running a finished pipeline would otherwise print one plaque per skipped stage.
- **Plaque subtitle comes from `stage_purpose()`.** The Python plaque CLI does not keep its own copy of the purpose text. Shell passes the text in, or Python reads it from one shared place. Exact mechanism is decided in step 2.
- **Color rules.** Color is on only when stdout is a TTY and `NO_COLOR` is unset. Non-TTY output (CI, `tee`, redirected logs) is plain text.
- **Art rules.** Art stays on when output is redirected, because the ASCII borders are greppable and help scrolling saved logs. `NANOCHAT_PLAIN=1` is the only opt-out, and it works in a TTY too.
- **Two color helpers, not merged.** `ColoredFormatter` in `common.py` serves `logging` records. `nanochat/term_style.py` serves direct `print` / shell ANSI output. Different concerns, so no shared code.
- **Trend color lives at the call site.** `emit_metric_trend` takes a `print_fn`. The `base_train` caller passes a color-aware wrapper. `metrics_trend.py` does not import `term_style`.
- **Contract is frozen.** Do not change `START` / `DONE` / `SKIP` / `FAIL` / `FORCE` tokens, field order, or wording in `runs/_stage_lib.sh`. Color and plaques are added around them.

## Goals and acceptance (measurable)

When someone reads a console or log, each stage boundary is visible without scrolling back:

1. Every stage START, DONE, and FAIL emits exactly one plaque and one contract line (SKIP emits only the contract line).
2. The `learn:` line has at most 3 terms, is one line, and is 100 columns or fewer for the `base_train` stage.
3. Zero enrichment lines are printed inside the train-step loop.
4. With `NO_COLOR=1`, `NANOCHAT_PLAIN` unset, and output piped: contract lines match the current text exactly (diff against a captured baseline is empty except for plaque lines).
5. With `NANOCHAT_PLAIN=1`: no plaques, no banner change, contract and `learn:` lines still print.

## Scope (MoSCoW)

**Must** (this slice ships only if these pass)
- `nanochat/glossary.py`: term catalog, `STAGE_TERMS`, `format_learn_line`, `emit_define`, CLI `learn --stage`.
- `nanochat/term_style.py`: `enabled()`, `paint(role, text)`.
- `runs/_stage_lib.sh`: color on START / DONE / SKIP / FAIL; `learn:` line after `purpose:`; `NANOCHAT_STAGE_BRIEFED` export; plaques on START / DONE / FAIL; color on `pipeline:` and `tip:` prefixes.
- Docs: RUNNING.md duplicate blurb removed (§ top, lines 10 and 12 are identical), new §6b Jargon table, Contents line updated, §5 and §6 document plaques, color roles, `learn:` / `define:`, `NO_COLOR`, `NANOCHAT_PLAIN`. README.md: one bullet under "Running on CPU / MPS".
- Tests (see below) and ledger entries.

**Should**
- `nanochat/stage_art.py`: plaque renderer and CLI (`plaque --stage X --kind start|done|fail`), ASCII-safe borders.
- `scripts/base_train.py`: color `Validation bpb` and `CORE metric` lines and the `mfu` field; `emit_define` for bpb, CORE, mfu, Muon when unbriefed; color trend via call-site `print_fn`.

**Could** (listed in the plan, not promised for this slice)
- `scripts/tok_train.py`, `scripts/tok_eval.py`, `scripts/chat_sft.py`, `scripts/chat_eval.py`: color key lines and define lines.
- `scripts/chat_cli.py`: welcome plaque, `User:` / `Assistant:` label colors.
- RUNNING "Example stage block" diagram (§6b neighbour).

**Won't** (this slice)
- A separate script-start banner in shell (`base_train` already prints one).
- Routing `print_banner()` through `stage_art` (no second owner for the banner).
- Rewriting `scripts/base_eval.py` bare `print0("="*80)` headers (lines 190, 224, 250) into stage-art section frames. Output is already readable, and the rewrite changes existing script output.
- Changing contract wording or field order, per-train-step output, tip rotation, giant figlet, heartbeat or ETA work, Web UI.
- Full DIVIO restructure into `docs/tutorials|howto|…` (single L2 RUNNING stays).

## Implementation steps

Each step is one reviewable commit on `feat/cli-learn-color-ux`. Run `uv run pytest` and `bash -n runs/_stage_lib.sh` before each commit.

1. **Glossary and color SSOT (Must).** Add `nanochat/glossary.py` and `nanochat/term_style.py`. Terms: BPE, vocab, compression ratio, BPB / val_bpb, CORE, ChatCORE, MFU, Muon, FLOPs. Skip softcap and DDP: no `STAGE_TERMS` entry uses them. Unit tests in `tests/test_glossary.py` and `tests/test_term_style.py`.
2. **Shell color, learn line, plaques (Must).** Edit `runs/_stage_lib.sh`: color helpers gated on TTY and `NO_COLOR`; plaque before START, DONE, and FAIL (FAIL plaque prints before the existing `exit "$rc"`); `learn:` line after `purpose:`; export `NANOCHAT_STAGE_BRIEFED=1` before `"$@"`. Color `pipeline:` and `tip:` prefixes. Extend `tests/test_phase15_enrichments.py` and `tests/test_stage_completion_markers.sh` (contract tokens unchanged, SKIP path has no plaque).
3. **Plaque renderer (Should).** Add `nanochat/stage_art.py` plus CLI. Subtitle comes from `stage_purpose()` (see decisions). `NANOCHAT_PLAIN=1` short-circuits to empty output. Tests in `tests/test_stage_art.py`.
4. **Python define and metric color (Should).** `base_train.py`: `emit_define` for bpb / CORE / mfu / Muon, no-op under `NANOCHAT_STAGE_BRIEFED`; color metric lines and trend via call-site `print_fn`. Verify `test_loss_eval_progress.py` and `test_format_duration.py` still pass.
5. **Docs (Must).** RUNNING.md: remove the duplicate blurb, add §6b Jargon (9 rows matching `STAGE_TERMS` keys, each cell 12 words or fewer, links from `glossary.py`), update the Contents line, document plaques and color in §5 and §6. README.md: one bullet under "Running on CPU / MPS" pointing to RUNNING §5–§6b. Add a test in `tests/test_glossary.py` that parses the RUNNING §6b table and asserts each term key exists in `glossary.py` with the same URL.
6. **Ledger and smoke (Must).** `docs/plan/GROWTH.md` (Added / Changed) and `docs/plan/PROGRESS.md` (one row). Smoke: `python -m nanochat.glossary learn --stage base_train`, `python -m nanochat.stage_art plaque --stage base_train --kind start`, `bash runs/runcpu.sh` with `NO_COLOR=1` and piped output (contract lines diff-clean), same with `NANOCHAT_PLAIN=1`.

## Docs audit (verified)

| Layer | Today | Gap this plan closes |
|-------|-------|----------------------|
| L0 console | `purpose:`, `health:`, `tip:`, `pipeline:`, `next:`, `trend:` enrichments | No `learn:`, no plaques, no color |
| L1 README | CPU/MPS section (line ~134) with a paragraph and bullets, links to RUNNING | No pointer that the console explains terms |
| L1 README terms | `val_bpb` defined inline at line 126; `CORE` named at lines 24 and 127 with no definition or link | Add the CORE link via RUNNING §6b, not a README glossary |
| L2 RUNNING | 12 sections; top blurb repeated at lines 10 and 12 | Remove duplicate; add §6b; document new console output |
| L3 deep | README Guides, external papers | No change. Do not paste papers into RUNNING |

### Disclosure ladder

1. **Console (L0):** plaque → `START` → `purpose:` → `learn:` → work → colored metrics.
2. **README (L1):** one bullet. Do not grow README into a glossary.
3. **RUNNING (L2):** how-to sections §1–4, §8–10 first; reference §5, §6, §6b, §9. One file so a stuck runner stays on one page.
4. **Deeper (L3):** links only, from `learn:` lines and the §6b table.

### DX checks (testable)

- The RUNNING Contents line links §5, §6, §6b, and §9.
- §6b term keys and URLs match `nanochat/glossary.py` (enforced by the test in step 5).
- README gains exactly one bullet; RUNNING gains no tutorial prose in README.

## Risks and open items

- **Modal path shares `_stage_lib.sh`.** `runs/runmodal.sh` runs through `tee`, so color stays off and the archived `run.log` is plain. The plaques will appear in `run.log` too. Confirm the Modal tests (`tests/test_runmodal_mode.py`) still pass in step 2.
- **Define gating depends on the env var.** If a future script runs a stage without `run_stage`, it will print `define:` lines. That is intended: it is the standalone case.
- **`base_eval` headers are Won't this slice.** They stay as three bare `====` lines. Changing them would alter existing script output.

## Out of scope

- Changing START / DONE / SKIP / FAIL / FORCE tokens, field order, or wording
- Per-step definition spam, tip rotation, giant figlet every train step
- Heartbeat / ETA train-loop work (still deferred)
- Web UI (this repo is CLI and shell)
- Full DIVIO restructure
