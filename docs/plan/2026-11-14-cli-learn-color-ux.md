# CLI stage marking, glossary, and color UX

> **Overview:** Make pipeline stages easy to spot in console and log output (color plus a small ASCII plaque at each stage start, done, or fail), teach each stage's jargon once with a definition and link, keep the frozen echo contract intact, and align RUNNING.md and README.md with a progressive-disclosure ladder (README teaser, then RUNNING legend and terms, then deep links).

## Plan home

Canonical copy: [`docs/plan/2026-11-14-cli-learn-color-ux.md`](2026-11-14-cli-learn-color-ux.md) (this file, git-tracked SSOT).

**Status:** IMPLEMENTED on branch `feat/cli-learn-color-impl` (from master). Reviewed by nw software-crafter, acceptance-designer, and solution-architect reviewers; blockers fixed. PR pending.
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
| Run tee | `runs/runmodal.sh:79–80` pipes output to `tee "$ARCHIVE_DIR/run.log"` | Stdout is not a TTY there, so color stays off and the archived log stays plain |
| Modal env | `runs/modal_run.py` `_modal_env()` (line 92) builds the container env | Set `NO_COLOR=1` there explicitly; do not rely on TTY detection in the container |
| Run record spans | `run_stage()` calls `record_span` on success, failure, and hook failure; writes `$NANOCHAT_RUN_DIR/events.jsonl` via `nanochat/run_record.py` | Must keep every `record_span` call and its order. Plaques and `learn:` print text only and never touch the record |
| Post-stage hook | `NANOCHAT_POST_STAGE_HOOK` runs after the DONE line (Modal volume commit); on failure `run_stage()` logs FAIL and exits in three branches (hook-failure `case`) | A hook failure after DONE is a legitimate DONE-then-FAIL pair: two plaques, one per contract line |
| Modal docs | `docs/MODAL.md` stage table and marker text; links to RUNNING for stage detail | No edit needed. Plaques show in `run.log`, which MODAL.md does not describe |
| Chat stages | `runs/speedrun.sh` runs `chat_eval` via `scripts/chat_eval.py`; `chat_cli.py` is the interactive chat, not a stage | `chat_eval` has a STAGE_TERMS entry (ChatCORE), so `scripts/chat_eval.py` joins the Could list |
| Tests to extend | `tests/test_phase15_enrichments.py`, `tests/test_stage_completion_markers.sh`, `tests/test_run_record.py`, `tests/test_runmodal_mode.py` | Extend existing files for shell, enrichment, and Modal env checks. New test files only for `glossary` / `stage_art` / `term_style` |
| Ledger | `docs/plan/GROWTH.md` (Added / Changed / Deferred sections), `docs/plan/PROGRESS.md` (table: date, PR, summary) | One entry each |

## Decisions (locked)

- **Definitions once per stage.** The shell `learn:` line at `START` is the briefing. `run_stage()` sets `NANOCHAT_STAGE_BRIEFED=1` only on the stage command itself (`NANOCHAT_STAGE_BRIEFED=1 "$@"`), never as a shell-wide export, so later stages and standalone Python runs are unaffected. `emit_define` is a no-op when that variable is set. A Python script run alone (no shell briefing) prints `define:` once per process.
- **Terms per stage.** `learn:` shows every term in that stage's `STAGE_TERMS` entry. The largest entry is `base_train` (BPB, Muon, MFU, FLOPs = 4 terms), so the cap is 4, not 3. No term is dropped.
- **Banner prints once.** `base_train.py:42` already prints the banner. The shell adds no separate script-start banner. This avoids a double banner under `runcpu.sh` and `speedrun.sh`.
- **Plaques only on START, DONE, FAIL.** SKIP gets color only. Re-running a finished pipeline would otherwise print one plaque per skipped stage.
- **Plaque subtitle comes from `stage_purpose()` (Option A).** The shell passes it in: `python3 -m nanochat.stage_art plaque --stage X --kind start --subtitle "$(stage_purpose "$stage")"`. No purpose text exists in any Python module, so there is one owner: `_stage_lib.sh`.
- **Color rules.** Color is on only when stdout is a TTY and `NO_COLOR` is unset. Non-TTY output (CI, `tee`, redirected logs) is plain text. Modal sets `NO_COLOR=1` explicitly.
- **Color roles.** `START` / `FORCE` bold cyan, `DONE` bold green, `SKIP` dim yellow, `FAIL` bold red. Every role is applied to the contract line's existing text; no token changes.
- **Art rules.** Art stays on when output is redirected, because the ASCII borders are greppable and help scrolling saved logs. `NANOCHAT_PLAIN=1` is the only opt-out, and it works in a TTY too.
- **Two color helpers, not merged.** `ColoredFormatter` in `common.py` serves `logging` records. `nanochat/term_style.py` serves direct `print` / shell ANSI output. Different concerns, so no shared code.
- **Trend color lives at the call site.** `emit_metric_trend` takes a `print_fn`. The `base_train` caller passes a color-aware wrapper. `metrics_trend.py` does not import `term_style`.
- **Contract is frozen.** Do not change `START` / `DONE` / `SKIP` / `FAIL` / `FORCE` tokens, field order, or wording in `runs/_stage_lib.sh`. Color and plaques are added around them.
- **Contract revision (ETA wording).** The `START` line's ETA text changed from `ETA: not yet knowable` to `ETA: unknown until first train step`. The `ETA:` token and field position are unchanged, so greps on `ETA:` still match. Reason: the old phrase read like a bug to new runners. Recorded here so the frozen-contract rule stays auditable.

## Goals and acceptance (measurable)

When someone reads a console or log, each stage boundary is visible without scrolling back:

1. Every stage START, DONE, and FAIL line is preceded by exactly one plaque. SKIP prints only its contract line. A post-stage hook failure after DONE prints DONE then FAIL, which is two plaques for two contract lines.
2. The `learn:` line is one line, includes URLs, and has at most 4 terms (the `base_train` maximum).
3. Zero enrichment lines are printed inside the train-step loop.
4. With `NO_COLOR=1`, `NANOCHAT_PLAIN` unset, and output piped: the diff against a captured baseline contains only added plaque lines and one added `learn:` line per START. Every existing contract line is unchanged, except the ETA wording on `START`, which is the recorded contract revision in Decisions.
5. With `NANOCHAT_PLAIN=1`: no plaques, no banner change; contract and `learn:` lines still print.

## Scope (MoSCoW)

**Must** (this slice ships only if these pass)
- `nanochat/glossary.py`: term catalog, `STAGE_TERMS`, `format_learn_line`, `emit_define`, CLI `learn --stage`.
- `nanochat/term_style.py`: `enabled()`, `paint(role, text)`.
- `nanochat/stage_art.py`: plaque renderer and CLI (`plaque --stage X --kind start|done|fail --subtitle ...`), ASCII-safe borders, `NANOCHAT_PLAIN` short-circuit.
- `runs/_stage_lib.sh`: color on START / DONE / SKIP / FAIL; plaques on START / DONE / FAIL; `learn:` line after `purpose:`; per-command `NANOCHAT_STAGE_BRIEFED`; color on `pipeline:` and `tip:` prefixes.
- `runs/modal_run.py`: `NO_COLOR=1` in `_modal_env()`.
- Docs: RUNNING.md duplicate blurb removed (top, lines 10 and 12 are identical); new §6b Terms table; Contents line updated; §5 emission order, §6 enrichment list (START now has two enrichment lines: `purpose:`·`health:` and `learn:`), and §9 hook behavior updated. README.md: one bullet under "Running on CPU / MPS".
- Tests (see below) and ledger entries.

**Should**
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

1. **Glossary, color, and plaque SSOT (Must).** Add `nanochat/glossary.py`, `nanochat/term_style.py`, `nanochat/stage_art.py`. Terms: BPE, vocab, compression ratio, BPB / val_bpb, CORE, ChatCORE, MFU, Muon, FLOPs. Skip softcap and DDP: no `STAGE_TERMS` entry uses them. Unit tests in `tests/test_glossary.py`, `tests/test_term_style.py`, `tests/test_stage_art.py`.
2. **Shell color, learn line, plaques (Must).** Edit `runs/_stage_lib.sh`:
   - Color helpers gated on TTY and `NO_COLOR`.
   - Plaque before START, DONE, and FAIL. Print the DONE plaque with the DONE line, before the post-stage hook runs.
   - Print a FAIL plaque immediately before each `log FAIL` line, including all three hook-failure `exit 1` branches. Plaque output goes to stdout, matching the contract lines.
   - `learn:` line after `purpose:`.
   - `NANOCHAT_STAGE_BRIEFED=1 "$@"` per command, not exported.
   - Color `pipeline:` and `tip:` prefixes.
   - Set `NO_COLOR=1` in `runs/modal_run.py` `_modal_env()`.
   - Extend `tests/test_phase15_enrichments.py`, `tests/test_stage_completion_markers.sh`, and `tests/test_runmodal_mode.py`. Assert contract tokens are unchanged, SKIP has no plaque, and the Modal env carries `NO_COLOR=1`.
3. **Python define and metric color (Should).** `base_train.py`: `emit_define` for bpb / CORE / mfu / Muon, no-op under `NANOCHAT_STAGE_BRIEFED`; color metric lines and trend via call-site `print_fn`. Verify `test_loss_eval_progress.py` and `test_format_duration.py` still pass.
4. **Docs (Must).** RUNNING.md:
   - Remove the duplicate blurb.
   - Add §6b Terms: 9 rows, one per distinct term key referenced by `STAGE_TERMS`. Each cell is 12 words or fewer. Links come from `glossary.py`.
   - Update the Contents line to link §5, §6, §6b, and §9.
   - Update §5 emission order: plaque, then START, then `purpose:`·`health:`, then `learn:`.
   - Update §6: START has two enrichment lines.
   - Update §9: a hook failure turns DONE into FAIL.
   - README.md: one bullet under "Running on CPU / MPS" pointing to RUNNING §5–§6b.
   - Add a test in `tests/test_glossary.py` that parses the RUNNING §6b table and asserts each term key exists in `glossary.py` with the same URL.
5. **Ledger and smoke (Must).** `docs/plan/GROWTH.md` (Added / Changed) and `docs/plan/PROGRESS.md` (one row). Smoke: `python -m nanochat.glossary learn --stage base_train`, `python -m nanochat.stage_art plaque --stage base_train --kind start --subtitle "teach raw next-token prediction on web text"`, `bash runs/runcpu.sh` with `NO_COLOR=1` and piped output (diff against baseline per Goal 4), same with `NANOCHAT_PLAIN=1`.

## Docs audit (verified)

| Layer | Today | Gap this plan closes |
|-------|-------|----------------------|
| L0 console | `purpose:`, `health:`, `tip:`, `pipeline:`, `next:`, `trend:` enrichments | No `learn:`, no plaques, no color |
| L1 README | CPU/MPS section (line ~134) with a paragraph and bullets; existing pointer to RUNNING at line ~143 | No pointer that the console explains terms |
| L1 README terms | `val_bpb` defined inline at line 126; `CORE` at lines 12, 14, 24, 124, 127 (line 127 gives "DCLM CORE score") | Link CORE to RUNNING §6b, not a README glossary |
| L2 RUNNING | 12 sections; top blurb repeated at lines 10 and 12 | Remove duplicate; add §6b; update §5, §6, §9 for new console output |
| L3 deep | README Guides, external papers | No change. Do not paste papers into RUNNING |

### Disclosure ladder

1. **Console (L0):** plaque → `START` → `purpose:` → `learn:` → work → colored metrics.
2. **README (L1):** one bullet. Do not grow README into a glossary.
3. **RUNNING (L2):** how-to sections §1–4, §8–10 first; reference §5, §6, §6b, §9. One file so a stuck runner stays on one page.
4. **Deeper (L3):** links only, from `learn:` lines and the §6b table.

### DX checks (testable)

- The RUNNING Contents line links §5, §6, §6b, and §9.
- §6b term keys and URLs match `nanochat/glossary.py` (enforced by the test in step 4).
- §6b cells are 12 words or fewer (enforced by the same test).
- README CPU/MPS section gains exactly one new `- ` bullet and no new paragraphs.

## Risks and open items

- **Modal path shares `_stage_lib.sh`.** `runs/runmodal.sh` runs through `tee`, and the container env sets `NO_COLOR=1`, so color is off by construction. Plaques appear in `run.log` as plain text. Covered by step 2's Modal test.
- **Define gating depends on the env var.** If a future script runs a stage without `run_stage`, it prints `define:` lines. That is intended: it is the standalone case.
- **`base_eval` headers are Won't this slice.** They stay as three bare `====` lines. Changing them would alter existing script output.

## Out of scope

- Changing START / DONE / SKIP / FAIL / FORCE tokens, field order, or wording
- Per-step definition spam, tip rotation, giant figlet every train step
- Heartbeat / ETA train-loop work (still deferred)
- Web UI (this repo is CLI and shell)
- Full DIVIO restructure
