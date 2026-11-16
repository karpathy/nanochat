# HANDOFF — nanochat SFT stall fix and Modal CPU run

Written: 2026-11-16. Replaces the STALE 2026-11-15 handoff.

## Where things stand

- Branch: `fix/sft-stall-resume-guard` (pushed, head `bd065a8`, in sync with remote, clean tree).
- PR #47 (open, ready, not draft): https://github.com/neomatrix369/nanochat/pull/47
  - Body updated 2026-11-16: Verification section now reports the finished run. Title, Test Results, Quality Gates unchanged.
- PR #46 (tracker cleanup, branch `docs/tracker-cleanup-2026-11-15`): open.
- Modal app for run `20261116-003831-cpu` is completed. Nothing billable is running.
- Local run logs: `~/.cache/nanochat-modal/runs/20261116-003831-cpu/run.log`.

## What was found and fixed

- Root cause of the SFT stall: conversations longer than one packed row (513 tokens) could never be placed. The buffer filled with them, the loader stopped advancing, and every later batch was padding (loss 0.0, zero gradients). The NaN guard does not catch 0.0, so the run looked alive.
- The step-572 checkpoint held 100 buffered conversations, minimum 517 tokens, none packable.
- All SFT checkpoints from the broken run (steps 40–967) had NaN EMA and a stuck cursor. They were deleted from the Volume. Restart point: base checkpoint `cpu/base_checkpoints/d6` step 5000.
- Fix in `bd065a8`:
  - `is_packable` skips oversize conversations (counted, logged every 1000).
  - `next_zero_token_streak` halts after `ZERO_TOKEN_STREAK_LIMIT` (10) empty steps.
  - `user_config` now saves the resolved batch and sequence settings, so the resume guard compares real values.
  - Tests: `tests/test_sft_loader_guards.py`. Suite: 379 passed, 15 skipped, 76.3% coverage.

## Result of the rerun (`20261116-003831-cpu`, code `bd065a8`)

- Completed 1500 of 1500 steps, exit 0, about 2h 10m training (3h 50m stage).
- No NaN or inf losses. No zero-token halt. No all-masked warnings.
- Oversize skips: 2000+ conversations skipped over the run.
- Checkpoint written: `cpu/chatsft_checkpoints/d6` (2.0 GiB) on the Modal Volume.

## NOT resolved (start here)

1. **Loss did not clearly drop.** About 1.77 at step 1, about 1.96 at step 1500, per-step range 1.77–3.31. Find out whether this is data (large skip count), learning rate, or the base checkpoint.
2. **Chat quality is very low.** ChatCORE 0.009, HumanEval 0%. Do not trust this checkpoint for quality until (1) is explained.
3. **B1 live download is broken.** `modal volume get` fails on nested folders ("Is a directory"). Planned fix: fetch files individually. Verify with a real download after the run.
4. **Trackers are behind.** `docs/plan/TRAIL.md`, `docs/plan/PROGRESS.md`, and `docs/plan/2026-11-15-eval-resume-and-tracker-cleanup.md` do not yet record the NaN and stall findings, the fix, or the rerun.
5. **Backup branch.** `backup/modal-pre-rebase` (2 commits, local only). Recommended: delete, do not PR (a PR would revert about 82 files of newer master work).
6. **Remaining plan items:** S1 per-task eval results (needs a test interview first), S2 eval marker only on full runs, `chat_sft` stale-`.tmp` cleanup, ruff errors in `tests/`, duplicate eval on resume (nit).
7. **Checkpoint not on your laptop.** Download it only after (1) is answered.

## Rules to keep

- Never commit to `main` or `master`. PRs target `neomatrix369/nanochat`, not karpathy.
- No attribution lines in commits or PR bodies (no Co-authored-by, no "Generated with").
- Get explicit OK before any billable Modal run or any remote mutation (push, PR edit, branch delete).
- Modal stop: `modal app stop <app-id> --yes`. Ctrl-C only stops the local client.

## Useful commands

- `modal app list` / `modal app logs <app-id>`
- `modal volume ls nanochat-modal cpu/` (inspect the Volume)
- `NANOCHAT_BASE_DIR=~/.cache/nanochat-modal/cpu python3 -m scripts.chat_cli -p "What is the capital of France?"` (chat check on the Mac, after download)
- Tests: `uv run python3 -m pytest`; lint: `uvx ruff check runs/ tests/`
