# LS Autoresearch Program

This repo contains an LS-enabled nanochat fork and a restartable autoresearch
runner in `scripts/ls_autoresearch.py`.

The goal is to improve the LS recurrent implementations on the 4090 box, not
to search dense-transformer settings. The intended workflow is multi-agent and
commit-based: each agent should land a small LS-focused commit, then use the
runner to evaluate that commit and sync the artifacts back locally.

## In-Scope Files

Read these before starting:

- `dev/AUTORESEARCH_LS.md`
- `scripts/ls_autoresearch.py`
- `nanochat/gpt.py`
- `scripts/base_train.py`
- `scripts/compare_loss_curves.py`
- `scripts/compare_lslinear.py`
- `scripts/compare_lsrecurrent.py`

## Scope Rules

You may edit:

- `nanochat/gpt.py`
- `scripts/base_train.py`
- LS-specific helper scripts
- LS benchmark scripts

You should avoid unrelated edits outside the LS search workflow.

Dense runs may appear as a comparison baseline in logs, but dense is not the
search target and should not be the thing you tune for.

Prioritize LS recurrent and LSRes ideas over LSLinear. LSLinear can remain as a
secondary control, but the search should be driven by LS recurrent metrics.

## Setup

For a new run:

1. Pick a tag like `apr19-ls`.
2. Initialize a run:

   ```bash
   python scripts/ls_autoresearch.py init --tag apr19-ls --preset short_train --create-branch
   ```

3. Confirm the run state:

   ```bash
   python scripts/ls_autoresearch.py status --tag apr19-ls
   ```

4. The first committed run should be the baseline.

The tag directory under `autoresearch_runs/<tag>/` is the durable run record.
If the search is interrupted, resume from the same tag instead of
reinitializing it.

## Multi-Agent Loop

For each experiment:

1. Take one LS recurrent hypothesis at a time.
2. Make one coherent LS-focused change.
3. Commit it before running anything.
4. Keep the commit message specific enough that the run log stays interpretable.
5. Run:

   ```bash
   python scripts/ls_autoresearch.py run --tag apr19-ls --description "short note"
   ```

6. Inspect status:

   ```bash
   python scripts/ls_autoresearch.py status --tag apr19-ls
   ```

7. Review the synced local artifacts in `autoresearch_runs/<tag>/logs/`,
   `autoresearch_runs/<tag>/runs/`, and `autoresearch_runs/<tag>/results.tsv`.
8. Continue iterating from the current branch or from the tracked best commit.

The run state is persisted in `autoresearch_runs/<tag>/`, so the search can be
stopped and resumed later without rebuilding the setup.

If multiple agents are working at once, they should each operate on isolated
commits or branches and run the harness on those committed states. Do not use
uncommitted edits as the unit of evaluation.

## Artifact Sync

The runner executes on the remote 4090 box but copies logs and run summaries
back into the local `autoresearch_runs/<tag>/` directory after each run. That
local directory is the source of truth for notebook analysis, plotting, and
cross-run comparison.

## Search Priorities

Focus on LS recurrent ideas first:

- `LSRecurrent` scan-driven dynamics
- `K`, `n_mem`, `log_dt_init`
- LS-specific optimizer routing or scaling
- simplifications that preserve or improve LS recurrent metrics

Use `LSLinear` only as a secondary control when it helps explain LS recurrent
behavior.

Prefer changes that are principled and survive the short-train or smoke suite.
