# LS Autoresearch

This repo has a small restartable autoresearch harness adapted from
`jake-molnia/autoresearch`, but aimed at the LS implementations in this
LS-enabled nanochat fork. The intended use is multi-agent LS recurrent tuning:
agents make focused commits, the runner evaluates those commits on the 4090
box, and the resulting logs and summaries are synced back locally.

The upstream repo assumes:

- one editable file
- one local GPU
- one fixed benchmark

This fork does not fit that shape. The LS work spans multiple files and the
actual benchmark machine is the 4090 box at
`jrmolnia@paffenroth23-1.dyn.wpi.edu`.

## What Was Added

- `scripts/ls_autoresearch.py`
  Restartable remote experiment runner.
- `dev/AUTORESEARCH_LS_PROGRAM.md`
  Agent-facing instructions adapted from upstream `program.md`.
- `autoresearch_runs/<tag>/`
  Local state, logs, and result tracking. This directory is ignored by git.

## Design

Each run tag stores:

- `config.json`: remote host, ssh key, experiment suite, primary metric
- `state.json`: baseline commit, best commit, best metrics, run count
- `results.tsv`: append-only tab-separated results log
- `logs/`: synced copies of remote experiment logs
- `runs/`: one JSON summary per suite execution

The harness is commit-oriented. The intended loop is:

1. Create or switch to an autoresearch branch.
2. Have one agent make one LS-focused change.
3. Commit that change with a message that captures the hypothesis.
4. Run `scripts/ls_autoresearch.py run --tag ...` on the committed state.
5. Inspect `status`, local logs, and local result tables.
6. Decide whether to continue from the current tip or the tracked best commit.

The script persists enough state to stop and restart later with the same tag.
Resuming the same tag reuses the saved state instead of rebuilding the search.

## Intended Workflow

This setup is meant to follow the upstream autoresearch style, but adapted to
remote 4090 execution and LS recurrent objectives:

1. Initialize a tag and branch for an LS recurrent search.
2. Spawn one or more agents to try focused LS recurrent ideas.
3. Each agent should produce a small, reviewable commit before running the
   harness.
4. Evaluate that committed state with `scripts/ls_autoresearch.py run`.
5. Let `autoresearch_runs/<tag>/` accumulate the restartable local record.
6. Use the synced local artifacts for notebook analysis and follow-up runs.

Dense transformer behavior may appear as a comparison baseline in some
benchmarks, but dense is not the thing being tuned.

## Quick Start

Initialize a smoke suite:

```bash
python scripts/ls_autoresearch.py init --tag apr19-ls --preset smoke --create-branch
```

Initialize a short real-data suite:

```bash
python scripts/ls_autoresearch.py init --tag apr19-ls-train --preset short_train --create-branch
```

Run the current commit:

```bash
python scripts/ls_autoresearch.py run --tag apr19-ls --description "baseline"
```

The `run` command syncs the repo to the remote GPU machine, executes the
configured LS suite there, and then syncs logs and run summaries back into
`autoresearch_runs/<tag>/` locally.

Inspect state:

```bash
python scripts/ls_autoresearch.py status --tag apr19-ls
```

## Presets

`smoke`

- uses `scripts.compare_loss_curves`
- fast deterministic loss-curve checks
- primary objective defaults to `lsres_k4_curve`

`short_train`

- uses `scripts.base_train`
- short real-data LS-only training probes
- primary objective defaults to `lsres_k4_train`

Both presets include the tuned LSRecurrent settings recovered from earlier runs:

- practical preset: `K=4`, `n_mem=128`, `log_dt_init=-3.5`
- raw-quality preset: `K=8`, `n_mem=64`, `log_dt_init=-3.5`

## Scope

This harness is intentionally LS-only:

- `LSLinear` transformer runs
- `LSRecurrent` scan-driven runs

The primary tuning target is `LSRecurrent` / LSRes behavior. It does not search
dense-transformer settings as a primary objective.

## Notes

- Remote sync uses `rsync` and excludes `.git`, `.venv`, `wandb`, and `autoresearch_runs`.
- The default remote Python is `/home/jrmolnia/slowbenchmark/.venv/bin/python`.
- The default remote repo root is `/tmp/nanochat-ls-autoresearch/<tag>/repo`.
- You can edit `autoresearch_runs/<tag>/config.json` after init if you want a different experiment suite.
- Notebook analysis should read from the local `autoresearch_runs/<tag>/`
  artifacts, not from the remote box.
