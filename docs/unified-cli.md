---
title: "Unified CLI Entry Point"
summary: "Design for a single `nanochat` CLI with subcommands, consistent --config/--base-dir support across all entry points."
read_when: "Planning CLI improvements or adding a new entry point."
status: draft
last_updated: 2026-03-15
---

# Unified CLI Entry Point

## Problem

nanochat has 10 independent entry points, each with its own argparse setup and inconsistent support for `--config` and `--base-dir`:

| Entry point                        | `--config` | `--base-dir` |
|------------------------------------|------------|--------------|
| `nanochat.scripts.base_train`      | ✅          | ✅            |
| `nanochat.scripts.tok_train`       | ❌          | ✅            |
| `nanochat.data.dataset`            | ❌          | ✅            |
| `nanochat.scripts.base_eval`       | ❌          | ❌            |
| `nanochat.scripts.chat_sft`        | ❌          | ❌            |
| `nanochat.scripts.chat_rl`         | ❌          | ❌            |
| `nanochat.scripts.chat_eval`       | ❌          | ❌            |
| `nanochat.scripts.tok_eval`        | ❌          | ❌            |
| `nanochat.scripts.chat_cli`        | ❌          | ❌            |
| `nanochat.scripts.chat_web`        | ❌          | ❌            |

This means:
- You can't drive a full pipeline from a single TOML config
- `--base-dir` must be repeated on every command
- No discoverability — users must know each module path

## Proposed Solution

### 1. Single `nanochat` CLI with subcommands

```
nanochat data download           # nanochat.data.dataset
nanochat data tokenizer train    # nanochat.scripts.tok_train
nanochat data tokenizer eval     # nanochat.scripts.tok_eval
nanochat train base              # nanochat.scripts.base_train
nanochat train sft               # nanochat.scripts.chat_sft
nanochat train rl                # nanochat.scripts.chat_rl
nanochat eval base               # nanochat.scripts.base_eval
nanochat eval chat               # nanochat.scripts.chat_eval
nanochat chat                    # nanochat.scripts.chat_cli
nanochat serve                   # nanochat.scripts.chat_web
```

Implemented as a `nanochat/cli.py` entry point registered in `pyproject.toml`:

```toml
[project.scripts]
nanochat = "nanochat.cli:main"
```

### 2. Global `--config` and `--base-dir` flags

Both flags live on the top-level parser and are inherited by all subcommands:

```bash
nanochat --config nanochat.toml train base --num-iterations=2000
nanochat --base-dir /data/nanochat tok train
```

### 3. Consistent config/CLI override mechanism

All subcommands follow the same pattern already established in `base_train`:

1. Load TOML config if `--config` is provided
2. Re-parse with `argparse.SUPPRESS` to detect only explicitly passed CLI args
3. CLI args override config values

This logic should live in a shared helper so it's not duplicated across 10 entry points.

## Package Restructure

Currently all scripts live flat in `nanochat/scripts/` with name prefixes (`base_`, `chat_`, `tok_`) doing the grouping. The subcommand structure should mirror the package layout:

```
nanochat/
├── data/
│   ├── dataset.py        # nanochat data download
│   └── tokenizer/
│       ├── train.py      # nanochat data tokenizer train
│       └── eval.py       # nanochat data tokenizer eval
├── train/
│   ├── base.py           # nanochat train base
│   ├── sft.py            # nanochat train sft
│   └── rl.py             # nanochat train rl
├── eval/
│   ├── base.py           # nanochat eval base
│   └── chat.py           # nanochat eval chat
├── chat/
│   ├── cli.py            # nanochat chat
│   └── web.py            # nanochat serve
└── cli.py                # entry point
```

`nanochat train base` maps to `nanochat.train.base`. No more prefixes, no more `scripts/` indirection. The existing `python -m` invocation paths will change, but the unified CLI makes that irrelevant for day-to-day use.

## Implementation Plan

1. Add `--config` and `--base-dir` to the remaining 8 entry points (can be done incrementally)
2. Restructure `nanochat/scripts/` into `train/`, `eval/`, `chat/` packages; move tokenizer scripts under `data/tokenizer/`
3. Create `nanochat/cli.py` with subcommand dispatch
4. Register `nanochat` script in `pyproject.toml`
5. Extract the config/CLI override logic into a shared helper in `nanochat/common/`
6. Update all documentation and references:
   - `README.md` — usage examples, quickstart commands
   - `docs/configuration.md` — `--config` flag coverage for all entry points
   - `docs/data-layout.md` — updated module paths
   - `CHANGELOG.md` — breaking change: `python -m nanochat.scripts.*` paths change
   - Any validation checklists referencing old `python -m` invocations
   - Bash scripts in `runs/` and `dev/` — `speedrun.sh`, `runcpu.sh`, `scaling_laws.sh`, `miniseries.sh`, `nanochat-helper.sh`

## Notes

- The existing `python -m nanochat.scripts.*` invocation style should continue to work — `cli.py` just wraps the existing `main()` functions
- `chat_cli` and `chat_web` don't use `TrainingConfig` — they only need `--base-dir` for checkpoint loading, not `--config`
- `base_train` already has the full implementation and can serve as the reference
