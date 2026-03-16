---
title: "Unified CLI Entry Point"
summary: "Design for a single `nanochat` CLI with subcommands, consistent --config/--base-dir support across all entry points."
read_when: "Planning CLI improvements or adding a new entry point."
status: draft
last_updated: 2026-06-10
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

A `nanochat/__main__.py` delegates to the same entry point, enabling `python -m nanochat` as an alternative to the installed `nanochat` command.

### 2. Global `--config` and `--base-dir` flags

Both flags live on the top-level parser and are inherited by all subcommands:

```bash
nanochat --config nanochat.toml train base --num-iterations=2000
nanochat --base-dir /data/nanochat train base
```

Config resolution order (later steps override earlier):

1. If `--base-dir` is provided and `<base_dir>/config.toml` exists, load it automatically
2. If `--config` is provided explicitly, load that file (overrides the auto-discovered one)
3. CLI args override any value from the loaded config

This means a project directory is self-describing: drop a `config.toml` in your `base_dir` and all commands pick it up without any extra flags.

### 3. Sectioned TOML config with matching dataclasses ✅

**Implemented** in `src/nanochat/common/config.py` on `feat/unified-cli`.

Key decisions made during implementation:

- No inheritance — `CommonConfig`, `TrainingConfig`, `SFTConfig`, `RLConfig`, `EvaluationConfig` are independent dataclasses. Common fields are not inherited into sections; `[common]` is its own section in the TOML.
- `update_parser(cls, parser)` classmethod on each dataclass replaces standalone `add_*_args` functions.
- `Config.generate_default() -> str` returns a TOML string. Each dataclass has its own `generate_default() -> str` returning its section lines.
- `Config.save(path)` uses `asdict(self)` with a one-liner dict comprehension, omitting `None` fields.
- `Config.load(path)` loads TOML and populates only sections present in the file, using the module-level `_SECTION_CLS` dispatch dict.
- `ConfigLoader` replaces `from_args` / `apply_args` / `load` on `Config`. It owns parser construction, section registration, and resolution.

#### `ConfigLoader` — the single entry point for config resolution ✅

```python
cfg = ConfigLoader().add_training().parse()
cfg = ConfigLoader().add_sft().parse(["--config", "path/to/config.toml"])
```

- `__init__` always registers `[common]` and its parser args.
- `add_training()` / `add_sft()` / `add_rl()` / `add_evaluation()` register one section. Calling more than one raises `RuntimeError` — sections share CLI flag names (e.g. `--model-tag`, `--device-batch-size`) and cannot coexist in one parser.
- `parse(args)` resolves: dataclass defaults → TOML → CLI, using `argparse.SUPPRESS` so only explicitly passed CLI args override TOML values.
- `_SECTION_CLS` dict at module level maps section name → dataclass class, used by both `ConfigLoader.parse` and `Config.load`.

See [configuration.md](configuration.md) for the full field reference for all sections.

#### `[chat]` — interactive CLI and web server (`chat_cli`, `chat_web`)

These scripts only need `[common]` for `base_dir` and `device_type`. Script-specific flags (model loading, generation params, server port) remain CLI-only as they are typically not persisted in a config file.

### 4. Argparse with per-section builders ✅

**Implemented** as `update_parser(cls, parser)` classmethods on each dataclass, replacing standalone `add_*_args` functions. `ConfigLoader` composes them internally — scripts no longer call `add_*_args` directly.

**Note on shared flag names**: Several flags appear in multiple sections (`--model-tag`, `--device-batch-size`, `--embedding-lr`, etc.) with different semantics per section. Because of this, `ConfigLoader` enforces single-section use — calling `add_training().add_sft()` raises `RuntimeError`. Each entry point registers exactly one section.

### 5. Default config generation ✅

**Implemented**. `Config.generate_default() -> str` returns a fully-commented TOML string with all sections. Each dataclass has its own `generate_default() -> str` returning its section lines. Wiring into `nanochat config init` is pending (step 7).

```bash
nanochat config init                        # writes config.toml in current directory
nanochat config init --output my.toml      # custom path
```

### 6. Wandb consolidation via `CommonConfig.wandb`

The `wandb` field in `[common]` replaces two current hacks:
- `--run="dummy"` magic value to disable wandb
- `WANDB_MODE=disabled` env var check (only in `base_train`)

A shared `init_wandb(config: CommonConfig, user_config: dict)` helper in `nanochat/common/wandb.py` handles all three cases. Wiring into all training scripts is pending (step 3).

## Package Restructure

Currently all scripts live flat in `nanochat/scripts/` with name prefixes (`base_`, `chat_`, `tok_`) doing the grouping. The subcommand structure should mirror the existing package layout — most target packages already exist:

```
nanochat/
├── config/
│   └── cli.py            # nanochat config init / show  ✅ done
├── data/
│   ├── dataset.py        # nanochat data download
│   └── tokenizer/
│       ├── train.py      # nanochat data tokenizer train
│       └── eval.py       # nanochat data tokenizer eval
├── training/             # already exists
│   ├── base.py           # nanochat train base
│   ├── sft.py            # nanochat train sft
│   └── rl.py             # nanochat train rl
├── evaluation/           # already exists
│   ├── base.py           # nanochat eval base
│   └── chat.py           # nanochat eval chat
├── chat/                 # new package
│   ├── cli.py            # nanochat chat
│   └── web.py            # nanochat serve
└── cli.py                # entry point
```

`nanochat train base` maps to `nanochat.training.base`, `nanochat eval base` to `nanochat.evaluation.base`, and so on. No more prefixes, no more `scripts/` indirection. The only new package needed is `chat/` — everything else lands in an already-existing package.

## Implementation Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | `Config` + dataclasses + `ConfigLoader` in `common/config.py` | ✅ Done |
| 2 | `update_parser` classmethods, `common/__init__.py` exports updated | ✅ Done |
| 3 | Wandb consolidation via `CommonConfig.wandb` | ⏳ Pending |
| 4 | Wire `ConfigLoader` into existing scripts | ⏳ Pending |
| 5 | Add `--config` / `--base-dir` to remaining entry points | ⏳ Pending |
| 6 | `nanochat/__main__.py` | 🔜 Next |
| 7 | `nanochat/cli.py` + subcommand dispatch + empty per-script clients | 🔜 Next |
| 8a | Migrate `scripts/base_train.py` → `training/base.py` — wire `ConfigLoader`, wandb consolidation, remove manual re-parse hack | ⏳ Pending |
| 8b | Migrate `scripts/chat_sft.py` → `training/sft.py` — wire `ConfigLoader`, wandb consolidation, remove `--run="dummy"` magic | ⏳ Pending |
| 8c | Migrate `scripts/chat_rl.py` → `training/rl.py` — fix broken `nanochat.common.config` import, wire `ConfigLoader`, wandb consolidation | ⏳ Pending |
| 8d | Migrate `scripts/base_eval.py` → `evaluation/base.py` — wire `ConfigLoader`, rename `--eval` → `--modes` to match `EvaluationConfig` | ⏳ Pending |
| 8e | Migrate `scripts/chat_eval.py` → `evaluation/chat.py` — wire `ConfigLoader`, add `--config`/`--base-dir` | ⏳ Pending |
| 8f | Migrate `scripts/chat_cli.py` → `chat/cli.py` — wire `CommonConfig` | ⏳ Pending |
| 8g | Migrate `scripts/chat_web.py` → `chat/web.py` — wire `CommonConfig` | ⏳ Pending |
| 8h | Migrate `scripts/tok_train.py` → `data/tokenizer/train.py` — replace manual `--base-dir` with `CommonConfig` | ⏳ Pending |
| 8i | Migrate `scripts/tok_eval.py` → `data/tokenizer/eval.py` — add `CommonConfig` for `base_dir` | ⏳ Pending |
| 8j | Migrate `data/dataset.py` `__main__` → proper `main()` — replace manual `--base-dir`/env-var hack with `CommonConfig`, add `--num-files`/`--num-workers` to `CommonConfig` or keep as local args | ⏳ Pending |
| 9 | Unit tests | ✅ Done (35 tests) |
| 10 | `docs/configuration.md` | ✅ Done |
| 11 | `docs/cli.md` | ⏳ Pending |
| 12 | Update README, CHANGELOG, bash scripts | ⏳ Pending |

## Notes

- Steps 1–2 done — config foundation is solid
- Steps 6–7 next — create `__main__.py`, `cli.py` with subcommand dispatch, and empty client stubs for each entry point
- Scripts migrated one by one after the CLI skeleton is in place (step 8)
- Steps 3–5 (wandb consolidation, wiring `ConfigLoader` into scripts) happen naturally as each script is migrated in step 8
- Keep `scripts/` shims so `python -m nanochat.scripts.*` keeps working during transition
- `chat_cli` and `chat_web` only need `[common]` from config — their script-specific flags stay CLI-only
- `tok_eval` has no args at all today — it only needs `[common]` for `base_dir`
- `nanochat config init` is the recommended starting point for any new experiment
