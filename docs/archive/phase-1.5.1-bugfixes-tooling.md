---
title: "Phase 1.5.1 Bugfixes & Tooling"
summary: "Config plumbing, argparse fix, compression console output, LocalWandb offline logger."
read_when: "Understanding what was fixed during Phase 1.5.1 validation setup."
status: archived
last_updated: 2026-03-15
---

# Phase 1.5.1 Bugfixes & Tooling

## Summary

Four fixes and improvements made during Phase 1.5.1 validation setup.

## Changes

### argparse SUPPRESS fix in `base_train`

When `--config` is used, argparse default values were silently overwriting TOML config fields (e.g. `depth=20` default clobbering `depth=12` from TOML). Fixed by re-parsing with `argparse.SUPPRESS` so only explicitly passed CLI args override the config.

### Compression metrics console output

`CompressionMetrics.log_metrics()` computed and returned metrics but never printed them. Added a `print0()` call in `base_train` after `log_metrics()` returns, producing a `[compression]` line every `--compression-log-every` steps.

### `LocalWandb` offline logger

Added `LocalWandb` to `nanochat/common/wandb.py` alongside `DummyWandb`. When `WANDB_MODE=disabled`, `LocalWandb` is used instead of `DummyWandb`, writing each `log()` call as a JSON line to `base_dir/runs/<run_name>/wandb.jsonl`. Includes 3 unit tests.
