---
title: "Phase 1.5.0.1 — Script Entry-Point Refactor"
summary: "Wrapped all training scripts in main() functions for importability without side effects."
status: archived
archived_date: "2026-03-15"
archived_reason: "All three scripts refactored. Scripts importable without triggering setup."
---

# Phase 1.5.0.1 — Script Entry-Point Refactor ✅ COMPLETED

**Goal**: Wrap top-level script code behind `main()` so scripts are importable without side effects.

## Key Features Delivered

### base_train.py
- Wrapped ~214 lines of top-level setup into `main()`
- Extracted `build_parser()` function
- Converted `train_base_model` from 30-param function to zero-param closure
- Fixed `config` shadowing bug in `build_model_meta()`
- Added missing `Path` import

### chat_sft.py
- Wrapped ~238 lines of top-level setup into `main()`
- Extracted `build_parser()` function
- Converted `global` to `nonlocal` in `sft_data_generator_bos_bestfit`

### chat_rl.py
- Wrapped ~103 lines of top-level setup into `main()`
- Extracted `build_parser()` function
- `get_batch()` and `run_gsm8k_eval()` as nested closures

## Sub-tasks

- [x] `base_train.py` — committed as `f732b9a`
- [x] `chat_sft.py` — committed with chat_rl.py as `d287503`
- [x] `chat_rl.py` — committed as `d287503`
- [x] Verify scripts work via `python -m` and `torchrun`
- [x] Confirm no tests import from scripts directly

## Tests

- 56 passed, 10 skipped (unchanged from before refactor)
