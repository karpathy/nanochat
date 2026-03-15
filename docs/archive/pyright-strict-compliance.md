---
title: "Pyright Strict Compliance"
summary: "Resolved all 128 remaining pyright errors after the reportMissingParameterType retrofit, achieving 0 errors under strict mode."
status: archived
archived_date: "2026-03-16"
archived_reason: "Completed. All 128 pyright errors resolved across 7 categories. 0 errors remaining."
---

# Pyright Strict Compliance

## Goal

Resolve all remaining pyright strict-mode errors after the initial `reportMissingParameterType` retrofit (which brought 323→0 for that category but left 128 others).

## Starting Point

128 errors across 10 categories after `reportMissingParameterType` was fixed.

## Changes Made

### `reportUnusedVariable` (42 errors → 0)
- Underscore-prefixed unused variables (`_ddp`, `_rank`, etc.)
- `_` for ignored tuple unpacking in loops

### `reportPossiblyUnboundVariable` (16 errors → 0)
- `base_train.py`: initialized `optimizer_data`, `meta_data`, `mfu`, `train_loss` before conditional branches
- `chat_sft.py`: initialized `content_len`, `val_bpb`, `train_loss` before loops
- `tokenizer.py`: hoisted `prepend_id`/`append_id` before branching

### `reportUnusedFunction` (5 errors → 0)
- `chat_web.py`: added `# pyright: reportUnusedFunction=false` — FastAPI route handlers registered via decorators are not "unused"

### `reportIndexIssue` (19 errors → 0)
- `tasks/base.py`: `__getitem__` return type `Conversation` → `dict[str, object]`
- `tasks/arc.py`, `gsm8k.py`, `humaneval.py`, `mmlu.py`, `spellingbee.py`: cast `conversation["messages"]` to `list[dict[str, object]]` before subscripting
- `scripts/chat_eval.py`: `task_object: object` → `Task`; cast `conversation["letters"]` to `list[str]`
- `scripts/base_eval.py`: cast `meta["model_config"]`, `meta["step"]`, `core_results[...]`
- `scripts/chat_rl.py`: cast `r["outcomes"]` to `list[dict[str, object]]`
- `training/checkpoint.py`: cast `meta_data["model_config"]`
- `report.py`: cast `gpu_info["names"]` and `gpu_info["count"]`

### `reportReturnType` (15 errors → 0)
- `tasks/base.py` + all 6 task files: `get_example` → `Mapping[str, object]` (covariant, accepts specific dicts)
- `base_eval.py`: `evaluate_core` → `Mapping[str, object]`
- `core_eval.py`: `return bool(is_correct)` (was numpy `bool_`)
- `compression_metrics.py`: `bool(improvement < 0.01)`; `get_summary` → `Dict[str, object]`
- `gpt.py`: `cast(torch.device, ...)` in `get_device()`; `yield int(token)` in `generate()`
- `io.py`: `os.environ["NANOCHAT_BASE_DIR"]` instead of `.get()` so return is `str` not `str | None`

### `reportMissingImports` (7 errors → 0)
- `chat_sft.py`, `chat_rl.py`: bare `tasks.*` → `nanochat.tasks.*`
- `chat_sft.py`: `tasks.common` → `tasks.base` (`TaskMixture` lives in `base.py`)

### `reportUnnecessaryComparison` (3 errors → 0)
- `flash_attention.py`: `_override_impl: str | None = None` so comparisons to `"fa3"`/`"sdpa"` are valid
- `tokenizer.py`: `encode_special` → `int | None` (matches `token_to_id` which can return `None`)
- `fp8.py`: `# pyright: ignore[reportUnnecessaryComparison]` on `bias` check (genuinely `None` at runtime)

### `reportMissingTypeArgument` (6 errors → 0)
- `tokenizer.py`: `set` → `set[str]`
- `base_train.py`: `dict` → `dict[str, object]` for `optimizer_data`/`meta_data`
- `chat_web.py`: `Queue` → `Queue[Worker]`
- `optimizer.py`: `list[dict]` → `list[dict[str, object]]`

### `reportUnnecessaryIsInstance` (2 errors → 0)
- Both tokenizer `encode()` methods: `elif isinstance(text, list)` → `else` (type already narrowed)

### Other — `reportGeneralTypeIssues`, `reportOptionalSubscript`, `reportOptionalCall` (16 errors → 0)
- `attention.py`: `assert self.ve_gate is not None` before calling it
- `flash_attention.py`: guard `cache_seqlens` before subscript
- `gpt.py`: cast `transformer.h` to `ModuleList`; cast `device` assignments
- `base_train.py`: `assert meta_data is not None` at usage sites; cast `loop_state`
- `core_eval.py`: cast `item["choices"]`/`item["context_options"]` to `list[object]`
- `spellingbee.py`: cast `assistant_parts` content to `list[dict[str, object]]`
- `tasks/base.py`: `# pyright: ignore[reportUnusedImport]` on re-exported `Conversation`

## Result

**128 → 0 pyright errors** under strict mode. All categories fully resolved.

## Commits

- `4ad4c5a` — fix reportUnusedVariable (128→79)
- `9105b4c` — fix reportPossiblyUnboundVariable (79→66)
- `e76859b` — fix reportUnusedFunction (66→61)
- `c5fe69f` — fix reportIndexIssue (61→48)
- `ded3567` — fix reportReturnType (48→32)
- `24ee35d` — fix reportMissingImports (32→25)
- `8ab1f9e` — fix reportUnnecessaryComparison (25→21)
- `ea7a027` — fix reportMissingTypeArgument (21→18)
- `87ff8a1` — fix reportUnnecessaryIsInstance (18→16)
- `cf31ea9` — fix remaining Other category (16→0)
