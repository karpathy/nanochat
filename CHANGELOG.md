# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

#### Documentation
- `docs/guides/` with 5 markdown guides mirrored from upstream GitHub discussions:
  - `introducing-nanochat.md` — original Oct 2025 nanochat post (discussion #1)
  - `infusing-identity.md` — synthetic identity data guide (discussion #139)
  - `counting-letters-adding-abilities.md` — SpellingBee task walkthrough (discussion #164)
  - `miniseries-v1.md` — scaling laws and miniseries v1 results (discussion #420)
  - `beating-gpt2-nanochat-journey.md` — Jan 2026 GPT-2 speedrun deep dive (discussion #481)
- `docs/data-layout.md` — hierarchical `NANOCHAT_BASE_DIR` directory structure
- `docs/m3-max-guide.md` — MPS backend guide covering device detection, dtype behavior, and training guidelines
- `docs/archive/pyright-strict-compliance.md` — pyright strict mode compliance archive
- `docs/archive/type-annotations-pyright-compliance.md` — type annotation compliance archive
- `docs/archive/refactor-common-package.md` — common package refactor archive
- `docs/archive/phase-1.5.0.2-code-review-cleanup.md` — code review cleanup archive

#### Configuration
- `--config` flag to load `TrainingConfig` from TOML file (CLI args override file values)
- `--base-dir` flag to override `NANOCHAT_BASE_DIR` env var
- `build_parser()` extracted in all three training scripts for reusable CLI setup
- TOML config save/load to `TrainingConfig` (replaces JSON); auto-saved to checkpoint dir on start
- Compression fields to `TrainingConfig`: `track_compression`, `compression_log_every`, `track_layer_compression`, `compression_early_stop`
- `base_dir` field to `TrainingConfig`
- `tomli` and `tomli-w` dependencies for TOML support on Python 3.10

#### Testing & CI
- `tests/test_paths.py` with 8 tests covering all path functions
- 3 new config tests: TOML round-trip, `from_args` mapping, `base_dir` default
- CI/CD workflow with GitHub Actions (pytest, ruff, pyright)
- Comprehensive test suite (30 tests: models, training, tasks, integration)

#### Package & API
- Console entry points: `nanochat-train`, `nanochat-eval`, `nanochat-chat`, `nanochat-sft`, `nanochat-rl`, `nanochat-chat-eval`, `nanochat-web`, `nanochat-tok-train`, `nanochat-tok-eval`
- Public API in `__init__.py`: `GPT`, `Engine`, `get_tokenizer`, optimizers
- Importable training functions: `train_base_model`, `train_sft_model`, `train_rl_model`
- Importable evaluation functions: `evaluate_base_model`, `evaluate_validation_loss`, `evaluate_chat_model`
- `TrainingConfig.from_args()` method for CLI integration
- `LR schedulers` module (warmup/constant/warmdown, Muon momentum, weight decay)
- Conversation type definitions (`Message`, `Conversation`)

#### Features
- Flash Attention 3 with automatic fallback to PyTorch SDPA for non-Hopper GPUs
- Learnable per-layer residual scalars (`resid_lambdas`, `x0_lambdas`)
- Sliding window attention with configurable patterns (default `SSSL`: 3 short, 1 long)
- BOS-aligned dataloaders with BestFit-Crop packing and epoch tracking
- Pretraining resumption from checkpoints
- SpellingBee task for character counting/spelling ability
- Python calculator tool support in inference engine
- Identity/personality system via synthetic data generation
- Web UI: slash commands, click-to-edit messages, click-to-regenerate responses
- Basic logging and abuse prevention in `chat_web`
- Multi-GPU inference (data parallel)
- CPU and MPS (Apple Silicon) support with auto-detection
- FP8 training via `torchao` (tensorwise, H100 only, default off)
- Muon optimizer with Polar Express orthogonalization and Adafactor-style variance reduction
- Cautious weight decay with linear schedule to zero
- Scaling law for optimal weight decay (∝ 1/channels²)
- Auto-calculated optimal batch size per model depth
- Miniseries and scaling laws training scripts (`runs/miniseries.sh`, `runs/scaling_laws.sh`)
- CORE score evaluation for HuggingFace models
- `--save-every` and `--resume-from-step` flags for checkpoint management
- MIT License

### Changed

#### Documentation
- README `## Guides` links updated from upstream GitHub discussion URLs to local `docs/guides/*.md`
- Commands in guides corrected to current module paths (`nanochat.scripts.*`, `nanochat.data.dataset`)
- Deprecation notes added in `introducing-nanochat.md` for removed scripts (`base_loss`, `mid_train`)

#### Type Safety (pyright strict, 128→0 errors)
- `tasks/base.py`: `get_example`/`__getitem__` return `Mapping[str, object]`; all task subclasses updated
- `tasks/*.py`: bare `tasks.*` imports replaced with `nanochat.tasks.*`; `tasks.common` → `tasks.base`
- `flash_attention.py`: `_override_impl: str | None = None`; guard `cache_seqlens` before subscript
- `tokenizer.py`: `encode_special` → `int | None`; redundant `isinstance` branches replaced with `else`
- `gpt.py`: cast `get_device()` and `device` assignments; `yield int(token)` in `generate()`
- `compression_metrics.py`: `bool(...)` wrapping for numpy `bool_` returns; `get_summary` → `Dict[str, object]`
- `io.py`: `os.environ[]` instead of `.get()` to return `str` not `str | None`
- `attention.py`: `assert self.ve_gate is not None` before calling it
- `base_train.py`: `assert meta_data is not None` at usage sites; cast `loop_state`
- Parameter type annotations added across all 29 files (323→0 `reportMissingParameterType`)

#### MPS / dtype
- MPS backend uses `float16` instead of `float32` (~10-30% faster, halves memory)
- MPS training uses `torch.mps.synchronize()` for accurate step timing and `torch.mps.current_allocated_memory()` for memory reporting
- MPS training calls `torch.mps.empty_cache()` between eval and training steps
- KV cache in `engine.py` uses `get_compute_dtype()` instead of hardcoded dtype
- Removed `torch.amp.autocast`; precision managed explicitly via `COMPUTE_DTYPE` / `get_compute_dtype()`
- `COMPUTE_DTYPE` deferred to lazy `get_compute_dtype()` / `get_compute_dtype_reason()`
- `fp16` training automatically enables `GradScaler` in `base_train.py` and `chat_sft.py`

#### Package structure
- Migrated to `src/nanochat/` layout with proper package structure
- Split `gpt.py` into `models/` submodules: `config.py`, `attention.py`, `mlp.py`, `gpt.py`
- Reorganized into domain modules: `models/`, `training/`, `evaluation/`, `data/`, `tasks/`, `scripts/`
- Refactored monolithic `common.py` into `common/` package with 7 focused modules
- Absorbed `paths.py` into `common/paths.py`; `__init__.py` re-exports all public names
- Renamed `tasks/common.py` → `tasks/base.py`
- Updated `pyproject.toml` to hatchling build backend
- Replaced `Configurator` with `argparse`

#### Scripts
- `base_train.py`, `chat_sft.py`, `chat_rl.py`: all top-level setup wrapped in `main()`, `build_parser()` extracted, importable without side effects
- Remaining 5 scripts wrapped in `main()`: `tok_train.py`, `tok_eval.py`, `chat_cli.py`, `chat_web.py`, `chat_eval.py`
- `WorkerPool` in `chat_web.py` accepts `device_type` parameter instead of reading module-level variable
- `checkpoint.py` uses `paths.checkpoints_dir()` instead of manual path construction
- `USE_FA3` in `flash_attention.py` deferred to lazy `_use_fa3()` with caching
- `DATA_DIR` in `dataset.py` replaced with lazy `_get_data_dir()`
- `setup_default_logging()` made idempotent, called from `compute_init()`
- `base_train.py` uses `TrainingConfig` throughout — raw `args` usage replaced
- `TrainingConfig.save()` / `load()` switched from JSON to TOML

#### Data layout
- `NANOCHAT_BASE_DIR` restructured from flat to hierarchical:
  - `base_data_climbmix/` → `data/climbmix/`
  - `eval_bundle/` → `data/eval_tasks/`
  - `base_checkpoints/` → `checkpoints/base/`
  - `chatsft_checkpoints/` → `checkpoints/sft/`
  - `chatrl_checkpoints/` → `checkpoints/rl/`
  - `base_eval/` → `eval/`
  - `identity_conversations.jsonl` → `identity.jsonl`
- Switched pretraining dataset from FineWeb-EDU to NVIDIA ClimbMix — time to GPT-2 reduced from 2.76h to 1.80h

#### Training hyperparameters
- Vocab size default: 50K → 32K
- D:N ratio: 20 → 8 (compute-optimal for nanochat)
- Warmdown ratio: 0.2 → 0.4
- Embedding learning rate: 0.2 → 0.3
- Adam beta1: 0.8 → 0.96
- Optimal tokens:params ratio tuned to ~4 (with value embeddings)
- Fused AdamW into single compiled kernel (1.7× faster)
- Combined AdamW and Muon into single `MuonAdamW` optimizer
- Pad vocab size to 64 for DDP optimizer efficiency

#### Misc
- Upgraded to PyTorch 2.9.1
- Upgraded synthetic data generation to Gemini 3
- Renamed `checkpoint_dir` → `checkpoints_dir` for consistency
- Renamed `max_iterations` → `num_iterations` in SFT
- Replaced `files-to-prompt` with `git ls-files` for bloat metrics
- Updated all shell scripts in `runs/` to use new module paths

### Fixed
- `config` shadowing bug in `build_model_meta()` (`GPTConfig` named `config` shadowed outer `TrainingConfig`)
- Syntax error: extra `)` in `dataset.py` `__main__` block
- Bare `except:` → `except Exception:` in `report.py` — no longer swallows `KeyboardInterrupt`/`SystemExit`
- All E402 module import errors
- All ruff linting errors (146 auto-fixed, 74 manually fixed)
- Completion-only loss masking in SFT dataloader
- FP8 application skipping tiny matmuls
- Attention window preservation in `chat_sft`
- KV-cache decode respecting sliding window
- Safe DDP cleanup (check initialized process group)
- TF32 deprecated API warnings
- `Random.seed()` footgun bug in SpellingBee data generation
- Missing `val_bpb` on resume
- Distributed Parquet dataloader resume for multi-epoch training
- KV cache indexing to include head dimension
- Float32 cast before logits softcapping
- Tok/sec calculation bug when `grad_accum_steps > 1`
- Memory leak in Rust tokenizer
- Tokenization bug (no space before first letter)
- Learning rate multiplier ramping direction
- `lstrip` bug (changed to `removeprefix`)
- CPU bfloat16 tensor loading (convert to float32)
- Grad clip bug (clipping per GPU before synchronization)
- Sample first token independently for multi-sample generation
- Typo "Addapted" → "Adapted" in `optimizer.py` docstring
- Missing `from pathlib import Path` import in `base_train.py`

### Removed
- `autocast` — precision now managed explicitly
- Midtraining stage — replaced by BOS-aligned dataloader + `chat_sft`
- Gradient clipping — not necessary, costs 2% MFU
- `base_loss` script — functionality merged into `base_eval`
- Numpy dependency
- Pandas dependency in `base_eval` (replaced with `csv`)
- Inline `rustbpe` project (now a separate package)
- Redundant exception handling
- Unnecessary tensor allocation in `DistAdamW`
- Old directories after `src/` migration

### Security
- Hardened eval: prevent calculator tool from accessing globals/locals
