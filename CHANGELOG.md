# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Parameter type annotations across all 29 files (323→0 `reportMissingParameterType` under pyright strict mode)
- `docs/archive/type-annotations-pyright-compliance.md`
- `docs/archive/refactor-common-package.md`

### Changed
- MPS backend now uses float16 instead of float32 (~10-30% faster, halves memory usage)
- MPS training uses `torch.mps.synchronize()` for accurate step timing and `torch.mps.current_allocated_memory()` for memory reporting
- MPS training calls `torch.mps.empty_cache()` between eval and training steps
- KV cache in `evaluation/engine.py` uses `get_compute_dtype()` instead of hardcoded dtype hack
- Rewrote `docs/m3-max-guide.md` from brainstorming notes into proper MPS backend documentation covering device detection, dtype behavior, limitations, and practical training guidelines
- Refactored monolithic `common.py` into `common/` package with 7 focused modules (dtype, logging, distributed, io, hardware, wandb, paths)
- Absorbed top-level `paths.py` into `common/paths.py`
- `__init__.py` re-exports all public names — 16 consumer files required zero import changes
- Updated 5 files importing from `nanochat.paths` → `nanochat.common.paths`

### Changed
- `checkpoint.py` now uses `paths.checkpoints_dir()` instead of manual `get_base_dir()` + `os.path.join()` in `load_model()` and `load_optimizer_state()`
- `USE_FA3` in `flash_attention.py` deferred from module-level constant to lazy `_use_fa3()` with caching — no longer triggers dtype detection at import time
- Remaining 5 scripts wrapped in `main()`: `tok_train.py`, `tok_eval.py`, `chat_cli.py`, `chat_web.py`, `chat_eval.py` — all 9 console scripts now have proper entry points
- `WorkerPool` in `chat_web.py` accepts `device_type` parameter instead of reading module-level variable

### Fixed
- Syntax error: extra `)` in `dataset.py` `__main__` block
- Bare `except:` → `except Exception:` in `report.py` `run_command()` and `extract_timestamp()` — no longer swallows `KeyboardInterrupt`/`SystemExit`
- Typo "Addapted" → "Adapted" in `optimizer.py` docstring

### Added
- `docs/archive/phase-1.5.0.2-code-review-cleanup.md`
- `docs/data-layout.md` documenting hierarchical directory structure
- `tests/test_paths.py` with 8 tests covering all path functions
- 3 new config tests: TOML round-trip, from_args mapping, base_dir default
- `build_parser()` in all three training scripts for reusable CLI setup
- `--config` flag to load `TrainingConfig` from TOML file (CLI args override file values)
- `--base-dir` flag to override `NANOCHAT_BASE_DIR` env var
- TOML config save/load to `TrainingConfig` (replaces JSON)
- Compression fields to `TrainingConfig` (`track_compression`, `compression_log_every`, `track_layer_compression`, `compression_early_stop`)
- `base_dir` field to `TrainingConfig`
- Auto-save `config.toml` to checkpoint directory on training start
- `tomli` and `tomli-w` dependencies for TOML support on Python 3.10

### Changed
- Restructured `NANOCHAT_BASE_DIR` from flat to hierarchical layout:
  - `base_data_climbmix/` → `data/climbmix/`
  - `eval_bundle/` → `data/eval_tasks/`
  - `base_checkpoints/` → `checkpoints/base/`
  - `chatsft_checkpoints/` → `checkpoints/sft/`
  - `chatrl_checkpoints/` → `checkpoints/rl/`
  - `base_eval/` → `eval/`
  - `identity_conversations.jsonl` → `identity.jsonl`
- All 8 consumer files updated to use `paths.py` instead of inline path construction
- `COMPUTE_DTYPE` deferred to lazy `get_compute_dtype()` / `get_compute_dtype_reason()`
- `setup_default_logging()` made idempotent, called from `compute_init()` instead of module level
- `DATA_DIR` in `dataset.py` replaced with lazy `_get_data_dir()` function
- `base_train.py`: all top-level setup wrapped in `main()`, `train_base_model` converted from 30-param function to closure, `build_parser()` extracted
- `chat_sft.py`: all top-level setup wrapped in `main()`, `build_parser()` extracted, global vars converted to nonlocal
- `chat_rl.py`: all top-level setup wrapped in `main()`, `build_parser()` extracted, nested closures for get_batch/run_gsm8k_eval
- All three scripts now importable without side effects

### Fixed
- `config` shadowing bug in `build_model_meta()` (local `GPTConfig` named `config` shadowed outer `TrainingConfig`)
- Missing `from pathlib import Path` import in `base_train.py`
- Removed redundant `setup_default_logging()` call from `checkpoint.py`

### Changed
- `base_train.py` now uses `TrainingConfig` throughout — raw `args` usage replaced
- `TrainingConfig.save()` / `load()` switched from JSON to TOML format
- Console scripts for CLI commands (nanochat-train, nanochat-eval, nanochat-chat, nanochat-sft, nanochat-rl, nanochat-chat-eval, nanochat-web, nanochat-tok-train, nanochat-tok-eval)
- Comprehensive test suite (30 tests: models, training, tasks, integration)
- CI/CD workflow with GitHub Actions (pytest, ruff, pyright)
- Public API in main __init__.py (GPT, Engine, get_tokenizer, optimizers)
- Type hints across core modules (models, training, data, tasks)
- Conversation type definitions (Message, Conversation)
- Importable training functions (train_base_model, train_sft_model, train_rl_model)
- Importable evaluation functions (evaluate_base_model, evaluate_validation_loss, evaluate_chat_model)
- LR schedulers module (warmup/constant/warmdown, Muon momentum, weight decay)
- TrainingConfig.from_args() method for CLI integration
- Flash Attention 3 with fallback to PyTorch SDPA
- Learnable lambdas for residual and skip connections
- Alternating window size patterns (SSSL: 3 short, 1 long)
- BOS-aligned dataloaders with epoch tracking
- Pretraining resumption logic with checkpoints
- SpellingBee task for character counting
- Calculator count function support
- Personality system for nanochat
- Web UI slash commands
- Click-to-edit messages in web UI
- Click-to-regenerate assistant messages
- Basic logging and abuse prevention for chat_web
- Multi-GPU inference support (data parallel)
- CPU and MPS (Apple Silicon) support with autodetect
- MIT License file
- FP8 training with torchao
- Muon optimizer with Polar Express and Adafactor-style variance reduction
- Cautious weight decay with linear schedule
- Scaling law for optimal weight decay (∝ 1/channels²)
- Auto-calculated optimal batch size per model size
- Miniseries script and scaling laws analysis
- Jupyter notebook support
- CORE score evaluation for HuggingFace models
- --dry_run option for experimentation
- --save_every and --resume_from_step flags for checkpoint management
- Eval bundle lazy download in Python code

### Changed
- Migrated to src/nanochat/ layout with proper package structure
- Split gpt.py into modular components (config.py, attention.py, mlp.py, gpt.py)
- Reorganized into domain modules:
  - models/ (config, attention, mlp, gpt)
  - training/ (optimizer, dataloader, checkpoint, schedulers)
  - evaluation/ (core_eval, loss_eval, engine)
  - data/ (tokenizer, dataset)
  - tasks/ (base, mmlu, arc, gsm8k, humaneval, smoltalk, spellingbee, customjson)
  - scripts/ (all training/eval scripts)
- Extracted importable functions from scripts:
  - train_base_model() from base_train.py
  - train_sft_model() and train_rl_model() from chat scripts
  - evaluate_base_model(), evaluate_validation_loss(), evaluate_chat_model()
- Updated all shell scripts in runs/ to use new module paths
- Renamed tasks/common.py → tasks/base.py
- Updated pyproject.toml to hatchling build backend with proper package configuration
- Replaced Configurator with argparse
- Updated TrainingConfig to match base_train.py arguments
- Simplified model initialization
- Upgraded to PyTorch 2.9.1
- Replaced files-to-prompt with git ls-files for bloat metrics
- Vocab size default from 50K to 32K
- D:N ratio from 20 to 8
- Warmdown ratio from 0.2 to 0.4
- Embedding learning rate from 0.2 to 0.3
- Adam beta1 from 0.8 to 0.96
- Optimal ratio tuned to ~4
- Switched from FineWeb-EDU to NVIDIA ClimbMix dataset
- Time to GPT-2 reduced from 2.76 hours to 1.80 hours
- Upgraded synthetic data generation to Gemini 3
- Fused AdamW into single compiled kernel (1.7x faster)
- Combined AdamW and Muon into single MuonAdamW optimizer
- Pad vocab size to 64 for DDP optimizer efficiency
- Renamed checkpoint_dir to checkpoints_dir for consistency
- Renamed max_iterations to num_iterations in SFT for consistency

### Fixed
- All E402 module import errors (moved imports to top of files)
- Import conflicts by renaming old nanochat/ → nanochat_old/
- Restored working chat_rl.py and chat_sft.py after refactor
- All ruff linting errors (146 auto-fixed, 74 manually fixed)
- Indentation in train_base_model function
- Completion-only loss masking in SFT dataloader
- Bug in setting precision
- MockModel device definition
- FP8 application to skip tiny matmuls
- Attention window preservation in chat_sft
- Broken import after refactor
- KV-cache decode to respect sliding window
- Safe DDP cleanup (check initialized process group)
- Batch encode speedup test assertion
- TF32 deprecated API warnings
- Random.seed() footgun bug for SpellingBee data
- Missing val_bpb on resume
- Distributed Parquet dataloader resume for multi-epoch training
- KV cache indexing to include head dimension
- Float32 cast before logits softcapping
- Tok/sec calculation bug when grad_accum_steps > 1
- Memory leak in Rust tokenizer
- Tokenization bug (no space before first letter)
- Learning rate multiplier ramping direction
- UTF-8 encoding issues for portability
- lstrip bug (changed to removeprefix)
- CPU bfloat16 tensor loading (convert to float32)
- Grad clip bug (clipping per GPU before synchronization)
- Sample first token independently for multi-sample generation

### Removed
- Old directories after migration
- Autocast (manage dtypes directly)
- Midtraining (replaced by BOS-aligned dataloader)
- Grad clip (not necessary, costs 2% MFU)
- Numpy dependency
- Pandas dependency in base_eval (use csv instead)
- Inline rustbpe project (now separate package)
- Redundant exception handling
- Unnecessary tensor allocation in DistAdamW

### Security
- Hardened eval: prevent calc tool from accessing globals/locals
