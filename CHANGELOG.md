# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Full modular architecture with src/nanochat/ layout
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
