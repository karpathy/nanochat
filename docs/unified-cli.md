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

A `nanochat/__main__.py` delegates to the same entry point, enabling `python -m nanochat` as an alternative to the installed `nanochat` command:

```python
from nanochat.cli import main
main()
```

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

### 3. Sectioned TOML config with matching dataclasses

Replace the flat `TrainingConfig` TOML with a sectioned format. Each section maps to a focused dataclass. Below is the complete specification with all fields and their defaults.

#### `[common]` — shared by all entry points

```toml
[common]
base_dir = ""              # override NANOCHAT_BASE_DIR env var (empty = use env var)
device_type = ""           # cuda | cpu | mps (empty = autodetect)
run = "my-run"             # wandb run name
wandb = "local"            # online | local | disabled
wandb_project = "nanochat" # wandb project name
```

```python
@dataclass
class CommonConfig:
    base_dir: Optional[str] = None
    device_type: str = ""
    run: str = "unnamed"
    wandb: str = "local"           # replaces --run=dummy magic + WANDB_MODE env var
    wandb_project: str = "nanochat"
```

#### `[training]` — base model pretraining (`base_train`)

```toml
[training]
# Model architecture
depth = 20
aspect_ratio = 64          # model_dim = depth * aspect_ratio
head_dim = 128
max_seq_len = 2048
window_pattern = "SSSL"    # L=full context, S=half context, tiled across layers

# Training horizon (first positive value wins)
num_iterations = -1        # explicit step count (-1 = disabled)
target_flops = -1.0        # compute budget in FLOPs (-1 = disabled)
target_param_data_ratio = 10.5  # tokens:params ratio (Chinchilla=20)

# Batch
device_batch_size = 32
total_batch_size = -1      # -1 = auto-compute optimal

# Optimizer
embedding_lr = 0.3
unembedding_lr = 0.008
matrix_lr = 0.02
scalar_lr = 0.5
weight_decay = 0.28
warmup_steps = 40
warmdown_ratio = 0.65
final_lr_frac = 0.05
resume_from_step = -1      # -1 = disabled

# Evaluation
eval_every = 250           # -1 = disabled
eval_tokens = 41943040     # 80 * 524288
core_metric_every = 2000   # -1 = disabled
core_metric_max_per_task = 500
sample_every = 2000        # -1 = disabled
save_every = -1            # -1 = only at end

# FP8
fp8 = false
fp8_recipe = "tensorwise"  # tensorwise | rowwise

# Output
model_tag = ""             # empty = auto (e.g. "d20")

# Compression metrics
track_compression = false
compression_log_every = 100
track_layer_compression = false
compression_early_stop = false
```

```python
@dataclass
class TrainingConfig(CommonConfig):
    # Model architecture
    depth: int = 20
    aspect_ratio: int = 64
    head_dim: int = 128
    max_seq_len: int = 2048
    window_pattern: str = "SSSL"
    # Training horizon
    num_iterations: int = -1
    target_flops: float = -1.0
    target_param_data_ratio: float = 10.5
    # Batch
    device_batch_size: int = 32
    total_batch_size: int = -1
    # Optimizer
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.28
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    resume_from_step: int = -1
    # Evaluation
    eval_every: int = 250
    eval_tokens: int = 80 * 524288
    core_metric_every: int = 2000
    core_metric_max_per_task: int = 500
    sample_every: int = 2000
    save_every: int = -1
    # FP8
    fp8: bool = False
    fp8_recipe: str = "tensorwise"
    # Output
    model_tag: Optional[str] = None
    # Compression
    track_compression: bool = False
    compression_log_every: int = 100
    track_layer_compression: bool = False
    compression_early_stop: bool = False
```

#### `[sft]` — supervised fine-tuning (`chat_sft`)

```toml
[sft]
# Model loading
model_tag = ""             # empty = auto (e.g. "d20")
model_step = -1            # -1 = last checkpoint
load_optimizer = true

# Training horizon
num_iterations = -1        # -1 = full epoch

# Batch (empty = inherit from pretrain checkpoint)
max_seq_len = -1           # -1 = inherit
device_batch_size = -1     # -1 = inherit
total_batch_size = -1      # -1 = inherit

# Optimizer (empty = inherit from pretrain checkpoint)
embedding_lr = -1.0        # -1 = inherit
unembedding_lr = -1.0      # -1 = inherit
matrix_lr = -1.0           # -1 = inherit
init_lr_frac = 0.8
warmup_ratio = 0.0
warmdown_ratio = 0.5
final_lr_frac = 0.0

# Evaluation
eval_every = 200
eval_tokens = 20971520     # 40 * 524288
chatcore_every = 200
chatcore_max_cat = -1      # -1 = no limit
chatcore_max_sample = 24

# Data mixture
mmlu_epochs = 3
gsm8k_epochs = 4
```

```python
@dataclass
class SFTConfig(CommonConfig):
    model_tag: Optional[str] = None
    model_step: Optional[int] = None
    load_optimizer: bool = True
    num_iterations: int = -1
    max_seq_len: Optional[int] = None
    device_batch_size: Optional[int] = None
    total_batch_size: Optional[int] = None
    embedding_lr: Optional[float] = None
    unembedding_lr: Optional[float] = None
    matrix_lr: Optional[float] = None
    init_lr_frac: float = 0.8
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0
    eval_every: int = 200
    eval_tokens: int = 40 * 524288
    chatcore_every: int = 200
    chatcore_max_cat: int = -1
    chatcore_max_sample: int = 24
    mmlu_epochs: int = 3
    gsm8k_epochs: int = 4
```

#### `[rl]` — reinforcement learning (`chat_rl`)

```toml
[rl]
# Model loading
model_tag = ""             # empty = auto
model_step = -1            # -1 = last checkpoint

# Training horizon
num_epochs = 1

# Batch / sampling
device_batch_size = 8
examples_per_step = 16
num_samples = 16

# Generation
max_new_tokens = 256
temperature = 1.0
top_k = 50

# Optimizer
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05

# Evaluation / checkpointing
eval_every = 60
eval_examples = 400
save_every = 60
```

```python
@dataclass
class RLConfig(CommonConfig):
    model_tag: Optional[str] = None
    model_step: Optional[int] = None
    num_epochs: int = 1
    device_batch_size: int = 8
    examples_per_step: int = 16
    num_samples: int = 16
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.02
    weight_decay: float = 0.0
    init_lr_frac: float = 0.05
    eval_every: int = 60
    eval_examples: int = 400
    save_every: int = 60
```

#### `[evaluation]` — base model evaluation (`base_eval`)

```toml
[evaluation]
modes = "core,bpb,sample"  # comma-separated: core | bpb | sample
hf_path = ""               # HuggingFace model path (empty = use nanochat checkpoint)
model_tag = ""             # empty = auto
step = -1                  # -1 = last checkpoint
max_per_task = -1          # -1 = all examples
device_batch_size = 32
split_tokens = 20971520    # 40 * 524288
```

```python
@dataclass
class EvaluationConfig(CommonConfig):
    modes: str = "core,bpb,sample"
    hf_path: Optional[str] = None
    model_tag: Optional[str] = None
    step: Optional[int] = None
    max_per_task: int = -1
    device_batch_size: int = 32
    split_tokens: int = 40 * 524288
```

#### `[chat]` — interactive CLI and web server (`chat_cli`, `chat_web`)

These scripts only need `[common]` for `base_dir` and `device_type`. Script-specific flags (model loading, generation params, server port) remain CLI-only as they are typically not persisted in a config file.

#### Root `Config`

A root `Config` dataclass composes all sections and owns all I/O. This is the single object passed around — scripts access only the section they need:

```python
@dataclass
class Config:
    common: CommonConfig = field(default_factory=CommonConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load from a TOML file. Each section inherits [common] fields then overrides."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        common = CommonConfig(**data.get("common", {}))
        def make(section_cls, key):
            return section_cls(**{**asdict(common), **data.get(key, {})})
        return cls(
            common=common,
            training=make(TrainingConfig, "training"),
            sft=make(SFTConfig, "sft"),
            rl=make(RLConfig, "rl"),
            evaluation=make(EvaluationConfig, "evaluation"),
        )

    def save(self, path: Path) -> None:
        """Save to a TOML file, one table per section, omitting common fields from sub-sections."""
        common_keys = set(asdict(self.common))
        def section_data(section):
            return {k: v for k, v in asdict(section).items() if k not in common_keys and v is not None}
        data = {
            "common": {k: v for k, v in asdict(self.common).items() if v is not None},
            "training": section_data(self.training),
            "sft": section_data(self.sft),
            "rl": section_data(self.rl),
            "evaluation": section_data(self.evaluation),
        }
        with open(path, "wb") as f:
            tomli_w.dump({k: v for k, v in data.items() if v}, f)

    @classmethod
    def from_args(cls, args: argparse.Namespace, section: str) -> "Config":
        """Build a Config from a parsed argparse Namespace, targeting one section."""
        config = cls()
        for field_name in config.common.__dataclass_fields__:
            if hasattr(args, field_name):
                setattr(config.common, field_name, getattr(args, field_name))
        section_obj = getattr(config, section)
        for field_name in section_obj.__dataclass_fields__:
            if hasattr(args, field_name):
                setattr(section_obj, field_name, getattr(args, field_name))
        return config

    def apply_args(self, args: argparse.Namespace, section: str) -> None:
        """Override fields with explicitly passed CLI args (SUPPRESS pattern)."""
        section_obj = getattr(self, section)
        for k, v in vars(args).items():
            if hasattr(self.common, k):
                setattr(self.common, k, v)
            elif hasattr(section_obj, k):
                setattr(section_obj, k, v)
```

Typical usage in a script:

```python
# Load from file (auto-discovered or explicit --config)
config = Config.load(path)
# Apply any explicit CLI overrides on top
config.apply_args(explicit_args, section="training")
# Access the relevant section
train(config.training)
```

### 4. Argparse with per-section builders

Instead of one monolithic parser or a third-party framework, each config section gets a matching `add_*_args` function. Scripts compose only what they need:

```python
def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--run", type=str, default="unnamed")
    parser.add_argument("--wandb", type=str, default="local", choices=["online", "local", "disabled"])
    parser.add_argument("--wandb-project", type=str, default="nanochat")

def add_training_args(parser: argparse.ArgumentParser) -> None:
    # ... all TrainingConfig fields

def add_sft_args(parser: argparse.ArgumentParser) -> None:
    # ... all SFTConfig fields

def add_rl_args(parser: argparse.ArgumentParser) -> None:
    # ... all RLConfig fields

def add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    # ... all EvaluationConfig fields
```

Each script composes its parser from the relevant builders:

```python
# base_train
parser = argparse.ArgumentParser()
add_common_args(parser)
add_training_args(parser)

# chat_sft
parser = argparse.ArgumentParser()
add_common_args(parser)
add_sft_args(parser)
```

This keeps the builders in sync with their dataclasses naturally — when a field is added to `TrainingConfig`, `add_training_args` is the only other place to update.

No third-party CLI framework is needed. Typer and Click don't integrate cleanly with the TOML-first + SUPPRESS override pattern: detecting which args were explicitly passed (vs defaulted) is argparse-specific and would require workarounds in other frameworks.

### 5. Default config generation

A `Config.generate_default(path)` classmethod writes a fully-commented `config.toml` with all sections and their defaults, so users have a starting point without reading source:

```python
@classmethod
def generate_default(cls, path: Path) -> None:
    """Write a fully-commented config.toml with all sections and default values."""
```

Example output:

```toml
[common]
base_dir = ""              # override NANOCHAT_BASE_DIR env var (empty = use env var)
device_type = ""           # cuda | cpu | mps (empty = autodetect)
run = "unnamed"            # wandb run name
wandb = "local"            # online | local | disabled
wandb_project = "nanochat"

[training]
depth = 20
aspect_ratio = 64          # model_dim = depth * aspect_ratio
# ... all fields with inline comments

[sft]
# ... etc.
```

Invoked via:

```bash
nanochat config init                        # writes config.toml in current directory
nanochat config init --output my.toml      # custom path
```

This makes `nanochat config init` the recommended starting point for any new experiment.

### 6. Wandb consolidation via `CommonConfig.wandb`

The `wandb` field in `[common]` replaces two current hacks:
- `--run="dummy"` magic value to disable wandb
- `WANDB_MODE=disabled` env var check (only in `base_train`)

A shared `init_wandb(config: CommonConfig) -> WandbRun` helper in `nanochat/common/wandb.py` handles all three cases:

```python
def init_wandb(config: CommonConfig, user_config: dict) -> DummyWandb | LocalWandb | wandb.Run:
    if config.wandb == "disabled":
        return DummyWandb()
    if config.wandb == "local":
        return LocalWandb(config.run, base_dir=config.base_dir)
    return wandb.init(project=config.wandb_project, name=config.run, config=user_config)
```

All training scripts (`base_train`, `chat_sft`, `chat_rl`) call `init_wandb` instead of their current inline logic.

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

1. **`Config` + dataclasses**: Define `CommonConfig`, `TrainingConfig(CommonConfig)`, `SFTConfig(CommonConfig)`, `RLConfig(CommonConfig)`, `EvaluationConfig(CommonConfig)`, and root `Config` with `load()`, `save()`, `from_args()`, `apply_args()`, `generate_default()` in `nanochat/common/config.py`.
2. **Per-section argparse builders**: Define `add_common_args()`, `add_training_args()`, `add_sft_args()`, `add_rl_args()`, `add_evaluation_args()` in `nanochat/common/config.py`. Replace all existing `build_parser()` functions to compose from these builders.
3. **Wandb consolidation**: Add `init_wandb(config: CommonConfig, user_config: dict)` to `nanochat/common/wandb.py`. Wire into `base_train`, `chat_sft`, `chat_rl`. Remove `--run=dummy` magic and `WANDB_MODE` env var checks.
4. **Wire `Config` into existing scripts**: Replace inline argparse + `TrainingConfig.from_args()` in `base_train`, `chat_sft`, `chat_rl`, `base_eval` with `Config.load()` / `Config.from_args()` / `Config.apply_args()`.
5. **Add `--config` and `--base-dir`** to the remaining entry points (`tok_train`, `tok_eval`, `chat_cli`, `chat_web`, `chat_eval`, `data download`) using the shared builders.
6. **Create `nanochat/__main__.py`** delegating to `nanochat.cli:main`.
7. **Create `nanochat/cli.py`** with subcommand dispatch including `nanochat config init`. Register `nanochat` script in `pyproject.toml`.
8. **Restructure `nanochat/scripts/`** into `train/`, `eval/`, `chat/` packages; move tokenizer scripts under `data/tokenizer/`. Keep `scripts/` shims for backward compatibility.
9. **Unit tests** for the new config system in `tests/test_common/test_config.py`:
   - `test_load_empty_toml` — `Config.load()` with only `[common]` populates section defaults
   - `test_common_fields_inherited` — section fields inherit `[common]` values
   - `test_cli_overrides_toml` — `apply_args()` with SUPPRESS only overrides explicitly passed args
   - `test_from_args_no_config` — `Config.from_args()` builds correct config without a TOML file
   - `test_save_roundtrip` — `save()` then `load()` produces identical `Config`
   - `test_save_omits_common_fields` — saved TOML does not duplicate common fields in sub-sections
   - `test_base_dir_autodiscovery` — `Config.load()` from `base_dir/config.toml` when only `--base-dir` is passed
   - `test_generate_default_is_valid_toml` — `generate_default()` output parses without error
   - `test_init_wandb_modes` — `init_wandb()` returns correct type for each of `online/local/disabled`
10. **Config reference doc** `docs/configuration.md` — complete reference covering:
    - All sections and fields with types, defaults, and descriptions (generated from `generate_default()` output)
    - Config resolution order (`base_dir` auto-discovery → `--config` → CLI args)
    - Annotated example `config.toml` for a full pretraining run
    - Annotated example `config.toml` for a full SFT + RL run
11. **CLI reference doc** `docs/cli.md` — complete reference covering:
    - All subcommands with descriptions
    - Global flags (`--config`, `--base-dir`)
    - Per-subcommand flag reference (generated from argparse)
    - Common usage examples: quick CPU run, MPS smoke test, full GPU pretraining, SFT, RL, eval
12. **Update existing docs**:
    - `README.md` — replace `python -m nanochat.scripts.*` examples with `nanochat` subcommands; add `nanochat config init` to quickstart
    - `docs/data-layout.md` — update module paths to new package structure
    - `docs/configuration.md` — replace flat `TrainingConfig` TOML description with sectioned format
    - `CHANGELOG.md` — breaking changes: `python -m nanochat.scripts.*` paths, `--run=dummy` removed, `TrainingConfig` flat TOML replaced
    - Bash scripts in `runs/` and `dev/` — `speedrun.sh`, `runcpu.sh`, `scaling_laws.sh`, `miniseries.sh`, `nanochat-helper.sh`

## Notes

- Steps 1–4 are self-contained and deliver immediate value (consistent config, wandb cleanup) without touching the CLI structure
- Steps 6–8 are the breaking changes — keep `scripts/` shims so `python -m nanochat.scripts.*` keeps working during transition
- `chat_cli` and `chat_web` only need `[common]` from config — their script-specific flags (generation params, server port) stay CLI-only
- `tok_eval` has no args at all today — it only needs `add_common_args()` for `base_dir`
- `nanochat config init` is the recommended starting point for any new experiment
