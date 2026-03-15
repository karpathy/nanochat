"""
Unified configuration dataclasses and argparse builders for nanochat.

Layout:
  CommonConfig          — fields shared by all entry points
  TrainingConfig        — base model pretraining (inherits CommonConfig)
  SFTConfig             — supervised fine-tuning (inherits CommonConfig)
  RLConfig              — reinforcement learning (inherits CommonConfig)
  EvaluationConfig      — base model evaluation (inherits CommonConfig)
  Config                — root object; owns load/save/from_args/apply_args/generate_default

Config resolution order (later overrides earlier):
  1. Defaults in dataclasses
  2. [common] section of TOML file
  3. Per-section TOML values
  4. Explicit CLI args (via apply_args with SUPPRESS pattern)
"""

import argparse
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

import tomli_w


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CommonConfig:
    base_dir: Optional[str] = None
    device_type: str = ""
    run: str = "unnamed"
    wandb: str = "local"          # online | local | disabled
    wandb_project: str = "nanochat"


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


@dataclass
class EvaluationConfig(CommonConfig):
    modes: str = "core,bpb,sample"
    hf_path: Optional[str] = None
    model_tag: Optional[str] = None
    step: Optional[int] = None
    max_per_task: int = -1
    device_batch_size: int = 32
    split_tokens: int = 40 * 524288


# ---------------------------------------------------------------------------
# Root Config
# ---------------------------------------------------------------------------

_SECTION_MAP: dict[str, type] = {
    "training": TrainingConfig,
    "sft": SFTConfig,
    "rl": RLConfig,
    "evaluation": EvaluationConfig,
}

_COMMON_KEYS: frozenset[str] = frozenset(f.name for f in fields(CommonConfig))


@dataclass
class Config:
    common: CommonConfig = field(default_factory=CommonConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load from a TOML file. [common] fields seed each section; section keys override."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        common_data = data.get("common", {})
        common = CommonConfig(**common_data)
        common_dict = asdict(common)

        def make(section_cls: type, key: str) -> object:
            section_data = data.get(key, {})
            # common fields are the base; section-specific keys override
            merged = {**common_dict, **section_data}
            # only pass keys that the dataclass actually accepts
            valid = {f.name for f in fields(section_cls)}
            return section_cls(**{k: v for k, v in merged.items() if k in valid})

        return cls(
            common=common,
            training=make(TrainingConfig, "training"),  # type: ignore[arg-type]
            sft=make(SFTConfig, "sft"),                 # type: ignore[arg-type]
            rl=make(RLConfig, "rl"),                    # type: ignore[arg-type]
            evaluation=make(EvaluationConfig, "evaluation"),  # type: ignore[arg-type]
        )

    def save(self, path: Path) -> None:
        """Save to TOML. Common fields go in [common]; sub-sections omit them."""
        def _common_section() -> dict:
            return {k: v for k, v in asdict(self.common).items() if v is not None}

        def _sub_section(section: object) -> dict:
            return {
                k: v for k, v in asdict(section).items()  # type: ignore[call-overload]
                if k not in _COMMON_KEYS and v is not None
            }

        data: dict[str, dict] = {}
        common_d = _common_section()
        if common_d:
            data["common"] = common_d
        for key, section in [
            ("training", self.training),
            ("sft", self.sft),
            ("rl", self.rl),
            ("evaluation", self.evaluation),
        ]:
            sub = _sub_section(section)
            if sub:
                data[key] = sub

        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    @classmethod
    def from_args(cls, args: argparse.Namespace, section: str) -> "Config":
        """Build a Config purely from a parsed argparse Namespace (no TOML file)."""
        config = cls()
        for f in fields(config.common):
            if hasattr(args, f.name):
                setattr(config.common, f.name, getattr(args, f.name))
        section_obj = getattr(config, section)
        for f in fields(section_obj):
            if hasattr(args, f.name):
                setattr(section_obj, f.name, getattr(args, f.name))
        return config

    def apply_args(self, args: argparse.Namespace, section: str) -> None:
        """Override config fields with explicitly-passed CLI args (SUPPRESS pattern).

        Only keys present in vars(args) are applied — with SUPPRESS defaults,
        this means only args the user actually typed on the command line.
        """
        section_obj = getattr(self, section)
        for k, v in vars(args).items():
            if k in _COMMON_KEYS:
                setattr(self.common, k, v)
            elif hasattr(section_obj, k):
                setattr(section_obj, k, v)

    @classmethod
    def generate_default(cls, path: Path) -> None:
        """Write a fully-commented config.toml with all sections and default values."""
        lines = [
            "[common]",
            'base_dir = ""              # override NANOCHAT_BASE_DIR env var (empty = use env var)',
            'device_type = ""           # cuda | cpu | mps (empty = autodetect)',
            'run = "unnamed"            # wandb run name',
            'wandb = "local"            # online | local | disabled',
            'wandb_project = "nanochat"',
            "",
            "[training]",
            "depth = 20",
            "aspect_ratio = 64          # model_dim = depth * aspect_ratio",
            "head_dim = 128",
            "max_seq_len = 2048",
            'window_pattern = "SSSL"    # L=full context, S=half context, tiled across layers',
            "num_iterations = -1        # explicit step count (-1 = disabled)",
            "target_flops = -1.0        # compute budget in FLOPs (-1 = disabled)",
            "target_param_data_ratio = 10.5  # tokens:params ratio (Chinchilla=20)",
            "device_batch_size = 32",
            "total_batch_size = -1      # -1 = auto-compute optimal",
            "embedding_lr = 0.3",
            "unembedding_lr = 0.008",
            "matrix_lr = 0.02",
            "scalar_lr = 0.5",
            "weight_decay = 0.28",
            "warmup_steps = 40",
            "warmdown_ratio = 0.65",
            "final_lr_frac = 0.05",
            "resume_from_step = -1      # -1 = disabled",
            "eval_every = 250           # -1 = disabled",
            f"eval_tokens = {80 * 524288}       # 80 * 524288",
            "core_metric_every = 2000   # -1 = disabled",
            "core_metric_max_per_task = 500",
            "sample_every = 2000        # -1 = disabled",
            "save_every = -1            # -1 = only at end",
            "fp8 = false",
            'fp8_recipe = "tensorwise"  # tensorwise | rowwise',
            '# model_tag = ""           # empty = auto (e.g. "d20")',
            "track_compression = false",
            "compression_log_every = 100",
            "track_layer_compression = false",
            "compression_early_stop = false",
            "",
            "[sft]",
            '# model_tag = ""           # empty = auto',
            "# model_step = -1          # -1 = last checkpoint",
            "load_optimizer = true",
            "num_iterations = -1        # -1 = full epoch",
            "# max_seq_len = -1         # -1 = inherit from pretrain",
            "# device_batch_size = -1   # -1 = inherit from pretrain",
            "# total_batch_size = -1    # -1 = inherit from pretrain",
            "# embedding_lr = -1.0      # -1 = inherit from pretrain",
            "# unembedding_lr = -1.0    # -1 = inherit from pretrain",
            "# matrix_lr = -1.0        # -1 = inherit from pretrain",
            "init_lr_frac = 0.8",
            "warmup_ratio = 0.0",
            "warmdown_ratio = 0.5",
            "final_lr_frac = 0.0",
            "eval_every = 200",
            f"eval_tokens = {40 * 524288}       # 40 * 524288",
            "chatcore_every = 200",
            "chatcore_max_cat = -1      # -1 = no limit",
            "chatcore_max_sample = 24",
            "mmlu_epochs = 3",
            "gsm8k_epochs = 4",
            "",
            "[rl]",
            '# model_tag = ""           # empty = auto',
            "# model_step = -1          # -1 = last checkpoint",
            "num_epochs = 1",
            "device_batch_size = 8",
            "examples_per_step = 16",
            "num_samples = 16",
            "max_new_tokens = 256",
            "temperature = 1.0",
            "top_k = 50",
            "embedding_lr = 0.2",
            "unembedding_lr = 0.004",
            "matrix_lr = 0.02",
            "weight_decay = 0.0",
            "init_lr_frac = 0.05",
            "eval_every = 60",
            "eval_examples = 400",
            "save_every = 60",
            "",
            "[evaluation]",
            'modes = "core,bpb,sample"  # comma-separated: core | bpb | sample',
            '# hf_path = ""             # HuggingFace model path (empty = use nanochat checkpoint)',
            '# model_tag = ""           # empty = auto',
            "# step = -1                # -1 = last checkpoint",
            "max_per_task = -1          # -1 = all examples",
            "device_batch_size = 32",
            f"split_tokens = {40 * 524288}       # 40 * 524288",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Argparse builders
# ---------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-dir", type=str, default=argparse.SUPPRESS, help="override NANOCHAT_BASE_DIR env var")
    parser.add_argument("--device-type", type=str, default=argparse.SUPPRESS, help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--run", type=str, default=argparse.SUPPRESS, help="wandb run name")
    parser.add_argument(
        "--wandb", type=str, default=argparse.SUPPRESS,
        choices=["online", "local", "disabled"],
        help="wandb mode: online | local | disabled",
    )
    parser.add_argument("--wandb-project", type=str, default=argparse.SUPPRESS, help="wandb project name")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    # Model architecture
    parser.add_argument("--depth", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--aspect-ratio", type=int, default=argparse.SUPPRESS, help="model_dim = depth * aspect_ratio")
    parser.add_argument("--head-dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-seq-len", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--window-pattern", type=str, default=argparse.SUPPRESS, help="L=full, S=half context, tiled")
    # Training horizon
    parser.add_argument("--num-iterations", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
    parser.add_argument("--target-flops", type=float, default=argparse.SUPPRESS, help="-1 = disabled")
    parser.add_argument("--target-param-data-ratio", type=float, default=argparse.SUPPRESS)
    # Batch
    parser.add_argument("--device-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--total-batch-size", type=int, default=argparse.SUPPRESS, help="-1 = auto")
    # Optimizer
    parser.add_argument("--embedding-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--unembedding-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--matrix-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--scalar-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--warmup-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--warmdown-ratio", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--final-lr-frac", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--resume-from-step", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
    # Evaluation
    parser.add_argument("--eval-every", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
    parser.add_argument("--eval-tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--core-metric-every", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
    parser.add_argument("--core-metric-max-per-task", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sample-every", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
    parser.add_argument("--save-every", type=int, default=argparse.SUPPRESS, help="-1 = only at end")
    # FP8
    parser.add_argument("--fp8", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--fp8-recipe", type=str, default=argparse.SUPPRESS, choices=["tensorwise", "rowwise"])
    # Output
    parser.add_argument("--model-tag", type=str, default=argparse.SUPPRESS)
    # Compression
    parser.add_argument("--track-compression", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--compression-log-every", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--track-layer-compression", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--compression-early-stop", action="store_true", default=argparse.SUPPRESS)


def add_sft_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-tag", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--model-step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--load-optimizer", type=int, default=argparse.SUPPRESS, help="0=no, 1=yes")
    parser.add_argument("--num-iterations", type=int, default=argparse.SUPPRESS, help="-1 = full epoch")
    parser.add_argument("--max-seq-len", type=int, default=argparse.SUPPRESS, help="None = inherit from pretrain")
    parser.add_argument("--device-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--total-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--embedding-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--unembedding-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--matrix-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--init-lr-frac", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--warmup-ratio", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--warmdown-ratio", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--final-lr-frac", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--eval-every", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
    parser.add_argument("--eval-tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--chatcore-every", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
    parser.add_argument("--chatcore-max-cat", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--chatcore-max-sample", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mmlu-epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--gsm8k-epochs", type=int, default=argparse.SUPPRESS)


def add_rl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-tag", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--model-step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--examples-per-step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-samples", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-new-tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--top-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--embedding-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--unembedding-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--matrix-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--init-lr-frac", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--eval-every", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--eval-examples", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--save-every", type=int, default=argparse.SUPPRESS)


def add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--modes", type=str, default=argparse.SUPPRESS, help="comma-separated: core,bpb,sample")
    parser.add_argument("--hf-path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--model-tag", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-per-task", type=int, default=argparse.SUPPRESS, help="-1 = all")
    parser.add_argument("--device-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--split-tokens", type=int, default=argparse.SUPPRESS)
