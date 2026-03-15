"""
Unified configuration dataclasses and argparse builders for nanochat.

Layout:
  CommonConfig          — shared fields (base_dir, device_type, run, wandb)
  TrainingConfig        — base model pretraining
  SFTConfig             — supervised fine-tuning
  RLConfig              — reinforcement learning
  EvaluationConfig      — evaluation
  Config                — root object owning all sections; save() and generate_default()
  ConfigLoader          — builds argparse parser and resolves Config from TOML + CLI

Config resolution order (later overrides earlier):
  1. Dataclass field defaults
  2. TOML file values (per section)
  3. Explicit CLI args (argparse.SUPPRESS ensures only provided args override)

Usage:
  cfg = ConfigLoader().add_training().parse()
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
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

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--config", type=str, default=argparse.SUPPRESS, help="path to TOML config file (CLI args override file values)")
        parser.add_argument("--base-dir", type=str, default=argparse.SUPPRESS, help="override NANOCHAT_BASE_DIR env var")
        parser.add_argument("--device-type", type=str, default=argparse.SUPPRESS, help="cuda|cpu|mps (empty = autodetect)")
        parser.add_argument("--run", type=str, default=argparse.SUPPRESS, help="wandb run name")
        parser.add_argument("--wandb", type=str, default=argparse.SUPPRESS, choices=["online", "local", "disabled"], help="wandb mode: online | local | disabled")
        parser.add_argument("--wandb-project", type=str, default=argparse.SUPPRESS, help="wandb project name")

    @classmethod
    def generate_default(cls) -> str:
        return (
            'base_dir = ""              # override NANOCHAT_BASE_DIR env var (empty = use env var)\n'
            'device_type = ""           # cuda | cpu | mps (empty = autodetect)\n'
            'run = "unnamed"            # wandb run name\n'
            'wandb = "local"            # online | local | disabled\n'
            'wandb_project = "nanochat"\n'
        )


@dataclass
class TrainingConfig:
    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
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

    @classmethod
    def generate_default(cls) -> str:
        return (
            "depth = 20\n"
            "aspect_ratio = 64          # model_dim = depth * aspect_ratio\n"
            "head_dim = 128\n"
            "max_seq_len = 2048\n"
            'window_pattern = "SSSL"    # L=full context, S=half context, tiled across layers\n'
            "num_iterations = -1        # explicit step count (-1 = disabled)\n"
            "target_flops = -1.0        # compute budget in FLOPs (-1 = disabled)\n"
            "target_param_data_ratio = 10.5  # tokens:params ratio (Chinchilla=20)\n"
            "device_batch_size = 32\n"
            "total_batch_size = -1      # -1 = auto-compute optimal\n"
            "embedding_lr = 0.3\n"
            "unembedding_lr = 0.008\n"
            "matrix_lr = 0.02\n"
            "scalar_lr = 0.5\n"
            "weight_decay = 0.28\n"
            "warmup_steps = 40\n"
            "warmdown_ratio = 0.65\n"
            "final_lr_frac = 0.05\n"
            "resume_from_step = -1      # -1 = disabled\n"
            "eval_every = 250           # -1 = disabled\n"
            f"eval_tokens = {80 * 524288}       # 80 * 524288\n"
            "core_metric_every = 2000   # -1 = disabled\n"
            "core_metric_max_per_task = 500\n"
            "sample_every = 2000        # -1 = disabled\n"
            "save_every = -1            # -1 = only at end\n"
            "fp8 = false\n"
            'fp8_recipe = "tensorwise"  # tensorwise | rowwise\n'
            '# model_tag = ""           # empty = auto (e.g. "d20")\n'
            "track_compression = false\n"
            "compression_log_every = 100\n"
            "track_layer_compression = false\n"
            "compression_early_stop = false\n"
        )

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
class SFTConfig:
    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model-tag", type=str, default=argparse.SUPPRESS)
        parser.add_argument("--model-step", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--load-optimizer", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
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

    @classmethod
    def generate_default(cls) -> str:
        return (
            '# model_tag = ""           # empty = auto\n'
            "# model_step = -1          # -1 = last checkpoint\n"
            "load_optimizer = true\n"
            "num_iterations = -1        # -1 = full epoch\n"
            "# max_seq_len = -1         # -1 = inherit from pretrain\n"
            "# device_batch_size = -1   # -1 = inherit from pretrain\n"
            "# total_batch_size = -1    # -1 = inherit from pretrain\n"
            "# embedding_lr = -1.0      # -1 = inherit from pretrain\n"
            "# unembedding_lr = -1.0    # -1 = inherit from pretrain\n"
            "# matrix_lr = -1.0        # -1 = inherit from pretrain\n"
            "init_lr_frac = 0.8\n"
            "warmup_ratio = 0.0\n"
            "warmdown_ratio = 0.5\n"
            "final_lr_frac = 0.0\n"
            "eval_every = 200\n"
            f"eval_tokens = {40 * 524288}       # 40 * 524288\n"
            "chatcore_every = 200\n"
            "chatcore_max_cat = -1      # -1 = no limit\n"
            "chatcore_max_sample = 24\n"
            "mmlu_epochs = 3\n"
            "gsm8k_epochs = 4\n"
        )

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
class RLConfig:
    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
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

    @classmethod
    def generate_default(cls) -> str:
        return (
            '# model_tag = ""           # empty = auto\n'
            "# model_step = -1          # -1 = last checkpoint\n"
            "num_epochs = 1\n"
            "device_batch_size = 8\n"
            "examples_per_step = 16\n"
            "num_samples = 16\n"
            "max_new_tokens = 256\n"
            "temperature = 1.0\n"
            "top_k = 50\n"
            "embedding_lr = 0.2\n"
            "unembedding_lr = 0.004\n"
            "matrix_lr = 0.02\n"
            "weight_decay = 0.0\n"
            "init_lr_frac = 0.05\n"
            "eval_every = 60\n"
            "eval_examples = 400\n"
            "save_every = 60\n"
        )

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
class EvaluationConfig:
    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--modes", type=str, default=argparse.SUPPRESS, help="comma-separated: core,bpb,sample")
        parser.add_argument("--hf-path", type=str, default=argparse.SUPPRESS)
        parser.add_argument("--model-tag", type=str, default=argparse.SUPPRESS)
        parser.add_argument("--step", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--max-per-task", type=int, default=argparse.SUPPRESS, help="-1 = all")
        parser.add_argument("--device-batch-size", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--split-tokens", type=int, default=argparse.SUPPRESS)

    @classmethod
    def generate_default(cls) -> str:
        return (
            'modes = "core,bpb,sample"  # comma-separated: core | bpb | sample\n'
            '# hf_path = ""             # HuggingFace model path (empty = use nanochat checkpoint)\n'
            '# model_tag = ""           # empty = auto\n'
            "# step = -1                # -1 = last checkpoint\n"
            "max_per_task = -1          # -1 = all examples\n"
            "device_batch_size = 32\n"
            f"split_tokens = {40 * 524288}       # 40 * 524288\n"
        )

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


_SECTION_CLS: dict[str, type] = {
    "common": CommonConfig,
    "training": TrainingConfig,
    "sft": SFTConfig,
    "rl": RLConfig,
    "evaluation": EvaluationConfig,
}


@dataclass
class Config:
    common: CommonConfig = field(default_factory=CommonConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def save(self, path: Path) -> None:
        """Save to TOML."""
        data = {k: {fk: fv for fk, fv in v.items() if fv is not None} for k, v in asdict(self).items()}
        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> Config:
        """Load from TOML, populating only the sections present in the file."""
        import tomllib
        with open(path, "rb") as f:
            data = tomllib.load(f)
        cfg = cls()
        for section, values in data.items():
            section_cls = _SECTION_CLS.get(section)
            if section_cls is None:
                raise ValueError(f"Unknown config section: {section!r}")
            setattr(cfg, section, section_cls(**values))
        return cfg

    @classmethod
    def generate_default(cls) -> str:
        return (
            "[common]\n" + CommonConfig.generate_default() + "\n"
            "[training]\n" + TrainingConfig.generate_default() + "\n"
            "[sft]\n" + SFTConfig.generate_default() + "\n"
            "[rl]\n" + RLConfig.generate_default() + "\n"
            "[evaluation]\n" + EvaluationConfig.generate_default()
        )


class ConfigLoader:

    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser()
        self._config = Config()
        self._sections: set[str] = {"common"}
        CommonConfig.update_parser(self._parser)

    def _add_section(self, name: str, cls: type) -> ConfigLoader:
        if self._sections - {"common"}:
            raise RuntimeError("ConfigLoader only supports one section per instance")
        self._sections.add(name)
        cls.update_parser(self._parser)
        return self

    def add_training(self) -> ConfigLoader:
        return self._add_section("training", TrainingConfig)

    def add_sft(self) -> ConfigLoader:
        return self._add_section("sft", SFTConfig)

    def add_rl(self) -> ConfigLoader:
        return self._add_section("rl", RLConfig)

    def add_evaluation(self) -> ConfigLoader:
        return self._add_section("evaluation", EvaluationConfig)

    def parse(self, args: list[str] | None = None) -> Config:
        import tomllib
        ns = self._parser.parse_args(args)
        cli = vars(ns)

        toml_path: Path | None = None
        if "base_dir" in cli and cli["base_dir"] is not None:
            candidate = Path(cli["base_dir"]) / "config.toml"
            if candidate.exists():
                toml_path = candidate
        if "config" in cli and cli["config"] is not None:
            toml_path = Path(cli["config"])

        toml_data: dict = {}
        if toml_path is not None:
            with open(toml_path, "rb") as f:
                toml_data = tomllib.load(f)

        for section in self._sections:
            cls = _SECTION_CLS[section]
            valid = cls.__dataclass_fields__
            merged = {**toml_data.get(section, {}), **{k: v for k, v in cli.items() if k in valid}}
            setattr(self._config, section, cls(**merged))

        return self._config
