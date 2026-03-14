"""
Configuration classes for GPT model and training.
"""

import tomllib
import tomli_w
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class GPTConfig:
    """Configuration for GPT model architecture."""

    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    # Model architecture
    depth: int
    aspect_ratio: int = 64
    head_dim: int = 128
    max_seq_len: int = 2048
    window_pattern: str = "SSSL"

    # Training horizon
    num_iterations: int = -1
    target_flops: float = -1.0
    target_param_data_ratio: float = 10.5

    # Optimization
    device_batch_size: int = 32
    total_batch_size: int = -1
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    weight_decay: float = 0.28
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
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

    # Runtime
    device_type: str = ""
    fp8: bool = False
    fp8_recipe: str = "tensorwise"

    # Output
    model_tag: Optional[str] = None
    run: str = "dummy"  # wandb run name

    # Compression metrics
    track_compression: bool = False
    compression_log_every: int = 100
    track_layer_compression: bool = False
    compression_early_stop: bool = False

    # Data / paths
    base_dir: Optional[str] = None  # overrides NANOCHAT_BASE_DIR env var

    def save(self, path: Path) -> None:
        """Save configuration to TOML file."""
        data = {k: v for k, v in asdict(self).items() if v is not None}
        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load configuration from TOML file."""
        with open(path, "rb") as f:
            return cls(**tomllib.load(f))

    @classmethod
    def from_args(cls, args: Any) -> "TrainingConfig":
        """Create TrainingConfig from argparse Namespace."""
        config_dict = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(args, field_name):
                config_dict[field_name] = getattr(args, field_name)
        return cls(**config_dict)
