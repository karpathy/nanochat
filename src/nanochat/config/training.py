"""Config for base model pre-training (architecture, optimizer, schedule, eval)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


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
