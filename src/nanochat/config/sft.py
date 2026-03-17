"""Config for supervised fine-tuning (SFT) on chat data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


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
