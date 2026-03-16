"""Config for reinforcement learning (RLHF/GRPO) fine-tuning."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


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
