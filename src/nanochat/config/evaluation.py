"""Config for standalone model evaluation runs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

VALID_EVAL_MODES = frozenset({"core", "bpb", "sample"})


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

    def __post_init__(self) -> None:
        invalid = {m.strip() for m in self.modes.split(",")} - VALID_EVAL_MODES
        if invalid:
            raise ValueError(f"Invalid eval modes: {invalid}. Valid: {VALID_EVAL_MODES}")

    modes: str = "core,bpb,sample"
    hf_path: Optional[str] = None
    model_tag: Optional[str] = None
    step: Optional[int] = None
    max_per_task: int = -1
    device_batch_size: int = 32
    split_tokens: int = 40 * 524288
