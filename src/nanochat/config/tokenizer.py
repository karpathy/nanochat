"""Config for BPE tokenizer training."""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    vocab_size: int = 32768
    max_chars: int = 2_000_000_000
    doc_cap: int = 10_000

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--vocab-size", type=int, default=argparse.SUPPRESS, help="vocabulary size (default: 32768)")
        parser.add_argument("--max-chars", type=int, default=argparse.SUPPRESS, help="max characters to train on (default: 2B)")
        parser.add_argument("--doc-cap", type=int, default=argparse.SUPPRESS, help="max characters per document (default: 10,000)")

    @classmethod
    def generate_default(cls) -> str:
        return (
            "vocab_size = 32768         # 2^15\n"
            "max_chars = 2000000000     # 2B characters\n"
            "doc_cap = 10000            # max characters per document\n"
        )
