"""
Module docstring for nanochat.glossary.

Author: Claude Haiku 4.5
Created: 2026-11-14
Scope: Glossary catalog and stage-term mapping for CLI learning output.
  - STAGE_TERMS: mapping of stage name to tuple of term keys
  - Term definitions with URLs (one short sentence each, ELI15 style)
  - format_learn_line(stage) → one line of formatted terms, or empty string
  - emit_define(term) → prints define line once per process when NANOCHAT_STAGE_BRIEFED is not set
"""

import argparse
import os
import sys
from typing import Dict, Tuple

# Term catalog: term_key -> (short_definition, url)
GLOSSARY: Dict[str, Tuple[str, str]] = {
    "BPE": (
        "Byte Pair Encoding: a compression algorithm that learns the vocabulary of frequent byte sequences in text.",
        "docs/RUNNING.md#6b-terms"
    ),
    "vocab": (
        "The complete set of unique tokens the tokenizer recognizes, created by learning from training data.",
        "docs/RUNNING.md#6b-terms"
    ),
    "compression ratio": (
        "How much raw text shrinks when encoded into tokens; lower is better (fewer tokens per byte).",
        "docs/RUNNING.md#6b-terms"
    ),
    "BPB": (
        "Bits Per Byte: the number of bits needed to encode one byte of data; measures model compression efficiency.",
        "docs/RUNNING.md#6b-terms"
    ),
    "CORE": (
        "DCLM CORE score: a benchmark metric that rates how well a model predicts text from the web.",
        "https://github.com/mlfoundations/dclm"
    ),
    "ChatCORE": (
        "A variant of CORE score adapted to measure performance on chat and instruction-following tasks.",
        "docs/RUNNING.md#6b-terms"
    ),
    "MFU": (
        "Model Flops Utilization: the fraction of theoretical peak floating-point operations actually used during training.",
        "https://arxiv.org/abs/2204.02311"
    ),
    "Muon": (
        "An optimizer that orthogonalizes momentum updates for the weight matrices of the model.",
        "https://github.com/KellerJordan/Muon"
    ),
    "FLOPs": (
        "Floating Point Operations: the raw count of arithmetic operations performed during training or inference.",
        "docs/RUNNING.md#6b-terms"
    ),
}

# Stage to term keys mapping. Each stage maps to a tuple of term keys (max 4 per stage).
# base_train has the max: BPB, Muon, MFU, FLOPs = 4 terms.
STAGE_TERMS: Dict[str, Tuple[str, ...]] = {
    "tok_train": ("BPE", "vocab", "compression ratio"),
    "tok_eval": ("compression ratio",),
    "base_train": ("BPB", "Muon", "MFU", "FLOPs"),
    "base_eval": ("BPB", "CORE"),
    "chat_sft": (),
    "chat_eval": ("ChatCORE",),
}

# Track which terms have been defined in this process (for emit_define).
_defined_terms: set[str] = set()


def format_learn_line(stage: str) -> str:
    """
    Format a single-line learn briefing for a stage.

    Args:
        stage: Stage name (e.g., 'base_train').

    Returns:
        One line "learn: <Term> = <def> (<url>) · ..." or empty string if stage has no terms.
        Max 4 terms per line (enforced by STAGE_TERMS).
    """
    if stage not in STAGE_TERMS:
        return ""

    term_keys = STAGE_TERMS[stage]
    if not term_keys:
        return ""

    parts = []
    for key in term_keys:
        if key not in GLOSSARY:
            continue
        definition, url = GLOSSARY[key]
        parts.append(f"{key} = {definition} ({url})")

    if not parts:
        return ""

    return "learn: " + " · ".join(parts)


def emit_define(term: str) -> None:
    """
    Print a define line for a term, once per process, when NANOCHAT_STAGE_BRIEFED is not set.

    If NANOCHAT_STAGE_BRIEFED=1, this is a no-op (the shell has already briefed the stage).
    Otherwise, prints "define: <Term> = <def> (<url>)" the first time this term is called,
    then memoizes it to avoid repeating.

    Args:
        term: The term key to define (e.g., 'BPB').
    """
    # If briefed by the shell, do nothing.
    if os.environ.get("NANOCHAT_STAGE_BRIEFED") == "1":
        return

    # If already defined in this process, do nothing.
    if term in _defined_terms:
        return

    # If term not in glossary, do nothing.
    if term not in GLOSSARY:
        return

    # Mark it as defined and print.
    _defined_terms.add(term)
    definition, url = GLOSSARY[term]
    print(f"define: {term} = {definition} ({url})")


def main() -> None:
    """CLI entry point for glossary.learn."""
    parser = argparse.ArgumentParser(
        description="Glossary module CLI: emit learn lines per stage."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    learn_parser = subparsers.add_parser("learn", help="Emit learn line for a stage")
    learn_parser.add_argument(
        "--stage",
        required=True,
        help="Stage name (tok_train, tok_eval, base_train, base_eval, chat_sft, chat_eval)",
    )

    args = parser.parse_args()

    if args.command == "learn":
        line = format_learn_line(args.stage)
        if line:
            print(line)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
