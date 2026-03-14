"""
Single source of truth for all paths under NANOCHAT_BASE_DIR.

Every module that needs a path should import from here instead of
constructing os.path.join(base_dir, ...) inline.
"""

import os
from nanochat.common import get_base_dir


def _dir(base: str, *parts: str) -> str:
    path = os.path.join(base, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def data_dir(base_dir: str | None = None) -> str:
    return _dir(base_dir or get_base_dir(), "data", "climbmix")


def legacy_data_dir(base_dir: str | None = None) -> str:
    """Legacy FinewebEdu-100B fallback path (no auto-create)."""
    return os.path.join(base_dir or get_base_dir(), "data", "fineweb")


def tokenizer_dir(base_dir: str | None = None) -> str:
    return _dir(base_dir or get_base_dir(), "tokenizer")


def checkpoint_dir(phase: str, model_tag: str, base_dir: str | None = None) -> str:
    assert phase in ("base", "sft", "rl"), f"Unknown phase: {phase}"
    return _dir(base_dir or get_base_dir(), "checkpoints", phase, model_tag)


def checkpoints_dir(phase: str, base_dir: str | None = None) -> str:
    assert phase in ("base", "sft", "rl"), f"Unknown phase: {phase}"
    return _dir(base_dir or get_base_dir(), "checkpoints", phase)


def eval_tasks_dir(base_dir: str | None = None) -> str:
    return _dir(base_dir or get_base_dir(), "data", "eval_tasks")


def eval_results_dir(base_dir: str | None = None) -> str:
    return _dir(base_dir or get_base_dir(), "eval")


def identity_data_path(base_dir: str | None = None) -> str:
    return os.path.join(base_dir or get_base_dir(), "identity.jsonl")
