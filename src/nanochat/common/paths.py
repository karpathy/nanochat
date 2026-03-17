"""
Single source of truth for all paths under NANOCHAT_BASE_DIR.

Every module that needs a path should import from here instead of
constructing os.path.join(base_dir, ...) inline.
"""

import os


def _dir(base: str, *parts: str) -> str:
    """Join path parts under base, create the directory if absent, and return the path."""
    path = os.path.join(base, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def get_default_base_dir() -> str:
    """Return the nanochat base directory: --base-dir CLI > NANOCHAT_BASE_DIR env > ~/.cache/nanochat."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ["NANOCHAT_BASE_DIR"]
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def root_data_dir(base_dir: str) -> str:
    """Return the top-level data directory under base_dir."""
    return _dir(base_dir, "data")


def data_dir(base_dir: str) -> str:
    """Return the primary training data directory (climbmix mix)."""
    return _dir(root_data_dir(base_dir), "climbmix")


def legacy_data_dir(base_dir: str) -> str:
    """Legacy FinewebEdu-100B fallback path (no auto-create)."""
    return os.path.join(root_data_dir(base_dir), "fineweb")


def eval_tasks_dir(base_dir: str) -> str:
    """Return the directory for evaluation task datasets."""
    return _dir(root_data_dir(base_dir), "eval_tasks")


def tokenizer_dir(base_dir: str) -> str:
    """Return the directory where tokenizer files are stored."""
    return _dir(base_dir, "tokenizer")


def checkpoint_dir(base_dir: str, phase: str, model_tag: str | None = None) -> str:
    """Return the checkpoint directory for a training phase, optionally scoped to a model tag."""
    assert phase in ("base", "sft", "rl"), f"Unknown phase: {phase}"
    if model_tag is not None:
        return _dir(base_dir, "checkpoints", phase, model_tag)
    return _dir(base_dir, "checkpoints", phase)


def eval_results_dir(base_dir: str) -> str:
    """Return the directory where evaluation results are written."""
    return _dir(base_dir, "eval")


def identity_data_path(base_dir: str) -> str:
    """Return the path to the identity fine-tuning data file."""
    return os.path.join(base_dir, "identity.jsonl")


def report_dir(base_dir: str) -> str:
    """Return the directory for training and evaluation reports."""
    return _dir(base_dir, "report")
