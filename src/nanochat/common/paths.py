"""
Single source of truth for all paths under NANOCHAT_BASE_DIR.

Every module that needs a path should import from here instead of
constructing os.path.join(base_dir, ...) inline.
"""

import os

def _dir(base: str, *parts: str) -> str:
    path = os.path.join(base, *parts)
    os.makedirs(path, exist_ok=True)
    return path

def root_data_dir(base_dir: str) -> str:
    return _dir(base_dir, "data" )

def data_dir(base_dir: str ) -> str:
    return _dir(root_data_dir(base_dir) , "climbmix")

def legacy_data_dir(base_dir: str ) -> str:
    """Legacy FinewebEdu-100B fallback path (no auto-create)."""
    return os.path.join(root_data_dir(base_dir) , "fineweb")

def eval_tasks_dir(base_dir: str ) -> str:
    return _dir(root_data_dir(base_dir) , "eval_tasks")


def tokenizer_dir(base_dir: str ) -> str:
    return _dir(base_dir , "tokenizer")

def checkpoint_dir(phase: str, base_dir: str , model_tag: str | None=None) -> str:
    assert phase in ("base", "sft", "rl"), f"Unknown phase: {phase}"
    if model_tag is not None:
        return _dir(base_dir , "checkpoints", phase, model_tag)
    return _dir(base_dir , "checkpoints", phase)

def eval_results_dir(base_dir: str ) -> str:
    return _dir(base_dir , "eval")

def identity_data_path(base_dir: str ) -> str:
    return os.path.join(base_dir , "identity.jsonl")

def report_dir(base_dir: str ) -> str:
    return _dir(base_dir, "report")
