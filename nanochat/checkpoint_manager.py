"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import json
import logging
import os
import re

import torch

from nanochat.common import get_base_dir, setup_default_logging
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', '0')) == 0:
        logger.info(message)

_SOURCE_DIRS = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}
_SOURCE_HINTS = {
    "base": "run scripts/base_train.py first",
    "sft": "run scripts/chat_sft.py first (after base_train)",
    "rl": "run scripts/chat_rl.py first (after chat_sft)",
}

def _require_dir(path, source=None):
    """Raise a FileNotFoundError that says what is missing and how to get it."""
    if os.path.isdir(path):
        return
    msg = f"Checkpoint directory not found: {path}"
    if source is not None:
        msg += f"\n  Expected source: '{source}' ({_SOURCE_DIRS[source]}/). Hint: {_SOURCE_HINTS[source]}."
    msg += f"\n  Base dir: {get_base_dir()} (override with NANOCHAT_BASE_DIR). See docs/RUNNING.md."
    raise FileNotFoundError(msg)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0("Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0("Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0("Patching missing x0_lambdas in model data to 0.0")

def _atomic_write(path, write_fn):
    # A killed write leaves only *.tmp, which find_safe_resume_step never matches.
    tmp_path = path + ".tmp"
    write_fn(tmp_path)
    os.replace(tmp_path, path)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        _atomic_write(model_path, lambda p: torch.save(model_data, p))
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        def _write_meta(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)
        _atomic_write(meta_path, _write_meta)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        _atomic_write(optimizer_path, lambda p: torch.save(optimizer_data, p))
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    _require_dir(checkpoints_dir)
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    _require_dir(checkpoint_dir)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.search(r'model_(\d+)\.pt$', f)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = max(int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files)
    return last_step


def remove_stale_tmp(checkpoint_dir: str) -> int:
    """Delete *.tmp files left by a save that was killed mid-write. Returns the count removed."""
    if not os.path.isdir(checkpoint_dir):
        return 0
    stale = [f for f in os.listdir(checkpoint_dir) if f.endswith(".tmp")]
    for f in stale:
        os.remove(os.path.join(checkpoint_dir, f))
    return len(stale)


def should_save_checkpoint(step: int, resume_from_step: int, elapsed_secs: float,
                           save_every_secs: float, save_every: int, last_step: bool) -> bool:
    """Pure predicate: true when this step should write a checkpoint.

    Callers own the DDP broadcast so the decision is consistent across ranks.
    """
    if last_step:
        return True
    if step <= 0 or step == resume_from_step:
        return False
    time_due = save_every_secs > 0 and elapsed_secs >= save_every_secs
    step_due = save_every > 0 and step % save_every == 0
    return time_due or step_due


def assert_resume_config_matches(saved_meta: dict, current: dict, keys) -> None:
    """Raise ValueError if a training dimension changed since the checkpoint was saved.

    Keys missing from either side are not compared (older checkpoints lack some keys).
    """
    for key in keys:
        saved_val = saved_meta.get(key)
        current_val = current.get(key)
        if saved_val is not None and current_val is not None and saved_val != current_val:
            raise ValueError(
                f"Cannot resume SFT: '{key}' changed from {saved_val} (saved) to {current_val} (current). "
                "Use FORCE_STAGES=chat_sft to restart from scratch."
            )


def find_safe_resume_step(checkpoint_dir: str, world_size: int = 1) -> int:
    """Return the highest step whose model, meta, and all optim shards are present.

    Returns -1 if no fully-committed step exists (fresh start, empty dir, or missing artifacts).
    Validates all three artifact types so a crash mid-save never produces a permanent crash-loop.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not os.path.isdir(checkpoint_dir):
        return -1
    files = os.listdir(checkpoint_dir)
    model_steps = sorted(
        [int(m.group(1)) for f in files if (m := re.search(r"model_(\d+)\.pt$", f))],
        reverse=True,
    )
    for step in model_steps:
        meta = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        if not os.path.exists(meta):
            continue
        optim_ok = all(
            os.path.exists(os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{r}.pt"))
            for r in range(world_size)
        )
        if optim_ok:
            return step
    return -1

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None, source=None):
    _require_dir(checkpoints_dir, source)
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    _require_dir(checkpoint_dir, source)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = _SOURCE_DIRS[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, source=source, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = _SOURCE_DIRS[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    _require_dir(checkpoints_dir, source)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    _require_dir(checkpoint_dir, source)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) >= 3 and _sys.argv[1] == "find-resume-step":
        _ws = int(_sys.argv[3]) if len(_sys.argv) > 3 else 1
        print(find_safe_resume_step(_sys.argv[2], _ws))
        raise SystemExit(0)
