"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import json
import logging
import torch

from nanochat.common import get_experiment_dir, get_experiment_name
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from harness.runtime import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def get_checkpoint_dir(model_tag, source):
    """
    Checkpoints belong to the experiment: experiments/<name>/<model_tag>/<source>/
    e.g. experiments/my_exp/d12/base/. The model_tag defaults to d<depth> in the
    training scripts but can be overridden (the --model-tag escape hatch).
    """
    assert source in ["base", "chat"], f"Invalid source: {source}"
    experiment_dir = get_experiment_dir()
    checkpoint_dir = os.path.join(experiment_dir, model_tag, source)
    return checkpoint_dir

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Re-materialize each tensor as a standalone contiguous copy. The live params
        # are views into the optimizer's flat tapes (see MuonAdamW in optim.py), and
        # torch.save deduplicates shared storage - saving the views directly would
        # serialize the whole world_size-dependent tape layout into the file. The
        # tapes are a runtime optimization; checkpoints must stay independent of them.
        # Copy via CPU: torch.save stages GPU tensors through CPU anyway, and cloning
        # on-device would spike VRAM by a full model - a late OOM risk for a job that
        # has been stepping fine and only saves much later. copy=True forces a copy
        # even when the tensor is already on CPU (it might still be a tape view).
        model_data = {k: v.detach().to("cpu", copy=True) for k, v in model_data.items()}
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
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


def build_model(checkpoint_dir, step, device):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    # Adopt the loaded params directly (validates keys/shapes against the config's schema)
    model = GPT(model_config, device, params=model_data)
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(source):
    # attempt to guess the model tag: take the biggest model available in the experiment
    experiment_dir = get_experiment_dir()
    model_tags = [f for f in os.listdir(experiment_dir)
                  if os.path.isdir(os.path.join(experiment_dir, f, source))] if os.path.isdir(experiment_dir) else []
    if not model_tags:
        raise FileNotFoundError(f"No '{source}' models found in experiment '{get_experiment_name()}' ({experiment_dir}). "
                                f"Select an experiment with NANOCHAT_EXPERIMENT=<name>, or train a model first.")
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
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(experiment_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.search(r'model_(\d+)\.pt$', f)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = max(int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files)
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model(source, device, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model in the experiment
        model_tag = find_largest_model(source)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = get_checkpoint_dir(model_tag, source)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device)
    return model, tokenizer, meta_data
