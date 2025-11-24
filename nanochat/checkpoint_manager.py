"""
This module provides utilities for saving and loading model, optimizer, and training state
checkpoints. It is essential for resuming training and for deploying models for inference.

A typical use case involves:
1.  Calling `save_checkpoint` periodically during training.
2.  Calling `load_checkpoint` to resume training or for inference.
3.  Using `build_model` to reconstruct a model from a checkpoint.

Python equivalent for basic checkpointing:
import torch
# Saving
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "checkpoint.pt")
# Loading
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    """Logs a message only on the main process (rank 0)."""
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data):
    """
    Saves a checkpoint to the specified directory.

    Args:
        checkpoint_dir (str): The directory to save the checkpoint to.
        step (int): The current training step.
        model_data (dict): The model's state_dict.
        optimizer_data (dict): The optimizer's state_dict.
        meta_data (dict): A dictionary of metadata to save.
    """
    assert int(os.environ.get('RANK', 0)) == 0 # prevent footguns for now
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save the model state (parameters)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    torch.save(model_data, model_path)
    log0(f"Saved model file to: {model_path}")
    # Save the optimizer state (useful for SFT or any other fine-tuning)
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        torch.save(optimizer_data, optimizer_path)
        log0(f"Saved optimizer file to: {optimizer_path}")
    # Save the metadata dict as json
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    log0(f"Saved metadata file to: {meta_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False):
    """
    Loads a checkpoint from the specified directory.

    Args:
        checkpoint_dir (str): The directory to load the checkpoint from.
        step (int): The training step of the checkpoint to load.
        device (str): The device to load the tensors onto.
        load_optimizer (bool, optional): Whether to load the optimizer state. Defaults to False.

    Returns:
        tuple: A tuple containing the model data, optimizer data, and metadata.
    """
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    Builds a model from a given checkpoint.

    Args:
        checkpoint_dir (str): The directory of the checkpoint.
        step (int): The training step of the checkpoint.
        device (str): The device to build the model on.
        phase (str): The phase, either "train" or "eval".

    Returns:
        tuple: A tuple containing the model, tokenizer, and metadata.
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
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
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    """
    Finds the largest model in a checkpoint directory, assuming a "d<number>" naming convention.

    Args:
        checkpoint_dir (str): The directory to search for models.

    Returns:
        str: The tag of the largest model found.
    """
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
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
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    """
    Finds the last training step in a checkpoint directory.

    Args:
        checkpoint_dir (str): The directory to search for checkpoints.

    Returns:
        int: The last training step found.
    """
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    """
    Loads a model from a directory, automatically detecting the model tag and step if not provided.

    Args:
        checkpoints_dir (str): The directory containing model checkpoints.
        device (str): The device to load the model on.
        phase (str): The phase, either "train" or "eval".
        model_tag (str, optional): The model tag to load. Defaults to None.
        step (int, optional): The step to load. Defaults to None.

    Returns:
        tuple: A tuple containing the model, tokenizer, and metadata.
    """
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    """
    Loads a model from a specific source directory within the nanochat project.

    Args:
        source (str): The source of the model, one of "base", "mid", "sft", or "rl".
        *args: Positional arguments to pass to `load_model_from_dir`.
        **kwargs: Keyword arguments to pass to `load_model_from_dir`.

    Returns:
        tuple: A tuple containing the model, tokenizer, and metadata.
    """
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
