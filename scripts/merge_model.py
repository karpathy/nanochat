"""
Model checkpoint merging utility.

Merges multiple model checkpoints using various averaging strategies:
- simple: Equal weight average of all checkpoints
- ema: Exponential moving average (more weight on recent checkpoints)
- wma: Weighted moving average with exponential decay

Usage:
    python -m scripts.merge_model --model_dir ./checkpoints/d20 --num_models 5 --step_size 1 --merge_method simple
    python -m scripts.merge_model --model_dir ./checkpoints/d20 --num_models 8 --step_size 2 --merge_method ema --alpha 0.7
"""

import os
import glob
import argparse
from datetime import datetime

import torch


# =============================================================================
# Exceptions
# =============================================================================

class InsufficientCheckpointsError(Exception):
    """Raised when not enough checkpoints available for selection"""
    pass


class IncompatibleModelsError(Exception):
    """Raised when model architectures don't match"""
    pass


class InvalidParameterError(Exception):
    """Raised when parameter validation fails"""
    pass


# =============================================================================
# Checkpoint Discovery and Selection
# =============================================================================

def get_sorted_checkpoints(model_dir: str) -> list[tuple[str, int]]:
    """
    Discover and sort checkpoints by step number (descending, newest first).

    Args:
        model_dir: Path to directory containing model checkpoints

    Returns:
        List of (checkpoint_path, step_number) tuples sorted by step (descending)
    """
    # Find all model checkpoint files
    pattern = os.path.join(model_dir, "model_*.pt")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")

    # Extract step number from filename: model_000000.pt -> 0
    checkpoints = []
    for path in checkpoint_files:
        filename = os.path.basename(path)
        # Extract step from model_XXXXXX.pt
        step_str = filename.replace("model_", "").replace(".pt", "")
        try:
            step = int(step_str)
            checkpoints.append((path, step))
        except ValueError:
            print(f"Warning: Skipping file with invalid step number: {filename}")
            continue

    # Sort by step number descending (newest first)
    checkpoints.sort(key=lambda x: x[1], reverse=True)

    return checkpoints


def select_checkpoints(sorted_checkpoints: list[tuple[str, int]],
                       num_models: int,
                       step_size: int) -> list[str]:
    """
    Select checkpoints starting from newest, stepping backwards.

    Args:
        sorted_checkpoints: List of (path, step) sorted by step descending
        num_models: Number of checkpoints to select
        step_size: Step interval between selections

    Returns:
        List of checkpoint paths (newest to oldest)
    """
    if len(sorted_checkpoints) < num_models:
        raise InsufficientCheckpointsError(
            f"Need at least {num_models} checkpoints, but only found {len(sorted_checkpoints)}"
        )

    # Determine the expected step distance from the first two checkpoints
    step_distance = sorted_checkpoints[0][1] - sorted_checkpoints[1][1]
    if step_distance <= 0:
        raise InsufficientCheckpointsError(
            "Checkpoints are not in descending step order"
        )

    if step_size % step_distance != 0:
        raise InsufficientCheckpointsError(
            f"step_size={step_size} is not a multiple of the checkpoint "
            f"interval={step_distance}, so exact matches are impossible"
        )

    expected_distance = step_size

    newest_step = sorted_checkpoints[0][1]
    oldest_needed = newest_step - expected_distance * (num_models - 1)
    oldest_available = sorted_checkpoints[-1][1]
    if oldest_needed < oldest_available:
        raise InsufficientCheckpointsError(
            f"Need checkpoint at step {oldest_needed} but oldest available is "
            f"step {oldest_available} (selecting {num_models} models with "
            f"step_size={step_size}, checkpoint interval={step_distance})"
        )

    selected = [sorted_checkpoints[0][0]]
    prev_step = newest_step

    for i in range(1, num_models):
        target_step = prev_step - expected_distance
        # Find the checkpoint matching this target step
        match = None
        for path, step in sorted_checkpoints:
            if step == target_step:
                match = path
                break
        if match is None:
            raise InsufficientCheckpointsError(
                f"No checkpoint found at step {target_step} (expected step distance "
                f"of {expected_distance} between selected models)"
            )
        selected.append(match)
        prev_step = target_step

    return selected


# =============================================================================
# State Dict Operations
# =============================================================================

def load_state_dict(path: str, device: str) -> dict:
    """Load a state dict and strip _orig_mod. prefix from torch.compile."""
    state_dict = torch.load(path, map_location=device, weights_only=True)
    # Handle torch.compile prefix
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def add_state_dicts(sd1: dict, sd2: dict) -> dict:
    """Element-wise addition of two state dicts."""
    result = {}
    for key in sd1.keys():
        result[key] = sd1[key] + sd2[key]
    return result


def scale_state_dict(sd: dict, factor: float) -> dict:
    """Element-wise scaling of a state dict."""
    result = {}
    for key in sd.keys():
        result[key] = sd[key] * factor
    return result


def blend_state_dicts(sd1: dict, sd2: dict, alpha1: float, alpha2: float) -> dict:
    """Weighted blend of two state dicts: alpha1 * sd1 + alpha2 * sd2."""
    result = {}
    for key in sd1.keys():
        result[key] = alpha1 * sd1[key] + alpha2 * sd2[key]
    return result


def validate_state_dicts(state_dicts_info: list[tuple[str, dict]]) -> None:
    """
    Validate that all state dicts have matching keys and tensor shapes.

    Args:
        state_dicts_info: List of (path, state_dict) tuples
    """
    if len(state_dicts_info) < 2:
        return

    reference_path, reference_sd = state_dicts_info[0]
    reference_keys = set(reference_sd.keys())

    for path, sd in state_dicts_info[1:]:
        current_keys = set(sd.keys())

        # Check keys match
        if current_keys != reference_keys:
            missing = reference_keys - current_keys
            extra = current_keys - reference_keys
            raise IncompatibleModelsError(
                f"State dict keys mismatch between {reference_path} and {path}. "
                f"Missing: {missing}, Extra: {extra}"
            )

        # Check shapes match
        for key in reference_keys:
            if reference_sd[key].shape != sd[key].shape:
                raise IncompatibleModelsError(
                    f"Shape mismatch for key '{key}': "
                    f"{reference_path} has {reference_sd[key].shape}, "
                    f"{path} has {sd[key].shape}"
                )


# =============================================================================
# Merge Methods
# =============================================================================

def merge_simple_average(checkpoint_paths: list[str], device: str) -> dict:
    """
    Simple average: m_avg = (1/n) * sum(m_i)

    All checkpoints receive equal weight.
    """
    n = len(checkpoint_paths)
    accumulated = None

    for i, path in enumerate(checkpoint_paths):
        print(f"  Processing checkpoint {i+1}/{n}...")
        sd = load_state_dict(path, device)

        if accumulated is None:
            accumulated = sd
        else:
            accumulated = add_state_dicts(accumulated, sd)
        del sd

    # Divide by n
    return scale_state_dict(accumulated, 1.0 / n)


def merge_ema(checkpoint_paths: list[str], alpha: float, device: str) -> dict:
    """
    Exponential Moving Average:
        m_avg_1 = m_oldest
        m_avg_i = alpha * m_i + (1 - alpha) * m_avg_{i-1}

    Process from oldest to newest, giving more weight to recent checkpoints.
    """
    # Reverse to process oldest to newest
    reversed_paths = checkpoint_paths[::-1]
    n = len(reversed_paths)

    print(f"  Processing checkpoint 1/{n} (oldest, initialization)...")
    m_avg = load_state_dict(reversed_paths[0], device)

    for i, path in enumerate(reversed_paths[1:], start=2):
        print(f"  Processing checkpoint {i}/{n}...")
        m_i = load_state_dict(path, device)
        # m_avg = alpha * m_i + (1 - alpha) * m_avg
        m_avg = blend_state_dicts(m_i, m_avg, alpha, 1 - alpha)
        del m_i

    return m_avg


def merge_wma(checkpoint_paths: list[str], alpha: float, device: str) -> dict:
    """
    Weighted Moving Average:
        w_i = alpha * (1 - alpha)^(n - i) for i=1 to n
        m_avg = sum(w_i * m_i) / sum(w_i)

    Exponentially decaying weights favoring recent checkpoints.
    """
    # Process oldest to newest (index 1 = oldest, index n = newest)
    reversed_paths = checkpoint_paths[::-1]
    n = len(reversed_paths)

    # Compute weights
    weights = []
    for i in range(1, n + 1):
        w = alpha * ((1 - alpha) ** (n - i))
        weights.append(w)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    print(f"  Weights (oldest to newest): {[f'{w:.4f}' for w in weights]}")

    accumulated = None

    for i, (path, weight) in enumerate(zip(reversed_paths, weights)):
        print(f"  Processing checkpoint {i+1}/{n} (weight={weight:.4f})...")
        sd = load_state_dict(path, device)
        weighted_sd = scale_state_dict(sd, weight)

        if accumulated is None:
            accumulated = weighted_sd
        else:
            accumulated = add_state_dicts(accumulated, weighted_sd)
        del sd, weighted_sd

    return accumulated


# =============================================================================
# Main
# =============================================================================

def validate_args(args):
    """Validate command-line arguments."""
    if args.num_models < 2:
        raise InvalidParameterError("num_models must be >= 2")

    if args.step_size < 1:
        raise InvalidParameterError("step_size must be >= 1")

    if args.merge_method not in ['simple', 'ema', 'wma']:
        raise InvalidParameterError(f"merge_method must be 'simple', 'ema', or 'wma', got '{args.merge_method}'")

    if args.merge_method in ['ema', 'wma']:
        if not (0 < args.alpha <= 1):
            raise InvalidParameterError(f"alpha must be in (0, 1], got {args.alpha}")

    if not os.path.isdir(args.model_dir):
        raise InvalidParameterError(f"model_dir does not exist: {args.model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge model checkpoints using various averaging strategies")

    # Required arguments
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to directory containing model checkpoints")
    parser.add_argument("--num_models", type=int, required=True,
                        help="Number of most recent models to merge (must be >= 2)")
    parser.add_argument("--step_size", type=int, required=True,
                        help="Step interval between selected checkpoints")
    parser.add_argument("--merge_method", type=str, required=True, choices=['simple', 'ema', 'wma'],
                        help="Averaging method: simple, ema, or wma")

    # Optional arguments
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Smoothing parameter for EMA and WMA (0 < alpha <= 1, default: 0.5)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save merged model (default: {model_dir}/merged_model.pt)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for merging: cpu, cuda, cuda:0, etc. (default: cpu)")

    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Set default output path
    if args.output_path is None:
        args.output_path = os.path.join(args.model_dir, "merged_model.pt")

    # Print configuration
    print("=" * 60)
    print("Model Merging Configuration:")
    print("=" * 60)
    print(f"  Directory: {args.model_dir}")
    print(f"  Method: {args.merge_method}" + (f" (alpha={args.alpha})" if args.merge_method != 'simple' else ""))
    print(f"  Models to merge: {args.num_models}")
    print(f"  Step size: {args.step_size}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_path}")
    print()

    # Discover checkpoints
    print("Discovering checkpoints...")
    sorted_checkpoints = get_sorted_checkpoints(args.model_dir)
    print(f"Found {len(sorted_checkpoints)} checkpoints in {args.model_dir}")
    print()

    # Select checkpoints
    selected_paths = select_checkpoints(sorted_checkpoints, args.num_models, args.step_size)

    print("Selected checkpoints (newest to oldest):")
    for i, path in enumerate(selected_paths):
        step = int(os.path.basename(path).replace("model_", "").replace(".pt", ""))
        print(f"  [{i+1}/{args.num_models}] {os.path.basename(path)} (step {step})")
    print()

    # Validate compatibility (load first two to check)
    print("Validating checkpoint compatibility...")
    sd1 = load_state_dict(selected_paths[0], args.device)
    sd2 = load_state_dict(selected_paths[1], args.device)
    validate_state_dicts([(selected_paths[0], sd1), (selected_paths[1], sd2)])
    del sd1, sd2
    print("Checkpoints are compatible.")
    print()

    # Merge
    print("Merging models...")
    if args.merge_method == 'simple':
        merged_sd = merge_simple_average(selected_paths, args.device)
    elif args.merge_method == 'ema':
        merged_sd = merge_ema(selected_paths, args.alpha, args.device)
    elif args.merge_method == 'wma':
        merged_sd = merge_wma(selected_paths, args.alpha, args.device)
    print()

    # Save with metadata
    print(f"Saving merged model to: {args.output_path}")

    merge_metadata = {
        'method': args.merge_method,
        'alpha': args.alpha if args.merge_method != 'simple' else None,
        'num_models': args.num_models,
        'step_size': args.step_size,
        'checkpoints_used': [os.path.basename(p) for p in selected_paths],
        'merge_timestamp': datetime.now().isoformat(),
    }

    # Save just the state dict (compatible with existing load_checkpoint)
    torch.save(merged_sd, args.output_path)

    # Count parameters
    total_params = sum(p.numel() for p in merged_sd.values())

    print()
    print("=" * 60)
    print("Merge Complete!")
    print("=" * 60)
    print(f"  Output: {args.output_path}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Merge metadata: {merge_metadata}")


if __name__ == "__main__":
    main()
