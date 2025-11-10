"""
SIM-CoT utilities for step-level supervision during training.

This module provides helper functions to compute step-level weights
and losses for Supervised Implicit Chain-of-Thought training.
"""

import torch


def compute_step_weights(targets, step_boundaries, step_weight_multiplier=2.0):
    """
    Compute per-token weights for step-level supervision.

    Args:
        targets: Tensor of shape (B, T) with target token IDs (-1 for masked)
        step_boundaries: List of lists, where step_boundaries[i] contains
                        the token positions where reasoning steps start in example i
        step_weight_multiplier: How much to upweight the tokens at step boundaries

    Returns:
        Tensor of shape (B, T) with weights for each token (1.0 for normal, higher for step tokens)
    """
    B, T = targets.shape
    weights = torch.ones_like(targets, dtype=torch.float32)

    # For each example in the batch
    for batch_idx in range(B):
        if batch_idx >= len(step_boundaries) or step_boundaries[batch_idx] is None:
            continue

        # Mark tokens near step boundaries with higher weight
        for step_pos in step_boundaries[batch_idx]:
            # Upweight a window around the step boundary
            # This gives more importance to the reasoning steps
            start = max(0, step_pos - 2)
            end = min(T, step_pos + 5)  # Slightly longer window after the step
            weights[batch_idx, start:end] = step_weight_multiplier

    # Zero out weights where targets are masked (-1)
    weights[targets == -1] = 0.0

    return weights


def compute_step_accuracy(logits, targets, step_boundaries):
    """
    Compute per-step accuracy to track learning progress.

    Args:
        logits: Tensor of shape (B, T, vocab_size)
        targets: Tensor of shape (B, T)
        step_boundaries: List of lists with step positions

    Returns:
        Dict with 'step_accuracy' and 'overall_accuracy'
    """
    B, T, V = logits.shape

    # Get predictions
    predictions = torch.argmax(logits, dim=-1)  # (B, T)

    # Overall accuracy (excluding masked positions)
    mask = targets != -1
    correct = (predictions == targets) & mask
    overall_acc = correct.sum().float() / mask.sum().float()

    # Step-level accuracy
    step_correct = 0
    step_total = 0

    for batch_idx in range(B):
        if batch_idx >= len(step_boundaries) or step_boundaries[batch_idx] is None:
            continue

        for step_pos in step_boundaries[batch_idx]:
            if step_pos < T and targets[batch_idx, step_pos] != -1:
                # Check if the tokens around the step boundary are correct
                window_start = max(0, step_pos)
                window_end = min(T, step_pos + 3)
                window_mask = targets[batch_idx, window_start:window_end] != -1
                window_correct = (
                    predictions[batch_idx, window_start:window_end]
                    == targets[batch_idx, window_start:window_end]
                ) & window_mask

                if window_mask.sum() > 0:
                    step_correct += window_correct.sum().item()
                    step_total += window_mask.sum().item()

    step_acc = step_correct / step_total if step_total > 0 else 0.0

    return {
        'overall_accuracy': overall_acc.item(),
        'step_accuracy': step_acc,
        'step_total': step_total,
    }


def compute_weighted_loss(model_output, targets, step_boundaries, step_weight_multiplier=2.0):
    """
    Compute weighted cross-entropy loss with step-level supervision.

    This is the core SIM-CoT loss function that upweights important reasoning steps.

    Args:
        model_output: Logits from model, shape (B, T, vocab_size)
        targets: Target token IDs, shape (B, T)
        step_boundaries: List of lists with step positions for each example
        step_weight_multiplier: How much to upweight step tokens

    Returns:
        Scalar loss tensor
    """
    import torch.nn.functional as F

    B, T, V = model_output.shape

    # Compute per-token cross-entropy (no reduction yet)
    logits_flat = model_output.view(-1, V)
    targets_flat = targets.view(-1)

    # Get per-token loss (shape: B*T)
    per_token_loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=-1,
        reduction='none'
    )
    per_token_loss = per_token_loss.view(B, T)

    # Compute step weights
    weights = compute_step_weights(targets, step_boundaries, step_weight_multiplier)

    # Apply weights and compute mean
    weighted_loss = (per_token_loss * weights).sum() / weights.sum()

    return weighted_loss


def extract_step_boundaries_from_mask(ids, mask, special_tokens):
    """
    Extract step boundaries from tokenized conversation by finding tool call markers.

    Args:
        ids: List of token IDs
        mask: List of mask values (0 or 1)
        special_tokens: Dict with special token IDs (python_start, python_end, etc.)

    Returns:
        List of positions where reasoning steps begin
    """
    python_start = special_tokens.get('python_start')
    python_end = special_tokens.get('python_end')

    if python_start is None:
        return []

    step_positions = []
    for i, token_id in enumerate(ids):
        if token_id == python_start:
            # Mark this position as a step boundary
            step_positions.append(i)

    return step_positions


def prepare_simcot_batch(batch_data, tokenizer):
    """
    Prepare a batch for SIM-CoT training.

    Takes raw conversations with step_boundaries metadata and converts them
    to tensors suitable for training.

    Args:
        batch_data: List of (ids, mask, step_boundaries) tuples
        tokenizer: Tokenizer instance

    Returns:
        Tuple of (input_ids, targets, step_boundaries_batch)
    """
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    nrows = len(batch_data)
    ncols = max(len(ids) for ids, _, _ in batch_data) - 1

    inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
    targets = torch.full((nrows, ncols), -1, dtype=torch.long)
    step_boundaries_batch = []

    for i, (ids, mask, step_bounds) in enumerate(batch_data):
        n = len(ids)
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        mask_tensor = torch.tensor(mask[1:], dtype=torch.long)

        inputs[i, :n-1] = ids_tensor[:-1]

        # Set targets
        row_targets = ids_tensor[1:]
        row_targets[mask_tensor == 0] = -1
        targets[i, :n-1] = row_targets

        # Adjust step boundaries for the shift (input[t] -> target[t+1])
        adjusted_bounds = [pos - 1 for pos in step_bounds if pos > 0]
        step_boundaries_batch.append(adjusted_bounds)

    return inputs, targets, step_boundaries_batch
