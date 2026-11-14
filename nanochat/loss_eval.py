"""
A number of functions that help with evaluating a base model.
"""
import math
import time

import torch
import torch.distributed as dist

from nanochat.common import print0


def _bpb_progress_every(steps: int) -> int:
    """How often to print during val bpb so slow devices don't look hung."""
    if steps <= 0:
        return 1
    # ~10 updates for long runs; every step when tiny.
    return max(1, min(steps, steps // 10 or 1))


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, progress_every=None):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    log_every = _bpb_progress_every(steps) if progress_every is None else max(1, progress_every)
    t0 = time.time()
    if steps > 0:
        print0(f"val bpb: starting {steps} eval steps (progress every {log_every})")
    for step_i in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        done = step_i + 1
        if done == 1 or done % log_every == 0 or done == steps:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (steps - done) / rate if rate > 0 else 0.0
            print0(
                f"val bpb: {done}/{steps} steps ({100.0 * done / steps:.0f}%) | "
                f"elapsed {elapsed:.1f}s | eta {eta:.1f}s"
            )
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
