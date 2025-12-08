"""
Nested Learning Optimizer (Deep Optimizer / Nested Momentum)
Based on 'Nested Learning: The Illusion of Deep Learning Architecture' by Behrouz et al. (2025).

The Nested Learning paradigm views the optimizer as an associative memory system with multiple
levels of 'context flow', each operating at a different time-scale. This implementation
realizes this by maintaining multiple momentum buffers with distinct decay rates (betas),
forming a 'Continuum Memory System' for the gradients.

This allows the optimizer to capture both transient (fast) and persistent (slow) gradient
dynamics, mitigating catastrophic forgetting during training.

Key Concepts:
- **Continuum Memory System:** A set of momentum buffers (memory states) with different decay rates (`betas`).
- **Time-Scales:** Defined by the `betas`. Low beta (e.g., 0.9) = Fast/Short-term memory. High beta (e.g., 0.9999) = Slow/Long-term memory.
- **Nested Update:** The final parameter update is a weighted sum (`level_weights`) of updates from all memory levels.

Reference:
Behrouz, A., et al. "Nested Learning: The Illusion of Deep Learning Architecture." (2025).
"""

import torch
import torch.distributed as dist
from torch import Tensor

class NestedMomentum(torch.optim.Optimizer):
    """
    NestedMomentum Optimizer.

    Maintains multiple levels of momentum buffers, each with a different decay rate (beta),
    corresponding to different 'time-scales' of memory (e.g., Fast, Medium, Slow).
    The final update is a weighted sum of these memory states.

    The update rule for each level `i` with decay `beta_i`:
        m_{t,i} = beta_i * m_{t-1,i} + (1 - beta_i) * g_t

    The final update `u_t` applied to parameters:
        u_t = sum(w_i * m_{t,i}) for i in levels

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (tuple[float]): A tuple of decay rates for each memory level.
                              Example: (0.9, 0.99, 0.999) defines 3 levels of memory.
                              Higher beta = slower decay = longer-term memory.
        level_weights (tuple[float] | None): Weights for combining the memory levels.
                                             If None, uniform weighting (1.0) is used.
                                             Example: (0.5, 0.3, 0.2) prioritizes fast memory.
        weight_decay (float): Weight decay coefficient (L2 penalty).
        nesterov (bool): Whether to use Nesterov-style momentum update for the levels.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), level_weights=None, weight_decay=0.0, nesterov=False):
        if level_weights is None:
            level_weights = tuple(1.0 for _ in betas)

        assert len(betas) == len(level_weights), "Number of betas must match number of level_weights"

        defaults = dict(
            lr=lr,
            betas=betas,
            level_weights=level_weights,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            betas = group['betas']
            weights = group['level_weights']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                state = self.state[p]

                # Initialize state for each memory level
                if len(state) == 0:
                    state['step'] = 0
                    # Create a momentum buffer for each beta (time-scale)
                    state['memory_levels'] = [torch.zeros_like(p, memory_format=torch.preserve_format) for _ in betas]

                state['step'] += 1
                memory_levels = state['memory_levels']

                # Calculate the aggregated update from all memory levels
                final_update = torch.zeros_like(p)

                for i, beta in enumerate(betas):
                    buf = memory_levels[i]

                    # Update memory state: m_t = beta * m_{t-1} + (1 - beta) * g_t
                    buf.mul_(beta).add_(d_p, alpha=1 - beta)

                    # Compute the contribution of this level
                    if nesterov:
                        # Nesterov: update = beta * m_update + (1-beta) * g_t
                        level_update = buf.mul(beta).add(d_p, alpha=1 - beta)
                    else:
                        level_update = buf

                    # Add weighted contribution to final update
                    final_update.add_(level_update, alpha=weights[i])

                # Apply the update
                p.add_(final_update, alpha=-lr)

        return loss

class DistNestedMomentum(torch.optim.Optimizer):
    """
    Distributed version of NestedMomentum.

    Implements ZeRO-2 style sharding (Optimizer State Partitioning):
    1. Gradients are reduced-scattered across ranks (each rank gets a slice of the average gradient).
    2. Optimizer step (momentum update) is performed on the local shard.
    3. Updated parameters are all-gathered back to all ranks.

    This ensures that optimizer states (the momentum buffers) are sharded, significantly reducing
    memory usage per GPU in distributed training.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate.
        betas (tuple[float]): Decay rates for memory levels.
        level_weights (tuple[float] | None): Weights for memory levels.
        weight_decay (float): Weight decay coefficient.
        nesterov (bool): Enable Nesterov momentum.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), level_weights=None, weight_decay=0.0, nesterov=False):
        if level_weights is None:
            level_weights = tuple(1.0 for _ in betas)

        assert len(betas) == len(level_weights), "Number of betas must match number of level_weights"

        defaults = dict(
            lr=lr,
            betas=betas,
            level_weights=level_weights,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        super().__init__(params, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self, closure=None):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []
        grad_slices = []

        # 1. Kick off reduce_scatter for all gradients
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                if grad is None:
                    # If any param has no grad, we need to handle it or error.
                    # For now assume all params have grads as in AdamW implementation.
                    grad = torch.zeros_like(params[base_i])

                # Assume params are divisible by world_size on 0th dim (standard for this repo's models)
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])

                reduce_scatter_futures.append(
                    dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                )
                grad_slices.append(grad_slice)

        # 2. Update params on local shard
        idx = 0
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            betas = group['betas']
            weights = group['level_weights']
            nesterov = group['nesterov']

            params = group['params']
            for base in range(len(params)):
                # Wait for gradient reduction
                reduce_scatter_futures[idx].wait()

                p = params[base]
                rank_size = p.shape[0] // world_size

                # Get local shard of parameter
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                g_slice = grad_slices[idx]

                # Adjust LR if per-parameter scaling is present (e.g. from Muon/AdamW hacks)
                # Although NestedMomentum usually doesn't use lr_mul/wd_mul, we support it for compatibility
                eff_lr = lr * getattr(p, "lr_mul", 1.0)

                # Weight decay
                if weight_decay != 0:
                    eff_weight_decay = eff_lr * weight_decay * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)

                # Optimizer State (on local shard)
                state = self.state[p]
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    # Create memory buffers for this shard
                    state['memory_levels'] = [torch.zeros_like(p_slice) for _ in betas]

                state['step'] += 1
                memory_levels = state['memory_levels']

                # Calculate aggregated update
                final_update = torch.zeros_like(p_slice)

                for i, beta in enumerate(betas):
                    buf = memory_levels[i]

                    # buf = beta * buf + (1 - beta) * grad
                    buf.mul_(beta).add_(g_slice, alpha=1 - beta)

                    if nesterov:
                         # level_update = beta * (beta * buf + (1-beta)*grad) + (1-beta)*grad ... wait
                         # Nesterov in PyTorch usually:
                         # v = beta * v + g
                         # p = p - lr * (beta * v + g) (approx)
                         # OR
                         # v = beta * v + (1-beta) * g
                         # p = p - lr * (beta * v + (1-beta) * g)

                         # In NestedMomentum local step above:
                         # buf updated: buf' = beta*buf + (1-beta)*g
                         # level_update = buf'.mul(beta).add(d_p, alpha=1-beta)
                         level_update = buf.mul(beta).add(g_slice, alpha=1 - beta)
                    else:
                        level_update = buf

                    final_update.add_(level_update, alpha=weights[i])

                # Apply update
                p_slice.add_(final_update, alpha=-eff_lr)

                idx += 1

                # 3. Kick off All-Gather to synchronize updated parameters
                all_gather_futures.append(
                    dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                )

        # 4. Wait for all gathers to complete
        torch.futures.collect_all(all_gather_futures).wait()

        return None
