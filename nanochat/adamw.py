"""
This file implements a distributed version of the AdamW optimizer, tailored for nanochat.
AdamW is a variant of the Adam optimizer that decouples weight decay from the gradient update.
This distributed implementation is inspired by ZeRO-2, sharding optimizer states and gradients
across multiple devices to reduce memory consumption, enabling the training of larger models.

A standard, non-distributed AdamW optimizer in PyTorch would be used as follows:
import torch
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
optimizer.step()
This distributed version achieves a similar outcome but coordinates across multiple processes.
"""
import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer in the style of ZeRO-2.

    This optimizer shards optimizer states (e.g., moments) and gradients across multiple
    devices. During the `step`, it performs a `reduce_scatter` to average gradients,
    updates its portion of the parameters, and then uses `all_gather` to ensure all
    devices have the updated parameters.

    Args:
        param_groups: An iterable of parameter groups to optimize.
        lr (float): The learning rate.
        betas (tuple[float, float]): Coefficients for running averages of gradient and its square.
        eps (float): Term for numerical stability.
        weight_decay (float): Weight decay coefficient.
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []

        # 1. Asynchronously reduce-scatter gradients.
        # Each device receives a slice of the averaged gradient.
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                # 2. Wait for the gradient slice and update parameters.
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]

                # State initialization
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # Apply weight decay
                if wd != 0:
                    p_slice.mul_(1 - lr * wd * getattr(p, "wd_mul", 1.0))

                # AdamW updates
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)

                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t

                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)

                p_slice.addcdiv_(exp_avg, denom, value=-step_size)

                idx += 1
                # 3. Asynchronously gather updated parameter slices.
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())

        # 4. Wait for all gather operations to complete.
        torch.futures.collect_all(all_reduce_futures).wait()
