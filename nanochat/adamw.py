"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
"""
import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_futures: list[torch.Future] = []
        gather_futures: list[torch.Future] = []
        grad_slices = []
        is_small = []  # track which params are small (use all_reduce) vs large (use reduce_scatter)

        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                grad = p.grad
                # Small params: use all_reduce (no scatter/gather needed)
                if p.numel() < 1024:
                    is_small.append(True)
                    reduce_futures.append(dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                    grad_slices.append(grad)
                else:
                    is_small.append(False)
                    assert p.shape[0] % world_size == 0, f"First dim of parameter shape {p.shape} must be divisible by world size {world_size}"
                    rank_size = grad.shape[0] // world_size
                    grad_slice = torch.empty_like(grad[:rank_size])
                    reduce_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                    grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for p in params:
                reduce_futures[idx].wait()
                g_slice = grad_slices[idx]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]

                # For small params, operate on full param; for large, operate on slice
                if is_small[idx]:
                    p_slice = p
                else:
                    rank_size = p.shape[0] // world_size
                    p_slice = p[rank * rank_size:(rank + 1) * rank_size]

                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = (exp_avg_sq / bias2).sqrt().add_(eps)
                step_size = lr / bias1
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)

                # Only large params need all_gather
                if not is_small[idx]:
                    gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
                idx += 1

        if gather_futures:
            torch.futures.collect_all(gather_futures).wait()
