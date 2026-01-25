"""
Distributed AdamW optimizer with a fused step function.
A bunch of ideas (e.g. dist comms in slices) are borrowed from modded-nanogpt.
"""
import torch
import torch.distributed as dist
from torch import Tensor

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    beta1_t: Tensor,
    beta2_t: Tensor,
    eps_t: Tensor,
    wd_t: Tensor,
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Validate
        if rank == 0:
            for group in param_groups:
                assert isinstance(group, dict), "expecting param_groups to be a list of dicts"
                assert isinstance(group['params'], list), "expecting group['params'] to be a list of tensors"
                for p in group['params']:
                    sliced = p.numel() >= 1024
                    print(f"AdamW: 1 param of shape {p.shape}, sliced={sliced}")
                    if sliced: # large parameter tensors will be operated on in slices
                        assert p.shape[0] % world_size == 0, f"First dim of parameter shape {p.shape} must be divisible by world size {world_size}"
        super().__init__(param_groups, defaults)
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

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
                    rank_size = grad.shape[0] // world_size # p.shape[0] % world_size == 0 is checked in __init__
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
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                # Fill 0-D tensors with current values
                eff_wd = wd * getattr(p, "wd_mul", 1.0)
                self._step_t.fill_(state['step'])
                self._lr_t.fill_(lr)
                self._beta1_t.fill_(beta1)
                self._beta2_t.fill_(beta2)
                self._eps_t.fill_(eps)
                self._wd_t.fill_(eff_wd)

                # Fused update: weight_decay -> momentum -> bias_correction -> param_update
                adamw_step_fused(
                    p_slice, g_slice, exp_avg, exp_avg_sq,
                    self._step_t, self._lr_t, self._beta1_t, self._beta2_t, self._eps_t, self._wd_t,
                )

                # Only large params need all_gather
                if not is_small[idx]:
                    gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
                idx += 1

        if gather_futures:
            torch.futures.collect_all(gather_futures).wait()
