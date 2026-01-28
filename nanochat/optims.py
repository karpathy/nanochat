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

"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    X = g.bfloat16()
    if g.size(-2) > g.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if g.size(-2) > g.size(-1):
        X = X.mT
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others.

    AdamW - Distributed AdamW optimizer with a fused step function.

    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    AdamW Arguments:
        adamw_groups: List of dicts with 'params' and optional 'lr' for AdamW params
        muon_params: List of 2D tensors to optimize with Muon
        adamw_betas: Beta coefficients for AdamW (default: (0.9, 0.999))
        adamw_eps: Epsilon for AdamW numerical stability (default: 1e-8)    
        adamw_weight_decay: Weight decay for AdamW (default: 0.01)
    
    Muon Arguments:
        muon_lr: The learning rate used by the internal SGD.
        muon_momentum: The momentum used by the internal SGD.
        muon_ns_steps: The number of Newton-Schulz iteration steps to use.
        muon_beta2: The decay rate for the second moment (variance) estimate. Set to None to disable.
        muon_weight_decay: Cautious weight decay coefficient. Only decays where update and weight agree.
    """
    def __init__(
        self,
        adamw_groups: list[dict],
        muon_params,
        # AdamW hyperparams
        adamw_lr: float = 1e-3, # can be overridden per-group
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.01,
        # Muon hyperparams
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_beta2: float = 0.95,
        muon_weight_decay: float = 0.0,
    ):
        muon_params = list(muon_params)
        assert all(p.ndim == 2 for p in muon_params), "Muon expects 2D parameters only"
        
        # Build unified param_groups for the base Optimizer class
        param_groups = []
        
        # AdamW groups: each input group becomes one param_group
        for group in adamw_groups:
            assert isinstance(group, dict) and 'params' in group
            params = list(group['params'])
            lr = group.get('lr', adamw_lr) # AdamW supports per-group learning rates
            for p in params:
                print(f"AdamW: 1 param of shape {p.shape}")
            param_groups.append(dict(
                params=params, lr=lr, kind='adamw',
                betas=adamw_betas, eps=adamw_eps, weight_decay=adamw_weight_decay,
            ))
        
        # Muon groups: group by shape for stacking, with all Muon hyperparams in the group
        muon_shapes = sorted({p.shape for p in muon_params})
        for shape in muon_shapes:
            group_params = [p for p in muon_params if p.shape == shape]
            print(f"Muon: {len(group_params)} params of shape {shape}")
            param_groups.append(dict(
                params=group_params, lr=muon_lr, kind='muon',
                momentum=muon_momentum, ns_steps=muon_ns_steps,
                beta2=muon_beta2, weight_decay=muon_weight_decay,
            ))
        
        defaults = dict(lr=adamw_lr) # torch.optim.Optimizer requires a default lr
        super().__init__(param_groups, defaults)
        
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            kind = group['kind']
            
            if kind == 'adamw':
                beta1, beta2 = group['betas']
                eps = group['eps']
                wd = group['weight_decay']
                # AdamW update for each param individually
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    lr = group['lr']
                    state = self.state[p]
                    
                    # State init
                    if not state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    state['step'] += 1

                    # Fill 0-D tensors with current values
                    eff_wd = wd * getattr(p, "wd_mul", 1.0)
                    self._adamw_step_t.fill_(state['step'])
                    self._adamw_lr_t.fill_(lr)
                    self._adamw_beta1_t.fill_(beta1)
                    self._adamw_beta2_t.fill_(beta2)
                    self._adamw_eps_t.fill_(eps)
                    self._adamw_wd_t.fill_(eff_wd)

                    # Fused update: weight_decay -> momentum -> bias_correction -> param_update
                    adamw_step_fused(
                        p, grad, exp_avg, exp_avg_sq,
                        self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                        self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
                    )
            
            else:  # muon
                params: list[Tensor] = group['params']
                if not params:
                    continue
                
                # Get or create group-level buffers (stored in first param's state for convenience)
                state = self.state[params[0]]
                num_params = len(params) # e.g.: 12 (for a d12 model)
                # e.g.: shape = (768, 3072), device = cuda:0, dtype = torch.float32, for one of the MLP projections
                shape, device, dtype = params[0].shape, params[0].device, params[0].dtype

                # Momentum for every individual parameter
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
                momentum_buffer = state["momentum_buffer"] # e.g.: (12, 768, 3072)

                # Second momentum buffer is factored, either per-row or per-column
                if "second_momentum_buffer" not in state:
                    if shape[-2] >= shape[-1]:
                        state["second_momentum_buffer"] = torch.zeros(num_params, shape[-2], 1, dtype=dtype, device=device)
                    else:
                        state["second_momentum_buffer"] = torch.zeros(num_params, 1, shape[-1], dtype=dtype, device=device)
                second_momentum_buffer = state["second_momentum_buffer"] # (12, 1, 3072)
                red_dim = -1 if shape[-2] >= shape[-1] else -2 # e.g.: -2

                # Stack grads and params
                stacked_grads = torch.stack([p.grad for p in params]) # (12, 768, 3072)
                stacked_params = torch.stack(params) # (12, 768, 3072)

                # Fill all the 0-D tensors with current values
                self._muon_momentum_t.fill_(group["momentum"])
                self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
                self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
                self._muon_wd_t.fill_(group["weight_decay"])

                # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
                muon_step_fused(
                    stacked_grads,
                    stacked_params,
                    momentum_buffer,
                    second_momentum_buffer,
                    self._muon_momentum_t,
                    self._muon_lr_t,
                    self._muon_wd_t,
                    self._muon_beta2_t,
                    group["ns_steps"],
                    red_dim,
                )

                # Copy back to original params: [(768, 3072), (768, 3072), ...] <- (12, 768, 3072)
                torch._foreach_copy_(params, list(stacked_params.unbind(0)))

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.

    (See MuonAdamW for algorithmic details.)

    AdamW Communication:
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction.
    A bunch of ideas (e.g. dist comms in slices) are borrowed from modded-nanogpt.

    Muon Communication:
    Parameters are grouped by shape, then stacked into single Tensors for efficient communication.
    We launch comms largest-first, then process smallest-first so large comms finish in time.
    """
    def __init__(
        self,
        adamw_groups: list[dict],
        muon_params,
        # AdamW hyperparams
        adamw_lr: float = 1e-3, 
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.01,
        # Muon hyperparams
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_beta2: float = 0.95,
        muon_weight_decay: float = 0.0,
    ):
        assert all(p.ndim == 2 for p in muon_params), "Muon expects 2D parameters only"
        muon_params = list(muon_params)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Build unified param_groups for the base Optimizer class
        # Each group is tagged with 'kind' = 'adamw' or 'muon'
        param_groups = []

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # AdamW groups: each input group becomes one param_group
        for group in adamw_groups:
            # Validate
            if rank == 0:
                assert isinstance(group, dict), "expecting param_groups to be a list of dicts"
                assert isinstance(group['params'], list), "expecting group['params'] to be a list of tensors"
                for p in group['params']:
                    sliced = p.numel() >= 1024
                    print(f"AdamW: 1 param of shape {p.shape}, sliced={sliced}")
                    if sliced: # large parameter tensors will be operated on in slices
                        assert p.shape[0] % world_size == 0, f"First dim of parameter shape {p.shape} must be divisible by world size {world_size}"
            # Add to param_groups
            params = list(group['params'])
            lr = group.get('lr', adamw_lr) # AdamW supports per-group learning rates
            param_groups.append(dict(
                params=params, lr=lr, kind='adamw',
                betas=adamw_betas, eps=adamw_eps, weight_decay=adamw_weight_decay,
            ))
        
        # Muon groups: group by shape for stacking, with all Muon hyperparams in the group
        muon_shapes = sorted({p.shape for p in muon_params}) # sort for deterministic ordering across ranks
        for shape in muon_shapes:
            group_params = [p for p in muon_params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            # Compute chunk size for this group (how many params each rank owns)
            chunk_size = (len(group_params) + world_size - 1) // world_size
            if rank == 0:
                print(f"Muon: {len(group_params)} params of shape {shape}, chunk_size={chunk_size}")
            param_groups.append(dict(
                params=group_params, lr=muon_lr, kind='muon', chunk_size=chunk_size,
                momentum=muon_momentum, ns_steps=muon_ns_steps,
                beta2=muon_beta2, weight_decay=muon_weight_decay,
            ))
        
        defaults = dict(lr=adamw_lr) # torch.optim.Optimizer requires a default lr
        super().__init__(param_groups, defaults)

        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Ensure all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"

        # First pass: launch all async communications
        adamw_infos: dict[Tensor, dict] = {}  # param -> {reduce_future, grad_slice, is_small}
        muon_infos: dict[int, dict] = {}  # group_idx -> {reduce_future, grad_chunk, stacked_grads}

        for group_idx, group in enumerate(self.param_groups):
            if group['kind'] == 'adamw':
                params: list[Tensor] = group['params']
                for p in params:
                    grad = p.grad
                    # Small params: use all_reduce (no scatter/gather needed)
                    if p.numel() < 1024:
                        reduce_future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                        adamw_infos[p] = dict(reduce_future=reduce_future, grad_slice=grad, is_small=True)
                    # Large param: reduce_scatter
                    else:
                        rank_size = grad.shape[0] // world_size # p.shape[0] % world_size == 0 is checked in __init__
                        grad_slice = torch.empty_like(grad[:rank_size])
                        reduce_future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                        adamw_infos[p] = dict(reduce_future=reduce_future, grad_slice=grad_slice, is_small=False)

            else:  # muon
                params: list[Tensor] = group["params"]
                chunk_size = group["chunk_size"]
                padded_num_params = chunk_size * world_size
                shape = params[0].shape
                device, dtype = params[0].device, params[0].dtype

                # Stack all gradients into a single tensor (single kernel via torch.stack)
                grad_stack = torch.stack([p.grad for p in params])
                stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
                stacked_grads[:len(params)].copy_(grad_stack)
                # Zero-pad if we have fewer params than padded size
                if len(params) < padded_num_params:
                    stacked_grads[len(params):].zero_()

                # Output buffer for this rank's chunk
                grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
                # Async reduce_scatter on the stacked tensor to get the grad_chunk for this rank
                reduce_future = dist.reduce_scatter_tensor(
                    grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()

                muon_infos[group_idx] = dict(
                    grad_chunk=grad_chunk,
                    reduce_future=reduce_future,
                    stacked_grads=stacked_grads,  # reuse for all_gather output
                )

        # Second pass: wait for reduce, compute updates, kick off all_gather
        gather_infos: list[dict] = []  # unified list for both AdamW and Muon gathers

        for group_idx, group in enumerate(self.param_groups):
            if group['kind'] == 'adamw':
                beta1, beta2 = group['betas']
                eps = group['eps']
                wd = group['weight_decay']
                params = group['params']
                for p in params:
                    info = adamw_infos[p]
                    info['reduce_future'].wait()
                    g_slice = info['grad_slice']
                    lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                    state = self.state[p]

                    # For small params, operate on full param; for large, operate on slice
                    if info['is_small']:
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
                    self._adamw_step_t.fill_(state['step'])
                    self._adamw_lr_t.fill_(lr)
                    self._adamw_beta1_t.fill_(beta1)
                    self._adamw_beta2_t.fill_(beta2)
                    self._adamw_eps_t.fill_(eps)
                    self._adamw_wd_t.fill_(eff_wd)

                    # Fused update: weight_decay -> momentum -> bias_correction -> param_update
                    adamw_step_fused(
                        p_slice, g_slice, exp_avg, exp_avg_sq,
                        self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                        self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
                    )

                    # Only large params need all_gather
                    if not info['is_small']:
                        gather_future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                        gather_infos.append(dict(gather_future=gather_future, params=None))

            else:  # muon
                info = muon_infos[group_idx]
                info['reduce_future'].wait()

                params = group["params"]
                chunk_size = group["chunk_size"]
                shape = params[0].shape
                device, dtype = params[0].device, params[0].dtype
                grad_chunk = info["grad_chunk"]

                # How many params does this rank actually own?
                start_idx = rank * chunk_size
                num_owned = min(chunk_size, max(0, len(params) - start_idx))

                # Get or create group-level state (stored keyed by first param)
                state = self.state[params[0]]

                # Momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
                momentum_buffer = state["momentum_buffer"]

                # Second momentum buffer is factored, either per-row or per-column
                if "second_momentum_buffer" not in state:
                    if shape[-2] >= shape[-1]:
                        state["second_momentum_buffer"] = torch.zeros(chunk_size, shape[-2], 1, dtype=dtype, device=device)
                    else:
                        state["second_momentum_buffer"] = torch.zeros(chunk_size, 1, shape[-1], dtype=dtype, device=device)
                second_momentum_buffer = state["second_momentum_buffer"]
                red_dim = -1 if shape[-2] >= shape[-1] else -2

                # Build updated_params tensor for all_gather
                updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

                if num_owned > 0:
                    # Stack owned params (single kernel via torch.stack)
                    owned_params = [params[start_idx + i] for i in range(num_owned)]
                    stacked_owned_params = torch.stack(owned_params)

                    # Get owned slices of buffers and grads
                    owned_grads = grad_chunk[:num_owned]
                    owned_momentum = momentum_buffer[:num_owned]
                    owned_second_momentum = second_momentum_buffer[:num_owned]

                    # Fill 0-D tensors with current values
                    self._muon_momentum_t.fill_(group["momentum"])
                    self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
                    self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
                    self._muon_wd_t.fill_(group["weight_decay"])

                    # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
                    muon_step_fused(
                        owned_grads,
                        stacked_owned_params,
                        owned_momentum,
                        owned_second_momentum,
                        self._muon_momentum_t,
                        self._muon_lr_t,
                        self._muon_wd_t,
                        self._muon_beta2_t,
                        group["ns_steps"],
                        red_dim,
                    )

                    # Copy updated params to output buffer
                    updated_params[:num_owned].copy_(stacked_owned_params)

                # Zero-pad the rest (for ranks that own fewer params)
                if num_owned < chunk_size:
                    updated_params[num_owned:].zero_()

                # Reuse stacked_grads buffer for all_gather output
                stacked_params = info["stacked_grads"]

                # Async all_gather to replicate updated params to all ranks
                gather_future = dist.all_gather_into_tensor(
                    stacked_params, updated_params, async_op=True
                ).get_future()

                gather_infos.append(dict(
                    gather_future=gather_future,
                    stacked_params=stacked_params,
                    params=params,
                ))

        # Final pass: wait for all_gather and copy back to params (Muon only)
        for info in gather_infos:
            info["gather_future"].wait()
            # Muon params need to be copied back from stacked buffer
            if info["params"] is not None:
                stacked_params = info["stacked_params"]
                params = info["params"]
                # Batched copy back (single kernel instead of N individual copies)
                torch._foreach_copy_(params, list(stacked_params[:len(params)].unbind(0)))