"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
The same class handles both single GPU and distributed training: when there is no
multi-rank process group, the communication ops are simply skipped and every rank
(i.e. the only rank) owns all of the parameters.

Adapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import torch
import torch.distributed as dist
from torch import Tensor
from nanochat.common import COMPUTE_DTYPE

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    All params and optimizer state are fp32 (asserted in MuonAdamW.__init__).
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

# -----------------------------------------------------------------------------
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

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Two more (very) slight and optional improvements:
1) MuonEq row equilibration: rescale each row to the mean row norm so the spectrum
entering orthogonalization is better conditioned (https://arxiv.org/abs/2603.28254)
2) Muon+ renormalization: snap the Frobenius norm to sqrt(min(m, n)), the norm of an exactly
semi-orthogonal matrix, correcting for under-convergence of the polar iteration (https://arxiv.org/abs/2602.21545)

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
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
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

    # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g

    # MuonEq row equilibration: rescale each row to the mean row norm so the spectrum entering orthogonalization is better conditioned
    target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
    row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    X = X * (target / row_norm).to(X.dtype)

    # Polar Express orthogonalization: replace each update with the nearest orthogonal matrix
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    # Cast back to the param dtype (MPS errors on the mixed-dtype ops below when X is bf16)
    g = X.to(stacked_params.dtype)

    # Muon+ renormalization: snap Frobenius norm to sqrt(min(m, n))
    target_norm = min(g.size(-2), g.size(-1)) ** 0.5
    current_norm = g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    g = g * (target_norm / current_norm).to(g.dtype)

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

# -----------------------------------------------------------------------------

# AdamW params smaller than this are all_reduced and updated on every rank (their
# optimizer state is replicated, but they are tiny: the per-layer scalars).
# Larger params (the embedding tables, lm_head) are ZeRO-sharded by rows.
ADAMW_SMALL_NUMEL = 1024

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for the 2D matrix params, AdamW for the rest
    (embeddings, lm_head, per-layer scalars). Written for nanochat's model
    specifically, not as a general-purpose optimizer.

    Muon groups (one per matrix shape, built in gpt.setup_optimizer):
        All K matrices of a group permanently live in one (K_pad, m, n) stack;
        each param is a view stack[j], and its grad is a view into a matching
        grad stack, so autograd accumulates gradients directly into the stack.
        Rank r owns rows [r*k, (r+1)*k) where k = ceil(K / world_size) - for a
        single-shape group, sharding by rank is just sequential order. K is
        padded up to k * world_size with all-zero dummy matrices (Muon's update
        of a zero matrix is zero, so they ride along harmlessly).
        Per step: one in-place reduce_scatter of the grad stack, one fused Muon
        update of the owned rows in place, one in-place all_gather of the stack.
        Nothing is stacked or copied back - the params ARE the stack.

    AdamW params, ZeRO-2 style (as before):
        Small params (<1024 elements: the scalars): all_reduce the grad, update
        the full param on every rank (state replicated, but they are tiny).
        Large params (lm_head, wte, value_embeds): in-place reduce_scatter of
        the grad along dim 0, fused update of this rank's row slice, in-place
        all_gather of the param. Requires shape[0] divisible by world_size
        (true for the padded vocab embeddings this model has).

    Communication is async in 3 phases to overlap with compute: launch all
    reduces, then per group wait -> compute -> launch gather, then wait for the
    gathers. On a single rank all communication is skipped.

    Optimizer state is sharded by rank, so state resume requires the same
    world_size (enforced in load_state_dict; model checkpoints are unaffected).

    Muon p.grad tensors are views into the grad stacks and must stay that way
    (step() asserts it): if you ever set gradients manually, write them in
    place. Use this optimizer's zero_grad(), which zeroes the grad stacks
    instead of severing the views.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
        All params in a Muon group must have the same shape (caller's responsibility).
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # The fused kernels do their math directly in the storage dtype
        for group in self.param_groups:
            for p in group['params']:
                assert p.dtype == torch.float32, f"MuonAdamW expects fp32 params, got {p.dtype}"
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # The stack layout depends on world size, decided once at construction
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        # Build the permanent (K_pad, m, n) stack for each Muon group.
        # self._stacks is parallel to self.param_groups (None for AdamW groups).
        # Groups are always accessed by index: load_state_dict() replaces the group
        # dicts, so holding a reference would read stale hyperparams after a resume.
        self._stacks = []
        for group in self.param_groups:
            assert group['kind'] in ('adamw', 'muon'), f"Unknown optimizer kind: {group['kind']}"
            if group['kind'] != 'muon':
                self._stacks.append(None)
                continue
            params = group['params']
            shape, dtype, device = params[0].shape, params[0].dtype, params[0].device
            assert all(p.shape == shape for p in params), "all params in a Muon group must share one shape"
            k = -(-len(params) // self.world_size)  # matrices per rank (ceil)
            stack = torch.zeros(k * self.world_size, *shape, dtype=dtype, device=device)
            grad_stack = torch.zeros_like(stack)
            for j, p in enumerate(params):
                stack[j].copy_(p.detach())
                p.data = stack[j]        # p now aliases row j of the stack
                p.grad = grad_stack[j]   # autograd accumulates straight into the grad stack
            # Sharded per-rank state (this rank's k rows only), created eagerly and
            # keyed under the group's first param so state_dict/load_state_dict work.
            second_shape = (k, shape[-2], 1) if shape[-2] >= shape[-1] else (k, 1, shape[-1])
            self.state[params[0]] = dict(
                momentum_buffer=torch.zeros(k, *shape, dtype=dtype, device=device),
                second_momentum_buffer=torch.zeros(second_shape, dtype=dtype, device=device),
            )
            self._stacks.append(dict(stack=stack, grad_stack=grad_stack, k=k))

    def state_dict(self):
        # The sharded state layout is a function of world_size, so stamp it into the
        # state dict; load_state_dict validates it. Without the check, resuming at a
        # LARGER world size would slice smaller-but-in-bounds regions of the loaded
        # buffers and silently train on the wrong moments (no shape error anywhere).
        sd = super().state_dict()
        sd['world_size'] = self.world_size
        return sd

    def load_state_dict(self, state_dict):
        saved_world_size = state_dict.get('world_size')
        assert saved_world_size == self.world_size, (
            f"optimizer state was saved with world_size={saved_world_size} but this run has "
            f"world_size={self.world_size}. The sharded state layout depends on world_size, "
            f"so optimizer-state resume requires the same number of ranks. "
            f"(Model checkpoints are unaffected and load at any world size.)"
        )
        state_dict = {k: v for k, v in state_dict.items() if k != 'world_size'}
        super().load_state_dict(state_dict)

    def zero_grad(self, set_to_none=True):
        """Zero the Muon grad stacks in place (the p.grad views stay aliased into
        them) and drop the AdamW grads like a regular set_to_none zero_grad."""
        for group, stack in zip(self.param_groups, self._stacks):
            if group['kind'] == 'muon':
                stack['grad_stack'].zero_()
            else:
                for p in group['params']:
                    p.grad = None

    @torch.no_grad()
    def step(self):
        W, r = self.world_size, self.rank
        ddp = W > 1 # on a single rank, all communication is skipped
        # Muon grads accumulate directly into the grad stacks because every p.grad is
        # a stack view (set up in __init__). If p.grad were ever reassigned, the reduce
        # below would consume stale gradients, so fail loudly here.
        for group, stack in zip(self.param_groups, self._stacks):
            if group['kind'] == 'muon':
                for j, p in enumerate(group['params']):
                    assert p.grad is not None and p.grad.data_ptr() == stack['grad_stack'][j].data_ptr(), \
                        "Muon p.grad must be its grad-stack view; write gradients in place, don't reassign p.grad"

        # Phase 1: launch all async reduce ops. Everything is reduced in place:
        # the output chunk/slice is a view into the gradient it is reduced from.
        reduce_works = []  # parallel to param_groups; each entry is a list of works
        for group, stack in zip(self.param_groups, self._stacks):
            works = []
            if ddp and group['kind'] == 'muon':
                k = stack['k']
                grad_chunk = stack['grad_stack'][r * k:(r + 1) * k]
                work = dist.reduce_scatter_tensor(grad_chunk, stack['grad_stack'], op=dist.ReduceOp.AVG, async_op=True)
                works.append(work)
            elif ddp:
                for p in group['params']:
                    is_small = p.numel() < ADAMW_SMALL_NUMEL
                    if is_small:
                        work = dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
                        works.append(work)
                    else:
                        assert p.shape[0] % W == 0, f"AdamW reduce_scatter requires shape[0] ({p.shape[0]}) divisible by world_size ({W})"
                        rows = p.shape[0] // W
                        grad_slice = p.grad[r * rows:(r + 1) * rows]
                        work = dist.reduce_scatter_tensor(grad_slice, p.grad, op=dist.ReduceOp.AVG, async_op=True)
                        works.append(work)
            reduce_works.append(works)

        # Phase 2: as each group's reduce lands, compute this rank's updates in
        # place, then launch the in-place all_gather that rebroadcasts them
        gather_works = []
        for gi, group in enumerate(self.param_groups):
            for work in reduce_works[gi]:
                work.wait()
            if group['kind'] == 'muon':
                stack = self._stacks[gi]
                k = stack['k']
                state = self.state[group['params'][0]]
                shape = group['params'][0].shape
                param_chunk = stack['stack'][r * k:(r + 1) * k]
                grad_chunk = stack['grad_stack'][r * k:(r + 1) * k]
                red_dim = -1 if shape[-2] >= shape[-1] else -2
                self._muon_momentum_t.fill_(group['momentum'])
                self._muon_beta2_t.fill_(group['beta2'])
                self._muon_lr_t.fill_(group['lr'] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
                self._muon_wd_t.fill_(group['weight_decay'])
                muon_step_fused(
                    grad_chunk, param_chunk,
                    state['momentum_buffer'], state['second_momentum_buffer'],
                    self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                    group['ns_steps'], red_dim,
                )
                if ddp:
                    gather_works.append(dist.all_gather_into_tensor(stack['stack'], param_chunk, async_op=True))
            else:
                self._adamw_lr_t.fill_(group['lr'])
                self._adamw_beta1_t.fill_(group['betas'][0])
                self._adamw_beta2_t.fill_(group['betas'][1])
                self._adamw_eps_t.fill_(group['eps'])
                self._adamw_wd_t.fill_(group['weight_decay'])
                for p in group['params']:
                    sharded = ddp and p.numel() >= ADAMW_SMALL_NUMEL # ZeRO: each rank updates its slice of rows
                    if sharded:
                        rows = p.shape[0] // W
                        p_slice = p[r * rows:(r + 1) * rows]
                        grad_slice = p.grad[r * rows:(r + 1) * rows]
                    else:
                        p_slice = p
                        grad_slice = p.grad
                    state = self.state[p]
                    if not state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p_slice)
                        state['exp_avg_sq'] = torch.zeros_like(p_slice)
                    state['step'] += 1
                    self._adamw_step_t.fill_(state['step'])
                    adamw_step_fused(
                        p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                        self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                        self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
                    )
                    if sharded:
                        gather_works.append(dist.all_gather_into_tensor(p, p_slice, async_op=True))

        # Phase 3: wait for the gathers. Params alias the gathered storage
        # (the Muon stacks, or the AdamW params themselves) - nothing to copy back.
        for work in gather_works:
            work.wait()
