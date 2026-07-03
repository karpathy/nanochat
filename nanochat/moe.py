"""
Mixture of Experts (MoE) layer for nanochat.

Drop-in replacement for the dense MLP in each transformer block: a learned
sigmoid router sends each token to its top-K experts (out of num_experts),
plus a shared expert that processes all tokens. (No knob for the shared
expert: ablating it at d16 was the largest single-knob quality hit of the
sweep, +0.0044 bpb, and the literature agrees — it is always on.) Total
parameters scale with num_experts while per-token FLOPs stay constant
(iso-FLOP with the dense MLP): expert_hidden = 4*dim/(top_k+1), rounded to 128.

Design choices, benchmarked in dev/moe_bench.py (d24 shapes, H100):
- Scores are applied AFTER the experts. ReLU^2 is homogeneous of degree 2
  (expert(s*x) == s^2 * expert(x)), so pre-multiplying the input — what
  torchtitan's score_before_experts does — would gate by the SQUARED score.
  Applying to the output gates linearly, like DeepSeekV3.
- The score multiply runs in bf16. Scores are gates in (0,1); the fp32
  upcast/downcast round-trip tripled the memory traffic for no benefit.
- Combine is scatter-into-zeros + sum over K slots. This beat index_add_
  (atomics) by ~9% under torch.compile: scatter is a pure permutation (every
  destination written once) and the K-way sum is a clean reduction.
- Load balancing is DeepSeekV3's auxiliary-loss-free bias nudging, batched
  across all layers into a single all_reduce with no CPU-GPU sync points.
- Expert weights are 3D tensors (num_experts, hidden, dim). Muon's Polar
  Express orthogonalization operates on the last two dims, so the expert dim
  acts as a batch dim and each expert is independently orthogonalized.
- The router gate gets its own optimizer group with a dedicated LR (see
  GPT.setup_optimizer): it is a tiny classifier head feeding a sigmoid, not
  clearly analogous to any other parameter group in the model. Swept at d16:
  0.005 is best; at 0.02+ the learned routing is WORSE than a frozen random
  router, and 0.0025-0.01 is a flat optimum (see dev/LOG.md).
- An FP8 path for the routed experts (torch._scaled_grouped_mm, rowwise
  scales) was implemented and removed: net wall-clock loss at nanochat scales
  (per-microstep quantization overhead exceeds the GEMM savings) and slightly
  worse quality. See dev/moe_fp8.md and dev/LOG.md before re-attempting.

References:
- DeepSeekV3: https://arxiv.org/abs/2412.19437
- Aux-loss-free balancing: https://arxiv.org/abs/2408.15664
- torchtitan MoE (implementation reference): torchtitan/models/moe/moe.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    """Same as gpt.py's Linear (duplicated here to avoid a circular import):
    master weights stay fp32, cast to the activation dtype in forward."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class TopKRouter(nn.Module):
    """Sigmoid-gated top-K router. Each token independently picks K experts.

    Sigmoid (not softmax) so each expert's score is independent — no
    competition across experts — following DeepSeekV3. The expert_bias buffer
    implements aux-loss-free load balancing: it is added to the scores for the
    top-k SELECTION only, while the returned gating weights use the raw scores,
    so balancing steers which experts fire but never distorts their output.
    """
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.gate = Linear(dim, num_experts, bias=False)  # weight: (E, dim)
        self.num_experts = num_experts
        self.top_k = top_k
        # Load balancing state (persistent buffers => saved in checkpoints)
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.register_buffer("tokens_per_expert_counter", torch.zeros(num_experts))

    def forward(self, x):
        """
        Args:
            x: (T, dim) flattened token representations
        Returns:
            top_scores:            (T, K)  gating weights of the selected experts, fp32 in (0,1)
            selected_experts:      (T, K)  indices of the selected experts
            num_tokens_per_expert: (E,)    how many (token, slot) pairs each expert received
        """
        scores = self.gate(x)                    # (T, E)
        scores = torch.sigmoid(scores.float())   # fp32: routing decisions are cheap (T*E) and tie-prone, keep them exact
        # Bias affects expert SELECTION but not the gating weights
        biased_scores = scores + self.expert_bias
        # sorted=False: we don't care about the order of the K winners, skip the sort
        _, selected_experts = torch.topk(biased_scores, k=self.top_k, dim=-1, sorted=False)
        top_scores = scores.gather(dim=-1, index=selected_experts)  # (T, K)
        # Histogram of assignments, stays on GPU (no CPU sync); sums to T*K
        num_tokens_per_expert = torch.histc(
            selected_experts.float().view(-1),
            bins=self.num_experts, min=0, max=self.num_experts,
        )
        # Accumulate token counts for the balancing update (training only, so
        # that val/inference forwards don't contaminate the balancing signal)
        if self.training:
            self.tokens_per_expert_counter += num_tokens_per_expert
        return top_scores, selected_experts, num_tokens_per_expert


def run_experts_grouped_mm(w_up, w_down, x, num_tokens_per_expert):
    """All experts in one kernel per projection via torch._grouped_mm.

    x must be sorted by expert: rows [0, offsets[0]) belong to expert 0, etc.
    The token distribution lives in the VALUES of offsets, not in any tensor
    shape, so all shapes are static and torch.compile(dynamic=False) is happy.

    Args:
        w_up:   (E, H, D) stacked up-projections (fp32 master weights)
        w_down: (E, D, H) stacked down-projections
        x:      (R, D) expert-sorted rows, R = T * top_k
        num_tokens_per_expert: (E,) rows per expert, sums to R
    Returns:
        (R, D) in x's dtype
    """
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)  # (E,)
    # grouped_mm is bf16-only and wants the right operand column-major over the
    # last two dims — transpose of a contiguous tensor gives exactly that view.
    h = torch._grouped_mm(x.bfloat16(), w_up.bfloat16().transpose(-2, -1), offs=offsets)  # (R, H)
    h = F.relu(h).square()
    out = torch._grouped_mm(h, w_down.bfloat16().transpose(-2, -1), offs=offsets)  # (R, D)
    return out.type_as(x)


@torch.compiler.disable
def run_experts_for_loop(w_up, w_down, x, num_tokens_per_expert):
    """Fallback for CPU/MPS where grouped_mm isn't available. The .tolist()
    is a device sync, hence @torch.compiler.disable; only used off-CUDA."""
    token_counts = [int(c) for c in num_tokens_per_expert.tolist()]
    chunks = torch.split(x, token_counts, dim=0)
    outputs = []
    for i, chunk in enumerate(chunks):
        # No empty-chunk skip: matmul with (0, dim) tensors is valid and produces
        # zero gradients (vs None), which the optimizer needs for stacking.
        h = chunk @ w_up[i].to(chunk.dtype).T
        h = F.relu(h).square()
        h = h @ w_down[i].to(chunk.dtype).T
        outputs.append(h)
    return torch.cat(outputs, dim=0)


class ExpertGroup(nn.Module):
    """N independent expert MLPs stored as stacked 3D weight tensors."""
    def __init__(self, dim, expert_hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.w_up = nn.Parameter(torch.empty(num_experts, expert_hidden_dim, dim))
        self.w_down = nn.Parameter(torch.empty(num_experts, dim, expert_hidden_dim))

    def forward(self, x, num_tokens_per_expert):
        if x.is_cuda:
            return run_experts_grouped_mm(self.w_up, self.w_down, x, num_tokens_per_expert)
        return run_experts_for_loop(self.w_up, self.w_down, x, num_tokens_per_expert)


class SharedExpert(nn.Module):
    """Dense MLP shared expert — processes ALL tokens, no routing (DeepSeekV3).
    Provides baseline capacity for every token; the routed experts specialize."""
    def __init__(self, dim, expert_hidden_dim):
        super().__init__()
        self.w_up = Linear(dim, expert_hidden_dim, bias=False)
        self.w_down = Linear(expert_hidden_dim, dim, bias=False)

    def forward(self, x):
        h = F.relu(self.w_up(x)).square()
        return self.w_down(h)


class MoE(nn.Module):
    """Mixture of Experts layer — iso-FLOP replacement for the dense MLP.

    Per token: shared expert + top_k routed experts, outputs summed.
    """
    def __init__(self, config):
        super().__init__()
        dim = config.n_embd
        self.top_k = config.top_k
        # Iso-FLOP sizing: active experts per token = top_k routed + 1 shared,
        # each with hidden H = 4*dim / active, rounded to 128 for tensor cores.
        # (At d24: 4*1536/3 = 2048 exactly, the rounding is a no-op.)
        active_experts = config.top_k + 1
        expert_hidden_dim = round(4 * dim / active_experts / 128) * 128
        self.expert_hidden_dim = expert_hidden_dim
        self.router = TopKRouter(dim, config.num_experts, config.top_k)
        self.experts = ExpertGroup(dim, expert_hidden_dim, config.num_experts)
        self.shared_expert = SharedExpert(dim, expert_hidden_dim)

    def forward(self, x):
        # x: (B, S, D). Notation: T = B*S tokens, K = top_k, E = num_experts.
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # (T, D)

        # Route: each token picks its top-K experts
        top_scores, selected_experts, num_tokens_per_expert = self.router(x_flat)

        # Permute: sort the T*K (token, slot) assignments by expert id so each
        # expert's tokens form one contiguous block for grouped_mm.
        # stable=True keeps original token order within each expert's block.
        token_indices_sorted = torch.argsort(selected_experts.view(-1), stable=True)  # (T*K,)
        scores_sorted = top_scores.view(-1)[token_indices_sorted]                     # (T*K,)
        # The flat index enumerates (token 0 slot 0), (token 0 slot 1), (token 1 slot 0), ...
        # so integer-dividing by K recovers the original token id.
        token_ids = token_indices_sorted // self.top_k                                # (T*K,)
        routed_input = x_flat[token_ids]                                              # (T*K, D)

        # Shared expert: launched before the routed experts since it has no
        # data dependency on the routing — the kernels can overlap on the GPU
        shared_output = self.shared_expert(x_flat)

        # Expert MLPs on the expert-sorted rows, gated by their scores on the way out
        routed_output = self.experts(routed_input, num_tokens_per_expert)  # (T*K, D)
        routed_output = routed_output * scores_sorted.unsqueeze(-1).to(routed_output.dtype)

        # Combine: un-permute the rows back to their original (token, slot)
        # positions in a zeros buffer, then sum the K contributions per token.
        # (Not index_add_: every destination here is written exactly once, so this
        # scatter + clean sum reduction beats atomics by ~9% compiled, see dev/moe_bench.py)
        combined = torch.zeros(B * S * self.top_k, D, dtype=routed_output.dtype, device=routed_output.device)
        combined[token_indices_sorted] = routed_output       # (T*K, D)
        output = combined.view(B * S, self.top_k, D).sum(dim=1)  # (T, D)

        output = output + shared_output
        return output.view(B, S, D)


@torch.no_grad()
def update_expert_biases(routers, coeff=1e-3):
    """Auxiliary-loss-free load balancing update (DeepSeekV3), batched over layers.

    Underloaded experts get +coeff on their selection bias, overloaded -coeff.
    sign() makes the update discrete, avoiding drift; centering keeps the bias
    zero-mean. One stacked all_reduce for all layers, and no CPU-GPU syncs
    (sign(0) = 0 handles the counters-are-all-zero case for free).

    Call once per training step, before optimizer.step().
    """
    if len(routers) == 0:
        return
    counters = torch.stack([r.tokens_per_expert_counter for r in routers])  # (L, E)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counters)  # sum token counts across ranks, one op for all layers
    mean_counts = counters.mean(dim=-1, keepdim=True)                       # (L, 1)
    biases = torch.stack([r.expert_bias for r in routers])                  # (L, E)
    biases = biases + coeff * torch.sign(mean_counts - counters)
    biases = biases - biases.mean(dim=-1, keepdim=True)  # center to prevent drift
    for i, router in enumerate(routers):
        router.expert_bias.copy_(biases[i])
        router.tokens_per_expert_counter.zero_()


@torch.no_grad()
def compute_moe_stats(routers):
    """Routing statistics for logging. Uses this rank's local counts.
    Call BEFORE update_expert_biases (which resets the counters).
    Contains .item() calls (CPU syncs) — call sparingly, e.g. every 100 steps."""
    if len(routers) == 0:
        return {}
    counters = torch.stack([r.tokens_per_expert_counter for r in routers]).float()  # (L, E)
    biases = torch.stack([r.expert_bias for r in routers]).float()                  # (L, E)
    # Load imbalance: coefficient of variation (std/mean) per layer, averaged.
    # 0 = perfectly balanced; 1 = std as large as the mean (severe imbalance).
    counts_mean = counters.mean(dim=-1).clamp(min=1)
    load_imbalance = (counters.std(dim=-1) / counts_mean).mean().item()
    stats = {
        "moe/load_imbalance": load_imbalance,
        "moe/expert_bias_std": biases.std().item(),
        "moe/expert_bias_absmax": biases.abs().max().item(),
    }
    return stats
