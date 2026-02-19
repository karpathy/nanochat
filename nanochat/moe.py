"""
Mixture of Experts (MoE) layer for nanochat.

Replaces the standard dense MLP in each transformer block. Each token picks its
top-K experts via a learned sigmoid router, so total parameters scale with
num_experts but per-token FLOPs remain constant (iso-FLOP with the dense MLP).

Expert hidden dim = 4 * dim / top_k ensures FLOPs match:
  Dense:  4 * dim * (4*dim) = 16*dim²
  MoE:    top_k * 4 * dim * (4*dim/top_k) = 16*dim²

Expert weights are stored as separate 2D parameters (not stacked 3D) so they
integrate natively with the Muon optimizer, which expects 2D matrices. At forward
time, params are stacked into 3D and fed to torch._grouped_mm for efficient
batched computation (single kernel per projection instead of a Python for-loop).
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """Sigmoid-gated top-K router. Each token independently picks K experts."""

    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        # Auxiliary-loss-free load balancing (DeepSeekV3)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.register_buffer('tokens_per_expert_counter', torch.zeros(num_experts))

    def forward(self, x):
        """
        Args:
            x: (T, dim) flattened token representations
        Returns:
            top_scores:             (T, top_k)    routing weights for selected experts
            selected_experts:       (T, top_k)    which experts each token chose
            num_tokens_per_expert:  (num_experts,) how many tokens each expert received
        """
        scores = self.gate(x)                       # (T, num_experts)
        scores = torch.sigmoid(scores.float())      # values in (0, 1)
        # Bias affects expert SELECTION but not gating weights (DeepSeekV3)
        biased_scores = scores + self.expert_bias
        _, selected_experts = torch.topk(biased_scores, k=self.top_k, dim=-1)
        top_scores = scores.gather(dim=-1, index=selected_experts)
        num_tokens_per_expert = torch.histc(
            selected_experts.float().view(-1),
            bins=self.num_experts, min=0, max=self.num_experts,
        )
        # Accumulate token counts for load balancing updates
        self.tokens_per_expert_counter += num_tokens_per_expert
        return top_scores, selected_experts, num_tokens_per_expert

    def update_expert_bias(self, coeff=1e-3):
        """Auxiliary-loss-free bias update (DeepSeekV3). Call before optimizer.step()."""
        counts = self.tokens_per_expert_counter
        # Sync token counts across GPUs if distributed
        if dist.is_initialized():
            dist.all_reduce(counts)
        if counts.sum() == 0:
            return
        mean_count = counts.mean()
        # Nudge underloaded experts up, overloaded experts down
        self.expert_bias += coeff * torch.sign(mean_count - counts)
        self.expert_bias -= self.expert_bias.mean()  # center to prevent drift
        self.tokens_per_expert_counter.zero_()


def _run_experts_grouped_mm(w_up, w_down, x, num_tokens_per_expert):
    """Run all experts via grouped matmul — single kernel per projection.

    torch._grouped_mm handles variable tokens-per-expert internally via
    cumulative offsets, so no Python for-loop or .tolist() device sync needed.
    All tensor shapes are static (the dynamic token distribution is encoded
    in the offsets, not in tensor dimensions).

    Args:
        w_up:   (num_experts, expert_hidden_dim, dim) - stacked up-projections
        w_down: (num_experts, dim, expert_hidden_dim) - stacked down-projections
        x:      (total_tokens, dim) - tokens sorted by expert assignment
        num_tokens_per_expert: (num_experts,) - count per expert
    Returns:
        output: (total_tokens, dim)
    """
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    # Cast everything to bf16 upfront (weights are fp32 for Muon, need bf16 for grouped_mm)
    x_bf16 = x.bfloat16()
    w_up_bf16 = w_up.bfloat16().transpose(-2, -1)
    w_down_bf16 = w_down.bfloat16().transpose(-2, -1)
    # Up-project all experts at once: (total_tokens, dim) → (total_tokens, expert_hidden_dim)
    h = torch._grouped_mm(x_bf16, w_up_bf16, offs=offsets)
    h = F.relu(h).square()  # ReLU² activation
    # Down-project all experts at once: (total_tokens, expert_hidden_dim) → (total_tokens, dim)
    out = torch._grouped_mm(h.bfloat16(), w_down_bf16, offs=offsets)
    return out.type_as(x)


@torch.compiler.disable
def _run_experts_for_loop(w_up, w_down, x, num_tokens_per_expert):
    """Fallback for-loop implementation for CPU/MPS where grouped_mm isn't available.

    Decorated with @torch.compiler.disable because .tolist() causes a device-host
    sync that torch.compile can't handle. Only used on non-CUDA devices.
    """
    token_counts = num_tokens_per_expert.tolist()
    chunks = torch.split(x, [int(c) for c in token_counts], dim=0)
    outputs = []
    for i, chunk in enumerate(chunks):
        # No empty-chunk skip: matmul with (0, dim) tensors is valid and produces
        # zero gradients (vs None), which the optimizer needs for stacking.
        h = chunk @ w_up[i].T
        h = F.relu(h).square()
        h = h @ w_down[i].T
        outputs.append(h)
    return torch.cat(outputs, dim=0)


class ExpertGroup(nn.Module):
    """
    N independent expert MLPs, each with separate 2D weight matrices.
    Separate 2D params (not stacked 3D) for native Muon optimizer compatibility.
    At forward time, params are stacked and dispatched via torch._grouped_mm.
    """

    def __init__(self, dim, expert_hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.w_ups = nn.ParameterList([
            nn.Parameter(torch.empty(expert_hidden_dim, dim))
            for _ in range(num_experts)
        ])
        self.w_downs = nn.ParameterList([
            nn.Parameter(torch.empty(dim, expert_hidden_dim))
            for _ in range(num_experts)
        ])

    def forward(self, x, num_tokens_per_expert):
        """
        Args:
            x:                      (T*K, dim)     tokens sorted by expert assignment
            num_tokens_per_expert:  (num_experts,) count per expert
        Returns:
            output:                 (T*K, dim)
        """
        # Stack separate 2D params into 3D for grouped_mm
        # (autograd handles gradient propagation back to individual params)
        w_up = torch.stack(list(self.w_ups))     # (num_experts, expert_hidden_dim, dim)
        w_down = torch.stack(list(self.w_downs))  # (num_experts, dim, expert_hidden_dim)
        if x.is_cuda:
            return _run_experts_grouped_mm(w_up, w_down, x, num_tokens_per_expert)
        return _run_experts_for_loop(w_up, w_down, x, num_tokens_per_expert)


class MoE(nn.Module):
    """
    Mixture of Experts layer — iso-FLOP replacement for the dense MLP.

    For each token:
    1. Router scores all experts via sigmoid(gate(x))
    2. Top-K experts are selected
    3. Token is dispatched to those experts (weighted by routing score)
    4. Expert outputs are summed back together

    Total params are ~num_experts/top_k times larger than dense MLP,
    but per-token FLOPs are identical.
    """

    def __init__(self, config):
        super().__init__()
        dim = config.n_embd
        num_experts = config.num_experts
        top_k = config.top_k
        self.top_k = top_k
        # Iso-FLOP sizing: expert_hidden = 4*dim/top_k so MoE FLOPs = dense MLP FLOPs
        expert_hidden_dim = 4 * dim // top_k
        self.router = TopKRouter(dim, num_experts, top_k)
        self.experts = ExpertGroup(dim, expert_hidden_dim, num_experts)

    def forward(self, x):
        """
        Args:  x: (bs, slen, dim)
        Returns: output: (bs, slen, dim) — same shape, drop-in MLP replacement
        """
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)                                        # (T, dim)

        # Step 1: Route — each token picks its top-K experts
        top_scores, selected_experts, num_tokens_per_expert = self.router(x_flat)

        # Step 2: Sort tokens by expert assignment for contiguous expert processing
        # argsort groups all assignments to expert 0 first, then expert 1, etc.
        token_indices_sorted = torch.argsort(selected_experts.view(-1), stable=True)
        scores_sorted = top_scores.view(-1)[token_indices_sorted]       # (T*K,)
        token_ids = token_indices_sorted // self.top_k                  # map back to original token
        routed_input = x_flat[token_ids]                                # (T*K, dim)

        # Step 3: Pre-multiply by routing scores (score_before_experts strategy)
        routed_input = (routed_input.float() * scores_sorted.unsqueeze(-1)).to(x.dtype)

        # Step 4: Run experts on their assigned token blocks
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # Step 5: Scatter outputs back to original positions and sum over top-K
        combined = torch.zeros(
            bs * slen * self.top_k, dim,
            dtype=routed_output.dtype, device=routed_output.device,
        )
        combined[token_indices_sorted] = routed_output
        output = combined.view(bs * slen, self.top_k, dim).sum(dim=1)   # (T, dim)

        return output.view(bs, slen, dim)
