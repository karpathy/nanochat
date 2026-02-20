"""
Mixture of Experts (MoE) layer for nanochat.

Replaces the standard dense MLP in each transformer block. Each token picks its
top-K experts via a learned sigmoid router, so total parameters scale with
num_experts but per-token FLOPs remain constant (iso-FLOP with the dense MLP).

Expert hidden dim = 4 * dim / (top_k + num_shared), rounded to 128, ensures
approximately iso-FLOP with the dense MLP:
  Dense:            2 * dim * (4*dim) = 8*dim²
  MoE per token:    (top_k + num_shared) * 2 * dim * H ≈ 8*dim²

Expert weights are 3D tensors of shape (num_experts, hidden, dim). Muon's Polar
Express orthogonalization operates on the last two dims, so the expert dimension
acts as a batch dim and each expert is independently orthogonalized.

At forward time, torch._grouped_mm dispatches tokens to experts via cumulative
offsets — a single kernel per projection instead of a Python for-loop.
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
        _, selected_experts = torch.topk(biased_scores, k=self.top_k, dim=-1, sorted=False)
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


class SharedExpert(nn.Module):
    """Dense MLP shared expert — processes ALL tokens (no routing).

    Same architecture as each routed expert (up → ReLU² → down) but uses
    standard nn.Linear layers (2D weights, regular matmul) since there's
    no need for the grouped_mm dispatch machinery.
    """

    def __init__(self, dim, expert_hidden_dim):
        super().__init__()
        self.w_up = nn.Linear(dim, expert_hidden_dim, bias=False)
        self.w_down = nn.Linear(expert_hidden_dim, dim, bias=False)

    def forward(self, x):
        h = F.relu(self.w_up(x)).square()
        return self.w_down(h)


class ExpertGroup(nn.Module):
    """
    N independent expert MLPs stored as 3D weight tensors.
    Shape (num_experts, hidden, dim) — Muon's Polar Express operates on the
    last two dims, so each expert matrix is independently orthogonalized.
    """

    def __init__(self, dim, expert_hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.w_up = nn.Parameter(torch.empty(num_experts, expert_hidden_dim, dim))
        self.w_down = nn.Parameter(torch.empty(num_experts, dim, expert_hidden_dim))

    def forward(self, x, num_tokens_per_expert):
        """
        Args:
            x:                      (T*K, dim)     tokens sorted by expert assignment
            num_tokens_per_expert:  (num_experts,) count per expert
        Returns:
            output:                 (T*K, dim)
        """
        if x.is_cuda:
            return _run_experts_grouped_mm(self.w_up, self.w_down, x, num_tokens_per_expert)
        return _run_experts_for_loop(self.w_up, self.w_down, x, num_tokens_per_expert)


class MoE(nn.Module):
    """
    Mixture of Experts layer — approximately iso-FLOP replacement for the dense MLP.

    For each token:
    1. Shared expert processes all tokens via standard dense matmul
    2. Router scores all routed experts via sigmoid(gate(x))
    3. Top-K routed experts are selected
    4. Token is dispatched to those experts (weighted by routing score)
    5. Routed + shared expert outputs are summed together

    Total active experts per token = top_k + num_shared_experts.
    Expert hidden dim is sized so total active FLOPs ≈ dense MLP FLOPs.
    """

    def __init__(self, config):
        super().__init__()
        dim = config.n_embd
        num_experts = config.num_experts
        top_k = config.top_k
        num_shared = config.num_shared_experts
        self.top_k = top_k
        # Iso-FLOP sizing: total active experts per token = top_k + num_shared
        # Round to nearest 128 for tensor core alignment
        active_experts = top_k + num_shared
        expert_hidden_dim = round(4 * dim / active_experts / 128) * 128
        self.expert_hidden_dim = expert_hidden_dim
        self.router = TopKRouter(dim, num_experts, top_k)
        self.experts = ExpertGroup(dim, expert_hidden_dim, num_experts)
        self.shared_expert = SharedExpert(dim, expert_hidden_dim) if num_shared > 0 else None

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

        # Step 4: Shared expert — runs on ALL tokens via standard dense matmul
        # Launched before routed experts so compute can overlap (no data dependency)
        shared_output = self.shared_expert(x_flat) if self.shared_expert is not None else None

        # Step 5: Run routed experts on their assigned token blocks
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # Step 6: Scatter outputs back to original positions and sum over top-K
        combined = torch.zeros(
            bs * slen * self.top_k, dim,
            dtype=routed_output.dtype, device=routed_output.device,
        )
        combined[token_indices_sorted] = routed_output
        output = combined.view(bs * slen, self.top_k, dim).sum(dim=1)   # (T, dim)

        # Step 7: Add shared expert output
        if shared_output is not None:
            output = output + shared_output

        return output.view(bs, slen, dim)
