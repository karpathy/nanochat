"""
Mixture-of-Experts (MoE) block for nanochat.

Design (DeepSeek-style, see also Tian et al. 2507.17702 for scaling-law guidance):
- `num_experts` routed experts + `num_shared_experts` always-active experts.
- Top-k softmax routing, renormalized across chosen experts.
- Padded-capacity dispatch so all tensor shapes are static => torch.compile works.
- Switch-style aux loss: lambda * E * sum_e (f_e * p_e), where f_e is the top-1 token
  fraction (detached) and p_e is the mean router softmax prob (carries gradient).
- Expert weights stored as 3D nn.Parameter (num_experts, D_in, D_out). Muon orthogonalizes
  each expert slice independently via batched polar-express (needs leading-dim support
  in nanochat.optim, already added).
- Compute-matched default: expert_hidden_dim = 4 * n_embd / (top_k + num_shared_experts),
  rounded to multiples of 64. Keeps per-token active FFN params ~= dense MLP so that
  sweeping `num_experts` varies total capacity at roughly fixed active compute.

forward(x) returns (y, aux_loss) where aux_loss is a scalar fp32 tensor already scaled by
the aux-loss coefficient. The GPT model sums these across blocks and adds to the main CE loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _round_to(x: int, multiple: int) -> int:
    return max(multiple, ((x + multiple - 1) // multiple) * multiple)


def default_expert_hidden(n_embd: int, top_k: int, num_shared_experts: int, head_dim_multiple: int = 64) -> int:
    """Compute-matched sizing: total active FFN params per token ~= dense MLP (hidden=4*n_embd)."""
    active_factor = max(1, top_k + num_shared_experts)
    raw = (4 * n_embd) // active_factor
    return _round_to(raw, head_dim_multiple)


class MoE(nn.Module):
    """Mixture-of-Experts MLP replacement. Drop-in for nanochat.gpt.MLP when num_experts > 1."""

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_experts = config.num_experts
        self.top_k = min(config.top_k, config.num_experts)
        self.num_shared_experts = config.num_shared_experts
        self.capacity_factor = config.capacity_factor
        self.aux_loss_coef = config.moe_aux_loss_coef
        self.expert_hidden_dim = default_expert_hidden(
            config.n_embd, self.top_k, self.num_shared_experts
        )

        # Router: project n_embd -> num_experts. Shape (num_experts, n_embd) so it groups
        # with other 2D matrix params of like shape across layers in Muon.
        self.router_weight = nn.Parameter(torch.empty(self.num_experts, self.n_embd))

        # Routed experts as 3D tensors: Muon orthogonalizes each (D, H) / (H, D) slice independently.
        E, D, H = self.num_experts, self.n_embd, self.expert_hidden_dim
        self.w_fc = nn.Parameter(torch.empty(E, D, H))
        self.w_proj = nn.Parameter(torch.empty(E, H, D))

        # Shared experts (always active). Same per-expert sizing as routed.
        if self.num_shared_experts > 0:
            S = self.num_shared_experts
            self.ws_fc = nn.Parameter(torch.empty(S, D, H))
            self.ws_proj = nn.Parameter(torch.empty(S, H, D))
        else:
            self.register_parameter("ws_fc", None)
            self.register_parameter("ws_proj", None)

    @torch.no_grad()
    def init_weights(self):
        """Mirrors nanochat.gpt.MLP init: c_fc uniform +/- 0.4*s, c_proj zeros.

        With c_proj zeros, the MoE block contributes 0 to the residual stream at init
        (same as dense MLP), so the model starts in a well-behaved state.
        """
        s = (3.0 ** 0.5) * self.n_embd ** -0.5
        # Router: small init so routing starts near-uniform
        torch.nn.init.uniform_(self.router_weight, -s * 0.02, s * 0.02)
        # Routed experts
        torch.nn.init.uniform_(self.w_fc, -s * 0.4, s * 0.4)
        torch.nn.init.zeros_(self.w_proj)
        # Shared experts
        if self.ws_fc is not None:
            torch.nn.init.uniform_(self.ws_fc, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(self.ws_proj)

    def forward(self, x):
        B, T, D = x.shape
        N = B * T
        k = self.top_k
        E = self.num_experts
        x_flat = x.view(N, D)

        # ---- Shared-expert branch (always active, no routing) ------------------
        shared_out = torch.zeros_like(x_flat)
        if self.ws_fc is not None:
            ws_fc = self.ws_fc.to(x.dtype)
            ws_proj = self.ws_proj.to(x.dtype)
            # Unroll over a small (usually 1) number of shared experts.
            for s in range(self.num_shared_experts):
                h = x_flat @ ws_fc[s]
                h = F.relu(h).square()
                shared_out = shared_out + h @ ws_proj[s]

        # ---- Router ------------------------------------------------------------
        router_logits = F.linear(x_flat, self.router_weight.to(x.dtype))  # (N, E)
        # Aux-loss and top-k probs are computed in fp32 for numerical stability
        routing_probs = F.softmax(router_logits.float(), dim=-1)  # (N, E) fp32
        topk_probs_f, topk_idx = routing_probs.topk(k, dim=-1)    # (N, k)
        topk_probs_f = topk_probs_f / (topk_probs_f.sum(dim=-1, keepdim=True) + 1e-9)
        topk_probs = topk_probs_f.to(x.dtype)  # (N, k) compute dtype for gating math

        # ---- Aux loss (Switch load balancing) ----------------------------------
        # f_e = fraction of tokens routing top-1 to expert e (detached)
        # p_e = mean router softmax prob for expert e (carries gradient)
        # loss = lambda * E * sum_e(f_e * p_e)
        with torch.no_grad():
            one_hot_top1 = F.one_hot(topk_idx[:, 0], num_classes=E).to(routing_probs.dtype)
            f = one_hot_top1.mean(dim=0)  # (E,)
        p = routing_probs.mean(dim=0)     # (E,)
        aux_loss = self.aux_loss_coef * float(E) * (f * p).sum()

        # ---- Padded-capacity dispatch (static shapes) --------------------------
        capacity = max(1, int(math.ceil(self.capacity_factor * k * N / E)))

        flat_idx = topk_idx.reshape(-1)         # (N*k,) expert assignment
        flat_probs = topk_probs.reshape(-1)     # (N*k,) gate weight
        flat_tok = torch.arange(N, device=x.device).repeat_interleave(k)  # (N*k,)

        # Sort by expert id so assignments for the same expert land contiguously.
        sort_order = flat_idx.argsort(stable=True)
        sorted_idx = flat_idx[sort_order]
        sorted_tok = flat_tok[sort_order]
        sorted_probs = flat_probs[sort_order]

        # Compute each assignment's position within its expert's block:
        # counts[e] = #tokens routed to expert e; starts[e] = #tokens routed to experts < e.
        counts = torch.zeros(E, dtype=torch.long, device=x.device)
        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx))
        starts = F.pad(counts[:-1].cumsum(0), (1, 0))                   # (E,)
        pos_within = torch.arange(N * k, device=x.device) - starts[sorted_idx]  # (N*k,)
        keep = (pos_within < capacity).to(x.dtype)                      # 1.0 kept, 0.0 overflow
        pos_clamped = pos_within.clamp(max=capacity - 1)

        # Flat index into (E*capacity, D) dispatch buffer.
        flat_scatter_idx = sorted_idx * capacity + pos_clamped          # (N*k,) in [0, E*cap)

        # Dispatch via scatter_add. Overflow entries contribute 0 (keep=0) so they do not
        # corrupt the already-placed valid token at slot capacity-1.
        expert_inputs_flat = torch.zeros(E * capacity, D, dtype=x.dtype, device=x.device)
        expert_inputs_flat.scatter_add_(
            0,
            flat_scatter_idx.unsqueeze(-1).expand(-1, D),
            x_flat[sorted_tok] * keep.unsqueeze(-1),
        )
        expert_inputs = expert_inputs_flat.view(E, capacity, D)

        # ---- Per-expert FFN (batched matmul) -----------------------------------
        w_fc = self.w_fc.to(x.dtype)      # (E, D, H)
        w_proj = self.w_proj.to(x.dtype)  # (E, H, D)
        hidden = torch.bmm(expert_inputs, w_fc)   # (E, capacity, H)
        hidden = F.relu(hidden).square()
        expert_output = torch.bmm(hidden, w_proj) # (E, capacity, D)

        # ---- Combine: gather expert outputs and weighted-sum back to tokens ----
        expert_output_flat = expert_output.view(E * capacity, D)
        gathered = expert_output_flat[flat_scatter_idx]                      # (N*k, D)
        weighted = gathered * (sorted_probs * keep).unsqueeze(-1)            # zero for overflow

        routed_out = torch.zeros_like(x_flat)
        routed_out.scatter_add_(
            0,
            sorted_tok.unsqueeze(-1).expand(-1, D),
            weighted,
        )

        y = (routed_out + shared_out).view(B, T, D)
        return y, aux_loss
