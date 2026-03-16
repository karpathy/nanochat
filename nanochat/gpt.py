"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration with SDPA fallback
- Sliding window attention
"""

import math
from dataclasses import dataclass, field
from functools import partial

import stk
import stk.ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from megablocks import ops
from megablocks.layers.relu_squared import relu_squared

from cut_cross_entropy import linear_cross_entropy

from nanochat.adamw import DistAdamW
from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.muon import DistMuon, Muon
from nanochat.topology_var import topology_var
from nanochat.flash_attention import flash_attn

# Keys in aux_loss that should be aggregated with max across layers, not mean
_MAX_AGGREGATE_KEYS = frozenset({"router_logits_abs_max", "expert_bias_abs_max"})


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    window_pattern: str = "SSSL"
    use_moe: bool = True
    expert_sizes: list = field(
        default_factory=lambda: [(64, 256)]
    )  # 64 fine-grained experts
    num_active_experts: int = 8
    norm_topk_prob: bool = True
    block_size: int = 128  # Token padding granularity for MoE
    load_balance_loss_weight: float = 0.08
    router_z_loss_weight: float = 0.001
    compute_loss_weight: float = 0.004
    use_bias_balancing: bool = False
    bias_update_speed: float = 0.0005  # λ in SMEBU
    bias_momentum: float = 0.5  # β in SMEBU — smooths updates over time
    bias_kappa: float = 2.0  # κ in SMEBU — tanh saturation speed


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) — FA3's native layout
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for sliding window, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x, None, None


class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = sum(count for count, _ in config.expert_sizes)
        self.num_active_experts = config.num_active_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.block_size = 128  # always will be 128.

        # expert_widths: FFN width for each expert, expanded from config.expert_sizes tuples
        # e.g. config.expert_sizes=[(2, 1024), (1, 512)] -> expert_widths=[1024, 1024, 512]
        self.expert_widths = []
        self.expert_offsets = [0]
        for count, size in config.expert_sizes:
            assert size % 128 == 0, "expert sizes must be divisible by 128"
            for _ in range(count):
                self.expert_widths.append(size)
                self.expert_offsets.append(self.expert_offsets[-1] + size)
        self.total_expert_width = self.expert_offsets[-1]

        self.use_bias_balancing = config.use_bias_balancing
        self.bias_update_speed = config.bias_update_speed
        self.bias_momentum = config.bias_momentum
        self.bias_kappa = config.bias_kappa
        if self.use_bias_balancing:
            self.register_buffer("expert_bias", torch.zeros(self.num_experts))
            # SMEBU momentum buffer — smooths bias updates over time
            self.register_buffer("expert_bias_momentum", torch.zeros(self.num_experts))

        # compute normalized expert widths for aux losses
        mean_expert_width = sum(self.expert_widths) / self.num_experts
        self.register_buffer(
            "expert_widths_normalized",
            torch.tensor(
                [w / mean_expert_width for w in self.expert_widths], dtype=torch.float32
            ),
            persistent=False,
        )

        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)

        self.w1 = nn.Parameter(torch.empty(config.n_embd, self.total_expert_width))
        self.w2 = nn.Parameter(torch.empty(self.total_expert_width, config.n_embd))

        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02, a=-0.06, b=0.06)

        # need this for megablocks ops
        self.sort_end_bit = max(int(math.ceil(math.log2(self.num_experts))), 1)

        self.transpose_sort_end_bit = max(
            int(math.ceil(math.log2(self.num_experts))), 1
        )

        # Register buffers for efficient CUDA kernel access
        self.register_buffer(
            "expert_size_blocks",
            torch.tensor(
                [s // self.block_size for s in self.expert_widths], dtype=torch.int32
            ),
            persistent=False,
        )
        self.register_buffer(
            "expert_block_offsets",
            torch.tensor(
                [o // self.block_size for o in self.expert_offsets], dtype=torch.int32
            ),
            persistent=False,
        )

        # Precompute tensors for vectorized load balance loss computation
        expert_to_group = []
        group_sizes = []
        valid_group_idx = 0
        for count, size in config.expert_sizes:
            for _ in range(count):
                if count > 1:
                    expert_to_group.append(valid_group_idx)
                else:
                    expert_to_group.append(-1)
            if count > 1:
                group_sizes.append(count)
                valid_group_idx += 1

        self._num_valid_groups = len(group_sizes)
        self._all_experts_valid = all(g >= 0 for g in expert_to_group)

        if self._num_valid_groups > 0:
            group_membership = torch.zeros(self.num_experts, self._num_valid_groups)
            for i, g in enumerate(expert_to_group):
                if g >= 0:
                    group_membership[i, g] = 1.0

            self.register_buffer("group_membership", group_membership, persistent=False)
            self.register_buffer(
                "group_sizes",
                torch.tensor(group_sizes, dtype=torch.float32),
                persistent=False,
            )
            valid_indices = [i for i, g in enumerate(expert_to_group) if g >= 0]
            self.register_buffer(
                "valid_expert_indices",
                torch.tensor(valid_indices, dtype=torch.long),
                persistent=False,
            )
        else:
            self.register_buffer("group_membership", torch.empty(0), persistent=False)
            self.register_buffer("group_sizes", torch.tensor([1.0]), persistent=False)
            self.register_buffer(
                "valid_expert_indices",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )

    def _init_buffers(self):
        """Reinitialize non-persistent buffers after to_empty from meta device."""
        device = self.w1.device

        self.expert_size_blocks.copy_(torch.tensor(
            [s // self.block_size for s in self.expert_widths], dtype=torch.int32, device=device
        ))
        self.expert_block_offsets.copy_(torch.tensor(
            [o // self.block_size for o in self.expert_offsets], dtype=torch.int32, device=device
        ))

        mean_expert_width = sum(self.expert_widths) / self.num_experts
        self.expert_widths_normalized.copy_(torch.tensor(
            [w / mean_expert_width for w in self.expert_widths], dtype=torch.float32, device=device
        ))

        if self._num_valid_groups > 0:
            expert_to_group = []
            valid_group_idx = 0
            for count, _ in self.config.expert_sizes:
                for _ in range(count):
                    expert_to_group.append(valid_group_idx if count > 1 else -1)
                if count > 1:
                    valid_group_idx += 1

            group_membership = torch.zeros(
                self.num_experts, self._num_valid_groups, device=device
            )
            for i, g in enumerate(expert_to_group):
                if g >= 0:
                    group_membership[i, g] = 1.0
            self.group_membership.copy_(group_membership)

            group_sizes = [count for count, _ in self.config.expert_sizes if count > 1]
            self.group_sizes.copy_(torch.tensor(group_sizes, dtype=torch.float32, device=device))

            valid_indices = [i for i, g in enumerate(expert_to_group) if g >= 0]
            self.valid_expert_indices.copy_(torch.tensor(valid_indices, dtype=torch.long, device=device))

    # Disable torch.compile tracing for MoE - triton kernels (stk.ops.row_indices etc)
    # can't handle FakeTensors used during compile tracing
    @torch.compiler.disable
    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape

        x_flat = rearrange(
            x, "batch_size seq_len n_embd -> (batch_size seq_len) n_embd "
        )

        router_logits = self.router(x_flat)

        # router_probs = F.softmax(
        #     router_logits, dim=-1, dtype=torch.float32
        # )
        router_probs = F.sigmoid(
            router_logits.to(torch.float32)
        )  # seeing if we really need to cast to float32, guessing probably

        if self.use_bias_balancing:
            selection_scores = router_probs + self.expert_bias.unsqueeze(0)
            _, selected_experts = torch.topk(
                selection_scores, self.num_active_experts, dim=-1
            )
            top_k_weights = router_probs.gather(-1, selected_experts)

        else:
            top_k_weights, selected_experts = torch.topk(
                router_probs, self.num_active_experts, dim=-1
            )
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
        )  # epsilon so we don't divide by 0

        top_k_weights = top_k_weights.to(x.dtype)

        top_k_weights_flat = rearrange(top_k_weights, "... -> (...)")
        selected_experts_flat = rearrange(selected_experts, "... -> (...)")

        bin_ids, indices, tokens_per_expert = self._sort_tokens_by_expert(
            selected_experts_flat
        )

        if self.use_bias_balancing and self.training:
            with torch.no_grad():
                # SMEBU: Soft-clamped Momentum Expert Bias Updates (Trinity, 2025)
                n = tokens_per_expert.float()
                n_bar = n.mean()

                # 1. Normalized violation: how far each expert is from balanced
                #    v_i = (n_bar - n_i) / n_bar
                #    Positive = underloaded (needs more tokens), negative = overloaded
                v = (n_bar - n) / n_bar

                # 2. Soft clamp with tanh: near-balanced experts get tiny updates,
                #    far-from-balanced get updates approaching ±1 (not the full ±1 hammer)
                v_clamped = torch.tanh(self.bias_kappa * v)

                # 3. Scale by learning rate
                delta = self.bias_update_speed * v_clamped

                # 4. Zero-center: subtract mean so updates sum to zero across experts.
                #    This prevents the monotonic drift we were seeing — biases can only
                #    move relative to each other, never all grow together.
                delta = delta - delta.mean()

                # 5. Momentum: smooth out noisy updates over time (like momentum SGD)
                self.expert_bias_momentum.mul_(self.bias_momentum).add_(
                    delta, alpha=1 - self.bias_momentum
                )

                # 6. Apply
                self.expert_bias += self.expert_bias_momentum

        # Compute bins for gather/scatter
        bins = ops.inclusive_cumsum(tokens_per_expert, 0).contiguous()

        # Build topology dynamically each forward (like dMoE)
        padded_bins, topology = self._create_topology(x_flat, tokens_per_expert)

        x_permuted = ops.padded_gather(
            x_flat, indices, bin_ids, bins, padded_bins, self.num_active_experts
        )
        x_permuted = stk.ops.sdd(x_permuted, self.w1, topology)
        x_permuted = relu_squared(x_permuted)
        x_permuted = stk.ops.dsd(x_permuted, self.w2)
        x_permuted = ops.padded_scatter(
            x_permuted,
            indices,
            bin_ids,
            top_k_weights_flat,
            bins,
            padded_bins,
            self.num_active_experts,
        )
        output = rearrange(
            x_permuted,
            "(batch_size seq_len) n_embd -> batch_size seq_len n_embd",
            batch_size=batch_size,
            seq_len=seq_len,
        )

        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        # Batch-level f_i for expert usage logging
        f_i = (tokens_per_expert.float() / tokens_per_expert.sum()).to(x.dtype)

        # Load balance loss: sequence-level when bias balancing, batch-level otherwise
        if self.use_bias_balancing:
            selected_per_seq = selected_experts.view(batch_size, seq_len * self.num_active_experts)
            f_i_seq = torch.zeros(batch_size, self.num_experts, device=x.device, dtype=torch.float32)
            f_i_seq.scatter_add_(1, selected_per_seq.long(),
                                 torch.ones_like(selected_per_seq, dtype=torch.float32))
            f_i_seq = f_i_seq / (seq_len * self.num_active_experts)
            p_i_seq = router_probs.view(batch_size, seq_len, self.num_experts).mean(dim=1)
            load_balance_loss = self._compute_load_balance_loss(f_i_seq, p_i_seq)
        else:
            p_i = router_probs.mean(dim=0)
            load_balance_loss = self._compute_load_balance_loss(f_i, p_i)
        router_probs_flat = rearrange(
            router_probs,
            "(batch_size seq_len) n_embd -> (batch_size seq_len) n_embd",
            batch_size=batch_size,
            seq_len=seq_len,
        )
        compute_loss = (
            router_probs_flat
            @ self.expert_widths_normalized.to(router_probs_flat.dtype)
        ).mean()

        aux_loss = {
            "router_z_loss": router_z_loss,
            "load_balance_loss": load_balance_loss,
            "compute_loss": compute_loss,
            "router_logits_abs_max": router_logits.abs().max().detach(),
            "router_logits_abs_mean": router_logits.abs().mean().detach(),
        }
        if self.use_bias_balancing:
            aux_loss["expert_bias_abs_max"] = self.expert_bias.abs().max()
            aux_loss["expert_bias_abs_mean"] = self.expert_bias.abs().mean()
            aux_loss["expert_bias_vector"] = self.expert_bias.detach().clone()

        return output, aux_loss, f_i

    def _sort_tokens_by_expert(self, selected_experts_flat):
        """Group token assignments by expert id."""

        bin_ids, indices = ops.sort(selected_experts_flat, self.sort_end_bit)
        tokens_per_expert = ops.histogram(selected_experts_flat, self.num_experts)

        return bin_ids, indices, tokens_per_expert

    def _create_topology(self, x, tokens_per_expert):
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.block_size)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = padded_bins.contiguous()

        padded_tokens = padded_bins[-1].clamp_min(self.block_size)

        block_rows = padded_tokens // self.block_size

        # Use variable-size topology with per-expert block counts
        column_indices = topology_var(
            padded_bins,
            self.expert_size_blocks,  # Per-expert block counts
            self.expert_block_offsets,  # Cumulative block offsets
            self.block_size,
            block_rows,
        )

        # Compute all expert token blocks at once
        expert_token_blocks = padded_tokens_per_expert // self.block_size

        # Repeat each expert's size by how many token blocks it handles
        repeated_sizes = torch.repeat_interleave(
            self.expert_size_blocks, expert_token_blocks
        )

        # Cumulative sum gives you offsets
        offsets = torch.cat([repeated_sizes.new_zeros(1), repeated_sizes.cumsum(0)])

        column_indices = column_indices.to(torch.int32)
        offsets = offsets.to(torch.int32)

        shape = (padded_tokens, self.total_expert_width)

        num_blocks = column_indices.numel()
        data_placeholder = torch.empty(
            num_blocks,
            self.block_size,
            self.block_size,
            dtype=x.dtype,
            device="meta",
        )

        row_indices = stk.ops.row_indices(
            shape, data_placeholder, offsets, column_indices
        )
        row_indices = row_indices.to(torch.int32)

        column_indices_t, offsets_t, block_offsets_t = self._sparse_transpose(
            row_indices, column_indices
        )
        column_indices_t = column_indices_t.to(torch.int32)
        offsets_t = offsets_t.to(torch.int32)
        block_offsets_t = block_offsets_t.to(torch.int32)

        topology = stk.Matrix(
            shape,
            data_placeholder,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t,
        )

        return padded_bins, topology

    def _sparse_transpose(self, row_indices, column_indices):
        # Use total_expert_width instead of d_ffn * num_experts
        block_columns = self.total_expert_width // self.block_size

        _, gather_indices = ops.sort(
            column_indices.int(),
            self.transpose_sort_end_bit,
        )

        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        if nnz_per_column.dim() == 0:
            # This addresses an edge case when ffn_hidden_size is equal to self.block_size.
            nnz_per_column = nnz_per_column.unsqueeze(0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    def _gather_tokens(
        self, x, indices, bin_ids, tokens_per_expert, padded_bins, bins=None
    ):
        if bins is None:
            bins = ops.inclusive_cumsum(tokens_per_expert, 0)
            bins = bins.contiguous()

        return ops.padded_gather(
            x, indices, bin_ids, bins, padded_bins, self.num_active_experts
        )

    def _scatter_tokens(
        self, x, indices, bin_ids, weights, tokens_per_expert, padded_bins, bins=None
    ):
        """Un-permute tokens and apply expert weights."""
        if bins is None:
            bins = ops.inclusive_cumsum(tokens_per_expert, 0)
            bins = bins.contiguous()

        return ops.padded_scatter(
            x, indices, bin_ids, weights, bins, padded_bins, self.num_active_experts
        )

    def _compute_load_balance_loss(self, f_i, p_i):
        """Compute load balance loss within expert groups.

        Supports batch-level (f_i, p_i are 1D) and sequence-level (2D: [B, N]).
        """
        if len(set(self.expert_widths)) == 1:
            return self.num_experts * (f_i.float() * p_i.float()).sum(-1).mean()

        if self._num_valid_groups == 0:
            return (f_i.float() * p_i.float()).sum(-1).mean()

        fi_pi = f_i.float() * p_i.float()

        membership = self.group_membership.to(fi_pi.dtype)
        if self._all_experts_valid:
            group_sums = fi_pi @ membership
        else:
            fi_pi_valid = fi_pi[..., self.valid_expert_indices]
            membership_valid = membership[self.valid_expert_indices]
            group_sums = fi_pi_valid @ membership_valid

        group_losses = self.group_sizes.to(fi_pi.dtype) * group_sums
        return group_losses.mean()


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        if config.use_moe:
            self.mlp = MoEMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        mlp_x, aux_loss, f_i = self.mlp(norm(x))
        x = x + mlp_x
        return x, aux_loss, f_i


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        self.rotary_seq_len = (
            config.sequence_len * 10
        )  # 10X over-compute should be enough
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _compute_window_sizes(config):
        """Compute per-layer window sizes from the window_pattern string.
        L = full context (-1, 0), S = half context (seq_len // 2, 0).
        Pattern is tiled across layers, but the last layer always gets L (full context).
        """
        pattern = config.window_pattern.upper()
        half_ctx = config.sequence_len // 2
        window_map = {
            'L': (-1, 0),    # full context
            'S': (half_ctx, 0),  # half context sliding window
        }
        sizes = []
        for i in range(config.n_layer):
            if i == config.n_layer - 1:
                sizes.append((-1, 0))  # last layer always full context
            else:
                char = pattern[i % len(pattern)]
                sizes.append(window_map.get(char, (-1, 0)))
        return sizes

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks (MoE uses trunc_normal init for w2 instead)
        for block in self.transformer.h:
            if hasattr(block.mlp, "c_proj"):
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Reinitialize MoE buffers lost during meta device -> to_empty
        for block in self.transformer.h:
            if hasattr(block.mlp, "_init_buffers"):
                block.mlp._init_buffers()
        # Cast all floating-point params to compute dtype (bf16 on Ampere+)
        # Integer buffers (expert_block_counts etc.) are unaffected
        if self.transformer.wte.weight.device.type == "cuda":
            self.to(dtype=COMPUTE_DTYPE)
            # Keep expert_bias and momentum in float32 for precise updates
            for block in self.transformer.h:
                if hasattr(block.mlp, "expert_bias"):
                    block.mlp.expert_bias.data = block.mlp.expert_bias.data.float()
                if hasattr(block.mlp, "expert_bias_momentum"):
                    block.mlp.expert_bias_momentum.data = block.mlp.expert_bias_momentum.data.float()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311"""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        # For MoE, use active params instead of total
        if self.config.use_moe:
            total_expert_width = sum(
                count * size for count, size in self.config.expert_sizes
            )
            num_experts = sum(count for count, _ in self.config.expert_sizes)
            moe_params = (
                self.config.n_embd * total_expert_width * 2 * self.config.n_layer
            )
            inactive_moe = (
                moe_params
                * (num_experts - self.config.num_active_experts)
                // num_experts
            )
            nparams = nparams - inactive_moe
        l, h, q, t = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(
        self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(
            embedding_params
        ) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(
                f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
            )
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        # Use sharded optimizers only if world_size is power of 2 (divisibility requirement)
        use_sharded = ddp and (world_size & (world_size - 1)) == 0
        AdamWFactory = (
            DistAdamW if use_sharded else partial(torch.optim.AdamW, fused=True)
        )
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if use_sharded else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        )
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (
            self.cos[:, T0 : T0 + T],
            self.sin[:, T0 : T0 + T],
        )  # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)

        combined_aux_loss = None
        aux_loss_count = 0
        expert_usage_sum = None
        expert_usage_count = 0
        expert_usage_per_layer = []
        expert_bias_per_layer = []
        for layer_idx, block in enumerate(self.transformer.h):
            window_size = self.window_sizes[layer_idx]
            x, aux_loss, f_i = block(x, cos_sin, window_size, kv_cache)

            if f_i is not None:
                if expert_usage_sum is None:
                    expert_usage_sum = f_i.clone()
                else:
                    expert_usage_sum += f_i
                expert_usage_count += 1
                expert_usage_per_layer.append(f_i.clone())

            if aux_loss is not None:
                # expert_bias_vector is a raw tensor (not a loss) - collect per-layer, don't average
                if "expert_bias_vector" in aux_loss:
                    expert_bias_per_layer.append(aux_loss["expert_bias_vector"])
                scalar_aux = {
                    k: v for k, v in aux_loss.items() if k != "expert_bias_vector"
                }
                if combined_aux_loss is None:
                    combined_aux_loss = {k: v.clone() for k, v in scalar_aux.items()}
                else:
                    for key in scalar_aux:
                        if key in _MAX_AGGREGATE_KEYS:
                            combined_aux_loss[key] = torch.maximum(
                                combined_aux_loss[key], scalar_aux[key]
                            )
                        else:
                            combined_aux_loss[key] += scalar_aux[key]
                aux_loss_count += 1

        if combined_aux_loss is not None and aux_loss_count > 0:
            for key in combined_aux_loss:
                if key not in _MAX_AGGREGATE_KEYS:
                    combined_aux_loss[key] /= aux_loss_count

        if expert_usage_sum is not None and expert_usage_count > 0:
            avg_expert_usage = expert_usage_sum / expert_usage_count
            if combined_aux_loss is None:
                combined_aux_loss = {}
            combined_aux_loss["expert_usage"] = avg_expert_usage
            combined_aux_loss["expert_usage_per_layer"] = expert_usage_per_layer

        if expert_bias_per_layer:
            if combined_aux_loss is None:
                combined_aux_loss = {}
            combined_aux_loss["expert_bias_per_layer"] = expert_bias_per_layer

        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # CCE fuses lm_head + softcap + cross_entropy without materializing logits
            ce_loss = linear_cross_entropy(
                x, self.lm_head.weight, targets,
                softcap=float(softcap),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            loss = ce_loss
            if combined_aux_loss is not None:
                if self.config.load_balance_loss_weight > 0:
                    loss = loss + self.config.load_balance_loss_weight * combined_aux_loss["load_balance_loss"]
                if self.config.router_z_loss_weight > 0:
                    loss = loss + self.config.router_z_loss_weight * combined_aux_loss["router_z_loss"]
                if self.config.compute_loss_weight > 0 and "compute_loss" in combined_aux_loss:
                    loss = loss + self.config.compute_loss_weight * combined_aux_loss["compute_loss"]
                combined_aux_loss["ce_loss"] = ce_loss

            return None, loss, combined_aux_loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(
                ids
            )  # (B, T, vocab_size) - inference returns just logits
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
