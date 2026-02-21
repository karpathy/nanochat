"""
Hybrid HRM-Nanochat model combining:
- L-layers: Local attention (reusing nanochat's Block) for processing all tokens
- H-blocks: Full attention that only processes periodic scratchpad latents for "latent thinking"

The scratchpad latents are raw nn.Parameter vectors (not tokenized), concatenated after L-layers,
enabling long-range reasoning without O(n²) attention on all tokens.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import GPTConfig, Block, MLP, norm, has_ve
from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW


@dataclass
class HybridConfig(GPTConfig):
    """Extends GPTConfig with hybrid-specific parameters."""
    # Hybrid architecture
    n_l_layers: int = 8           # L-layer count (local attention)
    n_h_layers: int = 4           # H-block layer count (full attention)
    local_window_size: int = 64   # L-layer attention window

    # Scratchpad
    chunk_size: int = 16          # Tokens per chunk
    n_scratchpad: int = 2         # Latent vectors per chunk

    def __post_init__(self):
        """Set n_layer to n_l_layers for Block compatibility."""
        # GPTConfig.n_layer is used by Block's has_ve check
        object.__setattr__(self, 'n_layer', self.n_l_layers)


class HBlock(nn.Module):
    """H-block: Full attention that updates ALL positions (chunk + scratchpad).

    All positions attend to [prev_scratchpad | chunk | new_scratchpad] with causal masking.
    This allows chunk tokens to benefit from accumulated scratchpad context.
    NO RoPE applied - position-agnostic attention.
    """

    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.softcap = 15  # match nanochat's softcap

        # Projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, chunk_start_idx: int, chunk_len: int, n_scratchpad: int):
        """
        Args:
            x: [B, T_context, D] = [prev_scratchpad | chunk_tokens | new_scratchpad]
            chunk_start_idx: Start index of chunk tokens in x
            chunk_len: Number of chunk tokens
            n_scratchpad: Number of new scratchpad tokens (at the end)

        All positions that need updating (chunk + new_scratchpad) attend to the
        full context with appropriate causal masking.
        """
        B, T_total, D = x.shape

        # Positions to update: chunk tokens and new scratchpad
        update_start = chunk_start_idx
        update_end = T_total  # chunk + new_scratchpad
        n_update = update_end - update_start

        x_norm = norm(x)

        # Q for positions to update, K/V for all
        x_update = x_norm[:, update_start:update_end]  # [B, n_update, D]

        q = self.q_proj(x_update)  # [B, n_update, D]
        k = self.k_proj(x_norm)    # [B, T_total, n_kv_head * head_dim]
        v = self.v_proj(x_norm)

        # Reshape for attention
        q = q.view(B, n_update, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T_total, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T_total, self.n_kv_head, self.head_dim).transpose(1, 2)

        # QK normalization (always enabled for stability)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Expand k, v for GQA if needed
        if self.n_kv_head < self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        # Build causal mask: query at position i can attend to positions <= i
        query_positions = torch.arange(update_start, update_end, device=x.device)
        attn_mask = self._build_causal_mask(query_positions, T_total, x.device)

        # Attention (SDPA)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        attn = attn.transpose(1, 2).reshape(B, n_update, D)
        attn_out = self.out_proj(attn)

        # Softcap for stability
        attn_out = self.softcap * torch.tanh(attn_out / self.softcap)

        # Update positions with residual
        x_update_new = x[:, update_start:update_end] + attn_out
        x_update_new = x_update_new + self.mlp(norm(x_update_new))

        # Build output: [prev_scratchpad (unchanged) | updated chunk + new_scratchpad]
        if update_start > 0:
            x_out = torch.cat([x[:, :update_start], x_update_new], dim=1)
        else:
            x_out = x_update_new

        return x_out

    def _build_causal_mask(self, query_positions, kv_len, device):
        """Causal mask: query at pos i sees K/V at positions <= i."""
        kv_positions = torch.arange(kv_len, device=device)
        # Shape: [n_query, kv_len] -> [1, 1, n_query, kv_len] for broadcasting
        mask = kv_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        return mask.unsqueeze(0).unsqueeze(0).to(dtype=torch.bool)


class HybridGPT(nn.Module):
    def __init__(self, config: HybridConfig, pad_vocab_size_to=64):
        """
        NOTE: This __init__ function may run in meta device context.
        Therefore, actual initialization happens in init_weights().
        """
        super().__init__()
        self.config = config

        # Pad vocab for efficiency
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size

        # Token embedding
        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)

        # Scratchpad latents (simple nn.Parameter)
        self.scratchpad = nn.Parameter(
            torch.randn(config.n_scratchpad, config.n_embd) * 0.02
        )

        # L-layers: reuse nanochat's Block
        self.l_blocks = nn.ModuleList([
            Block(config, layer_idx=i)
            for i in range(config.n_l_layers)
        ])

        # H-blocks: new (full attention, scratchpad only)
        self.h_blocks = nn.ModuleList([
            HBlock(config, layer_idx=i)
            for i in range(config.n_h_layers)
        ])

        # Output head
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # RoPE buffers for L-layers
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        # Per-layer scalars (match nanochat)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_l_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_l_layers))

        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_l_layers) if has_ve(i, config.n_l_layers)
        })

        # Compute per-layer window sizes for L-layers (all use local window)
        self.window_sizes = [(config.local_window_size, 0)] * config.n_l_layers

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    @torch.no_grad()
    def init_weights(self):
        """Initialize all model parameters."""
        n_embd = self.config.n_embd

        # Embedding and unembedding
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Scratchpad latents
        torch.nn.init.normal_(self.scratchpad, mean=0.0, std=0.02)

        # L-blocks: uniform init
        s = 3**0.5 * n_embd**-0.5
        for block in self.l_blocks:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # H-blocks: uniform init
        for h_block in self.h_blocks:
            torch.nn.init.uniform_(h_block.q_proj.weight, -s, s)
            torch.nn.init.uniform_(h_block.k_proj.weight, -s, s)
            torch.nn.init.uniform_(h_block.v_proj.weight, -s, s)
            torch.nn.init.zeros_(h_block.out_proj.weight)
            torch.nn.init.uniform_(h_block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(h_block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero
        for block in self.l_blocks:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16 if on CUDA
        if self.wte.weight.device.type == "cuda":
            self.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def get_device(self):
        return self.wte.weight.device

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None, loss_reduction='mean'):
        B, T = tokens.shape
        cfg = self.config
        device = tokens.device

        # === Token Embedding ===
        x = self.wte(tokens)  # [B, T, D]
        x = norm(x)
        x0 = x  # Save for x0 residual

        # === L-layers: Full sequence with local attention ===
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for i, block in enumerate(self.l_blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](tokens) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache=None)

        # === H-blocks: Chunk and process scratchpad ===
        accumulated_scratchpad = None
        final_outputs = []

        for chunk_start in range(0, T, cfg.chunk_size):
            chunk_end = min(chunk_start + cfg.chunk_size, T)
            chunk = x[:, chunk_start:chunk_end]
            chunk_len = chunk_end - chunk_start

            # Build context: [prev_scratchpad | chunk | new_scratchpad]
            new_scratch = self.scratchpad.unsqueeze(0).expand(B, -1, -1)

            if accumulated_scratchpad is None:
                h_context = torch.cat([chunk, new_scratch], dim=1)
                chunk_start_idx = 0
            else:
                h_context = torch.cat([accumulated_scratchpad, chunk, new_scratch], dim=1)
                chunk_start_idx = accumulated_scratchpad.shape[1]

            # H-blocks update chunk tokens AND new scratchpad
            for h_block in self.h_blocks:
                h_context = h_block(h_context, chunk_start_idx, chunk_len, cfg.n_scratchpad)

            # Extract results
            if accumulated_scratchpad is None:
                chunk_output = h_context[:, :chunk_len]
                refined_scratch = h_context[:, chunk_len:]
            else:
                n_prev = accumulated_scratchpad.shape[1]
                chunk_output = h_context[:, n_prev:n_prev + chunk_len]
                refined_scratch = h_context[:, n_prev + chunk_len:]

            final_outputs.append(torch.cat([chunk_output, refined_scratch], dim=1))

            # Accumulate scratchpad
            if accumulated_scratchpad is None:
                accumulated_scratchpad = refined_scratch
            else:
                accumulated_scratchpad = torch.cat(
                    [accumulated_scratchpad, refined_scratch], dim=1
                )

        # === Combine chunks ===
        x = torch.cat(final_outputs, dim=1)  # [B, T + n_chunks * n_scratchpad, D]

        # === LM Head ===
        x = norm(x)
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]  # slice to remove padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation

        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            return self._compute_loss(logits, targets, loss_reduction)

        return logits

    def _compute_loss(self, logits, targets, reduction='mean'):
        """Compute loss, inserting -1 at scratchpad positions."""
        cfg = self.config
        B, T_targets = targets.shape

        # Expand targets with -1 at scratchpad positions
        expanded = []
        for chunk_start in range(0, T_targets, cfg.chunk_size):
            chunk_end = min(chunk_start + cfg.chunk_size, T_targets)
            chunk_targets = targets[:, chunk_start:chunk_end]

            # Scratchpad positions get -1 (ignore)
            scratch_ignore = torch.full(
                (B, cfg.n_scratchpad), -1, dtype=targets.dtype, device=targets.device
            )
            expanded.append(chunk_targets)
            expanded.append(scratch_ignore)

        targets_expanded = torch.cat(expanded, dim=1)

        # Verify shape alignment
        assert logits.shape[1] == targets_expanded.shape[1], \
            f"Shape mismatch: logits {logits.shape[1]} vs targets {targets_expanded.shape[1]}"

        # Flatten and compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets_expanded.view(-1),
            ignore_index=-1,
            reduction=reduction
        )

        return loss

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model (forward + backward)."""
        cfg = self.config
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings, scratchpad, and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.wte.weight.numel() +
            self.scratchpad.numel() +
            value_embeds_numel +
            self.resid_lambdas.numel() +
            self.x0_lambdas.numel()
        )

        h, q, t = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.sequence_len

        # L-layer attention FLOPs (local window)
        l_attn_flops = cfg.n_l_layers * 12 * h * q * min(cfg.local_window_size, t)

        # H-block attention FLOPs (full attention, but only n_scratchpad queries per chunk)
        n_chunks = (t + cfg.chunk_size - 1) // cfg.chunk_size
        # Each H-block: n_scratchpad queries attend to growing context
        # Approximate: average context size is ~t/2 + n_chunks*n_scratchpad/2
        avg_context = t // 2 + n_chunks * cfg.n_scratchpad // 2
        h_attn_flops = cfg.n_h_layers * n_chunks * cfg.n_scratchpad * 12 * h * q

        num_flops_per_token = 6 * (nparams - nparams_exclude) + l_attn_flops + h_attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """Return detailed parameter counts for scaling law analysis."""
        wte = sum(p.numel() for p in self.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        l_blocks = sum(p.numel() for p in self.l_blocks.parameters())
        h_blocks = sum(p.numel() for p in self.h_blocks.parameters())
        scratchpad = self.scratchpad.numel()
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + l_blocks + h_blocks + scratchpad + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'l_blocks': l_blocks,
            'h_blocks': h_blocks,
            'scratchpad': scratchpad,
            'scalars': scalars,
            'total': total,
            # For compatibility with base_train scaling calculations
            'transformer_matrices': l_blocks + h_blocks,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        l_block_params = list(self.l_blocks.parameters())
        h_block_params = list(self.h_blocks.parameters())
        matrix_params = l_block_params + h_block_params
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        scratchpad_params = [self.scratchpad]

        total_param_count = (
            len(matrix_params) + len(embedding_params) + len(lm_head_params) +
            len(value_embeds_params) + len(resid_params) + len(x0_params) + len(scratchpad_params)
        )
        assert len(list(self.parameters())) == total_param_count

        # Scale the LR for the AdamW parameters by ∝1/√dmodel
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups
        param_groups = [
            # AdamW groups
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            # Scratchpad gets AdamW with embedding-like LR
            dict(kind='adamw', params=scratchpad_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]

        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer
