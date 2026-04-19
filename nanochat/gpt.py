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
- Flash Attention 3 integration
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Experimental architecture knobs. Defaults preserve the upstream model.
    architecture: str = "transformer" # transformer|lsrecurrent
    linear_impl: str = "dense" # dense|ls
    ls_num_blocks: int = 16
    ls_rank: int = 128
    lsrec_h_dim: int = 0 # 0 => n_embd
    lsrec_n_iter: int = 4
    lsrec_n_mem: int = 0
    lsrec_log_dt_init: float = -2.3


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


def _find_common_num_blocks(in_f: int, out_f: int, desired: int) -> int:
    nb = max(1, min(int(desired), in_f, out_f))
    while nb > 1 and (in_f % nb != 0 or out_f % nb != 0):
        nb -= 1
    return nb


def _find_dynamic_num_blocks(in_f: int, out_f: int, min_blocks: int, target_block_size: int = 64) -> int:
    widest_shared = max(1, min(in_f, out_f))
    desired = max(int(min_blocks), widest_shared // max(1, int(target_block_size)))
    return _find_common_num_blocks(in_f, out_f, desired)


class LSLinear(nn.Module):
    """Low-rank + block-diagonal linear layer.

    y = blockdiag(W_1, ..., W_k) x + A B x
    The sparse block weights stay as 2D parameters so nanochat's Muon grouping
    can treat them like ordinary matrices.
    """
    def __init__(self, in_features, out_features, num_blocks, rank, bias=False):
        super().__init__()
        assert in_features % num_blocks == 0 and out_features % num_blocks == 0, (
            f"{in_features},{out_features} must divide by num_blocks={num_blocks}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.rank = min(rank, min(in_features, out_features))
        self.block_in = in_features // num_blocks
        self.block_out = out_features // num_blocks
        self.sparse_weight = nn.Parameter(torch.empty(num_blocks * self.block_out, self.block_in))
        self.A = nn.Parameter(torch.empty(out_features, self.rank))
        self.B = nn.Parameter(torch.empty(self.rank, in_features))
        self.register_parameter("bias", nn.Parameter(torch.empty(out_features)) if bias else None)

    @property
    def weight(self):
        # Convenience for code paths that only need shape/device information.
        return self.sparse_weight

    def forward(self, x):
        dtype = x.dtype
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        w = self.sparse_weight.to(dtype=dtype).reshape(self.num_blocks, self.block_out, self.block_in)
        x_b = x_flat.reshape(-1, self.num_blocks, self.block_in).transpose(0, 1)
        y = torch.bmm(w, x_b.transpose(-1, -2)).transpose(-1, -2)
        y = y.transpose(0, 1).reshape(-1, self.out_features)
        y = y + (x_flat @ self.B.to(dtype=dtype).t()) @ self.A.to(dtype=dtype).t()
        if self.bias is not None:
            y = y + self.bias.to(dtype=dtype)
        return y.reshape(*shape[:-1], self.out_features)


class BlockDiagonalLinear(nn.Module):
    """Pure block-diagonal linear layer with a BMM forward path."""
    def __init__(self, in_features, out_features, num_blocks, bias=False):
        super().__init__()
        assert in_features % num_blocks == 0 and out_features % num_blocks == 0, (
            f"{in_features},{out_features} must divide by num_blocks={num_blocks}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.block_in = in_features // num_blocks
        self.block_out = out_features // num_blocks
        self.sparse_weight = nn.Parameter(torch.empty(num_blocks * self.block_out, self.block_in))
        self.register_parameter("bias", nn.Parameter(torch.empty(out_features)) if bias else None)

    def forward(self, x):
        dtype = x.dtype
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        w = self.sparse_weight.to(dtype=dtype).reshape(self.num_blocks, self.block_out, self.block_in)
        x_b = x_flat.reshape(-1, self.num_blocks, self.block_in).transpose(0, 1)
        y = torch.bmm(w, x_b.transpose(-1, -2)).transpose(-1, -2)
        y = y.transpose(0, 1).reshape(-1, self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(dtype=dtype)
        return y.reshape(*shape[:-1], self.out_features)


class LowRankLinear(nn.Module):
    """Low-rank linear layer y = (x B^T) A^T."""
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.A = nn.Parameter(torch.empty(out_features, rank))
        self.B = nn.Parameter(torch.empty(rank, in_features))

    def forward(self, x):
        dtype = x.dtype
        return (x @ self.B.to(dtype=dtype).t()) @ self.A.to(dtype=dtype).t()


def make_blockdiag_linear(in_features, out_features, min_blocks, bias=False):
    nb = _find_dynamic_num_blocks(in_features, out_features, min_blocks)
    return BlockDiagonalLinear(in_features, out_features, nb, bias=bias)


def make_linear(config, in_features, out_features, bias=False, use_ls=True):
    if config.linear_impl == "ls" and use_ls:
        nb = _find_dynamic_num_blocks(in_features, out_features, config.ls_num_blocks)
        rank = min(config.ls_rank, min(in_features, out_features) // 2)
        return LSLinear(in_features, out_features, nb, rank, bias=bias)
    return Linear(in_features, out_features, bias=bias)


def _init_ls_uniform(layer, bound):
    w = layer.sparse_weight.data.reshape(layer.num_blocks, layer.block_out, layer.block_in)
    for i in range(layer.num_blocks):
        torch.nn.init.uniform_(w[i], -bound, bound)
    torch.nn.init.zeros_(layer.A)
    torch.nn.init.kaiming_uniform_(layer.B, a=math.sqrt(5))


def _init_ls_normal(layer, std):
    w = layer.sparse_weight.data.reshape(layer.num_blocks, layer.block_out, layer.block_in)
    for i in range(layer.num_blocks):
        torch.nn.init.normal_(w[i], mean=0.0, std=std)
    torch.nn.init.zeros_(layer.A)
    torch.nn.init.kaiming_uniform_(layer.B, a=math.sqrt(5))


def init_linear(layer, mode, scale):
    if isinstance(layer, LSLinear):
        if mode == "uniform":
            _init_ls_uniform(layer, scale)
        else:
            _init_ls_normal(layer, scale)
    else:
        if mode == "uniform":
            torch.nn.init.uniform_(layer.weight, -scale, scale)
        else:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=scale)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)


def init_blockdiag(layer, mode, scale):
    w = layer.sparse_weight.data.reshape(layer.num_blocks, layer.block_out, layer.block_in)
    for i in range(layer.num_blocks):
        if mode == "uniform":
            torch.nn.init.uniform_(w[i], -scale, scale)
        else:
            torch.nn.init.normal_(w[i], mean=0.0, std=scale)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)


def init_lowrank_zero(layer):
    torch.nn.init.zeros_(layer.A)
    torch.nn.init.kaiming_uniform_(layer.B, a=math.sqrt(5))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

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
        self.c_q = make_linear(config, self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = make_linear(config, self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = make_linear(config, self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = make_linear(config, self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
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
        self.c_fc = make_linear(config, config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = make_linear(config, 4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            init_linear(block.attn.c_q, "uniform", s) # weights use Uniform to avoid outliers
            init_linear(block.attn.c_k, "uniform", s)
            init_linear(block.attn.c_v, "uniform", s)
            init_linear(block.attn.c_proj, "uniform", 0.0) # projections are zero
            init_linear(block.mlp.c_fc, "uniform", s * 0.4)  # 0.4x init scale for c_fc
            init_linear(block.mlp.c_proj, "uniform", 0.0)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
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
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
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
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token


class LSRecurrentScanDrivenBlock(nn.Module):
    """Hybrid LS recurrent block with temporal scan and closed-form depth drive."""
    def __init__(self, config, layer_idx):
        super().__init__()
        h_dim = config.lsrec_h_dim or config.n_embd
        self.h_dim = h_dim
        self.n_embd = config.n_embd
        self.drop_prob = 0.0

        self.b_seq = make_blockdiag_linear(config.n_embd, h_dim, config.ls_num_blocks)
        self.log_A_seq = nn.Parameter(torch.zeros(h_dim))
        self.log_dt_seq = nn.Parameter(torch.full((h_dim,), config.lsrec_log_dt_init))

        self.b_depth = make_blockdiag_linear(config.n_embd, h_dim, config.ls_num_blocks)
        self.n_mem = max(0, min(config.lsrec_n_mem, h_dim))
        self.n_forg = h_dim - self.n_mem
        if self.n_forg > 0:
            self.log_A_depth = nn.Parameter(torch.zeros(self.n_forg))
            self.log_dt_depth = nn.Parameter(torch.full((self.n_forg,), config.lsrec_log_dt_init))

        self.w_post = make_blockdiag_linear(2 * config.n_embd + h_dim, h_dim, config.ls_num_blocks)
        self.w_local = make_blockdiag_linear(h_dim, config.n_embd, config.ls_num_blocks)
        self.w_lowrank = LowRankLinear(h_dim, config.n_embd, config.ls_rank)

    def a_bar_seq(self):
        A = -torch.exp(self.log_A_seq)
        dt = torch.exp(self.log_dt_seq)
        return torch.exp(A * dt)

    def a_bar_depth(self):
        if self.n_forg == 0:
            ref = next(self.parameters())
            return torch.ones(self.h_dim, device=ref.device, dtype=torch.float32)
        A = -torch.exp(self.log_A_depth)
        dt = torch.exp(self.log_dt_depth)
        a = torch.exp(A * dt)
        if self.n_mem == 0:
            return a
        return torch.cat([torch.ones(self.n_mem, device=a.device, dtype=a.dtype), a], dim=0)

    @staticmethod
    def geom_gain(a, K):
        one_minus_a = 1 - a
        safe_denom = one_minus_a.clamp_min(1e-8)
        return torch.where(
            one_minus_a.abs() < 1e-6,
            torch.full_like(a, float(K)),
            (1 - a.pow(K)) / safe_denom,
        )

    _geom_gain = geom_gain

    def causal_scan(self, drive):
        _B, T, _D = drive.shape
        a = self.a_bar_seq().to(device=drive.device, dtype=torch.float32)
        steps = torch.arange(T, device=drive.device, dtype=torch.float32)
        kernel = torch.exp(torch.log(a.clamp_min(1e-6)).unsqueeze(1) * steps.unsqueeze(0))
        n_fft = 1 << ((2 * T - 1).bit_length())
        drive_f = drive.float().transpose(1, 2)
        y = torch.fft.irfft(
            torch.fft.rfft(drive_f, n=n_fft, dim=-1)
            * torch.fft.rfft(kernel, n=n_fft, dim=-1).unsqueeze(0),
            n=n_fft,
            dim=-1,
        )[..., :T]
        return y.transpose(1, 2).to(drive.dtype)

    _causal_scan_conv = causal_scan

    def forward(self, x, active_k):
        K = int(active_k) if active_k is not None else 1
        e = norm(x)
        shifted_e = F.pad(e[:, :-1], (0, 0, 1, 0))
        h_seq = self.causal_scan(F.silu(self.b_seq(e)))

        a_depth = self.a_bar_depth().to(device=x.device, dtype=torch.float32)
        gain = self.geom_gain(a_depth, K).to(x.dtype)
        h_depth = gain * F.silu(self.b_depth(e))

        h = h_seq + h_depth
        r = F.silu(self.w_post(torch.cat([norm(h), e, shifted_e], dim=-1)))
        x = x + self.w_local(r) + self.w_lowrank(r)
        return x


class LSRecurrentScanDrivenGPT(nn.Module):
    """Scan-driven LSRecurrent language model for transformer-replacement experiments."""
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.sequence_len, config.n_embd),
            "h": nn.ModuleList([LSRecurrentScanDrivenBlock(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.active_k = config.lsrec_n_iter

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.wpe.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        out_std = self.config.n_embd ** -0.5
        for block in self.transformer.h:
            init_blockdiag(block.b_seq, "uniform", s)
            init_blockdiag(block.b_depth, "uniform", s)
            init_blockdiag(block.w_post, "uniform", s)
            init_blockdiag(block.w_local, "normal", out_std)
            init_lowrank_zero(block.w_lowrank)
            block.log_A_seq.zero_()
            block.log_dt_seq.fill_(self.config.lsrec_log_dt_init)
            if block.n_forg > 0:
                block.log_A_depth.zero_()
                block.log_dt_depth.fill_(self.config.lsrec_log_dt_init)
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            self.transformer.wpe.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.transformer.wte.weight.numel() + self.transformer.wpe.weight.numel()
        return 6 * (nparams - nparams_exclude)

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        wpe = sum(p.numel() for p in self.transformer.wpe.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        recurrent_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        total = wte + wpe + lm_head + recurrent_matrices
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            "wte": wte,
            "wpe": wpe,
            "value_embeds": 0,
            "lm_head": lm_head,
            "transformer_matrices": recurrent_matrices,
            "scalars": 0,
            "total": total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        recurrent_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in recurrent_params if p.ndim >= 2]
        dynamics_params = [p for p in recurrent_params if p.ndim < 2]
        embedding_params = list(self.transformer.wte.parameters()) + list(self.transformer.wpe.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(dynamics_params) + len(embedding_params) + len(lm_head_params)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=dynamics_params, lr=scalar_lr * 0.0025, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind="muon", params=group_params, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
        optimizer = (DistMuonAdamW if ddp else MuonAdamW)(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        if kv_cache is not None:
            raise NotImplementedError("LSRecurrentGPT does not support KV-cache inference")
        B, T = idx.size()
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.transformer.wte(idx) + self.transformer.wpe(positions)
        x = norm(x.to(COMPUTE_DTYPE))
        for block in self.transformer.h:
            x = block(x, self.active_k)
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids[:, -self.config.sequence_len:])[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()


def build_model_from_config(config, pad_vocab_size_to=64):
    if config.architecture == "transformer":
        return GPT(config, pad_vocab_size_to=pad_vocab_size_to)
    if config.architecture in ("lsrecurrent", "lsrecurrent-scan-driven"):
        if config.linear_impl != "ls":
            raise ValueError("LSRecurrent scan-driven requires linear_impl='ls'")
        return LSRecurrentScanDrivenGPT(config, pad_vocab_size_to=pad_vocab_size_to)
    raise ValueError(f"Unknown architecture: {config.architecture}")
