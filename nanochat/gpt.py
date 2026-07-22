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

from functools import partial
from dataclasses import dataclass

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
    # HC exapnsion rate N：hidden stream的形状从(B,T,D)转变成(B,T,N,D)
    hc_rate: int = 4
    hc_dynamic: bool = True


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


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
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
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


# HC
class HyperConnection(nn.Module):
    """
    hyper-connections over a small stream dimension.
    h has shape (B, T, n_stream, n_embd), matching the paper's (B, L, N, D).
    """
    def __init__(self, dim, rate, layer_id, dynamic=True, device=None):
        super().__init__()
        self.rate = rate
        self.layer_id = layer_id
        self.dynamic = dynamic

        # static_beta 对应论文里的 B，初始化为全 1
        self.static_beta = nn.Parameter(torch.empty(rate, device=device))
        # static_alpha 对应论文里的 WC = (Am, Ar)，形状是 (N, N + 1)
        self.static_alpha = nn.Parameter(torch.empty(rate, rate + 1, device=device))

        # dynamic=True 时才创建动态 B / WC 参数
        if self.dynamic:
            # dynamic_alpha_fn 生成动态 WC，每个 hidden row 预测 N + 1 个连接权重，初始化0
            self.dynamic_alpha_fn = nn.Parameter(torch.empty(dim, rate + 1, device=device))
            # dynamic_alpha_scale 是小尺度可学习因子，对应论文里的 s_alpha，初始化0.01
            self.dynamic_alpha_scale = nn.Parameter(torch.empty(1, device=device))
            # dynamic_beta_fn 生成动态 B，每个 hidden row 预测 1 个 beta，初始化全0
            self.dynamic_beta_fn = nn.Parameter(torch.empty(dim, device=device))
            # dynamic_beta_scale 是小尺度可学习因子，对应论文里的 s_beta，初始化0.01
            self.dynamic_beta_scale = nn.Parameter(torch.empty(1, device=device))

    @torch.no_grad()
    def reset_parameters(self):
        """初始化参数"""
        # 初始化 static B 为 1
        self.static_beta.fill_(1.0)

        # 初始化 static WC
        self.static_alpha.zero_()
        self.static_alpha[self.layer_id % self.rate, 0] = 1.0
        eye = torch.eye(self.rate, device=self.static_alpha.device, dtype=self.static_alpha.dtype)
        self.static_alpha[:, 1:].copy_(eye)

        # 只有 dynamic=True 时初始化动态参数
        if self.dynamic:
            self.dynamic_alpha_fn.zero_()
            self.dynamic_beta_fn.zero_()
            self.dynamic_alpha_scale.fill_(0.01)
            self.dynamic_beta_scale.fill_(0.01)

    def width_connection(self, h):
        # h: hyper hidden matrix，形状 (B, L, N, D)
        if self.dynamic:
            # dynamic HC: norm(H)
            norm_h = norm(h)

            # 动态 WC: tanh(norm(H) @ W_alpha) * s_alpha
            wc_weight = norm_h @ self.dynamic_alpha_fn.to(dtype=norm_h.dtype)
            dynamic_alpha = torch.tanh(wc_weight) * self.dynamic_alpha_scale.to(dtype=norm_h.dtype)
            # 总 WC = dynamic WC + static WC
            # 目的：dynamic 部分一开始初始化为 0，所以模型初始等价于 static HC，而 static HC 又被初始化成类似 Pre-Norm residual 的稳定结构。
            # 让动态调整WC只做微调：dynamic WC 通过 tanh 和很小的 scale，比如 0.01，只是在稳定连接基础上按 token / hidden 状态做动态调整，而不是从训练一开始就完全重写连接关系。
            alpha = dynamic_alpha + self.static_alpha.to(dtype=norm_h.dtype)[None, None, :, :]

            # 动态 B: tanh(norm(H) @ W_beta) * s_beta
            dc_weight = norm_h @ self.dynamic_beta_fn.to(dtype=norm_h.dtype)
            dynamic_beta = torch.tanh(dc_weight) * self.dynamic_beta_scale.to(dtype=norm_h.dtype)
            # 总 B = dynamic B + static B
            beta = dynamic_beta + self.static_beta.to(dtype=norm_h.dtype)[None, None, :]
        else:
            # static HC: 不使用动态调度
            alpha = self.static_alpha.to(dtype=h.dtype)[None, None, :, :]
            beta = self.static_beta.to(dtype=h.dtype)[None, None, :]

        # width connection: (B, L, N + 1, N) @ (B, L, N, D) -> (B, L, N + 1, D)
        # mix_h[..., 0, :] 是当前子层 T 的输入 h0
        # mix_h[..., 1:, :] 是 Ar^T H 得到的 H'
        mix_h = alpha.transpose(-1, -2) @ h
        return mix_h, beta

    def depth_connection(self, mix_h, h_0, beta):
        # h_0: 当前 attention/ffn 子层输出，形状 (B, L, D)
        # beta: 动态 B，形状 (B, L, N)

        # B^T T(h0): 把子层输出按 beta 写回 N 条 hyper hidden
        projected_output = torch.einsum("bld,bln->blnd", h_0, beta)
        # depth connection: B^T T(h0) + H'
        h = projected_output + mix_h[..., 1:, :]

        return h

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

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

        # attention 和 FFN 是两个连续子层，因此分别使用 2*i 和 2*i+1 初始化 HC
        self.attn_hc = HyperConnection(config.n_embd, config.hc_rate, 2 * layer_idx, dynamic=config.hc_dynamic)
        self.mlp_hc = HyperConnection(config.n_embd, config.hc_rate, 2 * layer_idx + 1, dynamic=config.hc_dynamic)


    def forward(self, h, x, ve, cos_sin, window_size, kv_cache):
        # Attention width connection: 从 H 混合出当前 attention 输入 h0 和保留路径 H'
        mix_h, beta = self.attn_hc.width_connection(h)
        # 对 h0 做 Pre-Norm，再送入 self-attention
        x = norm(mix_h[..., 0, :])
        x = self.attn(x, ve, cos_sin, window_size, kv_cache)
        # Attention depth connection: 用动态 beta 把 attention 输出写回 hyper hidden
        h = self.attn_hc.depth_connection(mix_h, x, beta)


        # FFN width connection: 从新的 H 混合出当前 FFN 输入 h0 和保留路径 H'
        mix_h, beta = self.mlp_hc.width_connection(h)
        # 对 h0 做 Pre-Norm，再送入 FFN
        x = norm(mix_h[..., 0, :])
        x = self.mlp(x)
        # FFN depth connection: 用动态 beta 把 FFN 输出写回 hyper hidden
        h = self.mlp_hc.depth_connection(mix_h, x, beta)

        return h


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
        # Separate parameters so they can have different optimizer treatment
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
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
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

            # 真实初始化 DHC 的 static B、static WC、dynamic B、dynamic WC
            block.attn_hc.reset_parameters()
            block.mlp_hc.reset_parameters()

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
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
        HC 版本额外统计 dynamic B / dynamic WC 和 width/depth connection 的开销。
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
                self.transformer.wte.weight.numel()
                + value_embeds_numel
                + self.lm_head.weight.numel()
                + self.smear_gate.weight.numel()
                + self.smear_lambda.numel()
            )        
        
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        d = self.config.n_embd
        n = self.config.hc_rate        # Sum attention FLOPs per layer, accounting for sliding window
        
        # Attention FLOPs 保持原逻辑，只按有效 attention window 统计
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq

        # 每个 Block 有 attention-HC 和 FFN-HC 两个 HC 子层
        hc_sublayers = 2 * self.config.n_layer

        # dynamic WC: H @ W_alpha, 约 N * D * (N + 1)
        # dynamic B: H @ W_beta, 约 N * D
        # width connection: alpha^T @ H, 约 (N + 1) * N * D
        # depth connection: beta * h_0 + H_prime, 约 N * D
        if getattr(self.config, "hc_dynamic", True):
            hc_flops = hc_sublayers * (
                n * d * (n + 1)
                + n * d
                + (n + 1) * n * d
                + n * d
            )
        else:
            hc_flops = hc_sublayers * (
                (n + 1) * n * d
                + n * d
            )
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops + hc_flops
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
        HC 版本额外统计 hyper-connections 参数。
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())

        # Transformer block params
        transformer_matrices = 0
        hc_static = 0
        hc_dynamic = 0

        for block in self.transformer.h:
            # Attention / MLP 原始矩阵参数
            transformer_matrices += sum(p.numel() for p in block.attn.parameters())
            transformer_matrices += sum(p.numel() for p in block.mlp.parameters())

            # HC static params: static_alpha / static_beta
            hc_static += block.attn_hc.static_alpha.numel()
            hc_static += block.attn_hc.static_beta.numel()
            hc_static += block.mlp_hc.static_alpha.numel()
            hc_static += block.mlp_hc.static_beta.numel()

            # HC dynamic params: dynamic_alpha / dynamic_beta
            if getattr(block.attn_hc, "dynamic", False):
                hc_dynamic += block.attn_hc.dynamic_alpha_fn.numel()
                hc_dynamic += block.attn_hc.dynamic_alpha_scale.numel()
                hc_dynamic += block.attn_hc.dynamic_beta_fn.numel()
                hc_dynamic += block.attn_hc.dynamic_beta_scale.numel()

            if getattr(block.mlp_hc, "dynamic", False):
                hc_dynamic += block.mlp_hc.dynamic_alpha_fn.numel()
                hc_dynamic += block.mlp_hc.dynamic_alpha_scale.numel()
                hc_dynamic += block.mlp_hc.dynamic_beta_fn.numel()
                hc_dynamic += block.mlp_hc.dynamic_beta_scale.numel()

        # 其他 scalar / gate 参数
        scalars = self.smear_gate.weight.numel() + self.smear_lambda.numel()

        total = (
                wte
                + value_embeds
                + lm_head
                + transformer_matrices
                + hc_static
                + hc_dynamic
                + scalars
            )
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            "hc_static": hc_static,
            "hc_dynamic": hc_dynamic,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        # 原 Transformer 矩阵参数：只包含 attention / MLP 本体，不包含 HC 参数
        matrix_params = []
        for block in self.transformer.h:
            matrix_params.extend(list(block.attn.parameters()))
            matrix_params.extend(list(block.mlp.parameters()))

        # HC static 参数：static B 和 static WC，论文要求 static component 不使用 weight decay
        hc_static_params = []
        for block in self.transformer.h:
            hc_static_params.extend([
                block.attn_hc.static_alpha,
                block.attn_hc.static_beta,
                block.mlp_hc.static_alpha,
                block.mlp_hc.static_beta,
            ])

        # HC dynamic 参数：dynamic B 和 dynamic WC，论文要求 dynamic component 使用 weight decay
        hc_dynamic_params = []
        for block in self.transformer.h:
            if getattr(block.attn_hc, "dynamic", False):
                hc_dynamic_params.extend([
                    block.attn_hc.dynamic_alpha_fn,
                    block.attn_hc.dynamic_alpha_scale,
                    block.attn_hc.dynamic_beta_fn,
                    block.attn_hc.dynamic_beta_scale,
                ])
            if getattr(block.mlp_hc, "dynamic", False):
                hc_dynamic_params.extend([
                    block.mlp_hc.dynamic_alpha_fn,
                    block.mlp_hc.dynamic_alpha_scale,
                    block.mlp_hc.dynamic_beta_fn,
                    block.mlp_hc.dynamic_beta_scale,
                ])

        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        smear_params = [self.smear_gate.weight, self.smear_lambda]

        # 注意：resid_lambdas / x0_lambdas / backout_lambda 不参与训练，因此这里不加入 optimizer
        optim_params = (
            matrix_params
            + hc_static_params
            + hc_dynamic_params
            + value_embeds_params
            + embedding_params
            + lm_head_params
            + smear_params
        )
        assert len([p for p in self.parameters() if p.requires_grad]) == len(optim_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),

            # HC static B / WC：可训练，但不做 weight decay
            dict(kind='adamw', params=hc_static_params, lr=scalar_lr, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),

            # HC dynamic B / WC：可训练，并使用 weight decay
            dict(kind='adamw', params=hc_dynamic_params, lr=matrix_lr * dmodel_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay),

            # Smear 参数保持独立
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

        # Forward the trunk of the Transformer with Dynamic Hyper-Connections
        rate = self.config.hc_rate
        # H0: 把普通 hidden x 复制成 hyper hidden matrix，形状 (B, T, N, D)
        h = x.unsqueeze(2).expand(-1, -1, rate, -1).contiguous()
        for i, block in enumerate(self.transformer.h):
            # value embedding 仍然按原 nanochat 逻辑提供给 attention
            ve = self.value_embeds[str(i)](idx).to(h.dtype) if str(i) in self.value_embeds else None
            # 每个 Block 内部执行 attention-HC 和 FFN-HC
            h = block(h, ve, cos_sin, self.window_sizes[i], kv_cache)
        # 论文做法：最后把 N 条 hyper hidden row-wise sum，回到普通 hidden
        x = h.sum(dim=2)
        # 最终 norm 后进入 lm_head
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
