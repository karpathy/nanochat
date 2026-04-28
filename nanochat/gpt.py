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
    # Engram: conditional memory via N-gram hash lookup (DeepSeek Engram paper)
    engram_enabled: bool = False
    engram_ngram_size: int = 3       # max N-gram order (uses 2-gram .. N-gram)
    engram_n_heads: int = 4          # hash heads per N-gram order
    engram_embed_dim: int = 128      # embedding dim per hash head
    engram_table_size: int = 0       # hash table size (0=auto: ~vocab_size*3)
    engram_layer_ids: tuple = ()     # layers to place Engram (empty=auto: (1, n_layer//2))
    engram_kernel_size: int = 4      # depthwise causal conv kernel size


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
    def __init__(self, config, layer_idx, engram=None):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.engram = engram

    def forward(self, x, input_ids, ve, cos_sin, window_size, kv_cache):
        if self.engram is not None:
            x = x + self.engram(x, input_ids, kv_cache)
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


def _next_prime(n):
    """Find the smallest prime >= n."""
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while True:
        if all(n % d for d in range(3, int(n**0.5) + 1, 2)):
            return n
        n += 2


class EngramModule(nn.Module):
    """Conditional memory via N-gram hash lookup (DeepSeek Engram).

    For each N-gram order (2..max_ngram) and each hash head (1..n_heads),
    computes a multiplicative hash over the last n tokens to index into
    prime-sized embedding tables. Retrieved embeddings are projected to
    key/value, gated by hidden state similarity, and refined via depthwise
    causal convolution.

    Known deviations from paper:
    - Token compression (Section 2.2, NFKC/lowercase normalization) is omitted.
      nanochat's 32K BPE vocab is already compact; the hash tables are prime-sized
      and larger than the vocab, so collision rates remain low.
    - Hash function uses polynomial rolling hash over the exact n-gram window
      rather than the demo's multiplicative-XOR accumulation.

    Paper: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs"
    """

    def __init__(self, config, layer_idx, padded_vocab_size):
        super().__init__()
        self.layer_idx = layer_idx
        self.max_ngram = config.engram_ngram_size
        self.vocab_size = padded_vocab_size
        self.n_heads = config.engram_n_heads
        self.embed_dim = config.engram_embed_dim
        self.n_embd = config.n_embd
        n_gram_orders = self.max_ngram - 1  # e.g. max_ngram=3 → orders 2,3
        self.d_mem = n_gram_orders * self.n_heads * self.embed_dim

        base_table_size = config.engram_table_size if config.engram_table_size > 0 else padded_vocab_size * 3
        # Nested nn.ModuleList[nn.ModuleList[Embedding]]: index as [n_idx][k]
        self.embed_tables = nn.ModuleList()
        self._powers = []  # list of LongTensor: precomputed hash powers per (n, k); may be meta
        self._power_configs = []  # list of (prime_size, seed, n) for reinit on real device
        for n in range(2, self.max_ngram + 1):
            tables_for_n = nn.ModuleList()
            for k in range(self.n_heads):
                prime_size = _next_prime(base_table_size + n * k * 997)
                tables_for_n.append(nn.Embedding(prime_size, self.embed_dim))
                seed = n * 2654435761 + k * 40503  # multiplicative hash constants
                powers = torch.tensor([pow(seed, n - j, prime_size) for j in range(n)], dtype=torch.long)
                self._powers.append(powers)
                self._power_configs.append((prime_size, seed, n))
            self.embed_tables.append(tables_for_n)

        self.key_proj = Linear(self.d_mem, config.n_embd, bias=False)
        self.value_proj = Linear(self.d_mem, config.n_embd, bias=False)
        dilation = config.engram_ngram_size  # paper Section 2.3: dilation = max N-gram order
        self.short_conv = nn.Conv1d(
            config.n_embd, config.n_embd,
            kernel_size=config.engram_kernel_size,
            padding=dilation * (config.engram_kernel_size - 1),
            dilation=dilation,
            groups=config.n_embd, bias=False,
        )

    def _ngram_hash(self, input_ids, n, powers, table_size):
        """Vectorized multiplicative hash over the last n tokens only.

        For order n, the hash at position t is:
            h(t) = (ids[t] * seed^1 + ids[t-1] * seed^2 + ... + ids[t-n+1] * seed^n) % table_size
        Positions with fewer than n preceding tokens use a zero-padded window.

        powers: LongTensor of shape (n,) — precomputed powers on the correct device.
        """
        B, T = input_ids.shape
        ids = input_ids.long()
        padded = F.pad(ids, (n - 1, 0), value=0)
        windows = padded.unfold(dimension=1, size=n, step=1)  # (B, T, n)
        h = (windows * powers).sum(dim=-1) % table_size
        return h

    def forward(self, x, input_ids, kv_cache=None):
        B, T, D = x.shape
        embeds = []
        idx = 0
        for n_idx, n in enumerate(range(2, self.max_ngram + 1)):
            for k in range(self.n_heads):
                table = self.embed_tables[n_idx][k]
                table_size = table.num_embeddings
                powers = self._powers[idx].to(input_ids.device)
                if kv_cache is None:
                    hash_idx = self._ngram_hash(input_ids, n, powers, table_size)
                elif T == 1 and kv_cache.ngram_history is not None:
                    # Decode: build n-gram window from cached history + current token
                    history = kv_cache.ngram_history  # (B, max_ngram-1)
                    current = input_ids[:, 0:1]  # (B, 1)
                    # Take last (n-1) tokens from history, append current
                    hist_slice = history[:, -(n - 1):]  # (B, n-1)
                    window = torch.cat([hist_slice, current], dim=1)  # (B, n)
                    hash_idx = (window * powers.unsqueeze(0)).sum(dim=-1, keepdim=True) % table_size  # (B, 1)
                else:
                    hash_idx = self._ngram_hash(input_ids, n, powers, table_size)
                embeds.append(table(hash_idx))
                idx += 1
        # Update n-gram history in KV cache after all orders processed
        if kv_cache is not None:
            max_hist = self.max_ngram - 1
            if T == 1:
                if kv_cache.ngram_history is None:
                    kv_cache.ngram_history = input_ids[:, -1:].expand(B, max_hist).clone()
                else:
                    # Shift left by 1 and append current token
                    new_hist = torch.cat([kv_cache.ngram_history[:, 1:], input_ids[:, -1:]], dim=1)
                    kv_cache.ngram_history = new_hist
            else:
                # Prefill: take last max_hist tokens
                if kv_cache.ngram_history is None:
                    pad_len = max(0, max_hist - T)
                    hist = F.pad(input_ids[:, -max_hist:], (pad_len, 0), value=0)
                    kv_cache.ngram_history = hist
                else:
                    combined = torch.cat([kv_cache.ngram_history, input_ids], dim=1)
                    kv_cache.ngram_history = combined[:, -max_hist:]
        e = torch.cat(embeds, dim=-1)  # (B, T, d_mem)

        k = norm(self.key_proj(e))
        v = self.value_proj(e)
        gate = torch.sigmoid((norm(x) * k).sum(dim=-1, keepdim=True) / (D ** 0.5))
        gated_v = gate * v
        normed_v = norm(gated_v)
        conv_out = self.short_conv(normed_v.transpose(1, 2))[:, :, :T].transpose(1, 2)
        return F.silu(conv_out) + gated_v


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
        # Engram: conditional memory via N-gram hash lookup
        # Resolve engram_layer_ids (including auto-default) before creating VE, so VE excludes Engram layers
        if config.engram_enabled:
            engram_layer_ids = config.engram_layer_ids if config.engram_layer_ids else (1, max(2, config.n_layer // 2 - 1))
            self.engram_layer_ids = engram_layer_ids
        else:
            engram_layer_ids = ()
            self.engram_layer_ids = ()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        # Skip VE for layers that will use Engram instead
        engram_layers = set(engram_layer_ids)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer) and i not in engram_layers})
        # Create Engram modules on the designated layers
        if config.engram_enabled:
            for i in engram_layer_ids:
                self.transformer.h[i] = Block(config, i, engram=EngramModule(config, i, padded_vocab_size))
                # Disable VE gate in Engram layers (ve is set to None, gate would have no gradient)
                self.transformer.h[i].attn.ve_gate = None
        else:
            self.engram_layer_ids = ()
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
            for block in self.transformer.h:
                if block.engram is not None:
                    for tables_n in block.engram.embed_tables:
                        for t in tables_n:
                            t.to(dtype=COMPUTE_DTYPE)
                    block.engram.short_conv.to(dtype=COMPUTE_DTYPE)

        # Engram: init embeddings (uniform), projections (uniform), conv (zeros for identity)
        for block in self.transformer.h:
            if block.engram is not None:
                em = block.engram
                for tables_n in em.embed_tables:
                    for t in tables_n:
                        torch.nn.init.uniform_(t.weight, -s, s)
                torch.nn.init.uniform_(em.key_proj.weight, -s, s)
                torch.nn.init.uniform_(em.value_proj.weight, -s, s)
                torch.nn.init.zeros_(em.short_conv.weight)
                # Recompute _powers on the correct device (needed when __init__ ran on meta device)
                em._powers = [
                    torch.tensor([pow(seed, n - j, prime_size) for j in range(n)], dtype=torch.long, device=em.embed_tables[0][0].weight.device)
                    for prime_size, seed, n in em._power_configs
                ]

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
        engram_embed_numel = 0
        for block in self.transformer.h:
            if block.engram is not None:
                for tables_n in block.engram.embed_tables:
                    for t in tables_n:
                        engram_embed_numel += t.weight.numel()
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel + engram_embed_numel +
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
        # Break down engram params for reporting
        engram_params = 0
        for block in self.transformer.h:
            if block.engram is not None:
                engram_params += sum(p.numel() for p in block.engram.parameters())
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'engram': engram_params,
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
        # Engram: extract embed weights (-> AdamW) and matrix weights (already in matrix_params)
        engram_embed_params = []
        engram_matrix_params_set = set()
        for block in self.transformer.h:
            if block.engram is not None:
                em = block.engram
                for tables_n in em.embed_tables:
                    for t in tables_n:
                        engram_embed_params.append(t.weight)
                engram_matrix_params_set.add(id(em.key_proj.weight))
                engram_matrix_params_set.add(id(em.value_proj.weight))
                engram_matrix_params_set.add(id(em.short_conv.weight))
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
        # Engram embedding tables: AdamW with 5x lr and no weight decay (per paper)
        # Remove engram embeds and non-2D params from matrix_params so they don't end up in Muon groups
        engram_embed_ids = {id(p) for p in engram_embed_params}
        muon_params = [p for p in matrix_params if id(p) not in engram_embed_ids and p.dim() == 2 and p.requires_grad]
        if engram_embed_params:
            param_groups.append(dict(kind='adamw', params=engram_embed_params, lr=embedding_lr * dmodel_lr_scale * 5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.0))
        # Non-2D engram params (conv weights, hash seeds) go to AdamW
        non_matrix_engram = [p for p in matrix_params if id(p) not in engram_embed_ids and id(p) not in {id(mp) for mp in muon_params}]
        if non_matrix_engram:
            param_groups.append(dict(kind='adamw', params=non_matrix_engram, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.0))
        # Muon groups (2D matrix params from blocks, grouped by shape for stacking)
        for shape in sorted({p.shape for p in muon_params}):
            group_params = [p for p in muon_params if p.shape == shape]
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
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if (str(i) in self.value_embeds and i not in self.engram_layer_ids) else None
            x = block(x, idx, ve, cos_sin, self.window_sizes[i], kv_cache)
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
