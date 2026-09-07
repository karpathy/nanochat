"""
GPT model, written in a functional style.

The model is not an nn.Module. Instead:
- init_params(config, device) returns a flat dict of name -> Tensor
- forward(params, idx, ...) is a free function that loops over the layers and
  indexes the dict by string key
- the GPT class below is a thin, stateful shell around these two functions that
  holds the params dict plus non-parameter constants (config, rotary embeddings),
  giving downstream code (Engine, evals, checkpointing) a familiar object API.

Why functional? All the structure that made the nn.Module tree awkward lives more
naturally in a loop: per-layer heterogeneity (value embeddings on alternating
layers, per-layer window sizes, the backout layer) becomes a one-line condition at
the use site, the layer index is just the loop variable, and there is no meta
device init dance - parameters are allocated once, on the right device, with real
values. Matmul behavior (e.g. FP8) is threaded in as a function argument instead
of by swapping module classes.

Notable model features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
- Value embeddings (ResFormer-style) with input-dependent gates
- per-layer residual scalars, x0 blending, mid-layer backout
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nanochat.common import print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
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
    # Characters: L=long (full context), S=short (short_seq_len window)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    short_seq_len: int = 512 # window of the S layers (absolute, independent of sequence_len)


# Number of leading channels of the (normed) block input that feed the value-embedding gate
VE_GATE_CHANNELS = 12


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

def bf16_matmul(x, w):
    """The default Linear: cast the fp32 master weight to the activation dtype
    (typically bf16), replacing autocast. FP8 training passes fp8_matmul instead."""
    return F.linear(x, w.to(x.dtype))

def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def padded_vocab_size(vocab_size, pad_to=64):
    """Pad vocab for efficiency (tensor cores). Just an optimization - logits are cropped in forward()."""
    padded = ((vocab_size + pad_to - 1) // pad_to) * pad_to
    return padded

def precompute_rotary_embeddings(seq_len, head_dim, base=100000, device=None):
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

def init_buffers(config, device):
    """
    Non-trainable constants needed by forward, as a flat dict: name -> tensor.
    Currently just the rotary embedding tables. They are small/cheap in memory, so
    we over-compute them by 10X the training context (forward asserts if exceeded).
    Unlike params, buffers are derived from the config and are never checkpointed.
    """
    rotary_seq_len = config.sequence_len * 10
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rotary_embeddings(rotary_seq_len, head_dim, device=device)
    buffers = {"cos": cos, "sin": sin}
    return buffers

def compute_window_sizes(config):
    """
    Compute per-layer window sizes for sliding window attention.

    Returns list of (left, right) tuples for FA3's window_size parameter:
    - left: how many tokens before current position to attend to (-1 = unlimited)
    - right: how many tokens after current position to attend to (0 for causal)

    Pattern string is tiled across layers. Final layer always gets L (full context).
    Characters: L=long (full context), S=short (short_seq_len window)
    """
    pattern = config.window_pattern.upper()
    assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
    # Map characters to window sizes
    long_window = config.sequence_len
    short_window = config.short_seq_len
    assert short_window % 128 == 0, f"short_seq_len ({short_window}) must be a multiple of the FA3 tile size (128)"
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


def init_params(config, device):
    """
    Initialize all model parameters and return them as a flat dict: name -> Tensor.
    The names match nn.Module state_dict conventions (e.g. "transformer.h.3.attn.c_q.weight")
    so checkpoints remain interchangeable with the previous module-based implementation.

    wte (embedding):     normal, std=0.8
    lm_head:             normal, std=0.001
    for each block:
        attn.c_q:        uniform, std=1/sqrt(n_embd)
        attn.c_k:        uniform, std=1/sqrt(n_embd)
        attn.c_v:        uniform, std=1/sqrt(n_embd)
        attn.c_proj:     zeros
        mlp.c_fc:        uniform, std=0.4/sqrt(n_embd)
        mlp.c_proj:      zeros
    value_embeds:        normal, std=1.0, output-rms parity with v
    ve_gates:            uniform, small positive so gates start slightly above neutral

    Note: weights use Uniform (bound = sqrt(3) * std, same standard deviation as the
    equivalent Normal) to avoid outliers.
    """
    n_layer = config.n_layer
    n_embd = config.n_embd
    head_dim = n_embd // config.n_head
    kv_dim = config.n_kv_head * head_dim
    padded_vocab = padded_vocab_size(config.vocab_size)
    if padded_vocab != config.vocab_size:
        print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab} for efficiency")
    assert n_embd % config.n_head == 0
    assert config.n_kv_head <= config.n_head and config.n_head % config.n_kv_head == 0

    # All params are created in fp32; embeddings are cast to COMPUTE_DTYPE at the end.
    # Linear weights are stored (out_features, in_features), the nn.Linear convention.
    device = torch.device(device)
    normal = lambda shape, std: torch.empty(shape, dtype=torch.float32, device=device).normal_(0.0, std)
    uniform = lambda shape, lo, hi: torch.empty(shape, dtype=torch.float32, device=device).uniform_(lo, hi)
    zeros = lambda shape: torch.zeros(shape, dtype=torch.float32, device=device)
    s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal

    params = {}
    # Embedding and unembedding
    params["transformer.wte.weight"] = normal((padded_vocab, n_embd), 0.8)
    params["lm_head.weight"] = normal((padded_vocab, n_embd), 0.001)
    # Transformer blocks
    for i in range(n_layer):
        prefix = f"transformer.h.{i}."
        params[prefix + "attn.c_q.weight"] = uniform((n_embd, n_embd), -s, s)
        params[prefix + "attn.c_k.weight"] = uniform((kv_dim, n_embd), -s, s)
        params[prefix + "attn.c_v.weight"] = uniform((kv_dim, n_embd), -s, s)
        params[prefix + "attn.c_proj.weight"] = zeros((n_embd, n_embd)) # projections are zero
        params[prefix + "mlp.c_fc.weight"] = uniform((4 * n_embd, n_embd), -s * 0.4, s * 0.4) # 0.4x init scale for c_fc
        params[prefix + "mlp.c_proj.weight"] = zeros((n_embd, 4 * n_embd))
    # Per-layer scalars
    # resid_lambdas scale the residual stream: stronger at early layers, weaker at deep layers
    # x0_lambdas blend the initial embedding back in: earlier layers get more
    denom = max(n_layer - 1, 1)
    resid_init = [1.15 - (0.10 * i / denom) for i in range(n_layer)]
    x0_init = [0.20 - (0.15 * i / denom) for i in range(n_layer)]
    params["resid_lambdas"] = torch.tensor(resid_init, dtype=torch.float32, device=device)
    params["x0_lambdas"] = torch.tensor(x0_init, dtype=torch.float32, device=device)
    # Backout: subtract cached mid-layer residual before final norm to remove low-level features
    params["backout_lambda"] = torch.full((1,), 0.2, dtype=torch.float32, device=device)
    # Value embeddings (ResFormer-style): alternating layers, last layer always included.
    # N(0,1) entries: a lookup's output rms IS its entry std, so this matches v's rms ~1
    # (the previous "init like c_v" matched element std, which left ve outputs ~sqrt(n_embd)x too small)
    for i in range(n_layer):
        if has_ve(i, n_layer):
            params[f"value_embeds.{i}.weight"] = normal((padded_vocab, kv_dim), 1.0)
    # Gate weights init with small positive values so gates start slightly above neutral
    for i in range(n_layer):
        if has_ve(i, n_layer):
            params[f"transformer.h.{i}.attn.ve_gate.weight"] = uniform((config.n_kv_head, VE_GATE_CHANNELS), 0.0, 0.02)

    # All params are trainable leaves
    for p in params.values():
        p.requires_grad_(True)
    return params


def forward(params, buffers, idx, *, config, targets=None, kv_cache=None, loss_reduction='mean', matmul=bf16_matmul):
    """
    The GPT forward pass, a pure function of (params, buffers, inputs).

    params: flat dict of name -> Tensor (trainable), see init_params
    buffers: flat dict of name -> Tensor (non-trainable constants), see init_buffers
    idx: (B, T) token ids
    matmul: the function used for all the big Linear matmuls (default bf16, or FP8).
            The tiny ve_gate matmul always runs in bf16 (dims not FP8-compatible).

    Returns the loss if targets are given, otherwise the logits.
    """
    B, T = idx.size()
    n_layer = config.n_layer
    n_head = config.n_head
    n_kv_head = config.n_kv_head
    head_dim = config.n_embd // n_head
    window_sizes = compute_window_sizes(config)

    # Grab the rotary embeddings for the current sequence positions
    cos, sin = buffers["cos"], buffers["sin"]
    assert T <= cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {cos.size(1)}"
    assert idx.device == cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {cos.device}"
    # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_t = cos[:, T0:T0+T] # (1, T, 1, head_dim/2)
    sin_t = sin[:, T0:T0+T]

    # Embed the tokens (F.embedding, not wte[idx]: same gather, but a much faster specialized backward)
    x = F.embedding(idx, params["transformer.wte.weight"])
    x = x.to(COMPUTE_DTYPE) # the embeddings are fp32; activations run in the compute dtype
    x = norm(x)

    # Forward the trunk of the Transformer
    x0 = x  # save initial normalized embedding for x0 residual
    resid_lambdas = params["resid_lambdas"]
    x0_lambdas = params["x0_lambdas"]
    backout_layer = n_layer // 2  # cache at halfway point
    x_backout = None
    for i in range(n_layer):
        p = lambda name: params[f"transformer.h.{i}.{name}"]  # noqa: E731
        x = resid_lambdas[i] * x + x0_lambdas[i] * x0

        # Attention block
        xn = norm(x)
        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = matmul(xn, p("attn.c_q.weight")).view(B, T, n_head, head_dim)
        k = matmul(xn, p("attn.c_k.weight")).view(B, T, n_kv_head, head_dim)
        v = matmul(xn, p("attn.c_v.weight")).view(B, T, n_kv_head, head_dim)
        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if f"value_embeds.{i}.weight" in params:
            ve = F.embedding(idx, params[f"value_embeds.{i}.weight"]).to(x.dtype).view(B, T, n_kv_head, head_dim)
            gate = 3 * torch.sigmoid(bf16_matmul(xn[..., :VE_GATE_CHANNELS], p("attn.ve_gate.weight")))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve
        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        q = apply_rotary_emb(q, cos_t, sin_t)
        k = apply_rotary_emb(k, cos_t, sin_t)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2
        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_sizes[i])
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(i)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_sizes[i],
            )
        # Re-assemble the heads and project back to the residual stream
        y = y.contiguous().view(B, T, -1)
        x = x + matmul(y, p("attn.c_proj.weight"))

        # MLP block
        xn = norm(x)
        h = matmul(xn, p("mlp.c_fc.weight"))
        h = F.relu(h).square()
        x = x + matmul(h, p("mlp.c_proj.weight"))

        if i == backout_layer:
            x_backout = x

    # All layers have now written their KV into the cache: advance its position
    if kv_cache is not None:
        kv_cache.advance(T)

    # Subtract mid-layer residual to remove low-level features before logit projection
    if x_backout is not None:
        x = x - params["backout_lambda"].to(x.dtype) * x_backout
    x = norm(x)

    # Forward the lm_head (compute logits)
    softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
    logits = matmul(x, params["lm_head.weight"]) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
    logits = logits[..., :config.vocab_size] # slice to remove padding
    logits = logits.float() # switch to fp32 for logit softcap and loss computation
    logits = softcap * torch.tanh(logits / softcap) # squash the logits

    if targets is not None:
        # training: given the targets, compute and return the loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        return loss
    else:
        # inference: just return the logits directly
        return logits


class GPT:
    """
    Thin, stateful shell around the functional core. NOT an nn.Module - it holds
    the params dict plus the non-parameter constants (config, rotary embeddings)
    and mirrors the parts of the nn.Module API that downstream code relies on
    (forward/__call__, state_dict, parameters, zero_grad, setup_optimizer, ...).

    The shell's forward runs the free function eagerly, which handles varying
    shapes (Engine/KV-cache inference, evals). Training scripts compile this same
    method - fwd = torch.compile(model.forward) - for the fixed-shape hot path:
    same weights, no split model, and the internals (params/buffers/config) stay internal.
    """

    def __init__(self, config, device, params=None):
        self.config = config
        self.window_sizes = compute_window_sizes(config)
        self.buffers = init_buffers(config, device)
        if params is None:
            self.params = init_params(config, device)
        else:
            # Adopt externally provided params (e.g. a loaded checkpoint).
            # Validate the key set against the schema (init_params on meta is free).
            schema = init_params(config, device="meta")
            missing = schema.keys() - params.keys()
            unexpected = params.keys() - schema.keys()
            assert not missing and not unexpected, f"params mismatch. missing: {sorted(missing)}, unexpected: {sorted(unexpected)}"
            for name, p in params.items():
                assert p.shape == schema[name].shape, f"shape mismatch for {name}: {p.shape} != {schema[name].shape}"
            self.params = dict(params)
            for p in self.params.values():
                p.requires_grad_(True)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', matmul=bf16_matmul):
        return forward(self.params, self.buffers, idx, config=self.config,
                       targets=targets, kv_cache=kv_cache, loss_reduction=loss_reduction, matmul=matmul)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_device(self):
        return self.params["transformer.wte.weight"].device

    def parameters(self):
        return list(self.params.values())

    def state_dict(self):
        # a params dict IS a state_dict; return a shallow copy
        return dict(self.params)

    # note: no zero_grad here - use optimizer.zero_grad(), which zeroes the grad
    # stacks without severing the p.grad views (see MuonAdamW in optim.py)

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
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted by name: every
        matmul weight lives under transformer.h.* (block matrices, incl. ve_gate)
        or is the lm_head; embeddings (lookups) and per-layer scalars are not matmuls.
        """
        matmul_params = sum(p.numel() for n, p in self.params.items()
                            if n.startswith("transformer.h.") or n == "lm_head.weight")
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

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
        # Count each group separately (mirrors the grouping in setup_optimizer)
        wte = self.params["transformer.wte.weight"].numel()
        value_embeds = sum(p.numel() for n, p in self.params.items() if n.startswith("value_embeds."))
        lm_head = self.params["lm_head.weight"].numel()
        transformer_matrices = sum(p.numel() for n, p in self.params.items() if n.startswith("transformer.h."))
        scalars = sum(self.params[n].numel() for n in ("resid_lambdas", "x0_lambdas", "backout_lambda"))
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.params.values()), "Parameter count mismatch"
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

        # Separate out all parameters into groups
        matrix_params = [p for n, p in self.params.items() if n.startswith("transformer.h.")]
        value_embeds_params = [p for n, p in self.params.items() if n.startswith("value_embeds.")]
        embedding_params = [self.params["transformer.wte.weight"]]
        lm_head_params = [self.params["lm_head.weight"]]
        resid_params = [self.params["resid_lambdas"]]
        x0_params = [self.params["x0_lambdas"]]
        backout_params = [self.params["backout_lambda"]]
        num_grouped = len(matrix_params) + len(value_embeds_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params) + len(backout_params)
        assert len(self.params) == num_grouped, "some parameters were not assigned to an optimizer group"

        # muP width scaling of the AdamW LRs (LRs are tuned at 768 dim). Per Tensor Programs V
        # (github.com/microsoft/mup): embeddings are unscaled (one-hot rows see no width-summed
        # accumulation) and the unembedding LR is ∝ 1/width (equivalent to MuReadout's 1/width
        # forward multiplier under Adam's per-coord updates). Validated by a d20 exponent sweep.
        width_mult = model_dim / 768
        unemb_lr_scale = 1 / width_mult
        print0(f"Scaling the unembedding LR ∝1/width: {unemb_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * unemb_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=backout_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

