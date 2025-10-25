# Transformer Architecture: The GPT Model

The Transformer is the neural network architecture that powers modern LLMs. nanochat implements a **GPT-style decoder-only Transformer** with several modern improvements.

## High-Level Architecture

```
Input: Token IDs [B, T]
  ↓
Token Embedding [B, T, D]
  ↓
Norm (RMSNorm)
  ↓
Transformer Block 1
  ├── Self-Attention + Residual
  └── MLP + Residual
  ↓
Transformer Block 2
  ...
  ↓
Transformer Block N
  ↓
Norm (RMSNorm)
  ↓
Language Model Head [B, T, V]
  ↓
Output: Logits for next token prediction
```

Where:
- B = Batch size
- T = Sequence length
- D = Model dimension (embedding size)
- V = Vocabulary size
- N = Number of layers

## Model Configuration: `nanochat/gpt.py:26`

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024    # Maximum context length
    vocab_size: int = 50304     # Vocabulary size (padded to multiple of 64)
    n_layer: int = 12          # Number of Transformer blocks
    n_head: int = 6            # Number of query heads
    n_kv_head: int = 6         # Number of key/value heads (MQA)
    n_embd: int = 768          # Model dimension
```

**Design choice:** All sizes are chosen for GPU efficiency (multiples of 64/128).

## The GPT Class

Full implementation: `nanochat/gpt.py:154`

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Core transformer components
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })

        # Language model head (unembedding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Precompute rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10  # Over-allocate
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Cast embeddings to BF16 (saves memory)
        self.transformer.wte.to(dtype=torch.bfloat16)
```

### Key Architectural Choices

1. **No Positional Embeddings**: Uses RoPE (Rotary Position Embeddings) instead
2. **Untied Embeddings**: `wte` (input) and `lm_head` (output) are **separate**
   - Allows different learning rates
   - More parameters but better performance
3. **BFloat16 Embeddings**: Saves memory with minimal quality loss

## Model Initialization: `nanochat/gpt.py:175`

```python
def init_weights(self):
    self.apply(self._init_weights)

    # Zero-initialize output layers (residual path trick)
    torch.nn.init.zeros_(self.lm_head.weight)
    for block in self.transformer.h:
        torch.nn.init.zeros_(block.mlp.c_proj.weight)
        torch.nn.init.zeros_(block.attn.c_proj.weight)

    # Initialize rotary embeddings
    head_dim = self.config.n_embd // self.config.n_head
    cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
    self.cos, self.sin = cos, sin
```

**Residual path trick:** Zero-initialize final layers in residual connections.
- At initialization, blocks are "identity functions"
- Training progressively "turns on" each layer
- Improves training stability

### Weight Initialization: `nanochat/gpt.py:188`

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        # Fan-in aware initialization
        fan_out = module.weight.size(0)
        fan_in = module.weight.size(1)
        std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
```

Uses **fan-in aware initialization** (inspired by Muon paper):
- Scale by $1/\sqrt{\text{fan\_in}}$
- Additional scaling for wide matrices
- Prevents gradient explosion/vanishing

## Rotary Position Embeddings (RoPE): `nanochat/gpt.py:201`

```python
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
    if device is None:
        device = self.transformer.wte.weight.device

    # Frequency for each dimension pair
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # Position indices
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Outer product: (seq_len, head_dim/2)
    freqs = torch.outer(t, inv_freq)

    # Precompute cos and sin
    cos, sin = freqs.cos(), freqs.sin()

    # Cast to BF16 and add batch/head dimensions
    cos, sin = cos.bfloat16(), sin.bfloat16()
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    # Shape: [1, seq_len, 1, head_dim/2]

    return cos, sin
```

**RoPE intuition:**
- Each dimension pair forms a 2D rotation
- Rotation angle depends on position: $\theta_m = m \cdot \theta$
- Relative position $m-n$ encoded in dot product
- Works better than absolute position embeddings for extrapolation

**Application:** See `apply_rotary_emb()` in attention section.

## Forward Pass: `nanochat/gpt.py:259`

```python
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    B, T = idx.size()

    # Get rotary embeddings for current sequence
    assert T <= self.cos.size(1), f"Sequence too long: {T} > {self.cos.size(1)}"
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

    # Token embedding + normalization
    x = self.transformer.wte(idx)  # [B, T, D]
    x = norm(x)                     # RMSNorm

    # Pass through transformer blocks
    for block in self.transformer.h:
        x = block(x, cos_sin, kv_cache)

    # Final normalization
    x = norm(x)

    # Language model head
    softcap = 15
    if targets is not None:
        # Training mode: compute loss
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)  # Softcap
        logits = logits.float()  # Use FP32 for numerical stability
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss
    else:
        # Inference mode: return logits
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        return logits
```

### Logit Softcapping

```python
logits = 15 * torch.tanh(logits / 15)
```

**Why?** Prevents extreme logit values:
- Improves training stability
- Prevents over-confidence
- Used in Gemini models

Without softcapping: logits can be [-100, 200, 50, ...]
With softcapping: logits bounded to roughly [-15, 15]

### Normalization Strategy

nanochat uses **Pre-Norm** architecture:
```
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

**Why Pre-Norm?**
- More stable training
- Can train deeper models
- Gradient flow is smoother

Alternative is **Post-Norm** (used in original Transformer):
```
x = Norm(x + Attention(x))
x = Norm(x + MLP(x))
```

## Transformer Block: `nanochat/gpt.py:142`

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        # Self-attention with residual connection
        x = x + self.attn(norm(x), cos_sin, kv_cache)

        # MLP with residual connection
        x = x + self.mlp(norm(x))

        return x
```

**Two key components:**
1. **Self-Attention**: Allows tokens to communicate
2. **MLP**: Processes each token independently

Both use **residual connections** (the `x +` part).

## Multi-Layer Perceptron (MLP): `nanochat/gpt.py:129`

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)           # [B, T, D] -> [B, T, 4D]
        x = F.relu(x).square()     # ReLU²
        x = self.c_proj(x)         # [B, T, 4D] -> [B, T, D]
        return x
```

**Architecture:**
```
Input [D]
  ↓
Linear (expand 4x) [4D]
  ↓
ReLU² activation
  ↓
Linear (project back) [D]
  ↓
Output [D]
```

### ReLU² Activation

```python
F.relu(x).square()  # max(0, x)²
```

**Why ReLU² instead of GELU?**
- Simpler (no approximations needed)
- Works well for small models
- Slightly faster

Comparison:
- **ReLU**: $\max(0, x)$
- **ReLU²**: $\max(0, x)^2$
- **GELU**: $x \cdot \Phi(x)$ (Gaussian CDF)

For large models, GELU often performs better. For small models, ReLU² is competitive.

### MLP Expansion Ratio

The MLP expands to $4 \times D$ in the hidden layer:
- Original Transformer used $4 \times$
- Some modern models use $\frac{8}{3} \times$ or $3.5 \times$
- nanochat keeps $4 \times$ for simplicity

**Parameter count:** MLP contributes ~$\frac{2}{3}$ of model parameters!

## Model Scaling

nanochat derives model dimensions from **depth**:

```python
# From scripts/base_train.py:74
depth = 20  # User sets this
num_layers = depth
model_dim = depth * 64         # Aspect ratio of 64
num_heads = max(1, (model_dim + 127) // 128)  # Head dim ~128
num_kv_heads = num_heads       # 1:1 MQA ratio
```

**Example scales:**

| Depth | Layers | Dim | Heads | Params | Description |
|-------|--------|-----|-------|--------|-------------|
| 6 | 6 | 384 | 3 | ~8M | Tiny |
| 12 | 12 | 768 | 6 | ~60M | Small |
| 20 | 20 | 1280 | 10 | ~270M | Base ($100) |
| 26 | 26 | 1664 | 13 | ~460M | GPT-2 level |

**Scaling law:** Parameters ≈ $12 \times \text{layers} \times \text{dim}^2$

## FLOPs Estimation: `nanochat/gpt.py:220`

```python
def estimate_flops(self):
    """Estimate FLOPs per token (Kaplan et al. 2020)"""
    nparams = sum(p.numel() for p in self.parameters())
    nparams_embedding = self.transformer.wte.weight.numel()
    l, h, q, t = (self.config.n_layer, self.config.n_head,
                  self.config.n_embd // self.config.n_head,
                  self.config.sequence_len)

    # Forward pass FLOPs
    num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

    return num_flops_per_token
```

**Formula breakdown:**
- $6N$: Linear layers (2 FLOPs per multiply-add, 3 layers per block)
- $12lhqT$: Attention computation

Used for compute budget planning and MFU (Model FLOPs Utilization) tracking.

## Memory and Efficiency

### Mixed Precision Training

```python
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

with autocast_ctx:
    loss = model(x, y)
```

**BFloat16 (BF16)** benefits:
- 2× memory reduction vs FP32
- 2× speedup on modern GPUs (Ampere+)
- Better numerical properties than FP16 (no loss scaling needed)

### Model Compilation

```python
model = torch.compile(model, dynamic=False)
```

PyTorch 2.0+ can compile the model to optimized kernels:
- Fuses operations
- Reduces memory overhead
- ~20-30% speedup

## Parameter Count Breakdown

For a d=20 model (~270M params):

| Component | Params | Fraction |
|-----------|--------|----------|
| Token embeddings | 32K × 1280 = 41M | 15% |
| LM head | 32K × 1280 = 41M | 15% |
| Attention | ~56M | 21% |
| MLP | ~132M | 49% |
| **Total** | **~270M** | **100%** |

**Key insight:** Most parameters are in MLPs and embeddings!

## Inference Generation: `nanochat/gpt.py:294`

```python
@torch.inference_mode()
def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
    """Autoregressive generation"""
    device = self.get_device()
    rng = torch.Generator(device=device).manual_seed(seed) if temperature > 0 else None

    ids = torch.tensor([tokens], dtype=torch.long, device=device)  # [1, T]

    for _ in range(max_tokens):
        # Forward pass
        logits = self.forward(ids)  # [1, T, V]
        logits = logits[:, -1, :]   # Take last token [1, V]

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Sample
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
        else:
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)

        # Append to sequence
        ids = torch.cat((ids, next_ids), dim=1)
        token = next_ids.item()
        yield token
```

**Generation strategies:**
- **Greedy** (temperature=0): Always pick highest probability
- **Sampling** (temperature=1): Sample from distribution
- **Top-k sampling**: Only sample from top k tokens

## Comparison: GPT-2 vs nanochat GPT

| Feature | GPT-2 | nanochat GPT |
|---------|-------|--------------|
| Position encoding | Learned absolute | Rotary (RoPE) |
| Normalization | LayerNorm | RMSNorm (no params) |
| Activation | GELU | ReLU² |
| Embedding | Tied | Untied |
| Attention | Standard | Multi-Query + QK Norm |
| Logits | Raw | Softcapped |
| Bias in linear | Yes | No |

**Result:** nanochat GPT is simpler, faster, and performs better at small scale!

## Next Steps

Now we'll dive deep into the **Attention Mechanism** - the core innovation that makes Transformers work.
