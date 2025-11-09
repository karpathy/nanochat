# The GPT Architecture: A Deep Dive

This document explains how the GPT (Generative Pre-trained Transformer) model works in nanochat. We'll build your understanding from the ground up, explaining every component in detail.

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Embeddings: From Tokens to Vectors](#embeddings-from-tokens-to-vectors)
3. [Attention: Looking at Previous Words](#attention-looking-at-previous-words)
4. [Rotary Position Embeddings](#rotary-position-embeddings)
5. [Multi-Query Attention (MQA)](#multi-query-attention-mqa)
6. [The MLP (Feed-Forward Network)](#the-mlp-feed-forward-network)
7. [Layer Normalization (RMSNorm)](#layer-normalization-rmsnorm)
8. [The Complete Transformer Block](#the-complete-transformer-block)
9. [The Full GPT Model](#the-full-gpt-model)
10. [Code Walkthrough](#code-walkthrough)

---

## The Big Picture

### What is the GPT Model?

The GPT model is a **function** that takes a sequence of token IDs and outputs probabilities for the next token.

```
Input:  [15496, 995]  (tokens for "Hello world")
        ↓
    [GPT Model]
        ↓
Output: [0.01, 0.02, ..., 0.35, ..., 0.001]  (65,536 probabilities)
         ↑ probability for each possible next token
```

The model is just a **very complex mathematical function** with 561 million to 1.9 billion adjustable parameters (numbers).

### The Transformer Architecture

The GPT model is built using the **Transformer** architecture, which consists of:

```
Input tokens: [1234, 5678, 9012]
    ↓
┌─────────────────────┐
│  Token Embedding    │  Convert token IDs to vectors
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Block 1            │  ┐
│  - Attention        │  │
│  - MLP              │  │  Repeated
└─────────────────────┘  │  N times
┌─────────────────────┐  │  (20-32 layers)
│  Block 2            │  │
│  - Attention        │  │
│  - MLP              │  │
└─────────────────────┘  ┘
    ...
┌─────────────────────┐
│  Language Model Head│  Convert final vectors to logits (scores for each token)
└─────────────────────┘
    ↓
Output logits: [one score per vocab token]
    ↓
Softmax → Probabilities
```

**Key components:**
1. **Embedding layer**: Converts tokens to vectors
2. **Transformer blocks** (repeated N times): Process the vectors
3. **Language model head**: Converts final vectors to predictions

Each **Transformer block** contains:
- **Attention mechanism**: Looks at previous tokens
- **MLP (Multi-Layer Perceptron)**: Processes each position

---

## Embeddings: From Tokens to Vectors

### Why Vectors?

Neural networks process continuous vectors (lists of numbers), not discrete token IDs.

**Token IDs are categorical:**
```
Token 100 is not "less than" token 200
Token IDs have no inherent relationship
```

**Vectors can encode meaning:**
```
"cat"   → [0.2, -0.5, 0.8, 0.3, ...]
"kitten" → [0.3, -0.4, 0.7, 0.4, ...]  # Similar to "cat"!
"dog"   → [0.1, -0.6, 0.9, 0.2, ...]   # Similar to "cat" (both animals)
"car"   → [0.9,  0.7, 0.1, -0.8, ...]  # Very different from "cat"
```

### How Embeddings Work

An **embedding** is a lookup table: token ID → vector

```python
# In nanochat/gpt.py
self.wte = nn.Embedding(vocab_size, n_embd)
# vocab_size = 65,536 (number of tokens)
# n_embd = 512 (size of each vector)
```

This creates a matrix of shape `[65536, 512]`:
- Row 0: vector for token 0
- Row 1: vector for token 1
- ...
- Row 65535: vector for token 65535

**Example:**
```python
# Input: token IDs
idx = torch.tensor([[1234, 5678, 9012]])  # Shape: (batch=1, seq_len=3)

# Embedding lookup
x = embedding(idx)  # Shape: (1, 3, 512)
# x[0, 0, :] = 512-dimensional vector for token 1234
# x[0, 1, :] = 512-dimensional vector for token 5678
# x[0, 2, :] = 512-dimensional vector for token 9012
```

### Learning Embeddings

Initially, embedding vectors are **random**. During training:
- Model tries to predict next tokens
- Backpropagation adjusts embeddings to make predictions better
- Similar words naturally get similar vectors (because they're used in similar contexts)

**Result:** After training, the embedding space captures semantic relationships!

---

## Attention: Looking at Previous Words

### The Core Problem

When predicting the next word, we need to look at **previous context**:

```
"The cat sat on the ___"
```

To predict the blank, we need to consider:
- "cat" → suggests mat, floor, chair (things cats sit on)
- "the" before the blank → suggests a noun
- "sat on" → suggests a physical object

**How do we let the model "look at" previous words?**

### Attention Mechanism: The Intuition

**Attention** allows each token to gather information from previous tokens.

Think of it like this:
```
Token "___" asks questions:
  - "Hey 'cat', do you have relevant info for me?" → Yes! (high attention)
  - "Hey 'the', do you have relevant info for me?" → Not really (low attention)
  - "Hey 'sat', do you have relevant info for me?" → Yes! (high attention)

Token "___" then creates a weighted combination:
  - 50% of information from "cat"
  - 10% from "the"
  - 30% from "sat"
  - 10% from "on"
```

### Attention Mechanism: The Math

Attention uses three transformations:
1. **Query (Q)**: "What am I looking for?"
2. **Key (K)**: "What information do I have?"
3. **Value (V)**: "Here's my actual information"

**Step-by-step:**

```python
# Starting with token vectors x: shape (batch, seq_len, embedding_dim)
x = [[vec1, vec2, vec3, vec4]]  # 4 tokens

# 1. Project to get Q, K, V
Q = x @ W_q  # "What each token is looking for"
K = x @ W_k  # "What each token offers"
V = x @ W_v  # "The actual content of each token"

# 2. Compute attention scores
scores = Q @ K.T  # How much each token should attend to others
# scores[i, j] = similarity between query_i and key_j

# 3. Scale and normalize
scores = scores / sqrt(head_dim)  # Scale to stabilize gradients
attention_weights = softmax(scores)  # Convert to probabilities

# 4. Gather information
output = attention_weights @ V  # Weighted combination of values
```

**Example:**
```
Tokens: ["The", "cat", "sat", "on"]

Attention weights for "on" (what "on" pays attention to):
  "The": 0.1  ─┐
  "cat": 0.3   ├─→ Weighted sum → new vector for "on"
  "sat": 0.5   │   that incorporates context
  "on":  0.1  ─┘
```

### Causal (Autoregressive) Attention

**Important constraint:** Each token can only look at **previous** tokens, not future ones!

```
Allowed:
"The cat sat on" can see: ["The", "cat", "sat"]

Not allowed:
"cat" cannot see "sat" (it's in the future!)
```

**Why?** During training, we predict each next token. If we could see future tokens, the model would cheat!

**Implementation:** Use an attention mask
```python
# Attention scores before masking:
[[q1·k1, q1·k2, q1·k3, q1·k4],
 [q2·k1, q2·k2, q2·k3, q2·k4],
 [q3·k1, q3·k2, q3·k3, q3·k4],
 [q4·k1, q4·k2, q4·k3, q4·k4]]

# Apply causal mask (set future positions to -∞):
[[q1·k1,   -∞,    -∞,    -∞  ],
 [q2·k1, q2·k2,   -∞,    -∞  ],
 [q3·k1, q3·k2, q3·k3,  -∞  ],
 [q4·k1, q4·k2, q4·k3, q4·k4]]

# After softmax, -∞ becomes 0 attention
```

### Multi-Head Attention

Instead of one attention mechanism, we use **multiple heads** (6-8 typically).

**Why?** Different heads can learn different patterns:
- Head 1: Focuses on subjects ("cat")
- Head 2: Focuses on verbs ("sat")
- Head 3: Focuses on most recent word
- etc.

```python
# Each head has its own Q, K, V projections
head1_output = attention(Q1, K1, V1)
head2_output = attention(Q2, K2, V2)
...
head6_output = attention(Q6, K6, V6)

# Concatenate and project back
output = concat([head1_output, head2_output, ..., head6_output]) @ W_o
```

---

## Rotary Position Embeddings

### The Position Problem

Attention is **position-invariant** - it doesn't know the order of tokens!

```
"The cat sat on the mat"
"mat the on sat cat The"
```
Without position information, these look the same to attention (just different weightings).

### Traditional Solution: Position Embeddings

Add a unique vector for each position:
```python
# Position embeddings
pos_emb = [vec_pos0, vec_pos1, vec_pos2, ...]

# Add to token embeddings
x = token_embeddings + pos_emb[:seq_len]
```

**Problem:** Requires learning fixed maximum length. Can't generalize to longer sequences.

### Rotary Position Embeddings (RoPE): The Better Way

Instead of adding position info, **rotate** the Q and K vectors based on position.

**Core idea:** Treat each pair of dimensions as a 2D vector and rotate it:

```python
# For position t, rotation angle is t * θ
# where θ varies per dimension pair

# Original vector (two dimensions)
[x, y]

# Rotated vector
[x * cos(t*θ) - y * sin(t*θ),
 x * sin(t*θ) + y * cos(t*θ)]
```

**Key property:** The dot product of rotated vectors naturally encodes relative positions!

```python
# Query at position i, Key at position j
Q_i @ K_j  =  Q_{rotated by i*θ} @ K_{rotated by j*θ}
           =  (function of i - j)  # Only depends on distance!
```

**Benefits:**
- ✅ Encodes relative positions (more general than absolute)
- ✅ Can extrapolate to longer sequences
- ✅ No extra parameters to learn
- ✅ More stable training

### Implementation in Nanochat

```python
# From nanochat/gpt.py

def _precompute_rotary_embeddings(seq_len, head_dim, base=10000):
    # Compute rotation frequencies for each dimension pair
    channel_range = torch.arange(0, head_dim, 2)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # Compute rotation angles for each position
    t = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)

    # Convert to cos and sin
    cos, sin = freqs.cos(), freqs.sin()
    return cos, sin

def apply_rotary_emb(x, cos, sin):
    # Split vector into pairs
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]

    # Rotate each pair
    y1 = x1 * cos - x2 * sin  # Real part
    y2 = x1 * sin + x2 * cos  # Imaginary part

    return torch.cat([y1, y2], dim=-1)
```

**Usage:**
```python
# In attention forward pass
q = self.c_q(x)  # Query vectors
k = self.c_k(x)  # Key vectors

# Apply rotary embeddings (encodes position)
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)

# Now when we compute q @ k.T, position info is baked in!
```

---

## Multi-Query Attention (MQA)

### The Problem with Standard Attention

In multi-head attention, we compute separate K and V for each head:

```python
# Standard Multi-Head Attention (6 heads example)
Q: (batch, 6 heads, seq_len, 64 dim)  # 6 query heads
K: (batch, 6 heads, seq_len, 64 dim)  # 6 key heads
V: (batch, 6 heads, seq_len, 64 dim)  # 6 value heads
```

**During inference with KV cache:**
- We cache K and V for all previous tokens
- Memory usage: `2 * num_heads * seq_len * head_dim * num_layers`
- For long sequences, this is HUGE!

### Multi-Query Attention (MQA): The Solution

**Key insight:** Share K and V across all heads, keep Q separate per head.

```python
# Multi-Query Attention
Q: (batch, 6 heads, seq_len, 64 dim)  # Still 6 query heads
K: (batch, 1 head,  seq_len, 64 dim)  # Only 1 key head (shared!)
V: (batch, 1 head,  seq_len, 64 dim)  # Only 1 value head (shared!)
```

**Benefits:**
- ✅ **6x less memory** for KV cache
- ✅ **Faster inference** (less data to move)
- ✅ **Similar quality** (minor accuracy drop, if any)

### Implementation in Nanochat

```python
# From nanochat/gpt.py

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        self.n_head = config.n_head       # e.g., 6 query heads
        self.n_kv_head = config.n_kv_head # e.g., 1 key/value head

        # Separate Q projection (for each head)
        self.c_q = nn.Linear(n_embd, n_head * head_dim)

        # Shared K, V projections (fewer heads)
        self.c_k = nn.Linear(n_embd, n_kv_head * head_dim)
        self.c_v = nn.Linear(n_embd, n_kv_head * head_dim)

    def forward(self, x, cos_sin, kv_cache):
        # Project to get Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, head_dim)      # 6 heads
        k = self.c_k(x).view(B, T, self.n_kv_head, head_dim)   # 1 head
        v = self.c_v(x).view(B, T, self.n_kv_head, head_dim)   # 1 head

        # Apply rotary embeddings and normalization
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm for stability

        # Attention computation
        # PyTorch's scaled_dot_product_attention handles broadcasting
        # It will duplicate k and v across query heads automatically
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        return y
```

**What enable_gqa does:** Automatically broadcasts (duplicates) the single K and V head across all 6 Q heads.

---

## The MLP (Feed-Forward Network)

### Purpose

After attention (which mixes information across positions), we need to **process each position independently**.

The **MLP (Multi-Layer Perceptron)** applies the same transformation to each position:

```
Position 1: [vec] → MLP → [new_vec]
Position 2: [vec] → MLP → [new_vec]  (same MLP)
Position 3: [vec] → MLP → [new_vec]  (same MLP)
```

### Architecture

Standard Transformer MLP:
```
Input (512 dim)
    ↓
Linear layer: 512 → 2048  (expand 4x)
    ↓
Activation function (nonlinearity)
    ↓
Linear layer: 2048 → 512  (contract back)
    ↓
Output (512 dim)
```

**Why expand then contract?**
- Expansion creates a higher-dimensional space
- More capacity to learn complex transformations
- Contraction projects back to residual stream

### Activation Functions

**Activation functions** introduce nonlinearity (without them, stacking layers is pointless).

Common choices:
- **ReLU**: `max(0, x)` - simple, works well
- **GELU**: Smooth approximation of ReLU - used in GPT-2/3
- **ReLU²**: `(max(0, x))²` - used in nanochat!

**Why ReLU²?**
```python
x = relu(x).square()  # (max(0, x))²
```
- Smooth gradient (unlike ReLU which has sharp corner at 0)
- Empirically works well for LLMs
- Simple to implement

### Implementation in Nanochat

```python
# From nanochat/gpt.py

class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        # n_embd = 512 → 4 * 512 = 2048 (expand)

        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        # 2048 → 512 (contract back)

    def forward(self, x):
        x = self.c_fc(x)          # (B, T, 512) → (B, T, 2048)
        x = F.relu(x).square()    # ReLU² activation
        x = self.c_proj(x)        # (B, T, 2048) → (B, T, 512)
        return x
```

**Note:** No biases in linear layers! This is a design choice that slightly reduces parameters and can improve training dynamics.

---

## Layer Normalization (RMSNorm)

### Why Normalize?

During training, activations can grow or shrink, making training unstable.

**Layer normalization** re-scales activations to have consistent statistics:
```
Before: [100, 200, 150, 175]  (large, varied)
After:  [0.5, 1.2, 0.8, 0.95] (normalized)
```

### Standard LayerNorm

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
x_normalized = (x - mean) / sqrt(var + eps)

# Learnable scale and shift
output = gamma * x_normalized + beta
```

**Components:**
- Subtract mean (center)
- Divide by std deviation (scale)
- Learnable gamma and beta (affine transform)

### RMSNorm: Simplified Normalization

**Nanochat uses RMSNorm** - a simpler variant:

```python
# Compute RMS (Root Mean Square)
rms = sqrt(mean(x²))

# Normalize
x_normalized = x / rms
```

**Differences from LayerNorm:**
- ✅ No mean subtraction (just scaling)
- ✅ No learnable parameters (gamma, beta)
- ✅ Faster to compute
- ✅ Works just as well in practice!

### Implementation in Nanochat

```python
# From nanochat/gpt.py

def norm(x):
    """Purely functional RMSNorm with no learnable params"""
    return F.rms_norm(x, (x.size(-1),))
```

That's it! Super simple. PyTorch's `F.rms_norm` does:
```python
# Equivalent to:
rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
return x / rms
```

---

## The Complete Transformer Block

A **Transformer block** combines attention and MLP with **residual connections** and normalization.

### The Architecture

```python
# From nanochat/gpt.py

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        # Attention with residual connection
        x = x + self.attn(norm(x), cos_sin, kv_cache)

        # MLP with residual connection
        x = x + self.mlp(norm(x))

        return x
```

### Residual Connections

**Residual connection:** Add the input back to the output.

```
x_new = x_old + transformation(x_old)
```

**Why?**
- **Gradient flow:** Gradients can flow directly through the `+` during backpropagation
- **Preserves information:** Original information always available
- **Enables deep networks:** Can stack 20-100+ layers without vanishing gradients

**Visual representation:**
```
x ─────────────────┬───→ (output)
    │              ↑
    ↓             add
  norm
    ↓
 attention ───────┘
```

### Pre-Norm vs Post-Norm

**Nanochat uses pre-norm** (normalize before attention/MLP):

```python
x = x + attention(norm(x))  # Normalize BEFORE attention
x = x + mlp(norm(x))        # Normalize BEFORE MLP
```

**Alternative: Post-norm** (normalize after):
```python
x = norm(x + attention(x))  # Normalize AFTER adding
x = norm(x + mlp(x))
```

**Why pre-norm?**
- ✅ More stable training
- ✅ Can use higher learning rates
- ✅ Standard in modern LLMs

---

## The Full GPT Model

### Complete Architecture

```python
# From nanochat/gpt.py

class GPT(nn.Module):
    def __init__(self, config):
        # Token embeddings: token ID → vector
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })

        # Language model head: vector → logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Precomputed rotary embeddings
        cos, sin = self._precompute_rotary_embeddings(...)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.size()  # batch size, sequence length

        # 1. Get rotary embeddings for current sequence
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # 2. Embed tokens
        x = self.transformer.wte(idx)  # (B, T, n_embd)
        x = norm(x)  # Normalize embeddings

        # 3. Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)

        # 4. Final normalization
        x = norm(x)

        # 5. Compute logits (scores for each possible next token)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        logits = 15 * torch.tanh(logits / 15)  # Logit softcapping

        # 6. If training, compute loss
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size),
                                   targets.view(-1))
            return loss
        else:
            return logits
```

### Data Flow Example

Let's trace what happens to the input `[1234, 5678, 9012]`:

```
Step 1: Token Embedding
[1234, 5678, 9012] → [[vec1], [vec2], [vec3]]
Shape: (1, 3) → (1, 3, 512)

Step 2: Normalize embeddings
[[vec1], [vec2], [vec3]] → [[norm_vec1], [norm_vec2], [norm_vec3]]

Step 3: Block 1
├─ Attention: Each token looks at previous tokens
│  [norm_vec1], [norm_vec2], [norm_vec3] → [att1], [att2], [att3]
├─ Add residual
│  [vec1 + att1], [vec2 + att2], [vec3 + att3]
├─ MLP: Process each position
│  [mlp1], [mlp2], [mlp3]
└─ Add residual
   [vec1 + att1 + mlp1], [vec2 + att2 + mlp2], [vec3 + att3 + mlp3]

Step 4: Block 2
[Same as Block 1, with different learned parameters]

... (repeat for 20-32 blocks total)

Step 20: Final norm
[[final_vec1], [final_vec2], [final_vec3]]

Step 21: Language model head
Convert each vector to logits (scores for 65,536 possible tokens)
[[65536 scores], [65536 scores], [65536 scores]]

Step 22: During inference, we only care about the last position
[65536 scores for position 3]
    ↓
Apply softmax
    ↓
Probabilities for next token!
```

### Model Configuration

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024    # Maximum sequence length
    vocab_size: int = 65536     # Number of tokens (2^16)
    n_layer: int = 20           # Number of transformer blocks
    n_head: int = 6             # Number of attention heads (queries)
    n_kv_head: int = 1          # Number of KV heads (MQA)
    n_embd: int = 512           # Embedding dimension
```

**Different model sizes:**
- **d20 (base)**: 20 layers, 512 dim → 561M parameters
- **d26**: 26 layers → ~1B parameters
- **d32 (large)**: 32 layers → 1.9B parameters

---

## Code Walkthrough

### File: `nanochat/gpt.py`

Let's walk through the key sections:

#### 1. Configuration (lines 26-33)

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
```

This defines all the hyperparameters for the model.

#### 2. RMSNorm (lines 36-38)

```python
def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))
```

Simple normalization function used throughout.

#### 3. Rotary Embeddings (lines 41-49, 186-200)

```python
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # (batch, heads, seq_len, head_dim)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], 3)

def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
    # Compute rotation frequencies
    channel_range = torch.arange(0, head_dim, 2)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # Rotation angles for each position
    t = torch.arange(seq_len)
    freqs = torch.outer(t, inv_freq)

    # Convert to cos/sin
    cos, sin = freqs.cos(), freqs.sin()
    return cos, sin
```

#### 4. Attention (lines 51-110)

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # Projections for Q, K, V
        self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)

        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        # Project to Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, head_dim)

        # Apply rotary embeddings
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Normalize queries and keys
        q, k = norm(q), norm(k)

        # Transpose for attention computation
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Handle KV cache (if using)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        # Compute attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
```

#### 5. MLP (lines 113-123)

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)  # Expand
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)  # Contract

    def forward(self, x):
        x = self.c_fc(x)          # Expand
        x = F.relu(x).square()    # ReLU² activation
        x = self.c_proj(x)        # Contract
        return x
```

#### 6. Transformer Block (lines 126-135)

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)  # Attention + residual
        x = x + self.mlp(norm(x))                      # MLP + residual
        return x
```

#### 7. Full GPT Model (lines 138-276)

```python
class GPT(nn.Module):
    def __init__(self, config):
        # Token embeddings
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(vocab_size, n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(n_layer)]),
        })

        # Output head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Rotary embeddings
        cos, sin = self._precompute_rotary_embeddings(...)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, idx, targets=None, kv_cache=None):
        # Get rotary embeddings for current position
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Embed tokens and normalize
        x = norm(self.transformer.wte(idx))

        # Pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)

        # Final norm
        x = norm(x)

        # Compute logits
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)  # Softcap

        # Compute loss if training
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size),
                                   targets.view(-1))
            return loss
        return logits
```

---

## Key Takeaways

1. **Embeddings** convert discrete tokens to continuous vectors

2. **Attention** allows tokens to gather information from previous tokens

3. **Rotary embeddings** encode position by rotating Q and K vectors

4. **Multi-Query Attention** shares K and V across heads for efficiency

5. **MLP** processes each position independently with nonlinear transformations

6. **RMSNorm** normalizes activations for stable training

7. **Residual connections** enable gradient flow in deep networks

8. **Transformer blocks** stack attention + MLP with residuals

9. **The full GPT model** is just: embedding → N blocks → output projection

10. **All parameters are learned** during training through backpropagation

---

## What's Next?

Now that you understand the architecture, let's see how it's trained!

**→ Next: [Document 4: The Training Pipeline](04_training.md)**

You'll learn:
- How the model learns from data
- The complete training pipeline
- Distributed training on multiple GPUs
- Fine-tuning for chat

---

## Self-Check

Before moving on, make sure you understand:

- [ ] What embeddings are and why we need them
- [ ] How attention lets tokens look at previous context
- [ ] Why we use causal (autoregressive) masking
- [ ] What rotary embeddings do (encode position)
- [ ] How Multi-Query Attention reduces memory usage
- [ ] What the MLP does (process each position independently)
- [ ] Why we use residual connections
- [ ] The overall flow: tokens → embeddings → blocks → logits
- [ ] Where the model code is (`nanochat/gpt.py`)
