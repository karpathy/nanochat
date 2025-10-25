# The Attention Mechanism

Attention is the core innovation that makes Transformers powerful. It allows each token to "look at" and aggregate information from other tokens in the sequence.

## Intuition: What is Attention?

Think of attention like a **database query**:
- **Queries (Q)**: "What am I looking for?"
- **Keys (K)**: "What do I contain?"
- **Values (V)**: "What information do I have?"

Each token computes **how much it should attend** to every other token, then aggregates their values.

### Example

Sentence: "The cat sat on the mat"

When processing "sat":
- High attention to "cat" (who is sitting?)
- High attention to "mat" (where sitting?)
- Low attention to "The" (less relevant)

Result: "sat" has context-aware representation incorporating info from "cat" and "mat".

## Scaled Dot-Product Attention

Mathematical formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's break this down step by step.

### Step 1: Compute Attention Scores

$$S = QK^T \in \mathbb{R}^{T \times T}$$

- $Q \in \mathbb{R}^{T \times d_k}$: Query matrix (one row per token)
- $K \in \mathbb{R}^{T \times d_k}$: Key matrix
- $S_{ij}$: similarity between query $i$ and key $j$

**Dot product** measures similarity:
- High dot product → queries and keys are aligned → high attention
- Low dot product → different directions → low attention

### Step 2: Scale

$$S' = \frac{S}{\sqrt{d_k}}$$

**Why divide by $\sqrt{d_k}$?**

For random vectors with dimension $d_k$:
- Dot product has mean 0
- Variance grows as $d_k$
- Scaling keeps variance stable at 1

Without scaling, large $d_k$ causes:
- Very large/small scores
- Softmax saturates (gradients vanish)

### Step 3: Softmax (Normalize to Probabilities)

$$A = \text{softmax}(S') \in \mathbb{R}^{T \times T}$$

Each row becomes a probability distribution:
$$A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=1}^{T} \exp(S'_{ik})}$$

Properties:
- $A_{ij} \geq 0$
- $\sum_j A_{ij} = 1$ (each query's attention sums to 1)

$A_{ij}$ = how much query $i$ attends to key $j$

### Step 4: Weighted Sum of Values

$$\text{Output} = AV \in \mathbb{R}^{T \times d_v}$$

For each token $i$:
$$\text{output}_i = \sum_{j=1}^{T} A_{ij} V_j$$

Aggregate values from all tokens, weighted by attention scores.

## Causal Self-Attention

In language modeling, we can't look at future tokens! We need **causal masking**.

### Masking

Before softmax, add a mask:

$$\text{mask}_{ij} = \begin{cases}
0 & \text{if } i \geq j \text{ (can attend)} \\
-\infty & \text{if } i < j \text{ (future, block)}
\end{cases}$$

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)$$

After softmax, $-\infty$ becomes 0, so future tokens contribute nothing.

### Visualization

```
Attention matrix (T=5):
     k0  k1  k2  k3  k4
q0  [✓  ✗  ✗  ✗  ✗]    ← q0 can only see k0
q1  [✓  ✓  ✗  ✗  ✗]    ← q1 can see k0, k1
q2  [✓  ✓  ✓  ✗  ✗]
q3  [✓  ✓  ✓  ✓  ✗]
q4  [✓  ✓  ✓  ✓  ✓]    ← q4 can see all
```

This creates a **lower triangular** attention pattern.

## Implementation: `nanochat/gpt.py:64`

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        # Projection matrices
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
```

**Key design choices:**
1. **No bias**: Modern practice removes bias from linear layers
2. **Separate K/V heads**: Allows Multi-Query Attention (MQA)
3. **Output projection**: Mix information from all heads

### Forward Pass: `nanochat/gpt.py:79`

```python
def forward(self, x, cos_sin, kv_cache):
    B, T, C = x.size()  # [batch, sequence, channels]

    # 1. Project to queries, keys, values
    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

    # 2. Apply Rotary Position Embeddings
    cos, sin = cos_sin
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

    # 3. QK Normalization (stability)
    q, k = norm(q), norm(k)

    # 4. Rearrange to [B, num_heads, T, head_dim]
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # 5. Handle KV cache (for inference)
    if kv_cache is not None:
        k, v = kv_cache.insert_kv(self.layer_idx, k, v)

    Tq = q.size(2)  # Number of queries
    Tk = k.size(2)  # Number of keys

    # 6. Multi-Query Attention: replicate K/V heads
    nrep = self.n_head // self.n_kv_head
    k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

    # 7. Compute attention
    if kv_cache is None or Tq == Tk:
        # Training: simple causal attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    elif Tq == 1:
        # Inference with single token: attend to all cached tokens
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    else:
        # Inference with multiple tokens: custom masking
        attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
        prefix_len = Tk - Tq
        if prefix_len > 0:
            attn_mask[:, :prefix_len] = True  # Can attend to prefix
        # Causal within new tokens
        attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    # 8. Concatenate heads and project
    y = y.transpose(1, 2).contiguous().view(B, T, -1)
    y = self.c_proj(y)
    return y
```

Let's examine each component in detail.

## 1. Projections to Q, K, V

```python
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
```

**What's happening:**
- Linear projection: $Q = XW^Q$, $K = XW^K$, $V = XW^V$
- Reshape to separate heads
- Each head operates on $d_{head} = d_{model} / n_{heads}$ dimensions

**Example:** $d_{model}=768$, $n_{heads}=6$
- Input: $[B, T, 768]$
- After projection: $[B, T, 6 \times 128]$
- After view: $[B, T, 6, 128]$

## 2. Rotary Position Embeddings (RoPE)

Implementation: `nanochat/gpt.py:41`

```python
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # [B, T, H, D] or [B, H, T, D]
    d = x.shape[3] // 2

    # Split into pairs
    x1, x2 = x[..., :d], x[..., d:]

    # Rotation in 2D
    y1 = x1 * cos + x2 * sin        # Rotate first element
    y2 = x1 * (-sin) + x2 * cos     # Rotate second element

    # Concatenate back
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out
```

**Mathematical formula:**

For a pair of dimensions $(x_1, x_2)$ at position $m$:

$$\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} \cos(m\theta) & \sin(m\theta) \\ -\sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

**Why RoPE is powerful:**

The dot product $Q_m \cdot K_n$ after RoPE only depends on relative position $m-n$:

$$Q_m \cdot K_n = \tilde{Q} \cdot \tilde{K} \cdot e^{i(m-n)\theta}$$

This gives the model a **strong inductive bias** for relative positions.

**Benefits over learned positions:**
- Works for sequence lengths longer than seen during training
- More parameter efficient (no learned position embeddings)
- Better performance on downstream tasks

## 3. QK Normalization

```python
q, k = norm(q), norm(k)
```

**Why normalize Q and K?**

Without normalization, the scale of Q and K can grow during training:
- Large Q/K → large attention scores
- Softmax saturates
- Gradients vanish

Normalization (RMSNorm) keeps scales stable:

$$\text{norm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2}}$$

This is a modern improvement not in original Transformers.

## 4. Multi-Query Attention (MQA)

```python
nrep = self.n_head // self.n_kv_head
k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)
```

**Idea:** Use fewer K/V heads than Q heads.

Standard Multi-Head Attention:
- 6 query heads
- 6 key heads
- 6 value heads

Multi-Query Attention:
- 6 query heads
- 1 key head (replicated 6 times)
- 1 value head (replicated 6 times)

**Benefits:**
- Fewer parameters
- **Much faster inference** (less KV cache memory)
- Minimal quality loss

**Implementation:** `nanochat/gpt.py:52`

```python
def repeat_kv(x, n_rep):
    """Repeat K/V heads to match number of Q heads"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )
```

In nanochat, we use **1:1 MQA** (same number of Q and KV heads) for simplicity. Real MQA would use fewer KV heads.

## 5. Flash Attention

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

PyTorch's `scaled_dot_product_attention` automatically uses **Flash Attention** when available.

**Standard attention:**
1. Compute $S = QK^T$ (materialize $T \times T$ matrix)
2. Apply softmax
3. Compute $SV$

**Memory:** $O(T^2)$ for storing attention matrix

**Flash Attention:**
- Fuses operations
- Tiles computation to fit in SRAM
- Never materializes full attention matrix

**Benefits:**
- $O(T)$ memory instead of $O(T^2)$
- 2-4× faster
- Enables longer context lengths

## 6. KV Cache (for Inference)

During inference, we generate tokens one at a time. **KV cache** avoids recomputing past tokens.

**Without cache:** For each new token, recompute K and V for all previous tokens
- Token 1: compute K,V for 1 token
- Token 2: compute K,V for 2 tokens
- Token 3: compute K,V for 3 tokens
- Total: $1 + 2 + 3 + \ldots + T = O(T^2)$ operations

**With cache:** Store K and V from previous tokens
- Token 1: compute K,V for 1 token, store
- Token 2: compute K,V for 1 NEW token, concatenate with cache
- Token 3: compute K,V for 1 NEW token, concatenate with cache
- Total: $O(T)$ operations

**Speedup:** $T$ times faster!

```python
if kv_cache is not None:
    k, v = kv_cache.insert_kv(self.layer_idx, k, v)
```

The cache stores K and V for all previous tokens and layers.

## Multi-Head Attention Intuition

**Why multiple heads?**

Different heads can learn different attention patterns:
- **Head 1**: Attend to previous word
- **Head 2**: Attend to subject of sentence
- **Head 3**: Attend to syntactically related words
- **Head 4**: Attend to semantically similar words

Each head operates independently, then outputs are concatenated and projected:

```python
y = y.transpose(1, 2).contiguous().view(B, T, -1)  # Concatenate heads
y = self.c_proj(y)  # Final projection
```

## Computational Complexity

For sequence length $T$ and dimension $d$:

| Operation | Complexity |
|-----------|------------|
| Q, K, V projections | $O(T \cdot d^2)$ |
| $QK^T$ | $O(T^2 \cdot d)$ |
| Softmax | $O(T^2)$ |
| Attention × V | $O(T^2 \cdot d)$ |
| Output projection | $O(T \cdot d^2)$ |
| **Total** | $O(T \cdot d^2 + T^2 \cdot d)$ |

For small sequences: $T < d$, so $O(T \cdot d^2)$ dominates
For long sequences: $T > d$, so $O(T^2 \cdot d)$ dominates

**Bottleneck:** Quadratic in sequence length!

This is why context length is expensive.

## Attention Patterns Visualization

Let's visualize what attention learns. Here's a simplified example:

**Sentence:** "The quick brown fox jumps"

```
Attention pattern for "jumps":

       The  quick  brown  fox  jumps
The    0.05  0.05   0.05   0.05  0.0   (can't attend to self)
quick  0.1   0.1    0.1    0.1   0.0
brown  0.05  0.05   0.15   0.15  0.0
fox    0.15  0.05   0.15   0.4   0.0   ← "fox" has high attention
jumps  0.1   0.1    0.1    0.5   0.2   ← we're here
```

"jumps" attends strongly to "fox" (the actor) - this is learned!

## Comparison: Different Attention Variants

| Variant | #Q Heads | #KV Heads | Memory | Speed |
|---------|----------|-----------|--------|-------|
| Multi-Head (MHA) | H | H | High | Baseline |
| Multi-Query (MQA) | H | 1 | Low | Fast |
| Grouped-Query (GQA) | H | H/G | Medium | Fast |

nanochat uses MHA with equal Q/KV heads, but the code supports MQA.

## Common Attention Issues and Solutions

### Problem 1: Attention Collapse
**Symptom:** All tokens attend uniformly to all positions
**Solution:**
- QK normalization
- Proper initialization
- Attention dropout (not used in nanochat)

### Problem 2: Over-attention to Certain Positions
**Symptom:** Strong attention to first/last token regardless of content
**Solution:**
- Better position embeddings (RoPE helps)
- Softcapping logits

### Problem 3: Softmax Saturation
**Symptom:** Gradients vanish, training stalls
**Solution:**
- Scale by $\sqrt{d_k}$
- QK normalization
- Lower learning rate

## Exercises to Understand Attention

1. **Implement attention from scratch:**
```python
def simple_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    return attn @ V
```

2. **Visualize attention patterns** on a real sentence

3. **Compare with and without scaling** by $\sqrt{d_k}$

## Next Steps

Now that we understand attention, we'll explore the **Training Process** - how we actually train these models on massive datasets.
