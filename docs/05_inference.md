# Inference: How Text Generation Works

This document explains how the trained GPT model generates text, from single token prediction to complete conversations. We'll cover the inference engine, sampling strategies, and optimizations.

## Table of Contents
1. [Autoregressive Generation](#autoregressive-generation)
2. [Sampling Strategies](#sampling-strategies)
3. [KV Cache Optimization](#kv-cache-optimization)
4. [The Conversation Format](#the-conversation-format)
5. [The Engine Architecture](#the-engine-architecture)
6. [Code Walkthrough](#code-walkthrough)

---

## Autoregressive Generation

### What is Autoregressive Generation?

**Autoregressive** means generating one token at a time, where each new token depends on all previous tokens.

```
Step 1: Input: "The cat"         → Model outputs: "sat" (probability 0.8)
Step 2: Input: "The cat sat"     → Model outputs: "on"  (probability 0.6)
Step 3: Input: "The cat sat on"  → Model outputs: "the" (probability 0.9)
Step 4: Input: "The cat sat on the" → Model outputs: "mat" (probability 0.7)
```

**Key point:** Each step requires running the full model again with the expanded input!

### The Generation Loop

```python
# Simplified generation
def generate(model, prompt_tokens, max_tokens):
    tokens = prompt_tokens.copy()

    for i in range(max_tokens):
        # Run model on all tokens so far
        logits = model(tokens)  # Shape: (seq_len, vocab_size)

        # Get logits for last position (what comes next?)
        next_token_logits = logits[-1]  # Shape: (vocab_size,)

        # Convert to probabilities
        probs = softmax(next_token_logits)

        # Sample next token
        next_token = sample(probs)

        # Add to sequence
        tokens.append(next_token)

        # Stop if we generate a special end token
        if next_token == END_TOKEN:
            break

    return tokens
```

### Why Autoregressive?

**Why not generate all tokens at once?**

Because language is **sequential** - the meaning and grammar depend on what came before:

```
"The dog chased the ___"
→ Likely: "cat", "ball", "squirrel"
→ Unlikely: "ate", "is", "the"

"The dog ate the ___"
→ Likely: "food", "bone", "treat"
→ Unlikely: "chased", "ran"
```

The model needs previous context to make good predictions.

---

## Sampling Strategies

### Greedy Sampling (Temperature = 0)

**Always pick the most likely token**:

```python
# Always select argmax
next_token = argmax(probabilities)
```

**Example:**
```
Probabilities: {"the": 0.4, "a": 0.3, "mat": 0.2, "floor": 0.1}
Greedy selects: "the" (highest probability)
```

**Pros:**
- ✅ Deterministic (same output every time)
- ✅ "Safe" choices

**Cons:**
- ❌ Repetitive and boring
- ❌ Gets stuck in loops ("very very very very...")
- ❌ No creativity

### Temperature Sampling

**Temperature** controls randomness by scaling probabilities:

```python
# Before sampling, scale logits by temperature
logits = logits / temperature
probs = softmax(logits)
next_token = sample(probs)
```

**Temperature effects:**
- **T = 0**: Greedy (deterministic)
- **T < 1** (e.g., 0.7): More focused, less random
- **T = 1**: Unchanged probabilities
- **T > 1** (e.g., 1.5): More random, more creative

**Example (temperature = 0.7):**
```
Original probs: {"the": 0.4, "a": 0.3, "mat": 0.2, "floor": 0.1}
After temp scaling:
  logits = log(probs) / 0.7
  new_probs = {"the": 0.5, "a": 0.3, "mat": 0.15, "floor": 0.05}
  (More peaked - "the" even more likely)

After temp = 1.5:
  new_probs = {"the": 0.35, "a": 0.28, "mat": 0.22, "floor": 0.15}
  (Flatter - more uniform)
```

**Use cases:**
- **Low temp (0.5-0.7)**: Factual Q&A, code generation
- **Medium temp (0.9-1.0)**: Conversation, general use
- **High temp (1.2-1.5)**: Creative writing, brainstorming

### Top-K Sampling

**Only sample from the top K most likely tokens**:

```python
# Get top-k tokens
top_k_values, top_k_indices = torch.topk(logits, k=top_k)

# Zero out everything else
logits[logits < top_k_values[-1]] = -inf

# Sample from filtered distribution
probs = softmax(logits)
next_token = sample(probs)
```

**Example (k = 3):**
```
All probs: {"the": 0.4, "a": 0.3, "mat": 0.2, "floor": 0.05, "cat": 0.03, "dog": 0.02}
Top-3: {"the": 0.4, "a": 0.3, "mat": 0.2}
Renormalized: {"the": 0.44, "a": 0.33, "mat": 0.22}
Sample from these three only!
```

**Pros:**
- ✅ Prevents sampling very unlikely tokens
- ✅ More coherent than pure temperature sampling

**Cons:**
- ❌ Fixed K may be too restrictive or too loose depending on context

### Top-P (Nucleus) Sampling

**Sample from the smallest set of tokens whose cumulative probability ≥ p**:

```python
# Sort probabilities descending
sorted_probs, sorted_indices = torch.sort(probs, descending=True)

# Cumulative probabilities
cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

# Find cutoff (first position where cumsum > p)
cutoff = (cumsum_probs > p).nonzero()[0]

# Zero out tokens beyond cutoff
probs[sorted_indices[cutoff+1:]] = 0

# Renormalize and sample
probs = probs / probs.sum()
next_token = sample(probs)
```

**Example (p = 0.9):**
```
Probs: {"the": 0.5, "a": 0.3, "mat": 0.15, "floor": 0.03, "cat": 0.02}
Cumulative: [0.5, 0.8, 0.95, ...]
              ↑    ↑    ↑ First to exceed 0.9!
Top-p nucleus: {"the", "a", "mat"}
```

**Pros:**
- ✅ Adaptive: Few tokens when confident, many when uncertain
- ✅ Natural-sounding text

**Nanochat default:** Temperature = 0.9 (slightly focused, but creative)

---

## KV Cache Optimization

### The Problem

**Naive generation is wasteful:**

```
Step 1: Run model on [The, cat]                    (2 tokens)
Step 2: Run model on [The, cat, sat]               (3 tokens)
Step 3: Run model on [The, cat, sat, on]           (4 tokens)
Step 4: Run model on [The, cat, sat, on, the]      (5 tokens)
```

**We're recomputing attention for old tokens every step!**

For "The" and "cat" we compute their attention contributions 4 times. Huge waste!

### The Solution: KV Cache

**Key insight:** In attention, the Key (K) and Value (V) for old tokens don't change!

```python
# Attention computation
Q = current_token_query
K = [k1, k2, k3, k4, ...]  # Keys for all previous tokens
V = [v1, v2, v3, v4, ...]  # Values for all previous tokens

attention_output = softmax(Q @ K.T) @ V
```

For each old token, K and V are always the same. **Cache them!**

### How KV Cache Works

```
Step 1: Generate "sat"
  - Compute K, V for ["The", "cat"]
  - Cache them
  - Compute attention output

Step 2: Generate "on"
  - Load cached K, V for ["The", "cat"]
  - Compute K, V for ["sat"] only
  - Append to cache
  - Compute attention using full cached K, V

Step 3: Generate "the"
  - Load cached K, V for ["The", "cat", "sat"]
  - Compute K, V for ["on"] only
  - Append to cache
  - Compute attention

...
```

**Benefit:** Instead of recomputing N tokens, we only compute 1 new token!

### Memory vs. Compute Trade-off

**Without KV cache:**
- Memory: O(1) - only store current batch
- Compute: O(N²) - recompute all tokens every step

**With KV cache:**
- Memory: O(N) - store K, V for all previous tokens
- Compute: O(N) - only compute new token

**For generation: KV cache is much faster!**

### Implementation

```python
# From nanochat/engine.py

class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Shape: (layers, 2, batch, heads, seq_len, head_dim)
        #                  ↑ 2 for K and V
        self.kv_cache = torch.empty(...)
        self.pos = 0  # Current position in sequence

    def insert_kv(self, layer_idx, k, v):
        # Insert new K, V at current position
        self.kv_cache[layer_idx, 0, :, :, self.pos] = k
        self.kv_cache[layer_idx, 1, :, :, self.pos] = v

        # Return full cached K, V so far
        return (
            self.kv_cache[layer_idx, 0, :, :, :self.pos+1],  # All keys
            self.kv_cache[layer_idx, 1, :, :, :self.pos+1],  # All values
        )

    # Auto-increment position after last layer
    if layer_idx == last_layer:
        self.pos += 1
```

**Usage in model forward pass:**

```python
# From nanochat/gpt.py

def forward(self, idx, kv_cache=None):
    for layer_idx, block in enumerate(self.blocks):
        # Compute Q, K, V for current token(s)
        q, k, v = compute_qkv(x)

        # If KV cache exists, insert and get full history
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(layer_idx, k, v)

        # Attention using cached K, V
        output = attention(q, k, v)
```

### Multi-Query Attention Benefits

**Remember MQA?** Fewer K, V heads = smaller cache!

```
Standard MHA (6 heads):
  Cache size = 2 * 6 * seq_len * head_dim

MQA (1 K/V head):
  Cache size = 2 * 1 * seq_len * head_dim
  (6x smaller!)
```

This is why nanochat uses MQA for efficient inference.

---

## The Conversation Format

### How Conversations are Structured

During fine-tuning and inference, conversations use special tokens:

```
<|bos|>
<|user_start|>Hello! What is 2+2?<|user_end|>
<|assistant_start|>The answer is 4.<|assistant_end|>
<|user_start|>Thanks!<|user_end|>
<|assistant_start|>You're welcome!<|assistant_end|>
```

### Token Breakdown

**Special tokens:**
- `<|bos|>`: Beginning of sequence (start of conversation)
- `<|user_start|>` / `<|user_end|>`: User message boundaries
- `<|assistant_start|>` / `<|assistant_end|>`: Assistant message boundaries

**Why these tokens?**
- Model knows when user is speaking vs. assistant
- Model knows when to start/stop generating
- Enables multi-turn conversations

### Generation Process

```python
# 1. User provides input
user_input = "What is the capital of France?"

# 2. Build prompt with special tokens
prompt = "<|bos|><|user_start|>" + user_input + "<|user_end|><|assistant_start|>"

# 3. Tokenize
prompt_tokens = tokenizer.encode(prompt)

# 4. Generate until <|assistant_end|>
response_tokens = model.generate(
    prompt_tokens,
    stop_token=tokenizer.encode_special("<|assistant_end|>")
)

# 5. Decode response
response = tokenizer.decode(response_tokens)
```

### Multi-turn Conversations

For follow-up questions, append to the conversation:

```python
# First turn
conversation = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
]

# User asks follow-up
conversation.append({"role": "user", "content": "What about 3+3?"})

# Render entire conversation to tokens
tokens, _ = tokenizer.render_conversation(conversation)

# Add <|assistant_start|> to prompt a response
tokens.append(tokenizer.encode_special("<|assistant_start|>"))

# Generate
response_tokens = model.generate(tokens)
```

---

## The Engine Architecture

### What is the Engine?

The **Engine** (`nanochat/engine.py`) is a wrapper around the model that handles:
- Efficient inference with KV caching
- Token sampling
- Tool use (calculator, code execution)
- Batch generation (multiple samples from one prompt)
- Streaming generation

### Basic Usage

```python
from nanochat.engine import Engine
from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT

# Load model and tokenizer
model = GPT.from_checkpoint("path/to/checkpoint")
tokenizer = get_tokenizer()

# Create engine
engine = Engine(model, tokenizer)

# Tokenize prompt
prompt = "The capital of France is"
tokens = tokenizer.encode(prompt, prepend="<|bos|>")

# Generate
samples, masks = engine.generate_batch(
    tokens,
    num_samples=1,
    max_tokens=50,
    temperature=0.9
)

# Decode
response = tokenizer.decode(samples[0])
print(response)
```

### Streaming Generation

```python
# Generator that yields tokens as they're produced
for token_column, token_masks in engine.generate(tokens, max_tokens=50):
    # token_column[i] = next token for sample i
    # token_masks[i] = 1 if sampled, 0 if forced (tool output)

    for i, token in enumerate(token_column):
        # Decode and print each token immediately
        print(tokenizer.decode([token]), end='', flush=True)
```

**Benefits:**
- ✅ See generation in real-time
- ✅ Can stop generation early
- ✅ Better user experience

### Batch Generation (Multiple Samples)

Generate multiple diverse responses from one prompt:

```python
# Generate 5 different responses
samples, masks = engine.generate_batch(
    tokens,
    num_samples=5,
    max_tokens=50,
    temperature=1.2,  # Higher temp for diversity
    seed=42
)

# Print all responses
for i, sample in enumerate(samples):
    print(f"Sample {i+1}: {tokenizer.decode(sample)}")
```

**How it works:**
1. Run prompt through model once (prefill)
2. Clone KV cache 5 times
3. Generate 5 sequences in parallel

**Benefits:**
- ✅ Efficient: Shares prompt computation
- ✅ Diversity: Different samples explore different possibilities
- ✅ Useful for best-of-N sampling

---

## Code Walkthrough

### File: `nanochat/engine.py`

#### 1. KVCache Class (lines 82-152)

```python
class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Allocate memory for cache
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None  # Lazy init
        self.pos = 0

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize on first use
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)

        # Insert new K, V at current position
        t0, t1 = self.pos, self.pos + k.size(2)
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v

        # Return full cached K, V
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]

        # Increment position after last layer
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1

        return key_view, value_view
```

#### 2. Sampling Function (lines 156-172)

```python
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    # Greedy decoding (deterministic)
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Top-k filtering
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)

    # Standard temperature sampling
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)
```

#### 3. Engine.generate Method (lines 191-296)

```python
def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
    # Setup
    device = self.model.get_device()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # 1. Prefill: Run prompt through model once
    kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), ...)
    ids = torch.tensor([tokens], device=device)
    logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
    logits = logits[:, -1, :]  # Last token logits
    next_ids = sample_next_token(logits, rng, temperature, top_k)

    # 2. Clone KV cache for each sample
    kv_cache_decode = KVCache(batch_size=num_samples, ...)
    kv_cache_decode.prefill(kv_cache_prefill)

    # 3. Initialize state for each sample
    row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

    # 4. Generation loop
    while not done:
        # Forward model with single token per sample
        logits = self.model.forward(ids, kv_cache=kv_cache_decode)
        next_ids = sample_next_token(logits[:, -1, :], rng, temperature, top_k)

        # Process each sample (tool use, completion check, etc.)
        for i, (state, token) in enumerate(zip(row_states, next_ids)):
            state.current_tokens.append(token)

            # Check for completion
            if token == assistant_end_token:
                state.completed = True

            # Handle tool use (explained in next document)
            if token == python_start:
                state.in_python_block = True
            # ... tool logic

        yield next_ids  # Stream tokens

        # Prepare next iteration
        ids = next_ids.unsqueeze(1)
```

---

## Key Takeaways

1. **Autoregressive generation** produces one token at a time, each depending on previous tokens

2. **Temperature** controls randomness: low = focused, high = creative

3. **Top-k/top-p sampling** prevents sampling very unlikely tokens

4. **KV cache** dramatically speeds up generation by caching attention keys and values

5. **Conversation format** uses special tokens to structure multi-turn dialogs

6. **The Engine** wraps the model for efficient inference with streaming and batching

7. **Batch generation** efficiently produces multiple diverse samples

8. **MQA** reduces KV cache size by 6x, enabling faster inference

---

## What's Next?

Now that you understand text generation, let's see how tools are integrated!

**→ Next: [Document 6: Tools and Capabilities](06_tools.md)**

You'll learn:
- Calculator tool integration
- Python code execution
- How the model decides when to use tools
- Special tokens for tool use

---

## Self-Check

Before moving on, make sure you understand:

- [ ] What autoregressive generation means
- [ ] How temperature affects sampling
- [ ] The difference between top-k and top-p sampling
- [ ] Why KV cache speeds up generation
- [ ] How conversations are formatted with special tokens
- [ ] What the Engine class does
- [ ] How batch generation works
- [ ] The trade-off between memory and compute
