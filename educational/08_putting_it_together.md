# Putting It All Together: Implementation Guide

This section walks through implementing your own LLM from scratch, using nanochat as a guide.

## Project Structure

A well-organized codebase is essential:

```
your_llm/
├── src/                    # Core library
│   ├── model.py           # Model architecture
│   ├── tokenizer.py       # BPE tokenizer
│   ├── trainer.py         # Training loop
│   ├── optimizer.py       # Custom optimizers
│   └── data.py            # Data loading
├── scripts/               # Entry points
│   ├── train_tokenizer.py
│   ├── train_model.py
│   └── generate.py
├── tests/                 # Unit tests
├── configs/               # Hyperparameter configs
└── README.md
```

## Step-by-Step Implementation

### Step 1: Implement BPE Tokenizer

**Start simple:** Python-only implementation

```python
class SimpleBPE:
    def __init__(self):
        self.merges = {}  # (pair) -> new_token_id
        self.vocab = {}   # token_id -> bytes

    def train(self, text_iterator, vocab_size):
        # 1. Initialize with bytes 0-255
        self.vocab = {i: bytes([i]) for i in range(256)}

        # 2. Count pairs in text
        pair_counts = count_pairs(text_iterator)

        # 3. Iteratively merge most frequent pairs
        for i in range(256, vocab_size):
            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)

            # Record merge
            self.merges[best_pair] = i
            left, right = best_pair
            self.vocab[i] = self.vocab[left] + self.vocab[right]

            # Update pair counts
            pair_counts = update_counts(pair_counts, best_pair, i)

    def encode(self, text):
        # Convert to bytes, apply merges
        tokens = list(text.encode('utf-8'))

        while len(tokens) >= 2:
            # Find best pair to merge
            best_pair = None
            best_idx = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    if best_pair is None or self.merges[pair] < self.merges[best_pair]:
                        best_pair = pair
                        best_idx = i

            if best_pair is None:
                break

            # Apply merge
            new_token = self.merges[best_pair]
            tokens = tokens[:best_idx] + [new_token] + tokens[best_idx + 2:]

        return tokens
```

**Then optimize:** Rewrite critical parts in Rust/C++ if needed.

### Step 2: Implement Transformer Model

**Core components:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.mlp = MLP(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # Embed tokens
        x = self.token_emb(idx)

        # Pass through blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and project to vocab
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            return loss

        return logits
```

### Step 3: Implement Training Loop

**Minimal training script:**

```python
def train(model, train_loader, optimizer, num_steps):
    model.train()

    for step in range(num_steps):
        # Get batch
        x, y = next(train_loader)

        # Forward pass
        loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
```

**Add features incrementally:**
1. Learning rate scheduling
2. Gradient clipping
3. Evaluation
4. Checkpointing
5. Distributed training

### Step 4: Data Pipeline

**Efficient streaming:**

```python
class StreamingDataLoader:
    def __init__(self, data_files, batch_size, seq_len, tokenizer):
        self.data_files = data_files
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.buffer = []

    def __iter__(self):
        for file in itertools.cycle(self.data_files):
            with open(file) as f:
                for line in f:
                    # Tokenize
                    tokens = self.tokenizer.encode(line)
                    self.buffer.extend(tokens)

                    # Yield batches
                    while len(self.buffer) >= self.batch_size * self.seq_len:
                        batch = self.buffer[:self.batch_size * self.seq_len]
                        self.buffer = self.buffer[self.batch_size * self.seq_len:]

                        # Reshape to [batch_size, seq_len]
                        x = torch.tensor(batch[:-1]).view(self.batch_size, -1)
                        y = torch.tensor(batch[1:]).view(self.batch_size, -1)

                        yield x, y
```

## Common Implementation Pitfalls

### 1. Shape Mismatches

**Problem:** Tensor dimensions don't align

**Debug:**
```python
print(f"Q shape: {Q.shape}")  # [B, H, T, D]
print(f"K shape: {K.shape}")  # [B, H, T, D]
print(f"V shape: {V.shape}")  # [B, H, T, D]

# Attention: Q @ K^T
scores = Q @ K.transpose(-2, -1)  # [B, H, T, T]
print(f"Scores shape: {scores.shape}")
```

**Solution:** Add shape assertions
```python
assert Q.shape == K.shape == V.shape
assert scores.shape == (B, H, T, T)
```

### 2. Gradient Flow Issues

**Problem:** Gradients vanish or explode

**Debug:**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm={grad_norm:.6f}")
```

**Solutions:**
- Gradient clipping
- Better initialization
- Layer normalization
- Residual connections

### 3. Memory Leaks

**Problem:** GPU memory grows over time

**Common causes:**
```python
# BAD: Storing loss with gradients
losses.append(loss)

# GOOD: Detach from graph
losses.append(loss.item())
```

```python
# BAD: Creating new tensors on GPU in loop
for _ in range(1000):
    temp = torch.zeros(1000, 1000, device='cuda')  # Leak!

# GOOD: Reuse tensors
temp = torch.zeros(1000, 1000, device='cuda')
for _ in range(1000):
    temp.zero_()
```

### 4. Incorrect Masking

**Problem:** Attention can see future tokens

**Test:**
```python
def test_causal_mask():
    B, T = 2, 5
    mask = torch.tril(torch.ones(T, T))

    # Future positions should be masked
    assert mask[0, 1] == 0  # Position 0 can't see position 1
    assert mask[1, 0] == 1  # Position 1 can see position 0
```

## Testing Your Implementation

### Unit Tests

```python
import unittest

class TestTransformer(unittest.TestCase):
    def test_forward_pass(self):
        model = GPTModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
        x = torch.randint(0, 100, (2, 10))  # [batch=2, seq=10]

        logits = model(x)

        self.assertEqual(logits.shape, (2, 10, 100))  # [B, T, vocab]

    def test_loss_computation(self):
        model = GPTModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
        x = torch.randint(0, 100, (2, 10))
        y = torch.randint(0, 100, (2, 10))

        loss = model(x, y)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)  # Scalar
        self.assertGreater(loss.item(), 0)  # Positive loss

    def test_generation(self):
        model = GPTModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
        model.eval()

        prompt = torch.tensor([[1, 2, 3]])  # [batch=1, seq=3]

        with torch.no_grad():
            for _ in range(5):
                logits = model(prompt)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_token], dim=1)

        self.assertEqual(prompt.shape[1], 8)  # 3 + 5 generated tokens
```

### Integration Tests

```python
def test_training_reduces_loss():
    """Test that training actually reduces loss"""
    model = GPTModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create dummy data
    x = torch.randint(0, 100, (8, 20))
    y = torch.randint(0, 100, (8, 20))

    # Initial loss
    with torch.no_grad():
        initial_loss = model(x, y).item()

    # Train for 100 steps
    for _ in range(100):
        loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final loss
    with torch.no_grad():
        final_loss = model(x, y).item()

    # Loss should decrease
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
```

## Debugging Techniques

### 1. Overfit Single Batch

**Goal:** Verify model can learn

```python
# Create single batch
x = torch.randint(0, 100, (8, 20))
y = torch.randint(0, 100, (8, 20))

# Train on just this batch
for step in range(1000):
    loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Loss should go to near 0
```

If loss doesn't decrease:
- Model has bugs
- Learning rate too low
- Gradient flow issues

### 2. Compare with Reference Implementation

```python
# Your implementation
your_output = your_model(x)

# Reference (e.g., HuggingFace)
ref_output = reference_model(x)

# Should be close
diff = (your_output - ref_output).abs().max()
print(f"Max difference: {diff.item()}")
assert diff < 1e-5, "Outputs don't match!"
```

### 3. Gradient Checking

```python
from torch.autograd import gradcheck

model = GPTModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
x = torch.randint(0, 100, (2, 10), dtype=torch.float64)  # Use float64 for precision

# Check gradients
test = gradcheck(model, x, eps=1e-6, atol=1e-4)
print(f"Gradient check: {'PASS' if test else 'FAIL'}")
```

### 4. Attention Visualization

```python
import matplotlib.pyplot as plt

def visualize_attention(attn_weights, tokens):
    """
    attn_weights: [num_heads, seq_len, seq_len]
    tokens: [seq_len]
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for head in range(8):
        ax = axes[head // 4, head % 4]
        im = ax.imshow(attn_weights[head].cpu().numpy(), cmap='viridis')
        ax.set_title(f'Head {head}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')

    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    plt.show()
```

## Performance Optimization

### 1. Profile Your Code

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Run training step
    loss = model(x, y)
    loss.backward()
    optimizer.step()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 2. Use torch.compile

```python
# PyTorch 2.0+
model = torch.compile(model)

# 20-30% speedup in many cases
```

### 3. Optimize Data Loading

```python
# Use pin_memory for faster CPU->GPU transfer
train_loader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,
    num_workers=4
)

# Prefetch to GPU
for x, y in train_loader:
    x = x.to('cuda', non_blocking=True)
    y = y.to('cuda', non_blocking=True)
```

### 4. Mixed Precision Training

```python
scaler = torch.cuda.amp.GradScaler()

for x, y in train_loader:
    optimizer.zero_grad()

    # Forward in BF16
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = model(x, y)

    # Backward in FP32
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Scaling Up

### From Single GPU to Multi-GPU

```python
# Wrap model in DDP
model = nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank]
)

# Use distributed sampler
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
train_loader = DataLoader(dataset, sampler=train_sampler)

# Run with torchrun
# torchrun --nproc_per_node=8 train.py
```

### From Small to Large Models

1. **Start small:** 10M params, verify everything works
2. **Scale gradually:** 50M → 100M → 500M
3. **Tune hyperparameters** at each scale
4. **Monitor metrics:** Loss, perplexity, downstream tasks

## Checklist for Production

- [ ] Model passes all unit tests
- [ ] Can overfit single batch
- [ ] Training loss decreases smoothly
- [ ] Validation loss tracks training loss
- [ ] Generated text is coherent
- [ ] Checkpoint saving/loading works
- [ ] Distributed training tested
- [ ] Memory usage is reasonable
- [ ] Training speed meets targets
- [ ] Code is documented

## Resources for Learning More

### Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (GPT-3, Brown et al., 2020)
- "Training Compute-Optimal LLMs" (Chinchilla, Hoffmann et al., 2022)

### Codebases
- **nanoGPT**: Minimal GPT implementation
- **minGPT**: Educational GPT in PyTorch
- **GPT-Neo**: Open source GPT models
- **llm.c**: GPT training in pure C/CUDA

### Courses
- Stanford CS224N (NLP with Deep Learning)
- Fast.ai (Practical Deep Learning)
- Hugging Face Course (Transformers)

## Next Steps

You now have all the knowledge to build your own LLM! The key is to:

1. **Start simple** - Get a minimal version working first
2. **Test thoroughly** - Write tests for every component
3. **Iterate** - Add features incrementally
4. **Measure** - Profile and optimize bottlenecks
5. **Scale** - Gradually increase model size and data

Good luck building! 🚀
