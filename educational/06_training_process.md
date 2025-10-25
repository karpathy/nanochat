# The Training Process

Training a language model involves teaching it to predict the next token given previous tokens. Let's understand how nanochat implements this end-to-end.

## Overview: The Complete Training Pipeline

```
1. Tokenization Training (~10 min)
   └─> BPE tokenizer vocabulary

2. Base Pretraining (~2-4 hours, $100)
   └─> Base model checkpoint

3. Midtraining (~30 min, $12)
   └─> Refined base model

4. Supervised Fine-Tuning (~15 min, $6)
   └─> Chat model

5. Reinforcement Learning (~10 min, $4)
   └─> Final optimized model
```

**Total cost:** ~$122, **Total time:** ~3-5 hours on 8×H100 GPUs

## 1. Language Modeling Objective

**Goal:** Learn probability distribution over sequences

$$P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t | w_1, \ldots, w_{t-1})$$

**Training objective:** Maximize log-likelihood

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \log P(w_t | w_1, \ldots, w_{t-1})$$

In practice, we minimize **negative log-likelihood** (cross-entropy loss):

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(w_t | w_1, \ldots, w_{t-1})$$

### Cross-Entropy Loss in Code

File: `nanochat/gpt.py:285`

```python
def forward(self, idx, targets=None, ...):
    # ... forward pass to get hidden states x ...

    if targets is not None:
        # Training mode
        logits = self.lm_head(x)  # [B, T, vocab_size]
        logits = 15 * torch.tanh(logits / 15)  # Softcap

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # [B*T, V]
            targets.view(-1),                   # [B*T]
            ignore_index=-1,
            reduction='mean'
        )
        return loss
```

**Key points:**
1. Reshape to 2D for loss computation
2. `ignore_index=-1`: Skip padding tokens
3. `reduction='mean'`: Average over all tokens

## 2. Data Loading: `nanochat/dataloader.py`

Efficient data loading is crucial for fast training.

```python
def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128):
    """Stream pretraining text from parquet files, tokenize, yield batches."""
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1  # +1 for target

    # Initialize tokenizer and buffer
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = deque()  # Streaming token buffer

    # Infinite iterator over documents
    def document_batches():
        while True:
            for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]

    batches = document_batches()

    while True:
        # Fill buffer with enough tokens
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            # Tokenize in parallel
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)

        # Extract tokens from buffer
        scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()

        # Create inputs and targets
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)   # [0, 1, 2, ..., T-1]
        targets_cpu = scratch[1:]                          # [1, 2, 3, ..., T]

        # Move to GPU
        inputs = inputs_cpu.view(B, T).to(device="cuda", non_blocking=True)
        targets = targets_cpu.view(B, T).to(device="cuda", non_blocking=True)

        yield inputs, targets
```

**Design highlights:**

1. **Streaming:** Never loads entire dataset into memory
2. **Distributed:** Each GPU processes different shards (`start=ddp_rank, step=ddp_world_size`)
3. **Parallel tokenization:** Uses multiple threads
4. **Pinned memory:** Faster CPU→GPU transfer
5. **Non-blocking transfers:** Overlap with computation

### Input/Target Relationship

For sequence `[0, 1, 2, 3, 4, 5]`:

```
Inputs:  [0, 1, 2, 3, 4]
Targets: [1, 2, 3, 4, 5]

Position 0: input=0, target=1  →  predict 1 given 0
Position 1: input=1, target=2  →  predict 2 given 0,1
Position 2: input=2, target=3  →  predict 3 given 0,1,2
...
```

Each position predicts the next token!

## 3. Training Loop: `scripts/base_train.py`

### Hyperparameters: `scripts/base_train.py:28`

```python
# Model architecture
depth = 20              # Number of layers
max_seq_len = 2048      # Context length

# Training horizon
target_param_data_ratio = 20  # Chinchilla optimal

# Optimization
device_batch_size = 32         # Per-GPU batch size
total_batch_size = 524288      # Total tokens per step
embedding_lr = 0.2             # AdamW for embeddings
unembedding_lr = 0.004         # AdamW for LM head
matrix_lr = 0.02               # Muon for linear layers
grad_clip = 1.0                # Gradient clipping

# Evaluation
eval_every = 250
core_metric_every = 2000
```

### Computing Training Length: `scripts/base_train.py:108`

```python
# Chinchilla scaling: 20 tokens per parameter
target_tokens = target_param_data_ratio * num_params
num_iterations = target_tokens // total_batch_size

print(f"Parameters: {num_params:,}")
print(f"Target tokens: {target_tokens:,}")
print(f"Iterations: {num_iterations:,}")
print(f"Total FLOPs: {num_flops_per_token * total_tokens:e}")
```

**Example for d=20 model:**
- Parameters: 270M
- Target tokens: 20 × 270M = 5.4B
- Batch size: 524K
- Iterations: 5.4B / 524K ≈ 10,300 steps

### Gradient Accumulation: `scripts/base_train.py:89`

```python
tokens_per_fwdbwd = device_batch_size * max_seq_len  # Per-GPU
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # All GPUs
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

print(f"Tokens / micro-batch / rank: {tokens_per_fwdbwd:,}")
print(f"Total batch size {total_batch_size:,}")
print(f"Gradient accumulation steps: {grad_accum_steps}")
```

**Example:**
- Device batch: 32 × 2048 = 65,536 tokens
- 8 GPUs: 8 × 65,536 = 524,288 tokens
- Grad accum: 524,288 / 524,288 = 1 (no accumulation needed)

But if we only had 4 GPUs:
- 4 GPUs: 4 × 65,536 = 262,144 tokens
- Grad accum: 524,288 / 262,144 = 2 steps

**Gradient accumulation** allows larger effective batch sizes than GPU memory permits.

### Main Training Loop: `scripts/base_train.py:172`

```python
for step in range(num_iterations + 1):
    last_step = step == num_iterations

    # ===== EVALUATION =====
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        wandb_run.log({"val/bpb": val_bpb})
        model.train()

    # ===== SAMPLING =====
    if master_process and (last_step or step % sample_every == 0):
        model.eval()
        prompts = ["The capital of France is", ...]
        engine = Engine(model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            sample, _ = engine.generate_batch(tokens, max_tokens=16, temperature=0)
            print(tokenizer.decode(sample[0]))
        model.train()

    # ===== CHECKPOINT =====
    if master_process and last_step:
        save_checkpoint(checkpoint_dir, step, model.state_dict(), ...)

    if last_step:
        break

    # ===== TRAINING STEP =====
    torch.cuda.synchronize()
    t0 = time.time()

    # Gradient accumulation loop
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps  # Normalize for accumulation
        loss.backward()
        x, y = next(train_loader)  # Prefetch next batch

    # Gradient clipping
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Update learning rates
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # Update momentum for Muon
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum

    # Optimizer step
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t1 = time.time()

    # Logging
    print(f"step {step:05d} | loss: {loss:.6f} | dt: {(t1-t0)*1000:.2f}ms | ...")
```

### Mixed Precision Training

```python
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

with autocast_ctx:
    loss = model(x, y)
```

**BFloat16 (BF16)** automatic mixed precision:
- Forward pass in BF16 (2× faster, 2× less memory)
- Backward pass in FP32 (for numerical stability)
- Automatic casting handled by PyTorch

**Why BF16 over FP16?**
- Same exponent range as FP32 (no loss scaling needed)
- Better numerical stability
- Supported on Ampere+ GPUs

### Learning Rate Schedule: `scripts/base_train.py:148`

```python
warmup_ratio = 0.0      # No warmup
warmdown_ratio = 0.2    # 20% of steps for decay
final_lr_frac = 0.0     # Decay to 0

def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)

    if it < warmup_iters:
        # Linear warmup
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        # Constant
        return 1.0
    else:
        # Linear decay
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac
```

**Schedule visualization:**
```
LR
 |
1.0|     ___________________
   |    /                    \
   |   /                      \
   |  /                        \
0.0|_/                          \___
   0   10%  20%  ...  80%  90%  100%
       warmup   constant   warmdown
```

### Gradient Clipping: `scripts/base_train.py:265`

```python
if grad_clip > 0.0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

**Why clip gradients?**
- Prevents exploding gradients
- Stabilizes training
- Allows higher learning rates

**How it works:**
$$\mathbf{g} \leftarrow \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\| \leq \text{max\_norm} \\
\frac{\text{max\_norm}}{\|\mathbf{g}\|} \mathbf{g} & \text{otherwise}
\end{cases}$$

Scales gradient to have maximum norm of `grad_clip`.

## 4. Distributed Training (DDP)

nanochat uses **DistributedDataParallel (DDP)** for multi-GPU training.

### Initialization: `nanochat/common.py`

```python
def compute_init():
    ddp = int(os.environ.get("RANK", -1)) != -1  # Is this DDP?

    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = "cuda"

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device
```

### Running DDP

```bash
# Single GPU
python -m scripts.base_train

# Multi-GPU (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train
```

**How DDP works:**

1. **Data parallelism:** Each GPU gets different data
2. **Model replication:** Same model on all GPUs
3. **Gradient averaging:** After backward, gradients are averaged across GPUs
4. **Synchronized updates:** All GPUs update identically

**Benefits:**
- Near-linear scaling (8 GPUs ≈ 8× faster)
- Same final model as single-GPU training
- Minimal code changes

## 5. Evaluation During Training

### Validation Loss (BPB): `nanochat/loss_eval.py`

```python
def evaluate_bpb(model, val_loader, eval_steps, token_bytes):
    """Evaluate bits-per-byte on validation set"""
    total_loss = 0
    total_tokens = 0

    for step in range(eval_steps):
        x, y = next(val_loader)
        with torch.no_grad():
            loss = model(x, y, loss_reduction='sum')
        total_loss += loss.item()
        total_tokens += (y != -1).sum().item()

    # Average loss per token
    avg_loss_per_token = total_loss / total_tokens

    # Convert to bits per byte
    bits_per_token = avg_loss_per_token / math.log(2)
    token_bytes_mean = token_bytes.float().mean().item()
    bits_per_byte = bits_per_token / token_bytes_mean

    return bits_per_byte
```

**Bits-per-byte (BPB)** measures compression:
- Lower BPB = better model
- Random model: ~8 BPB (1 byte = 8 bits, no compression)
- Good model: ~1.0-1.5 BPB

### CORE Metric: `scripts/base_eval.py`

CORE is a weighted average of multiple benchmarks:

```python
def evaluate_model(model, tokenizer, device, max_per_task=500):
    results = {}

    # Run each task
    for task_name, task_fn in tasks.items():
        acc = task_fn(model, tokenizer, device, max_per_task)
        results[task_name] = acc

    # Compute weighted average
    weights = {"task1": 0.3, "task2": 0.7, ...}
    core_metric = sum(weights[k] * results[k] for k in weights)

    return {"core_metric": core_metric, "results": results}
```

Evaluated periodically during training to track progress.

## 6. Checkpointing: `nanochat/checkpoint_manager.py`

```python
def save_checkpoint(checkpoint_dir, step, model_state, optimizer_states, metadata):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
    torch.save(model_state, model_path)

    # Save optimizers
    for i, opt_state in enumerate(optimizer_states):
        opt_path = os.path.join(checkpoint_dir, f"optimizer_{i}_step_{step}.pt")
        torch.save(opt_state, opt_path)

    # Save metadata
    meta_path = os.path.join(checkpoint_dir, f"metadata_step_{step}.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    print(f"Saved checkpoint to {checkpoint_dir}")
```

**What to save:**
- Model weights
- Optimizer states (for resuming training)
- Metadata (step number, config, metrics)

## 7. Supervised Fine-Tuning: `scripts/chat_sft.py`

After pretraining, we fine-tune on conversations.

### Data Format

```python
conversation = {
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Italy?"},
        {"role": "assistant", "content": "The capital of Italy is Rome."}
    ]
}
```

### Tokenization with Mask

```python
ids, mask = tokenizer.render_conversation(conversation)

# ids:  [<|bos|>, <|user_start|>, "What", "is", ..., <|assistant_end|>]
# mask: [0, 0, 0, 0, ..., 1, 1, 1, ..., 1]
#       ↑ don't train      ↑ train on assistant responses
```

### Loss Computation

```python
# Only compute loss on assistant tokens
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1),
    reduction='none'
)
# Apply mask
masked_loss = (loss * mask).sum() / mask.sum()
```

**Key difference from pretraining:**
- Pretraining: train on ALL tokens
- SFT: train ONLY on assistant responses

## 8. Reinforcement Learning: `scripts/chat_rl.py`

Final stage: optimize for quality using RL.

### Self-Improvement Loop

```python
# 1. Generate multiple responses
prompts = load_prompts()
for prompt in prompts:
    responses = model.generate(prompt, num_samples=8, temperature=0.8)

    # 2. Score responses (using a reward model or heuristic)
    scores = [reward_model(r) for r in responses]

    # 3. Keep best responses
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_response = responses[best_idx]

    # 4. Fine-tune on best response
    train_on(prompt, best_response)
```

**Simple but effective!**

## Performance Metrics

### Model FLOPs Utilization (MFU)

```python
flops_per_sec = num_flops_per_token * total_batch_size / dt
promised_flops_per_sec_h100 = 989e12 * ddp_world_size  # BF16 on H100
mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
```

**Good MFU:** 40-60% (nanochat achieves ~50%)

### Tokens per Second

```python
tok_per_sec = world_tokens_per_fwdbwd / dt
```

**Typical:** 500K - 1M tokens/sec on 8×H100

## Common Training Issues

### 1. Loss Spikes
**Symptoms:** Loss suddenly jumps
**Causes:** Bad batch, numerical instability, LR too high
**Solutions:**
- Gradient clipping
- Lower learning rate
- Skip bad batches

### 2. Loss Plateau
**Symptoms:** Loss stops improving
**Causes:** Learning rate too low, insufficient data, model capacity
**Solutions:**
- Increase LR
- More data
- Larger model

### 3. NaN Loss
**Symptoms:** Loss becomes NaN
**Causes:** Numerical overflow, bad initialization
**Solutions:**
- Lower learning rate
- Gradient clipping
- Check for bad data

## Next Steps

Now we'll explore the **Optimization Techniques** - Muon and AdamW optimizers that make training efficient.
