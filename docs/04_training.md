# The Training Pipeline: How the Model Learns

This document explains how the GPT model is trained in nanochat, from random initialization to a fully functional ChatGPT-like assistant. We'll cover the complete pipeline and explain each stage in detail.

## Table of Contents
1. [Training vs. Inference](#training-vs-inference)
2. [How Neural Networks Learn](#how-neural-networks-learn)
3. [The Complete Training Pipeline](#the-complete-training-pipeline)
4. [Stage 1: Pretraining](#stage-1-pretraining)
5. [Stage 2: Mid-training](#stage-2-mid-training)
6. [Stage 3: Supervised Fine-Tuning (SFT)](#stage-3-supervised-fine-tuning-sft)
7. [Stage 4: Reinforcement Learning (RL)](#stage-4-reinforcement-learning-rl)
8. [Distributed Training](#distributed-training)
9. [Optimization Algorithms](#optimization-algorithms)
10. [Checkpointing and Resuming](#checkpointing-and-resuming)
11. [Code Walkthrough](#code-walkthrough)

---

## Training vs. Inference

### Inference (Using the Model)

**Inference** is using a trained model to make predictions:

```python
# Input: "What is 2+2?"
# Model: Computes probabilities for next token
# Output: "The answer is 4."
```

- **Fast**: One forward pass through the network
- **No learning**: Model parameters don't change
- **Cheap**: Can run on CPU or small GPU

### Training (Teaching the Model)

**Training** is adjusting the model's parameters to make better predictions:

```python
# Show model: "The cat sat on the mat"
# Ask: Predict next word after "The cat sat on the"
# Model predicts: "floor" (wrong!)
# Adjust parameters to make "mat" more likely
# Repeat billions of times
```

- **Slow**: Requires forward pass + backward pass (gradient computation)
- **Learning**: Model parameters are updated
- **Expensive**: Requires powerful GPUs, lots of data

### The Key Difference

```
Inference:
Input → Model → Output
(parameters frozen)

Training:
Input → Model → Output
      ↓
   Compare to target
      ↓
   Compute error (loss)
      ↓
   Compute gradients
      ↓
   Update parameters ← Repeat millions of times
```

---

## How Neural Networks Learn

### The Learning Process

Think of training like learning to throw darts:

1. **Try**: Throw a dart (make a prediction)
2. **Measure**: See how far you missed the bullseye (compute loss)
3. **Adjust**: Slightly change your throw (update parameters)
4. **Repeat**: Keep trying until you consistently hit the target

For neural networks:

1. **Forward pass**: Feed input through the model, get prediction
2. **Loss computation**: Measure how wrong the prediction is
3. **Backward pass**: Compute how to adjust each parameter
4. **Update**: Adjust parameters slightly to reduce the loss
5. **Repeat**: Do this for millions of examples

### Loss Functions

The **loss function** quantifies how wrong the model is. Lower loss = better model.

**For language models: Cross-Entropy Loss**

```python
# Example: Predicting the next word after "The cat"

# Model outputs probabilities for each word:
predicted_probs = {
    "sat":  0.3,
    "ran":  0.2,
    "is":   0.1,
    "mat":  0.05,  # Actual next word
    ...
}

# Cross-entropy loss:
loss = -log(predicted_probs["mat"])
     = -log(0.05)
     = 2.996  (higher = worse)

# If model had predicted 0.9 for "mat":
loss = -log(0.9) = 0.105  (much better!)
```

**Key property:** Loss is high when model assigns low probability to the correct answer.

### Gradient Descent

**Gradient descent** is the algorithm for updating parameters.

**Intuition:** Imagine you're in a foggy valley trying to reach the lowest point:
- Feel the slope under your feet (gradient)
- Take a small step downhill (parameter update)
- Repeat until you reach the bottom (minimum loss)

**Math:**
```python
# For each parameter:
gradient = ∂loss/∂parameter  # How much loss changes if we change this parameter
parameter = parameter - learning_rate * gradient
```

**Learning rate** controls step size:
- Too large: You overshoot the minimum
- Too small: Takes forever to converge
- Just right: Smooth convergence

### Backpropagation

**Backpropagation** efficiently computes gradients for all parameters.

**The algorithm:**
```
1. Forward pass: Compute output and loss
2. Backward pass: Compute ∂loss/∂output
3. Chain rule: Propagate gradients backwards through each layer
   ∂loss/∂layer_N → ∂loss/∂layer_(N-1) → ... → ∂loss/∂layer_1
4. Result: Gradient for every parameter
```

**PyTorch does this automatically!**
```python
loss = model(x, targets)  # Forward pass
loss.backward()           # Backprop - computes gradients
optimizer.step()          # Update parameters using gradients
```

---

## The Complete Training Pipeline

Nanochat trains models in multiple stages, each building on the previous:

```
┌──────────────────────────────────────────────────────────────┐
│ STAGE 1: PRETRAINING                                         │
│ ------------------------------------------------------------ │
│ Data: Raw internet text (54B tokens)                         │
│ Task: Predict next word                                      │
│ Goal: Learn language patterns, facts, reasoning              │
│ Output: "Base model" - can predict text but not chat         │
│ Cost: ~$100 (4 hours on GPU)                                 │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 2: MID-TRAINING (Optional)                             │
│ ------------------------------------------------------------ │
│ Data: Conversation-formatted data                            │
│ Task: Continue next-word prediction on chat data             │
│ Goal: Familiarize model with chat format                     │
│ Output: Model that "knows" conversation structure            │
│ Cost: Included in main training                              │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 3: SUPERVISED FINE-TUNING (SFT)                        │
│ ------------------------------------------------------------ │
│ Data: Question-answer pairs, instructions                    │
│ Task: Predict assistant responses (not user inputs!)         │
│ Goal: Learn to follow instructions, be helpful               │
│ Output: "Chat model" - can have conversations                │
│ Cost: Included in main training                              │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 4: REINFORCEMENT LEARNING (RL) (Optional)              │
│ ------------------------------------------------------------ │
│ Data: Reward signals, preferences                            │
│ Task: Maximize reward (human preference)                     │
│ Goal: Align outputs with human values                        │
│ Output: "Aligned model" - safer, more helpful                │
│ Cost: Additional training time                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Pretraining

### What is Pretraining?

**Pretraining** trains the model from scratch on massive amounts of raw text to predict the next word.

**Data example:**
```
"The quick brown fox jumps over the lazy dog"
```

**Training examples created:**
```
Input: "The"                          → Target: "quick"
Input: "The quick"                    → Target: "brown"
Input: "The quick brown"              → Target: "fox"
Input: "The quick brown fox"          → Target: "jumps"
...
```

The model learns:
- **Grammar**: "The" is usually followed by a noun or adjective
- **Facts**: "Paris is the capital of France"
- **Patterns**: Common phrases, idioms, etc.
- **Reasoning**: Logical relationships, math, etc.

### The Pretraining Dataset

**Nanochat uses FineWeb** - a high-quality dataset from the internet:
- Scraped from CommonCrawl (web pages)
- Filtered for quality
- Deduplicated
- ~54 billion tokens used in training

**Why so much data?** The model has 561M-1.9B parameters to learn!

### The Training Loop

```python
# Simplified training loop from scripts/base_train.py

model = GPT(config)  # Initialize with random weights
optimizer = setup_optimizers(model)
data_loader = load_training_data()

for step in range(num_iterations):
    # 1. Get a batch of data
    x, y = next(data_loader)
    # x: input tokens  [batch_size, seq_len]
    # y: target tokens [batch_size, seq_len] (x shifted by 1)

    # 2. Forward pass: compute loss
    loss = model(x, y)

    # 3. Backward pass: compute gradients
    loss.backward()

    # 4. Update parameters
    optimizer.step()

    # 5. Reset gradients for next iteration
    optimizer.zero_grad()

    # Repeat millions of times!
```

### Training Hyperparameters

```python
# From speedrun.sh (default d20 model)
num_iterations = 5400           # Number of training steps
total_batch_size = 524288       # Tokens per step (0.5M)
device_batch_size = 32          # Batch size per GPU
learning_rate = 0.02            # Step size for updates
max_seq_len = 2048              # Context window
```

**Total training:**
- 5,400 steps × 524,288 tokens/step = **2.8 billion tokens**
- Actually trains on 54B tokens with different hyperparameters
- Takes about 4 hours on a powerful GPU (H100)
- Costs approximately $100

### What the Model Learns

After pretraining, the model can:
- ✅ Complete sentences grammatically
- ✅ Answer factual questions (sometimes)
- ✅ Generate coherent text
- ❌ Follow instructions reliably
- ❌ Have natural conversations
- ❌ Know when to stop generating

**Example:**
```
Prompt: "The capital of France is"
Base model: "Paris, which is located in the north-central part of France
            on the river Seine. The city is one of the most important..."
            (keeps rambling)
```

---

## Stage 2: Mid-training

### What is Mid-training?

**Mid-training** continues pretraining (next-word prediction) but on **conversation-formatted data**.

**Purpose:** Help the model understand the structure of conversations before fine-tuning.

**Data format:**
```
<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>4<|assistant_end|>
```

**Still training on next-word prediction**, but now the text has conversation structure.

### Why Mid-training Helps

Without mid-training, the model has never seen:
- Special tokens like `<|user_start|>`
- Conversation turn structure
- The format we'll use during fine-tuning

Mid-training bridges the gap between raw text and conversations.

### Implementation

```python
# From scripts/mid_train.py
# Very similar to pretraining, just different data

data_loader = load_conversation_data()  # Conversations formatted as text

for step in range(num_iterations):
    x, y = next(data_loader)
    loss = model(x, y)  # Still next-word prediction!
    loss.backward()
    optimizer.step()
```

---

## Stage 3: Supervised Fine-Tuning (SFT)

### What is SFT?

**Supervised Fine-Tuning** teaches the model to follow instructions by training on example conversations.

**Key difference from pretraining:** We only train on the **assistant's responses**, not the user's inputs!

### The Data

**Format:**
```json
{
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
  ]
}
```

**Converted to tokens with mask:**
```
Tokens: [<|bos|>, <|user_start|>, "What", "is", "2", "+", "2", "?", <|user_end|>,
         <|assistant_start|>, "The", "answer", "is", "4", ".", <|assistant_end|>]

Mask:   [   0,          0,         0,     0,   0,  0,  0,  0,      0,
               0,            1,      1,      1,   1,  1,       1           ]
         ↑ Don't train on this          ↑ Train on this!
```

**Why mask user inputs?** We want the model to learn to *generate* good responses, not memorize questions.

### The Training Loop

```python
# From scripts/chat_sft.py

for step in range(num_iterations):
    # Load a conversation
    conversation = next(data_loader)

    # Render to tokens with mask
    ids, mask = tokenizer.render_conversation(conversation)

    # Forward pass
    logits = model(ids)

    # Compute loss ONLY on masked positions
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        ids.view(-1),
        ignore_index=-1  # Achieved by setting non-masked positions to -1
    )

    # Backprop and update
    loss.backward()
    optimizer.step()
```

### What the Model Learns

After SFT, the model can:
- ✅ Follow instructions
- ✅ Answer questions directly
- ✅ Have natural conversations
- ✅ Know when to stop (at `<|assistant_end|>`)
- ✅ Use tools (if trained on tool examples)

**Example:**
```
User: What is 2+2?
Assistant: The answer is 4.
```
Much better than the base model!

### Common SFT Datasets

- **Instruction-following**: "Explain photosynthesis" → explanation
- **Q&A**: "What is the capital of France?" → "Paris"
- **Reasoning**: "If x+3=7, what is x?" → "x=4"
- **Coding**: "Write a function to reverse a string" → code
- **Tool use**: "Calculate 123*456" → uses calculator tool

---

## Stage 4: Reinforcement Learning (RL)

### What is RL?

**Reinforcement Learning** further refines the model using reward signals instead of fixed training examples.

**Key idea:** Generate multiple responses, score them, train the model to generate higher-scoring responses.

### Why RL?

Some qualities are hard to capture in supervised examples:
- **Helpfulness**: Is this response actually useful?
- **Safety**: Does this avoid harmful content?
- **Style**: Is this friendly and conversational?
- **Conciseness**: Is this the right length?

### How RL Works

```
1. GENERATE RESPONSES
   Prompt: "Explain gravity"
   Model generates: Response A, Response B, Response C

2. SCORE RESPONSES
   Response A: Score = 8.5
   Response B: Score = 6.2
   Response C: Score = 9.1

3. UPDATE MODEL
   Increase probability of generating responses like C
   Decrease probability of generating responses like B
```

### Reward Sources

**Reward models:** Separate neural network trained on human preferences
```
Input: (prompt, response)
Output: score (higher = better)
```

**Rule-based rewards:**
- Length penalty (too short or too long)
- Format checking (does it follow instructions?)
- Safety filters (does it contain harmful content?)

### Implementation

```python
# From scripts/chat_rl.py (simplified)

for step in range(num_iterations):
    # Sample a prompt
    prompt = sample_prompt()

    # Generate response from current model
    response = model.generate(prompt)

    # Compute reward
    reward = reward_model(prompt, response)

    # Compute loss (policy gradient)
    loss = -reward * log_prob(response)

    # Update model to increase probability of high-reward responses
    loss.backward()
    optimizer.step()
```

### What RL Improves

After RL, the model is:
- ✅ More helpful
- ✅ Safer (less likely to generate harmful content)
- ✅ Better aligned with human preferences
- ✅ More consistent in style

**Note:** RL is optional in nanochat. SFT alone produces good results.

---

## Distributed Training

### Why Distributed Training?

**Problem:** A single GPU can't handle:
- Large batch sizes needed for stable training
- Fast enough training (4 hours instead of days)
- Large models (1.9B parameters)

**Solution:** Use multiple GPUs in parallel!

### Data Parallelism

**Each GPU processes different data**, same model:

```
GPU 0: Batch 0 → Forward → Backward → Gradients 0
GPU 1: Batch 1 → Forward → Backward → Gradients 1
GPU 2: Batch 2 → Forward → Backward → Gradients 2
GPU 3: Batch 3 → Forward → Backward → Gradients 3
        ↓
Average gradients across all GPUs
        ↓
Update model parameters (synchronized)
```

### PyTorch DDP (Distributed Data Parallel)

Nanochat uses PyTorch's DDP:

```python
# Launch with: torchrun --nproc_per_node=8 base_train.py

# Initialize process group
torch.distributed.init_process_group(backend="nccl")

# Wrap model in DDP
model = torch.nn.parallel.DistributedDataParallel(model)

# Training loop (same as before!)
for step in range(num_iterations):
    loss = model(x, y)
    loss.backward()  # Gradients automatically averaged across GPUs!
    optimizer.step()
```

### Gradient Accumulation

**Problem:** GPU memory limits batch size per device.

**Solution:** Accumulate gradients over multiple micro-batches before updating:

```python
grad_accum_steps = total_batch_size // (device_batch_size * num_gpus)

for step in range(num_iterations):
    for micro_step in range(grad_accum_steps):
        # Forward and backward, but don't update yet
        loss = model(x, y) / grad_accum_steps  # Normalize loss
        loss.backward()  # Accumulate gradients

        x, y = next(data_loader)  # Next micro-batch

    # Now update with accumulated gradients
    optimizer.step()
    optimizer.zero_grad()
```

**Example:**
- Desired total batch size: 524,288 tokens
- Device batch size: 32 sequences × 2048 tokens = 65,536 tokens
- Number of GPUs: 8
- Tokens per step: 8 × 65,536 = 524,288 ✓
- Gradient accumulation steps: 1 (no accumulation needed)

If we had only 4 GPUs:
- Tokens per step: 4 × 65,536 = 262,144
- Gradient accumulation steps: 524,288 / 262,144 = 2
- Each GPU processes 2 micro-batches before updating

---

## Optimization Algorithms

### Nanochat's Two-Optimizer Setup

Nanochat uses **two different optimizers** for different parameters:

1. **AdamW** for embeddings and language model head
2. **Muon** for transformer layers (linear layers)

### AdamW Optimizer

**AdamW** (Adam with Weight Decay) adapts learning rates per parameter:

```python
# Simplified AdamW
for param in parameters:
    # Exponential moving average of gradients
    m = beta1 * m + (1 - beta1) * grad

    # Exponential moving average of squared gradients
    v = beta2 * v + (1 - beta2) * grad²

    # Adaptive learning rate
    param_update = lr * m / (sqrt(v) + eps)

    # Weight decay (regularization)
    param = param - lr * weight_decay * param

    # Update parameter
    param = param - param_update
```

**Benefits:**
- ✅ Adaptive learning rates (different per parameter)
- ✅ Momentum (smooths updates)
- ✅ Works well for embeddings

**Used for:**
- Token embeddings (`wte`)
- Language model head (`lm_head`)

### Muon Optimizer

**Muon** is a novel optimizer designed for transformers:

**Key idea:** Treat weight matrices as manifolds and use momentum in the tangent space.

```python
# Simplified Muon
for weight_matrix in linear_layers:
    # Project gradient to tangent space
    projected_grad = project_to_tangent(grad, weight_matrix)

    # Apply momentum
    momentum = beta * momentum + (1 - beta) * projected_grad

    # Update in tangent space
    weight_matrix = weight_matrix - lr * momentum
```

**Benefits:**
- ✅ Better optimization geometry for matrix parameters
- ✅ More stable training
- ✅ Can use higher learning rates

**Used for:**
- All linear layers in transformer blocks
- Attention projections (`c_q`, `c_k`, `c_v`, `c_proj`)
- MLP layers (`c_fc`, `c_proj`)

### Learning Rate Schedules

**Learning rate** changes during training:

```python
def get_lr_multiplier(step):
    # Warmup: Gradually increase LR (first 0% of training)
    if step < warmup_steps:
        return (step + 1) / warmup_steps

    # Stable: Full learning rate (middle 80% of training)
    elif step < num_iterations - warmdown_steps:
        return 1.0

    # Warmdown: Gradually decrease LR (last 20% of training)
    else:
        progress = (num_iterations - step) / warmdown_steps
        return progress + (1 - progress) * final_lr_frac
```

**Why?**
- **Warmup**: Prevents instability early in training
- **Warmdown**: Allows model to settle into a good minimum
- **Result**: More stable, better final performance

### Gradient Clipping

**Gradient clipping** prevents exploding gradients:

```python
# Clip gradients to maximum norm
max_grad_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

**Why?** Occasionally gradients become very large, causing unstable updates. Clipping prevents this.

---

## Checkpointing and Resuming

### What is Checkpointing?

**Checkpointing** saves the model's state to disk so you can:
- Resume training if interrupted
- Use the model for inference later
- Share the trained model

### What Gets Saved

```python
checkpoint = {
    "model": model.state_dict(),           # Model parameters
    "optimizer": optimizer.state_dict(),    # Optimizer state
    "step": step,                          # Current step
    "config": model_config,                # Model architecture
    "metrics": {"val_loss": 2.3, ...}      # Training metrics
}
torch.save(checkpoint, "checkpoint.pt")
```

### Loading a Checkpoint

```python
# Create model with same config
model = GPT(config)

# Load saved parameters
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])

# Optionally resume training
optimizer.load_state_dict(checkpoint["optimizer"])
start_step = checkpoint["step"]
```

### Checkpoint Manager

Nanochat has a dedicated checkpoint manager (`checkpoint_manager.py`):

```python
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    checkpoint_dir="out/checkpoints/d20",
    step=5400,
    model_state=model.state_dict(),
    optimizer_states=[opt.state_dict() for opt in optimizers],
    metadata={"val_bpb": 1.234, ...}
)

# Load
checkpoint = load_checkpoint("out/checkpoints/d20")
model.load_state_dict(checkpoint["model"])
```

**Features:**
- Automatic directory creation
- Metadata storage (config, metrics)
- Safe saving (atomic writes)

---

## Code Walkthrough

### File: `scripts/base_train.py`

Let's walk through the pretraining script:

#### 1. Setup (lines 33-85)

```python
# Hyperparameters
depth = 20
max_seq_len = 2048
device_batch_size = 32
total_batch_size = 524288
num_iterations = 5400

# Initialize distributed training
ddp, rank, world_size, device = compute_init(device_type)

# Load tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
```

#### 2. Model Initialization (lines 107-119)

```python
# Model config
config = GPTConfig(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=depth,
    n_head=num_heads,
    n_embd=model_dim
)

# Create model
model = GPT(config)
model.init_weights()
model = torch.compile(model)  # Compile for speed
```

#### 3. Optimizer Setup (lines 142-144)

```python
optimizers = model.setup_optimizers(
    unembedding_lr=0.004,
    embedding_lr=0.2,
    matrix_lr=0.02,
    weight_decay=0.0
)
adamw_optimizer, muon_optimizer = optimizers
```

#### 4. Data Loading (lines 147-151)

```python
train_loader = tokenizing_distributed_data_loader(
    device_batch_size,
    max_seq_len,
    split="train",
    device=device
)
x, y = next(train_loader)  # Prefetch first batch
```

#### 5. Training Loop (lines 181-300)

```python
for step in range(num_iterations + 1):
    # Evaluation (every N steps)
    if step % eval_every == 0:
        val_bpb = evaluate_bpb(model, val_loader, eval_steps)

    # Checkpoint saving (at end)
    if last_step:
        save_checkpoint(checkpoint_dir, step, model.state_dict(), ...)

    if last_step:
        break

    # Training step
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)  # Prefetch next batch

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Update learning rate
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # Optimizer step
    for opt in optimizers:
        opt.step()

    model.zero_grad(set_to_none=True)
```

### File: `scripts/chat_sft.py`

#### Key Differences from Pretraining

```python
# Load conversation data
data_loader = load_chat_data()

for conversation in data_loader:
    # Render conversation with mask
    ids, mask = tokenizer.render_conversation(conversation)

    # Set non-masked tokens to -1 (ignored in loss)
    targets = ids.clone()
    targets[mask == 0] = -1

    # Compute loss (only on masked positions)
    loss = model(ids, targets)

    # Rest is same as pretraining
    loss.backward()
    optimizer.step()
```

---

## Key Takeaways

1. **Training adjusts parameters** to minimize loss (prediction error)

2. **Gradient descent** iteratively updates parameters in the direction that reduces loss

3. **Pretraining** teaches the model language patterns from raw text

4. **Mid-training** familiarizes the model with conversation format

5. **SFT** teaches the model to follow instructions and chat

6. **RL** further aligns the model with human preferences

7. **Distributed training** uses multiple GPUs for faster training

8. **Two optimizers**: AdamW for embeddings, Muon for transformers

9. **Checkpointing** allows saving and resuming training

10. **The complete pipeline** takes about 4 hours and $100 for a 561M parameter model

---

## What's Next?

Now that you understand how the model is trained, let's see how it generates text!

**→ Next: [Document 5: Inference and Text Generation](05_inference.md)**

You'll learn:
- Autoregressive generation
- Sampling strategies (temperature, top-k, top-p)
- KV cache optimization
- How the conversation format works

---

## Self-Check

Before moving on, make sure you understand:

- [ ] The difference between training and inference
- [ ] How loss measures prediction error
- [ ] What gradient descent does
- [ ] The four training stages and their purposes
- [ ] Why we mask user inputs during SFT
- [ ] How distributed training works
- [ ] Why nanochat uses two different optimizers
- [ ] What checkpointing is for
- [ ] The overall flow: pretraining → SFT → chat model
