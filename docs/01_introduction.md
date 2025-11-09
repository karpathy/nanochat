# Introduction to LLMs and PyTorch

This document will teach you the fundamental concepts you need to understand how nanochat works. We'll start from absolute basics and build up to the complete system.

## Table of Contents
1. [What is a Large Language Model?](#what-is-a-large-language-model)
2. [What is PyTorch?](#what-is-pytorch)
3. [How Does Machine Learning Work?](#how-does-machine-learning-work)
4. [The Nanochat System Overview](#the-nanochat-system-overview)
5. [Key Concepts and Terminology](#key-concepts-and-terminology)

---

## What is a Large Language Model?

### The Simple Answer

A **Large Language Model (LLM)** is a computer program that has learned to predict what words come next in a sentence by reading billions of examples. This simple ability - predicting the next word - turns out to be powerful enough to:

- Write essays
- Answer questions
- Write code
- Have conversations
- Solve math problems
- And much more

### An Analogy

Imagine you're reading a sentence: "The cat sat on the ___"

You can probably guess the next word might be "mat" or "floor" or "chair". How did you know? Because you've read thousands of sentences in your life and learned patterns about how language works.

An LLM does the same thing, but:
- It has read **trillions** of words (most of the internet!)
- It uses **mathematics** instead of intuition
- It has **billions of parameters** (numbers) that capture patterns

### The Technical Explanation

An LLM is a **neural network** - a mathematical function with billions of adjustable numbers (parameters). The function takes text as input and outputs a probability distribution over all possible next words.

```
Input:  "The cat sat on the"
Output: [("mat", 0.35), ("floor", 0.25), ("chair", 0.15), ("table", 0.10), ...]
         ↑ word          ↑ probability that this word comes next
```

The model is **trained** by:
1. Showing it billions of examples of text
2. Asking it to predict the next word
3. Adjusting its parameters when it's wrong
4. Repeating this process until it gets good at predicting

### Why "Large"?

Modern LLMs are "large" because they have:
- **Billions of parameters**: The nanochat model has 561 million to 1.9 billion parameters
- **Trained on trillions of tokens**: nanochat trains on 54+ billion tokens
- **Require significant compute**: Training costs $100-$1000 in GPU time

---

## What is PyTorch?

### The Simple Answer

**PyTorch** is a Python library that makes it easy to:
1. Build neural networks
2. Train them on data
3. Run them on GPUs (graphics cards that are fast at math)

Think of PyTorch as a toolkit that handles all the complicated math and GPU programming so you can focus on building your model.

### Core PyTorch Concepts

#### 1. Tensors (Multi-dimensional Arrays)

A **tensor** is just a fancy word for a multi-dimensional array of numbers.

```python
import torch

# A 1D tensor (like a list)
x = torch.tensor([1, 2, 3, 4])
print(x.shape)  # torch.Size([4])

# A 2D tensor (like a table/matrix)
y = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(y.shape)  # torch.Size([3, 2]) - 3 rows, 2 columns

# A 3D tensor (like a stack of tables)
z = torch.randn(2, 3, 4)  # 2 "sheets", each 3x4
print(z.shape)  # torch.Size([2, 3, 4])
```

**Why tensors?** Neural networks process data as multi-dimensional arrays. For example:
- A sentence might be `[10, 512]` - 10 words, each represented by 512 numbers
- A batch of sentences might be `[32, 10, 512]` - 32 sentences in the batch

#### 2. Automatic Differentiation (Autograd)

PyTorch can automatically compute gradients (derivatives) of functions. This is essential for training because we need to know how to adjust parameters to reduce errors.

```python
# Create a tensor that requires tracking for gradients
x = torch.tensor([2.0], requires_grad=True)

# Do some computation
y = x * x * 3  # y = 3x²

# Compute gradient: dy/dx = 6x = 6*2 = 12
y.backward()
print(x.grad)  # tensor([12.])
```

**Why autograd?** During training, we need to compute how each of the billions of parameters affects the error. PyTorch does this automatically.

#### 3. GPU Acceleration

PyTorch can run computations on GPUs, which are much faster than CPUs for the type of math neural networks do.

```python
# Create tensor on CPU
x = torch.tensor([1, 2, 3])

# Move to GPU
if torch.cuda.is_available():
    x = x.cuda()  # or x.to('cuda')
    # Now operations on x will run on GPU
```

#### 4. Neural Network Modules (nn.Module)

PyTorch provides building blocks for neural networks in `torch.nn`.

```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
        self.linear = nn.Linear(10, 5)  # Input: 10 numbers, Output: 5 numbers

    def forward(self, x):
        # Define how data flows through the model
        return self.linear(x)

model = SimpleModel()
```

**In nanochat**: The entire GPT model is defined as a `nn.Module` in `nanochat/gpt.py`.

---

## How Does Machine Learning Work?

### The Big Picture

Machine learning is teaching a computer to find patterns in data by showing it lots of examples. For LLMs specifically:

```
1. START WITH: A neural network with random parameters (it's terrible at predicting)
2. SHOW IT: Billions of examples of text
3. FOR EACH EXAMPLE:
   a. Ask it to predict the next word
   b. Compare its prediction to the actual next word
   c. Calculate how wrong it was (the "loss")
   d. Adjust the parameters slightly to be less wrong
4. REPEAT: Until it gets good at predicting
5. END WITH: A model that understands language patterns
```

### Key Concepts

#### Training vs. Inference

- **Training**: The process of adjusting parameters to make the model better (slow, expensive)
- **Inference**: Using a trained model to make predictions (fast, cheap)

Analogy: Training is like studying for an exam (hard work). Inference is like taking the exam (applying what you learned).

#### Loss Function

The **loss** measures how wrong the model's predictions are. Lower loss = better model.

For LLMs, we typically use **cross-entropy loss**:
- Model predicts probabilities for each possible next word
- Loss is high if it assigns low probability to the correct word
- Loss is low if it assigns high probability to the correct word

```python
# Simplified example
predicted = [0.1, 0.3, 0.6]  # Probabilities for 3 possible words
actual = 2  # The correct word is word #2 (index 2)

# Cross-entropy loss:
# -log(predicted[actual]) = -log(0.6) = 0.51
# If it predicted 0.9 for the correct word: -log(0.9) = 0.11 (better!)
```

#### Optimization (Gradient Descent)

**Optimization** is the algorithm for adjusting parameters to reduce loss.

Basic idea:
1. Compute the gradient: which direction to change each parameter to reduce loss
2. Update parameters: move them slightly in that direction
3. Repeat

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward pass: compute predictions
        predictions = model(batch)

        # 2. Compute loss
        loss = loss_function(predictions, targets)

        # 3. Backward pass: compute gradients
        loss.backward()

        # 4. Update parameters
        optimizer.step()  # This adjusts model parameters
        optimizer.zero_grad()  # Reset gradients for next batch
```

**Optimizers**: Different algorithms for updating parameters:
- **SGD**: Simple gradient descent
- **Adam**: Adaptive learning rates (popular)
- **AdamW**: Adam with weight decay (used in nanochat)
- **Muon**: Novel optimizer (also used in nanochat)

#### Hyperparameters

**Hyperparameters** are settings you choose before training (as opposed to parameters, which are learned during training):

- **Learning rate**: How big of steps to take when updating parameters
- **Batch size**: How many examples to show before updating parameters
- **Number of epochs**: How many times to go through the entire dataset
- **Model size**: Number of layers, dimensions, etc.

---

## The Nanochat System Overview

### What Nanochat Does

Nanochat is a **complete implementation** of training a ChatGPT-like model from scratch. It includes:

1. **Tokenizer training**: Learn how to break text into pieces
2. **Pretraining**: Train base model on raw internet text
3. **Fine-tuning**: Teach the model to chat and follow instructions
4. **Evaluation**: Measure how good the model is
5. **Deployment**: Serve the model via a web interface

### The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: TOKENIZER TRAINING                                 │
│ Input:  Raw text from internet                              │
│ Output: A trained tokenizer (vocabulary)                    │
│ Script: scripts/tok_train.py                                │
│ Cost:   ~Free (fast on CPU)                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: BASE MODEL PRETRAINING                             │
│ Input:  54B tokens of raw internet text                     │
│ Output: A base model that can predict next words            │
│ Script: scripts/base_train.py                               │
│ Cost:   ~$100 (4 hours on GPU)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: MID-TRAINING (Optional)                            │
│ Input:  Conversation-formatted data                         │
│ Output: Model that understands chat format                  │
│ Script: scripts/mid_train.py                                │
│ Cost:   Included in main training                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: SUPERVISED FINE-TUNING (SFT)                       │
│ Input:  Question-answer pairs, instructions                 │
│ Output: Model that follows instructions                     │
│ Script: scripts/chat_sft.py                                 │
│ Cost:   Included in main training                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 5: REINFORCEMENT LEARNING (RL) (Optional)             │
│ Input:  Reward signals / preferences                        │
│ Output: Model aligned with human preferences                │
│ Script: scripts/chat_rl.py                                  │
│ Cost:   Additional time                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 6: EVALUATION                                         │
│ Input:  Trained model + benchmark tasks                     │
│ Output: Performance metrics (accuracy, etc.)                │
│ Script: scripts/chat_eval.py                                │
│ Cost:   ~Free (fast inference)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 7: DEPLOYMENT                                         │
│ Input:  Trained model                                       │
│ Output: Web UI for chatting                                 │
│ Script: scripts/chat_web.py                                 │
│ Cost:   ~Free (runs on your computer)                       │
└─────────────────────────────────────────────────────────────┘
```

### File Organization

```
nanochat/
├── nanochat/           # Core library (the "engine")
│   ├── gpt.py         # THE MODEL: Neural network architecture
│   ├── engine.py      # TEXT GENERATION: How model creates responses
│   ├── tokenizer.py   # TEXT → NUMBERS: Converting text to tokens
│   ├── dataloader.py  # DATA LOADING: Feeding data during training
│   ├── dataset.py     # DATA MANAGEMENT: Downloading/managing datasets
│   ├── adamw.py       # OPTIMIZER: How parameters are updated
│   ├── muon.py        # OPTIMIZER: Alternative optimization algorithm
│   └── ...            # Other utilities
│
├── scripts/            # Executable programs
│   ├── tok_train.py   # Train the tokenizer
│   ├── base_train.py  # Pretrain the base model
│   ├── chat_sft.py    # Fine-tune for chat
│   ├── chat_web.py    # Run the web interface
│   └── ...            # Other scripts
│
├── rustbpe/            # Fast tokenizer training (Rust code)
│   └── src/lib.rs     # BPE algorithm implementation
│
├── tasks/              # Evaluation benchmarks
│   ├── arc.py         # Science questions
│   ├── gsm8k.py       # Math problems
│   ├── humaneval.py   # Code generation
│   └── ...            # Other tasks
│
├── speedrun.sh         # ONE SCRIPT TO RUN EVERYTHING (4 hours, $100)
└── run1000.sh          # Larger model version (41.6 hours, $800)
```

### Design Philosophy

Nanochat is designed to be:

1. **Minimal**: ~5,700 lines of code (vs millions in typical ML codebases)
2. **Hackable**: Easy to understand and modify
3. **Self-contained**: Few dependencies, no complex frameworks
4. **Educational**: Clear code that teaches how LLMs work
5. **Practical**: Actually produces a working ChatGPT-like model

---

## Key Concepts and Terminology

### Model Architecture Terms

| Term | What It Means | Nanochat Example |
|------|---------------|------------------|
| **Parameter** | A number in the model that gets learned | 561 million parameters in base model |
| **Layer** | A single transformation step in the network | 20-32 layers depending on model size |
| **Embedding** | Converting a token ID to a vector | Token 42 → [0.1, -0.3, 0.7, ...] (512 numbers) |
| **Attention** | Mechanism for looking at previous tokens | Multi-Query Attention in nanochat |
| **Hidden dimension** | Size of internal vectors | 512 in nanochat |
| **Vocabulary** | All possible tokens the model knows | 65,536 tokens (2^16) |
| **Context window** | Maximum length of text model can process | 1024 tokens in nanochat |

### Training Terms

| Term | What It Means | Nanochat Example |
|------|---------------|------------------|
| **Token** | A piece of text (part of a word) | "Hello" might be token ID 1234 |
| **Batch** | Group of examples processed together | 32 sequences at once |
| **Epoch** | One pass through the entire dataset | Multiple epochs in SFT |
| **Learning rate** | How fast parameters are updated | Starts high, decreases over time |
| **Loss** | How wrong the model is (lower = better) | Cross-entropy loss |
| **Checkpoint** | Saved snapshot of model parameters | Saved every N steps |
| **Gradient** | Direction to change parameters | Computed via backpropagation |

### Data Terms

| Term | What It Means | Nanochat Example |
|------|---------------|------------------|
| **Pretraining data** | Raw text from internet | FineWeb dataset |
| **Fine-tuning data** | Instruction-response pairs | Q&A, conversations |
| **Dataset shard** | One chunk of a large dataset | Each shard is ~100MB |
| **Data loader** | Code that feeds data to model | `dataloader.py` |
| **Tokenization** | Converting text to token IDs | "Hi!" → [12, 0, 256] |

### Inference Terms

| Term | What It Means | Nanochat Example |
|------|---------------|------------------|
| **Autoregressive** | Generating one token at a time | Each token depends on previous ones |
| **Sampling** | Choosing the next token | Temperature, top-p, top-k |
| **Temperature** | Controls randomness (higher = more random) | 0.9 in nanochat |
| **KV cache** | Optimization to avoid recomputing | Stores previous attention states |
| **Prompt** | Input text you give the model | "What is 2+2?" |
| **Completion** | Text the model generates | "The answer is 4." |

### Special Nanochat Terms

| Term | What It Means | Location in Code |
|------|---------------|------------------|
| **CORE score** | Benchmark for base model quality | `core_eval.py` |
| **BPB** | Bits per byte (compression metric) | `loss_eval.py` |
| **Muon optimizer** | Novel optimization algorithm | `muon.py` |
| **RustBPE** | Fast tokenizer training in Rust | `rustbpe/` |
| **Tool use** | Model can use calculator, run code | `execution.py` |

---

## What's Next?

Now that you understand the big picture, let's dive into the details:

**→ Next: [Document 2: Tokenization](02_tokenization.md)**

You'll learn:
- Why we need tokenization
- How Byte Pair Encoding works
- How to train a tokenizer
- How the tokenizer is used in practice

---

## Quick Self-Check

Before moving on, make sure you understand:

- [ ] What an LLM does (predicts next word)
- [ ] What PyTorch is (a toolkit for building neural networks)
- [ ] The difference between training and inference
- [ ] The 7 stages of the nanochat pipeline
- [ ] Where to find the main model code (`gpt.py`)
- [ ] What a parameter is (a number that gets learned)
- [ ] What a token is (a piece of text)

If any of these are unclear, re-read the relevant section. These are foundational concepts that everything else builds on!
