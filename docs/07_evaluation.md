# Evaluation and Benchmarks

This document explains how to measure the quality of language models. We'll cover metrics used during training (bits-per-byte) and benchmark tasks that test specific capabilities.

## Table of Contents
1. [Why Evaluation Matters](#why-evaluation-matters)
2. [Bits-Per-Byte (BPB) Metric](#bits-per-byte-bpb-metric)
3. [CORE Score for Base Models](#core-score-for-base-models)
4. [Task-Based Evaluation](#task-based-evaluation)
5. [Benchmark Tasks](#benchmark-tasks)
6. [Running Evaluations](#running-evaluations)
7. [Interpreting Results](#interpreting-results)

---

## Why Evaluation Matters

### The Challenge of Measuring LLM Quality

How do you know if your model is good? Unlike traditional ML (where you have clear metrics like accuracy), LLMs are complex:

**Problems:**
- ❌ No single "correct" answer for most prompts
- ❌ Multiple valid responses ("The cat is on the mat" vs "A feline rests on the rug")
- ❌ Hard to quantify "quality" of free-form text
- ❌ Different use cases need different capabilities

### Nanochat's Evaluation Strategy

**Multiple complementary metrics:**

1. **Bits-per-byte (BPB)**: How well does the model compress/predict text?
2. **CORE score**: Aggregate performance on diverse tasks
3. **Task-specific benchmarks**: Test individual capabilities
   - Reasoning (ARC)
   - Math (GSM8K)
   - Knowledge (MMLU)
   - Coding (HumanEval)
   - Conversation (SmolTalk)

---

## Bits-Per-Byte (BPB) Metric

### What is BPB?

**Bits-per-byte** measures how efficiently the model compresses text. Lower is better.

**Intuition:** Think of the model as a compression algorithm:
- Good model: Predicts text well → high compression → low BPB
- Bad model: Can't predict text → poor compression → high BPB

### The Math

```python
# For each byte in the text:
# 1. Model predicts probability of that byte
# 2. Compute bits needed to encode it: -log2(probability)

total_bits = 0
for byte in text:
    prob = model.predict_probability(byte)
    bits = -log2(prob)
    total_bits += bits

bpb = total_bits / len(text)
```

**Example:**
```
Text: "The cat"
Bytes: [T, h, e, ' ', c, a, t] (7 bytes)

Model predictions:
  P("T" | "")      = 0.5  → -log2(0.5) = 1.0 bits
  P("h" | "T")     = 0.8  → -log2(0.8) = 0.32 bits
  P("e" | "Th")    = 0.9  → -log2(0.9) = 0.15 bits
  P(" " | "The")   = 0.95 → -log2(0.95) = 0.07 bits
  P("c" | "The ") = 0.6  → -log2(0.6) = 0.74 bits
  P("a" | "The c") = 0.85 → -log2(0.85) = 0.23 bits
  P("t" | "The ca")= 0.9  → -log2(0.9) = 0.15 bits

Total: 2.66 bits
BPB: 2.66 / 7 = 0.38
```

### Why BPB?

**Advantages over cross-entropy loss:**
- ✅ Interpretable: "Model needs 0.8 bits per byte on average"
- ✅ Comparable across datasets: Not affected by tokenization
- ✅ Connection to information theory: Theoretical minimum is entropy of language

**Typical values:**
- **Random model**: ~8 bits/byte (log2(256) for random byte)
- **Good small model**: ~1.5-2 bits/byte
- **Good large model**: ~0.7-1 bits/byte
- **Perfect model**: ~0.5-0.7 bits/byte (entropy of English)

### Implementation

```python
# From nanochat/loss_eval.py

def evaluate_bpb(model, data_loader, num_steps, token_bytes):
    """
    Evaluate bits-per-byte on a dataset.

    Args:
        model: The GPT model
        data_loader: Iterator of (input, target) batches
        num_steps: How many batches to evaluate
        token_bytes: Tensor mapping token IDs to byte counts

    Returns:
        bpb: Average bits per byte
    """
    total_loss = 0
    total_bytes = 0

    for step in range(num_steps):
        x, y = next(data_loader)

        # Forward pass (computes cross-entropy loss)
        loss = model(x, y, loss_reduction='sum')

        # Count bytes in this batch
        num_bytes = token_bytes[y].sum()

        total_loss += loss.item()
        total_bytes += num_bytes.item()

    # Convert nats to bits and normalize by bytes
    bpb = (total_loss / total_bytes) / math.log(2)

    return bpb
```

---

## CORE Score for Base Models

### What is CORE?

**CORE** (Commonsense Reasoning Evaluation) is an aggregate metric for base models (before fine-tuning).

It tests:
- Reading comprehension
- Commonsense reasoning
- World knowledge
- Language understanding

**Source:** From the DCLM (DataComp for Language Models) paper

### How it Works

```
1. Run model on diverse tasks from CORE benchmark
2. For each task:
   - Compute accuracy
   - Center the score (subtract mean, divide by std dev)
3. Average centered scores across tasks
4. Final CORE score = average
```

**Typical values:**
- **Random baseline**: ~0 (by construction of centering)
- **Small model (d20)**: 0.3 - 0.5
- **Medium model (d26)**: 0.5 - 0.7
- **Large model (d32)**: 0.7 - 1.0
- **SOTA models**: 1.0+

### Why Center Scores?

**Problem:** Different tasks have different difficulty
- Task A: 90% accuracy (easy)
- Task B: 60% accuracy (hard)

**Solution:** Center each task relative to a baseline:
```
centered_score = (accuracy - baseline_mean) / baseline_std
```

This makes scores comparable across tasks.

### Running CORE Evaluation

```bash
# From scripts/base_eval.py
python -m scripts.base_eval \
    --checkpoint_dir=out/base_checkpoints/d20 \
    --max_per_task=500
```

**Output:**
```
Task: arc_challenge    Accuracy: 0.45  Centered: 0.52
Task: hellaswag        Accuracy: 0.62  Centered: 0.48
Task: mmlu             Accuracy: 0.38  Centered: 0.41
...
CORE Score: 0.47
```

---

## Task-Based Evaluation

### Why Task-Specific Benchmarks?

**BPB and CORE are general**, but we also want to test specific capabilities:
- Can the model do math?
- Can it write code?
- Does it have factual knowledge?

**Solution:** Curated benchmark tasks

### Evaluation Format

Most tasks use **few-shot prompting**:

```
<Examples>
Q: What is 2+2?
A: 4

Q: What is 3+5?
A: 8
</Examples>

<Test Question>
Q: What is 7+6?
A: ___ (model fills in)
```

The model learns the pattern from examples and applies it to the test question.

---

## Benchmark Tasks

### 1. ARC Challenge (Reasoning)

**What:** Multiple-choice science questions (3rd-8th grade level)

**Example:**
```
Question: Which of these is an example of liquid water?
A) Frost
B) Ice
C) Snow
D) Rain

Answer: D
```

**Measures:** Scientific reasoning, reading comprehension

**Difficulty:** Moderate (even SOTA models struggle with some questions)

### 2. GSM8K (Math)

**What:** Grade school math word problems

**Example:**
```
Question: Janet has 24 apples. She gives 8 to her friend and then buys
12 more. How many apples does Janet have now?

Answer: 24 - 8 + 12 = 28 apples
```

**Measures:** Arithmetic, multi-step reasoning

**Challenge:** Requires breaking down problems into steps

### 3. MMLU (Knowledge)

**What:** Multiple-choice questions across 57 subjects (STEM, humanities, social sciences)

**Example:**
```
Subject: World History
Question: Which civilization built Machu Picchu?
A) Aztecs
B) Mayans
C) Incas
D) Olmecs

Answer: C
```

**Measures:** Breadth of knowledge

**Difficulty:** Very hard (covers college-level topics)

### 4. HumanEval (Coding)

**What:** Python function completion

**Example:**
```
def has_close_elements(numbers, threshold):
    """
    Check if in given list of numbers, are any two numbers closer to each other
    than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    # Model completes this function
```

**Measures:** Code generation ability

**Evaluation:** Run generated code against test cases

### 5. SmolTalk (Conversation)

**What:** Evaluates conversational quality

**Example:**
```
Conversation:
User: Hi! How are you?
Assistant: [Model generates response]

Evaluated on:
- Politeness
- Coherence
- Relevance
- Naturalness
```

**Measures:** Chat quality

**Note:** More subjective than other benchmarks

### 6. Spelling Bee (Letter Counting)

**What:** Count letters in words

**Example:**
```
Question: How many times does 's' appear in 'mississippi'?
Answer: 4
```

**Measures:** Ability to perform simple algorithmic tasks

**Why hard:** LLMs process tokens, not individual letters!

---

## Running Evaluations

### Evaluating a Base Model

```bash
# BPB on validation set
python -m scripts.base_loss \
    --checkpoint_dir=out/base_checkpoints/d20

# CORE score
python -m scripts.base_eval \
    --checkpoint_dir=out/base_checkpoints/d20 \
    --max_per_task=500
```

### Evaluating a Chat Model

```bash
# Run all chat benchmarks
python -m scripts.chat_eval \
    --checkpoint_dir=out/chat_checkpoints/d20 \
    --max_per_task=100
```

**Output:**
```
Task: arc_challenge
  Accuracy: 0.52
  Examples evaluated: 100

Task: gsm8k
  Accuracy: 0.41
  Examples evaluated: 100

Task: mmlu
  Accuracy: 0.38
  Examples evaluated: 100

Task: humaneval
  Pass@1: 0.15
  Examples evaluated: 50

Overall Summary:
  Average accuracy: 0.44
```

### Custom Evaluation

```python
# From tasks/common.py

from tasks.arc import ARC
from nanochat.engine import Engine

# Load model
engine = Engine(model, tokenizer)

# Load task
task = ARC()

# Evaluate
correct = 0
for example in task.get_examples(max_examples=100):
    # Get model's answer
    answer = task.evaluate(engine, example)

    # Check correctness
    if answer == example['answer']:
        correct += 1

accuracy = correct / 100
print(f"Accuracy: {accuracy:.2%}")
```

---

## Interpreting Results

### Typical Performance Ranges

**Base model (d20, 561M params):**
- BPB: 1.2-1.4
- CORE: 0.4-0.5
- ARC: 30-40%
- GSM8K: 10-20%
- MMLU: 25-35%
- HumanEval: 5-10%

**Chat model (after SFT):**
- ARC: 40-50%
- GSM8K: 30-45% (big improvement with tool use!)
- MMLU: 35-45%
- HumanEval: 10-20%

**Comparison (GPT-4 class models):**
- ARC: 90%+
- GSM8K: 85%+
- MMLU: 80%+
- HumanEval: 60%+

### What Affects Performance?

**Model size:**
- Larger models (more parameters) → better performance
- d20 (561M) < d26 (1B) < d32 (1.9B)

**Training data:**
- More data → better performance
- Higher quality data → better performance

**Fine-tuning:**
- SFT improves task-following
- Tool use dramatically improves math/counting

**Context length:**
- Longer context → can see more examples in few-shot prompts

### When to Be Concerned

**Red flags:**
- BPB > 2.0: Model not learning well
- Accuracy near random chance (25% for 4-choice questions)
- Performance decreasing during training (overfitting)
- Big gap between train and val metrics (overfitting)

**What to do:**
- Check data quality
- Reduce learning rate
- Check for bugs in code
- Try longer training (if underfitting)

---

## Key Takeaways

1. **BPB (bits-per-byte)** measures text prediction efficiency (lower is better)

2. **CORE score** aggregates performance on diverse reasoning tasks

3. **Task-specific benchmarks** test individual capabilities (math, code, knowledge)

4. **Few-shot prompting** lets models learn from examples without fine-tuning

5. **Evaluation is critical** for understanding model capabilities and limitations

6. **Multiple metrics** give a complete picture of model quality

7. **Tool use dramatically improves** math and algorithmic task performance

8. **Small models** can still be useful for many tasks, despite lower scores than SOTA

---

## What's Next?

Now that you understand evaluation, let's get hands-on with training and using models!

**→ Next: [Document 8: Quick Start Guide](08_quickstart.md)**

You'll learn:
- How to install dependencies
- Running your first training
- Using the web interface
- Customizing models
- Troubleshooting

---

## Self-Check

Before moving on, make sure you understand:

- [ ] What bits-per-byte measures
- [ ] Why lower BPB is better
- [ ] What CORE score evaluates
- [ ] The different benchmark tasks (ARC, GSM8K, MMLU, HumanEval)
- [ ] How to run evaluations
- [ ] What typical performance numbers look like
- [ ] Why we need multiple evaluation metrics
