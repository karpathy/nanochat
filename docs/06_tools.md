# Tools and Capabilities

This document explains how nanochat integrates tools (calculator, Python code execution) that the model can use during generation. This makes the model more powerful than pure text generation.

## Table of Contents
1. [Why Tools?](#why-tools)
2. [Calculator Tool](#calculator-tool)
3. [Python Code Execution](#python-code-execution)
4. [How the Model Learns to Use Tools](#how-the-model-learns-to-use-tools)
5. [Implementation Details](#implementation-details)
6. [Code Walkthrough](#code-walkthrough)

---

## Why Tools?

### The Problem with Pure Language Models

Language models are trained on text and can only output text. This creates limitations:

```
User: What is 123,456 * 789?
Model (no tools): "Approximately 97 million" ❌ (wrong! actual: 97,406,784)

User: How many times does 'a' appear in 'banana'?
Model (no tools): "Three times" ❌ (wrong! actual: 3 'a's)
```

**Why errors?** The model is **approximating** based on patterns in training data, not actually computing.

### The Solution: Tool Use

**Give the model access to tools** that can perform exact computations:

```
User: What is 123,456 * 789?
Model: <|python_start|>123456 * 789<|python_end|>
       <|output_start|>97406784<|output_end|>
       The answer is 97,406,784. ✓

User: How many times does 'a' appear in 'banana'?
Model: <|python_start|>"banana".count("a")<|python_end|>
       <|output_start|>3<|output_end|>
       The letter 'a' appears 3 times in 'banana'. ✓
```

**Benefits:**
- ✅ Exact calculations (no approximation errors)
- ✅ Can handle tasks impossible for pure LLMs
- ✅ More trustworthy for quantitative questions
- ✅ Separates "thinking" from "computing"

---

## Calculator Tool

### What it Does

The calculator tool evaluates simple mathematical expressions:

```python
"2 + 2"           → 4
"123.45 * 67.89"  → 8381.5005
"(100 - 20) / 4"  → 20.0
```

### How it Works

```python
# From nanochat/engine.py

def use_calculator(expr):
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Safety check: only allow math characters
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # Disallow power operator (too expensive)
            return None
        # Evaluate safely with timeout
        return eval_with_timeout(expr, max_time=3)

    return None
```

**Safety features:**
- ✅ Only allows digits, operators, parentheses
- ✅ No power operator (prevents `9**9**9` type bombs)
- ✅ 3-second timeout (prevents infinite loops)
- ✅ Restricted eval (no access to builtins, file system, etc.)

### Example Usage by Model

```
User: Calculate 15% of 240

Model generates:
<|assistant_start|>
Let me calculate that.
<|python_start|>0.15 * 240<|python_end|>
```

Engine intercepts `<|python_end|>`:
1. Extracts expression: `0.15 * 240`
2. Evaluates: `36.0`
3. Injects output tokens: `<|output_start|>36.0<|output_end|>`

Model continues:
```
<|output_start|>36.0<|output_end|>
15% of 240 is 36.
<|assistant_end|>
```

---

## Python Code Execution

### Extended Calculator Features

In addition to pure math, the calculator can handle string operations:

```python
'"hello".count("l")'  → 2
'"banana".count("a")' → 3
```

**Implementation:**
```python
def use_calculator(expr):
    # ... math expression handling ...

    # Allow string operations if safe
    allowed_chars = "abcdefghijklmnopqrstuvwxyz0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Only allow .count() method
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)
```

### Example: Letter Counting

```
User: How many times does the letter 'e' appear in 'cheese'?

Model:
<|assistant_start|>
Let me count that.
<|python_start|>"cheese".count("e")<|python_end|>
<|output_start|>3<|output_end|>
The letter 'e' appears 3 times in 'cheese'.
<|assistant_end|>
```

### Safety Considerations

**Why restricted eval?**

Python's `eval()` is dangerous if unrestricted:

```python
# Dangerous examples (BLOCKED in nanochat):
eval("__import__('os').system('rm -rf /')")  # Delete files!
eval("open('/etc/passwd').read()")           # Read sensitive files!
```

**Nanochat's protections:**
1. Character whitelist (only alphanumeric, quotes, parens, etc.)
2. Keyword blacklist (`import`, `exec`, `open`, `__`, etc.)
3. No builtins access (`{"__builtins__": {}}`)
4. Timeout (3 seconds max)
5. Only allows specific operations (`.count()`)

---

## How the Model Learns to Use Tools

### Training Data Format

To teach the model tool use, include examples in SFT data:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is 12 * 34?"
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Let me calculate that."},
        {"type": "python", "text": "12 * 34"},
        {"type": "python_output", "text": "408"},
        {"type": "text", "text": "The answer is 408."}
      ]
    }
  ]
}
```

**Converted to tokens:**
```
<|bos|>
<|user_start|>What is 12 * 34?<|user_end|>
<|assistant_start|>
  Let me calculate that.
  <|python_start|>12 * 34<|python_end|>
  <|output_start|>408<|output_end|>
  The answer is 408.
<|assistant_end|>
```

**Training mask:**
```
mask = [0, 0, 0, 0, 0,      # User input and special tokens
        1, 1, 1,            # "Let me calculate that."
        1, 1, 1, 1,         # <|python_start|>, "12", "*", "34", <|python_end|>
        0, 0, 0,            # <|output_start|>, "408", <|output_end|> (not trained!)
        1, 1, 1, 1, 1]      # "The answer is 408."
```

**Key point:** Model learns to:
- ✅ Generate `<|python_start|>`
- ✅ Generate the expression
- ✅ Generate `<|python_end|>`
- ❌ NOT generate the output (that comes from the tool!)

### What the Model Learns

After training on tool-use examples:

1. **When to use tools:** "If the question requires calculation, use Python"
2. **How to format:** Wrap expression in special tokens
3. **What expressions work:** Valid Python syntax
4. **How to use results:** Reference the output in the response

### Generating Synthetic Training Data

Create tool-use examples programmatically:

```python
# From dev/gen_synthetic_data.py

# Generate random math problems
for i in range(1000):
    a, b = random.randint(1, 1000), random.randint(1, 1000)
    question = f"What is {a} * {b}?"
    answer = a * b

    example = {
        "messages": [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": [
                    {"type": "python", "text": f"{a} * {b}"},
                    {"type": "python_output", "text": str(answer)},
                    {"type": "text", "text": f"The answer is {answer}."}
                ]
            }
        ]
    }
    # Add to training data
```

---

## Implementation Details

### The Tool Use State Machine

During generation, the engine tracks tool use state:

```python
class RowState:
    def __init__(self):
        self.in_python_block = False      # Are we inside <|python_start|>?
        self.python_expr_tokens = []      # Tokens of the expression
        self.forced_tokens = deque()      # Tokens to inject next
```

**State transitions:**

```
Normal generation
    ↓ (model generates <|python_start|>)
Enter Python block (in_python_block = True)
    ↓ (accumulate tokens)
Collecting expression (python_expr_tokens = [12, *, 34])
    ↓ (model generates <|python_end|>)
Exit Python block (in_python_block = False)
    ↓ (evaluate expression)
Inject output (<|output_start|>, 408, <|output_end|>)
    ↓
Resume normal generation
```

### The Generation Loop with Tools

```python
# From nanochat/engine.py

for step in generation_loop:
    # Sample next token from model
    next_token = sample_from_model(logits)

    # Check if we're entering/exiting Python block
    if next_token == python_start:
        state.in_python_block = True
        state.python_expr_tokens = []

    elif next_token == python_end and state.in_python_block:
        # We just exited a Python block
        state.in_python_block = False

        # Decode the expression
        expr = tokenizer.decode(state.python_expr_tokens)

        # Evaluate it
        result = use_calculator(expr)

        # Inject output tokens
        if result is not None:
            result_tokens = tokenizer.encode(str(result))
            state.forced_tokens.append(output_start)
            state.forced_tokens.extend(result_tokens)
            state.forced_tokens.append(output_end)

    elif state.in_python_block:
        # Accumulate expression tokens
        state.python_expr_tokens.append(next_token)

    # If we have forced tokens, use them instead of sampling
    if state.forced_tokens:
        next_token = state.forced_tokens.popleft()

    yield next_token
```

---

## Code Walkthrough

### File: `nanochat/engine.py`

#### 1. Calculator Function (lines 35-79)

```python
def use_calculator(expr):
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Pure math expression
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)

    # String operations
    allowed_chars = "abcdefghijklmnopqrstuvwxyz..."
    if not all([x in allowed_chars for x in expr]):
        return None

    # Dangerous patterns
    dangerous = ['__', 'import', 'exec', ...]
    if any(pattern in expr.lower() for pattern in dangerous):
        return None

    # Only .count() for now
    if '.count(' not in expr:
        return None

    return eval_with_timeout(expr)
```

#### 2. RowState Class (lines 176-183)

```python
class RowState:
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()      # Queue of tokens to force
        self.in_python_block = False      # Inside <|python_start|>?
        self.python_expr_tokens = []      # Expression being accumulated
        self.completed = False            # Generation finished?
```

#### 3. Tool Use Logic in Generation (lines 273-289)

```python
# In the main generation loop

if next_token == python_start:
    # Entering Python block
    state.in_python_block = True
    state.python_expr_tokens = []

elif next_token == python_end and state.in_python_block:
    # Exiting Python block
    state.in_python_block = False

    # Evaluate the expression
    if state.python_expr_tokens:
        expr = self.tokenizer.decode(state.python_expr_tokens)
        result = use_calculator(expr)

        # Inject output tokens
        if result is not None:
            result_tokens = self.tokenizer.encode(str(result))
            state.forced_tokens.append(output_start)
            state.forced_tokens.extend(result_tokens)
            state.forced_tokens.append(output_end)

elif state.in_python_block:
    # Accumulating expression
    state.python_expr_tokens.append(next_token)
```

### File: `nanochat/tokenizer.py`

#### Rendering Tool Use in Conversations (lines 310-336)

```python
def render_conversation(self, conversation):
    # ... normal conversation rendering ...

    for part in message["content"]:
        if part["type"] == "text":
            # Regular text
            add_tokens(self.encode(part["text"]), mask=1)

        elif part["type"] == "python":
            # Python expression (model generates this)
            add_tokens([python_start], mask=1)
            add_tokens(self.encode(part["text"]), mask=1)
            add_tokens([python_end], mask=1)

        elif part["type"] == "python_output":
            # Tool output (NOT generated by model!)
            add_tokens([output_start], mask=0)  # mask=0!
            add_tokens(self.encode(part["text"]), mask=0)
            add_tokens([output_end], mask=0)
```

---

## Key Takeaways

1. **Tools extend LLM capabilities** beyond pure text generation to exact computation

2. **Calculator tool** safely evaluates math expressions and simple string operations

3. **Special tokens** structure tool use: `<|python_start|>`, `<|python_end|>`, etc.

4. **Training data** teaches the model when and how to use tools

5. **State machine** tracks tool use during generation and injects outputs

6. **Safety is critical** - restricted eval with whitelists, blacklists, and timeouts

7. **Tool outputs are forced** - not generated by the model, injected by the engine

8. **Training mask = 0 for outputs** - model learns to generate expressions, not results

---

## What's Next?

Now that you understand how tools work, let's see how models are evaluated!

**→ Next: [Document 7: Evaluation and Benchmarks](07_evaluation.md)**

You'll learn:
- How to measure model quality
- CORE score for base models
- Task-based evaluation (ARC, GSM8K, MMLU, HumanEval)
- Bits-per-byte metrics

---

## Self-Check

Before moving on, make sure you understand:

- [ ] Why tools are needed (LLMs can't compute exactly)
- [ ] How the calculator tool works
- [ ] What safety measures protect against malicious eval
- [ ] How the model learns tool use from training data
- [ ] The tool use state machine (enter block → accumulate → exit → inject output)
- [ ] Why tool outputs have mask=0 in training
- [ ] The special tokens for tool use
