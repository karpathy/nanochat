# Custom Debugging Scripts (Mac M2 8GB RAM)

This directory (and fork) is aimed at adding custom debugging and visualization scripts to `nanochat`. All experiments and configurations here are optimized for a **Mac M2 with 8GB of RAM**.

## Available Scripts

### 1. Token Inspection (`inspect_tokens.py`)
Visualizes the trained vocabulary, showing special tokens and samples of merged tokens.

**How to run:**
From the root directory, run:
```bash
source .venv/bin/activate
python -m scripts.custom.inspect_tokens
```

**Output:**
- Total vocabulary size.
- Samples of "merges" (common character combinations).

### 2. Checkpoint Inspection (`inspect_checkpoint.py`)
Loads a saved checkpoint (weights, metadata, and optimizer state) and prints summary statistics to ensure training is healthy (no NaNs, weights are changing).

**How to run:**
```bash
python -m scripts.custom.inspect_checkpoint
```

### 3. Model Sampling (`sample.py` or `base_eval.py`)
Generates text from a saved base model checkpoint. Note that extremely small models or those early in training (like 100 steps) will likely produce gibberish.

**Using `base_eval.py` (standard):**
```bash
python -m scripts.base_eval --model-tag d6 --step 800 --eval sample
```

**Using `sample.py` (custom prompt/tokens):**
```bash
python -m scripts.custom.sample --prompt "The capital of France is" --num-tokens 50
```

### 4. Dataset Inspection (`inspect_dataset.py`)
Reads and prints the actual text content of the training shards (ClimbMix-400B) to understand what the model is learning from.

**How to run:**
```bash
python -m scripts.custom.inspect_dataset --num-shards 1 --num-docs 5
```

**Optional arguments:**
- `--num-shards`: Number of shards to sample from (default: 1).
- `--num-docs`: Number of documents to show per shard (default: 3).
- `--shard-index`: Specific shard index to inspect (e.g., `--shard-index 0`).

---

## Pretraining (Mac M2 Optimized)

To train a "tiny" model that fits within 8GB of RAM, use the following configuration:

```bash
source .venv/bin/activate
python -m scripts.base_train \
    --depth=2 \
    --max-seq-len=256 \
    --device-batch-size=1 \
    --total-batch-size=256 \
    --window-pattern=L \
    --num-iterations=100 \
    --core-metric-every=-1 \
    --sample-every=20 \
    --run=my-first-run \
    --wandb-log-every=1
```

## Training Log Explanation

When running `base_train.py`, you will see lines like this in your console:
`step 00499/04000 (12.47%) | loss: 7.457520 | lrm: 1.00 | dt: 19.05ms | tok/sec: 13,435 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.16m | eta: 1.1m`

- **step**: Current iteration / Total planned iterations (Percentage complete).
- **loss**: The "Cross-Entropy Loss". Measures how poorly the model predicted the next token. (10.4 is random; < 4.0 is getting coherent).
- **lrm**: Learning Rate Multiplier. Starts at 0 (warmup), goes to 1.0, then decays (warmdown) to 0.05 at the end.
- **dt**: Delta Time. Time taken for one full training step (forward + backward pass).
- **tok/sec**: Throughput. Number of text tokens processed per second. Higher is better!
- **bf16_mfu**: Model FLOPs Utilization. Efficiency of the GPU usage (Note: This is usually 0.00 on Mac/MPS as it's optimized for NVIDIA).
- **epoch**: How many times we've looped through the entire dataset.
- **pq / rg**: Internal data loader indices. Useful for technical debugging of data resumes.
- **total time**: Clock time since training started.
- **eta**: Estimated time remaining until the run finishes.
|
## Evaluation Metrics Explained

To understand how "smart" or "healthy" your model is, we use several key metrics across different scripts.

### 1. CORE Metric (`--eval core`)
**What it is:** The **DCLM CORE** benchmark. It measures the base model's zero/few-shot intelligence.
- **How it works:** The model is given a few examples (In-Context Learning) and asked to solve a variety of tasks (science, reasoning, etc.).
- **What to look for:** A score of **0.0** means random guessing. A score of **~0.25** is roughly GPT-2 capability. Higher is better.

### 2. Bits Per Byte (BPB) (`--eval bpb`)
**What it is:** A measure of how well a language model has "compressed" its training data.
- **Why it's better than Loss:** Raw loss depends on your tokenizer (vocab size). If Model A has 50k tokens and Model B has 100k, their raw losses are not comparable. BPB normalizes the loss by the **actual number of bytes** (UTF-8 characters) each token represents.
- **How it's calculated:** `BPB = Total Loss (Nats) / (ln(2) * Total Bytes)`. This converts the loss into bits per character.
- **Interpretations:**
    - **8.0 BPB**: Roughly random guessing for standard ASCII text.
    - **~1.0 BPB**: A very strong model (approaching GPT-2/GPT-3 level compression).
- **Overfitting check:** If your `train bpb` is significantly lower than your `val bpb`, your model is memorizing the training data rather than learning general patterns.

### 3. Sampling (`--eval sample`)
**What it is:** A qualitative "sanity check." The model is given a prompt to see if it generates something that makes sense to a human.
- **Conditioned Sampling:** Prompting with "The capital of France is" to see if it says "Paris."
- **Unconditioned Sampling:** Allowing the model to speak freely from a blank slate to see its "inner thoughts."

### 4. Loss Function (The "loss" in logs)
**What it is:** `nanochat` uses **Cross-Entropy Loss**, but with a modern stability trick called **Logit Soft-Capping**.
- **The Trick:** Before calculating the loss, logits are "squashed" into the range **[-15, 15]** using `F.tanh`. This prevents any single prediction from becoming too extreme, making training much more stable and preventing "loss spikes."
- **Precision:** While training happens in `bfloat16`, the loss is calculated in `float32` (full precision) to ensure numerical accuracy.
- **Conversion to BPB:** The raw loss (calculated in "nats") is converted to bits (base-2) and normalized by the number of bytes to produce the **BPB** score.
- **Theoretical Minimum:** 
    - In pure math, the minimum loss is **0.0** (if the model predicts the correct token with 100% probability).
    - In practice, the true minimum is determined by the **Entropy of the Source Data** (the "Bayes Error Rate").
    - For example, if your dataset was just the letter "A" repeated a billion times, the minimum loss would be 0.0. 
    - For **Natural Language**, the entropy is typically estimated to be around **0.7 to 1.1 BPB**. You will never reach 0.0 on real language data unless you are overfitting (memorizing).

#### A Note on Information Theory (Shannon Entropy)
When we say common English has an entropy of ~1.0 BPB, we are referring to the **Entropy Rate** of the language, which depends on **contextual knowledge**:
1.  **Symbolic Entropy (The Baseline)**: If you look at every byte in isolation, English has about **4.5 bits per character**. This is the entropy if you had *zero* knowledge of the language other than how often individual letters appear.
2.  **Conditional Entropy (Knowledge of Language)**: As you learn the "rules" of language (grammar, common phrases, logic), the predictability of the next character increases. Knowing "The capital of Fra..." makes "n" almost certain because your *model of language* allows you to use conditional probabilities.
3.  **Entropy Rate (The Limit)**: This is the limit as context goes to infinity. It represents the absolute minimum number of bits needed (on average) to describe each character if you know *all* the rules and facts about the world.
4.  **The Model's Job**: Modern LLMs like `nanochat` are essentially trying to internalize "standard knowledge of language" into their weights to reach this 1.0 limit. A lower BPB means your model has a more accurate "knowledge of language."

So, when adding human knowledge to the computation, we are effectively reducing the entropy of the language model from 4.5 BPB to 1.0 BPB.

### 5. ChatCORE (SFT/RL stage)
**What it is:** Specifically for models that have undergone Supervised Fine-Tuning (SFT). It tests the model's ability to follow instructions and chat.
- **Tasks included:**
  - **MMLU / ARC**: General knowledge and multiple-choice reasoning.
  - **GSM8K**: Grade school math problems.
  - **HumanEval**: Basic Python coding tasks.
  - **SpellingBee**: Testing the model's ability to count letters (e.g., "how many 'r' in strawberry?").
- **What to look for:** Since these are harder tasks, any score above **0.0** (random) means the model is actually starting to "understand" instructions.

---

---
