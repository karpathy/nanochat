# AIME Evaluation

This directory contains evaluation datasets and scripts for AIME (American Invitational Mathematics Examination) 2024 and 2025.

## Datasets

- **AIME 2024 Part I**: 15 problems (`MathArena/aime_2024_I`)
- **AIME 2024 Part II**: 15 problems (`MathArena/aime_2024_II`)
- **AIME 2024 (combined)**: 30 problems (`MathArena/aime_2024_I` + `MathArena/aime_2024_II`)
- **AIME 2025**: 30 problems (`MathArena/aime_2025`) - includes both Part I and Part II

## Preprocessing

To download and preprocess the datasets:

```bash
cd /path/to/nanochat-ws/nanochat
source .venv/bin/activate
python dev-aime/preprocess.py
```

This creates JSONL files in `dev-aime/data/` with the preprocessed data.

## Running Evaluation

AIME tasks are **not** included in the default auto-run list. You must explicitly specify them.

### Basic Usage (1 sample, greedy decoding)

```bash
cd nanochat
source .venv/bin/activate

# AIME 2024 Part I
python -m scripts.chat_eval -i sft -a AIME-2024-I

# AIME 2024 Part II
python -m scripts.chat_eval -i sft -a AIME-2024-II

# AIME 2024 (combined I + II by default)
python -m scripts.chat_eval -i sft -a AIME-2024

# AIME 2025
python -m scripts.chat_eval -i sft -a AIME-2025
```

### Pass@k Evaluation (temperature sampling)

For AIME, it's common to use pass@k with temperature sampling. Default is n=16 samples.

```bash
# Pass@k with default n=16, temperature=0.8
python -m scripts.chat_eval \
    -i sft \
    -a AIME-2024 \
    -n 16 \
    -t 0.8 \
    --passatk \
    --passatk-ks 1,2,4,8,16

# With distributed evaluation (multiple GPUs)
torchrun --nproc_per_node=2 -m scripts.chat_eval -- \
    -i sft \
    -a AIME-2025 \
    -n 16 \
    -t 0.8 \
    --passatk

# Save detailed results for later analysis
python -m scripts.chat_eval \
    -i sft \
    -a AIME-2024 \
    -n 16 \
    -t 0.8 \
    --passatk \
    --save-results dev-aime/results/aime_2024_results.jsonl
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --source` | Required | Model source: `sft`, `mid`, `rl`, or `base` |
| `-a, --task-name` | None | Task name: `AIME-2024-I`, `AIME-2024-II`, `AIME-2024`, `AIME-2025` |
| `-n, --num-samples` | 1 | Number of samples per problem (n in pass@k) |
| `-t, --temperature` | 0.0 | Sampling temperature (use >0 for pass@k) |
| `-k, --top-k` | 50 | Top-k sampling |
| `-m, --max-new-tokens` | 512 | Maximum tokens to generate |
| `--passatk` | False | Enable pass@k calculation |
| `--passatk-ks` | 1,2,4,8,16 | Comma-separated k values for pass@k |
| `--save-results` | None | Save detailed results to JSONL file |

### Answer Extraction

The evaluation extracts answers from `\boxed{...}` in the model's response and performs exact string matching with the ground truth integer answer.

Example response format expected:
```
Let me solve this step by step...
[reasoning]
So the answer is \boxed{42}
```

## Implementation Details

### Pass@k Calculation

We use the **unbiased pass@k estimator** from the OpenAI paper:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Where:
- `n` = total number of samples per question
- `c` = number of correct samples for that question
- `k` = the k in pass@k

This is computed efficiently as:
```
pass@k = 1 - prod(1 - k / (n - c + i)) for i in 1..k
```

### File Structure

```
dev-aime/
├── README.md                  # This file
├── preprocess.py              # Download and preprocess datasets
├── data/                      # Preprocessed datasets
│   ├── aime_2024_I.jsonl
│   ├── aime_2024_II.jsonl
│   ├── aime_2024.jsonl
│   └── aime_2025.jsonl
└── results/                   # Evaluation results (optional)
```

### Task Implementation

The task implementations are in `tasks/`:
- `aime_2024.py`: AIME2024I, AIME2024II, AIME2024 classes
- `aime_2025.py`: AIME2025 class

These tasks are registered in `scripts/chat_eval.py` but **not** in the default `all_tasks` list, so they won't run automatically during training.
