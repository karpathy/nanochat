# RL Training Instructions (CSC490 A4P3)

This guide is for the **teammate running the RL training** after completing SFT.
After training finishes, the repo owner will pull the eval logs and do the full analysis.

---

## Prerequisites

You need access to:
- This GitHub repo (clone it)
- The shared Modal workspace (`nanochat-ablation` app, `nanochat-vol` volume)
- The shared W&B project `nanochat-rl` under `yoyoliuuu`
- The shared HuggingFace repo `alvina-yang/csc490a4p2` (write access)

Install dependencies:
```bash
pip install uv
uv sync
```

Authenticate:
```bash
uv run modal setup          # follow browser login
uv run wandb login          # paste your W&B API key
```

---

## Step 1 — Upload your SFT checkpoint to HuggingFace

Your SFT training saves checkpoints as:
```
model_<step>.pt
meta_<step>.json
```

Upload **both files** to `alvina-yang/csc490a4p2` under a folder named `sft-teammate` (or any short name without spaces):

```
alvina-yang/csc490a4p2/
└── sft-teammate/
    ├── model_XXXXXX.pt
    └── meta_XXXXXX.json
```

You can upload via the HuggingFace web UI or CLI:
```bash
pip install huggingface_hub
huggingface-cli upload alvina-yang/csc490a4p2 ./path/to/model_XXXXXX.pt sft-teammate/model_XXXXXX.pt
huggingface-cli upload alvina-yang/csc490a4p2 ./path/to/meta_XXXXXX.json sft-teammate/meta_XXXXXX.json
```

---

## Step 2 — Configure the RL run

Open `nanochat_modal.py` and find the `TEAMMATE_*` block (around line 655):

```python
TEAMMATE_HF_REPO        = "alvina-yang/csc490a4p2"   # HuggingFace repo
TEAMMATE_CHECKPOINT     = "sft-teammate"              # ← folder name you used above
TEAMMATE_STEP           = None                        # ← set to your step number (e.g. 28000), or leave None to auto-detect
TEAMMATE_RUN_NAME       = "rl-gsm8k-teammate"        # W&B run name (keep this as-is)
TEAMMATE_GPU            = "H100:4"                   # 4x H100 recommended
TEAMMATE_EPOCHS         = 1
TEAMMATE_DEVICE_BATCH   = 8
TEAMMATE_EXAMPLES_STEP  = 64
TEAMMATE_NUM_SAMPLES    = 8
TEAMMATE_MAX_NEW_TOKENS = 512
```

Set `TEAMMATE_CHECKPOINT` to match the folder name you used in Step 1.
Set `TEAMMATE_STEP` to the exact step number (e.g. `28000`), or leave as `None` to auto-detect.

---

## Step 3 — Launch training (detached)

```bash
uv run modal run --detach nanochat_modal.py::run_rl_teammate
```

This will:
1. Download your SFT checkpoint from HuggingFace to the Modal volume
2. Start RL training on 4x H100 GPUs (~1 hour)
3. Return immediately — safe to close your terminal

**Monitor progress:**
- W&B: https://wandb.ai/yoyoliuuu/nanochat-rl  (run name: `rl-gsm8k-teammate`)
- Modal: https://modal.com/apps

---

## What gets logged

### W&B (live during training)
| Metric | Description |
|---|---|
| `reward` | Mean reward per step (fraction of correct answers across all rollouts) |
| `sequence_length` | Mean generated sequence length per step |
| `pass@1` … `pass@8` | Fraction of eval problems solved with k attempts (logged every 60 steps) |
| `examples` | W&B Table of 10 sample questions, model outputs, and correctness (every 60 steps) |
| `lrm` | Learning rate multiplier |

### Modal volume (JSON files for EDA)
Saved to `/vol/nanochat_cache/chatrl_eval_logs/rl-gsm8k-teammate/`:
```
eval_step_000000.json   ← before any training (baseline)
eval_step_000060.json
eval_step_000120.json
...
```

Each JSON contains:
```json
{
  "step": 60,
  "pass@k": {"pass@1": 0.05, "pass@2": 0.08, ...},
  "records": [
    {
      "idx": 0,
      "question": "Janet's ducks lay 16 eggs per day...",
      "reference": "...\n#### 18",
      "outcomes": [
        {"is_correct": 0, "generated_text": "...#### 12"},
        {"is_correct": 1, "generated_text": "...#### 18"},
        ...
      ]
    },
    ...
  ]
}
```

### RL checkpoints
Saved to `/vol/nanochat_cache/chatrl_checkpoints/rl-gsm8k-teammate/` every 60 steps.

---

## Step 4 — After training completes

Tell the repo owner training is done. They will run:

```bash
# Download eval logs locally for EDA
uv run modal run nanochat_modal.py::download_eval_logs

# Push the trained RL checkpoint to HuggingFace
uv run modal run nanochat_modal.py::push_rl_to_hf
```

---

## Troubleshooting

**"No checkpoints found"** — The `.pt` file wasn't found on the Modal volume.
→ Check that the HuggingFace upload completed and re-run `run_rl_teammate` (it will re-download).

**"All rewards are 0.0"** — The model isn't outputting `#### <number>` format.
→ Verify your SFT training used GSM8K data and the model learned to write `#### answer` at the end.
→ Check a few examples: `uv run modal run nanochat_modal.py::peek_eval`

**Job disappeared from Modal** — Timeout or preemption.
→ Check https://modal.com/apps for the exit reason. Re-run with `--detach`.

**Pass@k not improving after 60+ steps** — Model too small or SFT init too weak.
→ This is a valid finding to report in the analysis. Let the run finish.

---

## Expected results

For reference, Karpathy's original run (from nanochat discussions):
- Pass@1 reaches ~10–15% after 1 epoch on a well-initialized model
- Reward curve shows gradual increase over training
- Sequence length may increase then stabilize as model learns to reason before answering

Your results will depend on model size and SFT quality. Differences are expected and should be discussed in the writeup.
