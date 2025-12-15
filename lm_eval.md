# Running lm-eval with nanochat checkpoints

This repo ships its own evals (CORE, ARC/GSM8K/MMLU/HumanEval/SpellingBee), but you can also run the HuggingFace-compatible [lm-evaluation-harness](tools/lm-eval). Steps below assume you've already run `bash setup.sh` (installs uv, submodules, deps, Rust tokenizer).

## 1) Activate env
```bash
source .venv/bin/activate
```

## 2) Export a trained checkpoint to HF format
- `nanochat/to_hf.py` loads the latest checkpoint from `~/.cache/nanochat/<source>_checkpoints` and writes an HF folder.
- Choose source: `base` | `mid` | `sft` | `rl`.
```bash
# export latest base checkpoint to hf-export/base
uv run python -m nanochat.to_hf --source base --output hf-export/base

# export latest SFT checkpoint (chat model)
uv run python -m nanochat.to_hf --source sft --output hf-export/sft
```

## 3) Run lm-eval benchmarks on the exported model
Use the HF backend (`--model hf`). Pick tasks; nanochat's built-in evals cover these, so they're good starters in lm-eval too:
- `arc_easy`, `arc_challenge`
- `mmlu`
- `gsm8k`
- `humaneval`

Example runs:
```bash
# Single task (MMLU)
uv run lm-eval run --model hf \
  --model_args pretrained=hf-export/sft \
  --tasks mmlu \
  --batch_size 1

# A small suite similar to nanochat chat_eval coverage
uv run lm-eval run --model hf \
  --model_args pretrained=hf-export/sft \
  --tasks arc_easy,arc_challenge,gsm8k,mmlu,humaneval \
  --batch_size 1
```

Notes:
- If you exported to a different folder, change `pretrained=...` accordingly. You can also point to a remote HF repo name.
- `--batch_size auto` can help find the largest batch that fits GPU RAM. On CPU, keep it small.
- No KV cache is implemented in the HF wrapper; generation is standard `AutoModelForCausalLM` style.
