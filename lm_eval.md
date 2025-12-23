# Running lm-eval with nanochat checkpoints

This repo ships its own evals (CORE, ARC/GSM8K/MMLU/HumanEval/SpellingBee), but you can also run the HuggingFace-compatible [lm-evaluation-harness](tools/lm-eval). Steps below assume you've already run `bash setup.sh` (installs uv, submodules, deps, Rust tokenizer). `Please clone and run this repo in the local disk!` 

## 1) Activate env
```bash
source .venv/bin/activate
```

## 2) Export a trained checkpoint to HF format
- `nanochat/to_hf.py` loads the latest checkpoint from `~/.cache/nanochat/<source>_checkpoints` and writes an HF folder.
- Choose source: `base` | `mid` | `chatsft` | `chatrl`.
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
  --model_args pretrained=hf-export/sft,trust_remote_code=True \
  --tasks mmlu \
  --batch_size 1

# A small suite similar to nanochat chat_eval coverage (vanilla HF backend)
# HumanEval requires both flags below to allow executing generated code.
HF_ALLOW_CODE_EVAL=1 uv run lm-eval run --confirm_run_unsafe_code --model hf \
  --model_args pretrained=hf-export/sft,trust_remote_code=True \
  --tasks arc_easy,arc_challenge,mmlu \
  --batch_size 1 > log.log 2>&1

# Nanochat-aligned tool-use backend (matches nanochat eval formatting)
HF_ALLOW_CODE_EVAL=1 uv run lm-eval run \
    --include_path tools/lm-eval/lm_eval/tasks \
    --confirm_run_unsafe_code \
    --model hf-nanochat-tool \
    --model_args pretrained=hf-export/sft,trust_remote_code=True,tokenizer=hf-export/sft \
    --tasks gsm8k_nanochat,humaneval_nanochat \
    --batch_size 1 \
    --log_samples \
    --output_path lm_eval_sample_nanochat > log.log 2>&1
```

Notes:
- If you exported to a different folder, change `pretrained=...` accordingly. You can also point to a remote HF repo name.
- If you must stay offline, add `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, **but** ensure the datasets are already cached locally (e.g., `allenai/ai2_arc`, `openai_humaneval`, `gsm8k`, `cais/mmlu`). Otherwise, leave them unset so the harness can download once.
- `--batch_size auto` can help find the largest batch that fits GPU RAM. On CPU, keep it small.
- No KV cache is implemented in the HF wrapper; generation is standard `AutoModelForCausalLM` style. The `hf-nanochat-tool` wrapper runs a nanochat-style tool loop (greedy, batch=1) and does not need `--apply_chat_template` because the prompts already contain special tokens.
