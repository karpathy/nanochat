# Running lm-eval with nanochat checkpoints

This repo ships its own evals (CORE, ARC/GSM8K/MMLU/HumanEval/SpellingBee), but you can also run the HuggingFace-compatible [lm-evaluation-harness](tools/lm-eval). Steps below assume you've already run `bash setup.sh` (installs uv, submodules, deps, Rust tokenizer). `Please clone and run this repo in the local disk!` 

## 1) Activate env
```bash
source .venv/bin/activate
```

## 2) Export a trained checkpoint to HF format
- `nanochat/to_hf.py` (dense) loads the latest checkpoint from `~/.cache/nanochat/<source>_checkpoints` and tokenizer from `~/.cache/nanochat/tokenizer/`, then writes an HF folder.
- `nanochat_moe/to_hf.py` (MoE) loads the latest checkpoint from `~/.cache/nanochat/<source>_checkpoints` and, by default, exports with the `gpt2` tiktoken tokenizer. Use `--tokenizer cache` if you want the cached rustbpe tokenizer from `~/.cache/nanochat/tokenizer/`.
- Choose source: `base` | `mid` | `chatsft` | `chatrl` (same for MoE; `n_layer/n_embd` etc. come from checkpoint metadata).
- A checkpoint directory looks like: `~/.cache/nanochat/<source>_checkpoints/<model_tag>/model_XXXXXX.pt` + `meta_XXXXXX.json` (optimizer shards optional, ignored for export). The exporter auto-picks the largest `model_tag` and latest step if you don’t pass `--model-tag/--step`.
```bash
# export latest dense base checkpoint to hf-export/base
uv run python -m nanochat.to_hf --source base --output hf-export/base

# export latest MoE base checkpoint to hf-export/moe_gpt2 (gpt2 tokenizer)
uv run python -m nanochat_moe.to_hf --source base --model-tag d20 --step 1000 --output hf-export/moe_gpt2 --tokenizer gpt2

# export latest SFT checkpoint (chat model, dense)
uv run python -m nanochat.to_hf --source sft --output hf-export/sft
```
- An exported folder should contain (minimum): `config.json`, `pytorch_model.bin`, `tokenizer.pkl`, `tokenizer_config.json`, and the custom code files `configuration_nanochat.py`/`configuration_nanochat_moe.py`, `modeling_nanochat.py`/`modeling_nanochat_moe.py`, `tokenization_nanochat.py`, `gpt.py` (written for `trust_remote_code=True`).

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

# arc_easy,arc_challenge,mmlu
HF_ALLOW_CODE_EVAL=1 uv run lm-eval run --confirm_run_unsafe_code --model hf \
  --model_args pretrained=hf-export/pretrain,trust_remote_code=True \
  --tasks arc_easy,arc_challenge,mmlu \
  --batch_size 1 > log.log 2>&1

# Nanochat-aligned tool-use backend 
# gsm8k, humaneval
uv pip install -e tools/lm-eval
PYTHONPATH=tools/lm-eval HF_ALLOW_CODE_EVAL=1 uv run lm-eval run \
    --include_path tools/lm-eval/lm_eval/tasks \
    --confirm_run_unsafe_code \
    --model hf-nanochat-no-tool \
    --model_args pretrained=hf-export/std,trust_remote_code=True,tokenizer=hf-export/std \
    --tasks gsm8k_nanochat,humaneval_nanochat \
    --batch_size 1 \
    --log_samples \
    --output_path lm_eval_sample_nanochat_notool > notool_std_gsm8k_humaneval.log 2>&1

PYTHONPATH=tools/lm-eval HF_ALLOW_CODE_EVAL=1 uv run lm-eval run \
    --include_path tools/lm-eval/lm_eval/tasks \
    --confirm_run_unsafe_code \
    --model hf \
    --model_args pretrained=hf-export/pretrain,trust_remote_code=True,tokenizer=hf-export/pretrain \
    --tasks gsm8k,humaneval \
    --batch_size 1 \
    --log_samples \
    --output_path lm_eval_sample_nanochat_test > test_pretrain_gsm8k_humaneval.log 2>&1

PYTHONPATH=tools/lm-eval HF_ALLOW_CODE_EVAL=1 uv run lm-eval run \
    --include_path tools/lm-eval/lm_eval/tasks \
    --confirm_run_unsafe_code \
    --model hf-nanochat-no-tool \
    --model_args pretrained=hf-export/std,trust_remote_code=True,tokenizer=hf-export/std \
    --tasks gsm8k_nanochat,humaneval_nanochat \
    --batch_size 1 \
    --log_samples \
    --limit 3 \
    --output_path lm_eval_sample_nanochat_notool > notool_std_gsm8k_humaneval.log 2>&1


# for nanomoe based models - exported with tokenizer = `gpt2`
PYTHONPATH=tools/lm-eval HF_ALLOW_CODE_EVAL=1 uv run lm-eval run \
    --include_path tools/lm-eval/lm_eval/tasks \
    --confirm_run_unsafe_code \
    --model hf \
    --model_args pretrained=hf-export/moe_gpt2,trust_remote_code=True,tokenizer=hf-export/moe_gpt2,max_length=1024 \
    --gen_kwargs max_gen_toks=128 \
    --tasks arc_easy,arc_challenge,mmlu,gsm8k,humaneval \
    --batch_size 1 \
    --log_samples \
    --output_path lm_eval_sample_nanomoe > moe_gpt2.log 2>&1
```

Notes:
- If you exported to a different folder, change `pretrained=...` accordingly. You can also point to a remote HF repo name.
- If you must stay offline, add `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, **but** ensure the datasets are already cached locally (e.g., `allenai/ai2_arc`, `openai_humaneval`, `gsm8k`, `cais/mmlu`). Otherwise, leave them unset so the harness can download once.
- `--batch_size auto` can help find the largest batch that fits GPU RAM. On CPU, keep it small.
- No KV cache is implemented in the HF wrapper; generation is standard `AutoModelForCausalLM` style. The `hf-nanochat-tool` wrapper runs a nanochat-style tool loop (greedy, batch=1) and does not need `--apply_chat_template` because the prompts already contain special tokens. The `hf-nanochat-no-tool` wrapper uses the same greedy loop but does not execute tool-use blocks.
