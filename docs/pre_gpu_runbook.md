# Pre-GPU Runbook

This runbook is the minimum operational checklist before spending GPU time.

## 1. Local Prep

1. Build the seed tool datasets:

```bash
python -m scripts.build_tool_datasets
```

2. Import the starting checkpoint from Hugging Face into native nanochat format:

```bash
python -m scripts.import_hf_checkpoint \
  --repo-id ManmohanSharma/nanochat-d24 \
  --model-tag d24_hf_import
```

3. Validate tool tokenization and mock tool execution with local tests:

```bash
python -m pytest tests/test_engine.py tests/test_tools.py -v
```

4. Dry-run tool evaluation on CPU:

```bash
python -m scripts.chat_eval \
  -i sft \
  -a ToolJSON \
  --tool-jsonl seed_data/tool_eval_seed.jsonl \
  --device-type cpu \
  -x 3
```

## 2. 48-Hour GPU Schedule

1. Pilot CPT
   - Run a short continuation test from the imported base checkpoint.
   - Confirm loss is moving, checkpoint save works, and HF sync works.

2. Full CPT
   - Run the main continuation stage on ClimbMix backbone.
   - Save staged checkpoints at planned intervals.

3. SFT
   - Include the local tool SFT JSONL via `--extra-train-jsonl`.
   - Validate that calculator/web_search traces render correctly.

4. RL / tool tuning
   - Keep this stage narrow and short.
   - Focus on tool-choice correctness and grounded answers.

5. Eval
   - Run ARC, MMLU, GSM8K, HumanEval, and ToolJSON checks.
   - Do not ship if tool behavior regresses or citations are missing.

## 3. Checkpoint Upload Cadence

Upload every stage boundary and any explicit resume point:

```bash
python -m scripts.hf_sync_checkpoint \
  --repo-id ManmohanSharma/nanochat-d24 \
  --source base \
  --model-tag d24_hf_import \
  --step 0
```

If a whole checkpoint directory should be mirrored:

```bash
python -m scripts.hf_sync_checkpoint \
  --repo-id ManmohanSharma/nanochat-d24 \
  --source base \
  --model-tag d24_hf_import
```

## 4. Go / No-Go

Go only if:

- HF import works.
- HF sync works.
- Mock tool execution works.
- Tool seed datasets are generated.
- Tool eval runs locally.
- The search backend plan is explicit: search provider plus Cloudflare fetch/crawl.

No-Go if:

- Any tokenizer mismatch appears during HF import.
- Tool blocks fail to render.
- `web_search` still has no backend plan beyond fetch-only Cloudflare Browser Rendering.
- Local tool eval is missing or failing.
