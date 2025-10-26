# Synthetic Data Pipeline

Generate synthetic SWAP Commerce conversations for NanoChat fine-tuning. The workflow is split into small scripts so you can rerun or customize each stage independently, but every script is executed the same way: `uv run <script>.py`.

## Prerequisites

1. **Environment variables** – create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_gemini_key
   OPENAI_API_KEY=your_openai_key
   LOGFIRE_TOKEN=optional_logfire_token
   ```
2. **Source material** – place the facts you want to model at `data/swap_facts.md`.

## Pipeline Stages

Each stage reads from `output/` (if the previous step has already run) and writes its own JSONL artifact. Run them sequentially with `uv run`.

| Stage | Script | Purpose | Output |
| --- | --- | --- | --- |
| 1 | `1_extract_qa.py` | Chunk the facts file and create 3 grounded Q&A pairs per chunk. Filters out future dates. | `output/qa_pairs.jsonl` |
| 2 | `2_validate_qa.py` | Filter hallucinated or low-quality pairs. Saves full validation metadata and a `_passed` file. | `output/qa_pairs_validated.jsonl` + `output/qa_pairs_validated_passed.jsonl` |
| 3 | `3_generate_conversations.py` | Sample personas/styles and turn Q&A seeds into synthetic dialogues. Matches topics to personas. | `output/conversations_raw.jsonl` |
| 4 | `4_judge_and_save.py` | Judge conversations using **bool-only rubric** (factual, natural, on-topic, adds-value). Only passing conversations advance. | `output/conversations_judged.jsonl`, `output/conversations_final.jsonl` |
| 5 | `5_embed_conversations.py` | Create OpenAI embeddings for deduplication. | `output/conversations_embedded.jsonl` |
| 6 | `6_deduplicate.py` | Remove near-duplicates via cosine similarity. | `output/conversations_deduplicated.jsonl` |
| 7 | `7_select_top.py` | Export final training set of top-K conversations. | `output/conversations_final.jsonl` |

### Quick Start

**Run the full pipeline** (all 7 stages with deduplication):
```bash
make full
```

**Or run a quick trial** on a small dataset first:
```bash
make trial
```

### Running stages individually

If you prefer to run stages one at a time:

```bash
uv run 1_extract_qa.py
uv run 2_validate_qa.py
uv run 3_generate_conversations.py
uv run 4_judge_and_save.py
uv run 5_embed_conversations.py
uv run 6_deduplicate.py
uv run 7_select_top.py
```

Or use Makefile shortcuts: `make stage1`, `make stage2`, etc.

### Trial run (smoke test)

To test the pipeline on a small subset (~30 Q&A pairs, ~280 conversations):

```bash
make trial
# or
uv run trial_run.py
```

This produces `output/trial_*.jsonl` files plus quality statistics showing pass rates for each bool criterion.

## Makefile Commands

Run `make help` to see all available commands:
- `make trial` - Quick test on small dataset
- `make full` - Run complete pipeline with all 7 stages
- `make stage1` through `make stage7` - Run individual stages
- `make clean` - Remove all generated outputs
- `make stats` - Show pipeline statistics

## Repository structure

```
src/
└── synth_data_pipeline/
    ├── agents/                   # Model-facing helpers + prompts for each stage
    │   ├── base.py
    │   ├── qa_extractor.py
    │   ├── qa_validator.py
    │   ├── conversation_generator.py
    │   ├── conversation_judge.py
    │   └── prompts/*.txt
    ├── config.py                 # Shared constants and file paths
    ├── models.py                 # Pydantic schemas for every artifact
    ├── sampling.py               # Persona/style sampling utilities
    ├── utils.py                  # IO + concurrency helpers
    └── prompts/{system_prompts,personas}
```

Workflow scripts (e.g., `1_extract_qa.py`, `3_generate_conversations.py`, etc.) live in the repo root next to `trial_run.py`. All generated data lands under `output/`.
