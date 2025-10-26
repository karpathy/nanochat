.PHONY: help setup trial clean stage1 stage2 stage3 stage4 stage5 stage6 stage7 full

# Default target
help:
	@echo "Synthetic Data Pipeline - Makefile Commands"
	@echo "============================================"
	@echo ""
	@echo "Setup & Testing:"
	@echo "  make setup          - Install dependencies"
	@echo "  make trial          - Run trial pipeline (small dataset)"
	@echo ""
	@echo "Main Commands:"
	@echo "  make full           - Run full pipeline with deduplication (all stages)"
	@echo ""
	@echo "Individual Stages:"
	@echo "  make stage1         - Extract Q&A pairs from documentation"
	@echo "  make stage2         - Validate extracted Q&A pairs"
	@echo "  make stage3         - Generate conversations from validated Q&A"
	@echo "  make stage4         - Judge conversations and save passing ones"
	@echo "  make stage5         - Generate embeddings for deduplication"
	@echo "  make stage6         - Deduplicate similar conversations"
	@echo "  make stage7         - Select top-K final conversations"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Remove generated outputs"
	@echo "  make clean-trial    - Remove trial outputs only"
	@echo "  make stats          - Show statistics from latest run"
	@echo ""
	@echo "Development:"
	@echo "  make lint           - Run code formatting checks"
	@echo "  make format         - Format code with ruff"
	@echo ""

# Setup
setup:
	@echo "Installing dependencies..."
	uv sync
	@echo "✓ Dependencies installed"

# Testing
trial:
	@echo "Running trial pipeline (small dataset)..."
	uv run trial_run.py

# Pipeline Stages
stage1:
	@echo "Stage 1: Extracting Q&A pairs..."
	uv run 1_extract_qa.py

stage2:
	@echo "Stage 2: Validating Q&A pairs..."
	uv run 2_validate_qa.py

stage3:
	@echo "Stage 3: Generating conversations..."
	uv run 3_generate_conversations.py

stage4:
	@echo "Stage 4: Judging and saving passing conversations..."
	uv run 4_judge_and_save.py

stage5:
	@echo "Stage 5: Generating embeddings for deduplication..."
	uv run 5_embed_conversations.py

stage6:
	@echo "Stage 6: Deduplicating similar conversations..."
	uv run 6_deduplicate.py

stage7:
	@echo "Stage 7: Selecting top-K conversations..."
	uv run 7_select_top.py

# Run full pipeline (all stages with deduplication)
full: stage1 stage2 stage3 stage4 stage5 stage6 stage7
	@echo ""
	@echo "============================================"
	@echo "✓ Full pipeline completed!"
	@echo "============================================"
	@echo ""
	@echo "Final outputs:"
	@echo "  - output/qa_pairs.jsonl"
	@echo "  - output/conversations_raw.jsonl"
	@echo "  - output/conversations_judged.jsonl"
	@echo "  - output/conversations_deduplicated.jsonl"
	@echo "  - output/conversations_final.jsonl   <-- Use this for training"
	@echo ""

# Cleaning
clean:
	@echo "Removing all generated outputs..."
	rm -rf output/
	@echo "✓ Cleaned"

clean-trial:
	@echo "Removing trial outputs..."
	rm -f output/trial_*
	@echo "✓ Trial outputs cleaned"

# Statistics
stats:
	@echo "Pipeline Statistics:"
	@echo "==================="
	@if [ -f output/qa_pairs.jsonl ]; then \
		echo "Q&A Pairs: $$(wc -l < output/qa_pairs.jsonl) pairs"; \
	fi
	@if [ -f output/conversations_raw.jsonl ]; then \
		echo "Raw Conversations: $$(wc -l < output/conversations_raw.jsonl) conversations"; \
	fi
	@if [ -f output/conversations_judged.jsonl ]; then \
		echo "Judged Conversations: $$(wc -l < output/conversations_judged.jsonl) conversations"; \
	fi
	@if [ -f output/conversations_final.jsonl ]; then \
		echo "Top Conversations: $$(wc -l < output/conversations_final.jsonl) conversations"; \
	fi

# Development
lint:
	@echo "Running linting checks..."
	uv run ruff check src/ || true

format:
	@echo "Formatting code..."
	uv run ruff format src/ || true
	@echo "✓ Code formatted"

# Quick iteration workflow
quick: clean-trial trial
	@echo "✓ Quick iteration complete"
