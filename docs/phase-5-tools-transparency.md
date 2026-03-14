---
title: "Phase 5: Capabilities — Tool Use and Transparency"
summary: "Plan for tool use, function calling, and compression-based transparency"
read_when: "Planning tool use capabilities or transparency/interpretability features"
status: draft
last_updated: 2026-03-14
---

# Phase 5: Capabilities

## 5.1 Long Context (100K+)

- RoPE base theta extension (100K → 500K)
- YaRN scaling for context beyond training length
- Ring attention for sequences exceeding single-GPU memory
- Sliding window + global attention hybrid (nanochat has sliding window, needs global tokens)

## 5.2 Tool Use / Function Calling

### Goal

Extend nanochat beyond text generation to interact with external tools and APIs.

### Current State

Basic calculator tool in `engine.py`. No structured tool calling, no tool registry.

### Approach

- Tool registry with schema definitions (OpenAI-compatible function calling format)
- Core tools: Python execution (sandboxed), web search, file operations, calculator (safe, no eval)
- Integration into chat engine with multi-turn tool use loop
- Training data: tool-use conversation examples for SFT

### Dependencies

- Phase 4 (alignment/SFT) — model needs instruction following before tool use training

## 5.3 Transparency and Interpretability

### Goal

Make model reasoning transparent using compression-based pattern analysis.

### Approach

- **Pattern tracking**: Hook into attention and MLP layers to capture activation patterns during inference
- **Compression analysis**: Use SVD to measure how compressed each layer's representations are — high compression = structured reasoning
- **Explanation generation**: Map active patterns to human-readable reasoning chains
- **Safety auditing**: Detect harmful reasoning patterns by inspecting activation signatures

### SP Theory Advantage

SP Theory frames intelligence as information compression through pattern discovery. By exposing which patterns the model activates, we get transparency "for free" — the compression structure *is* the reasoning structure.

### Applications

- Alignment verification (inspect decision patterns)
- Safety auditing (detect harmful reasoning)
- Debugging (understand model failures)
- Trust building (transparent decision pathways)

## 5.4 Multimodal (Vision)

- Vision encoder (SigLIP or CLIP)
- Project image embeddings into text token space
- Train on image-text pairs

## Dependencies

- Phase 4 (alignment) must be complete before tool use training
- Phase 1.5 compression metrics provide foundation for transparency work
