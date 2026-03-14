---
title: "Apple Silicon (MPS) Documentation"
summary: "Archive of the MPS documentation task — rewrote brainstorming draft into proper technical documentation covering device detection, dtype behavior, limitations, and training guidelines."
status: archived
archived_date: "2026-03-15"
archived_reason: "Completed — MPS guide rewritten as proper project documentation (db8c79c)"
---

# Apple Silicon (MPS) Documentation

## What Was Done

Rewrote `docs/m3-max-guide.md` from a ~400-line brainstorming/planning draft into 135 lines
of accurate technical documentation based on codebase analysis of all 12 files referencing MPS.

## Completed Tasks

- [x] Document MPS backend setup and limitations
- [x] Add hardware-specific batch size and memory guidelines
- [x] Integrate MPS device support into training scripts (already integrated — documented existing support)

## Key Changes

- Replaced speculative code snippets with actual codebase behavior
- Fixed outdated CLI flags (`--device` → `--device-type`, `--batch-size` → `--device-batch-size`, `--sequence-len` → `--max-seq-len`)
- Documented: fp32 dtype (no bf16 on MPS), SDPA fallback, int64 workaround, FP8 gating, torch.compile status
- Added batch size recommendations per model depth for 128GB unified memory

## Artifacts

- [MPS Guide](../m3-max-guide.md) — the rewritten documentation
- Commit: `db8c79c`
