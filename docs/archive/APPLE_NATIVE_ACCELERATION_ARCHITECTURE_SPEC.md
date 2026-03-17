# Apple-Native Acceleration Architecture Spec

## Document Purpose

This document defines the architecture and strategy question for the next nanochat phase on Apple Silicon after MPS portability and initial M2 Ultra scaling.

It is a planning artifact only.

It does not authorize implementation work by itself.

## Current Baseline

The current branch has completed two prerequisites:

1. Apple Silicon portability.
2. Initial M2 Ultra scaling and benchmarking.

Relevant existing documents:

- [APPLE_SILICON_PORTABILITY.md](APPLE_SILICON_PORTABILITY.md)
- [M2_ULTRA_SCALING.md](M2_ULTRA_SCALING.md)

Current known baseline on this machine:

- Apple M2 Ultra is exposed as one `mps` accelerator, not 60 separate GPUs.
- The project can now execute meaningful larger-model training steps on MPS.
- The current practical default tier is approximately:
  - `depth=32`
  - `device-batch-size=2`
  - `max-seq-len=1024`
  - about `2.8B` parameters
- Larger runs are possible, but MPS memory pressure and framework limitations become more visible.

## Problem Statement

The next meaningful performance ceiling on this machine is unlikely to come from minor PyTorch tuning alone.

The strongest candidate sources of additional performance or capability are:

- Apple-native execution models built around unified memory
- more controllable kernel fusion
- fewer MPSGraph edge-case failures than the current PyTorch-on-MPS path
- inference and serving stacks that are better aligned with Apple Silicon

At the same time, a full Swift-native rewrite is high-cost and high-risk.

The architecture question for this phase is therefore:

Should nanochat remain primarily a PyTorch project on Apple Silicon, or should it add a second Apple-native execution path for training and/or inference?

## Goals

### Goal 1: Establish an Apple-native decision framework

Produce a defensible answer to which of the following should be pursued:

- PyTorch + MPS only
- hybrid path with selective Apple-native acceleration
- MLX-based Apple Silicon training backend
- Swift-native inference or runtime path
- full Swift + Metal or MPSGraph training rewrite

### Goal 2: Quantify upside, not just plausibility

For each viable option, estimate:

- expected speedup or slowdown
- memory efficiency gains or losses
- impact on model size frontier
- implementation cost
- maintenance burden
- portability tradeoffs

### Goal 3: Identify the best next technical investment

Define the one next implementation track that is most likely to improve real outcomes on the M2 Ultra.

### Goal 4: Preserve the existing repo’s strengths

Any future direction should preserve or consciously replace:

- current tokenizer pipeline
- checkpoint structure and reporting workflows
- evaluation workflows
- current training ergonomics
- the ability to continue using the existing PyTorch path when needed

## Non-Goals

This phase does not include:

- a full Swift rewrite of nanochat
- replacing all existing training scripts
- productionizing a Core ML app deployment path
- cross-platform backend abstraction for every accelerator
- distributed multi-machine training
- speculative implementation before architecture choice

## Research Summary

### MLX

MLX is the strongest Apple-native candidate for training-oriented experimentation.

Why it matters:

- designed specifically for Apple Silicon
- built around unified memory
- supports autodiff, optimizers, and graph optimization
- available beyond Python, including Swift bindings
- lower migration cost than a pure Swift or Metal rewrite

Implication:

- best candidate for a serious prototype backend for local training on Apple Silicon

### Core ML

Core ML is compelling for deployment and inference, and supports on-device model customization in some contexts.

However, for this repo’s current objective, it is not the leading candidate for large-scale pretraining or full trainer replacement.

Implication:

- good inference and export target
- weak primary candidate for replacing nanochat training workflows

### Metal, MPSGraph, and Swift-native path

This path offers the greatest theoretical control and possibly the highest ceiling on Apple hardware.

It also has the highest cost because it would require rebuilding most of the project’s core runtime behavior.

Implication:

- viable only if Apple Silicon becomes the project’s long-term primary platform
- not the first implementation move

## Option Space

### Option A: Stay PyTorch + MPS only

Description:

- continue improving current MPS path
- add profiling, memory tuning, and targeted kernel-safe fixes

Pros:

- lowest implementation cost
- keeps one codebase and one training stack
- fastest path to near-term iteration

Cons:

- remains constrained by PyTorch-on-MPS limitations
- limited control over deeper optimizer, attention, and runtime issues
- may plateau below hardware potential

When to choose:

- if experiments show Apple-native alternatives do not clearly outperform current throughput or scale

### Option B: Hybrid PyTorch + custom Apple-native hot paths

Description:

- keep core repo in Python and PyTorch
- selectively replace hot paths with Apple-native implementations
- likely candidates: attention, optimizer update kernels, inference kernels

Pros:

- preserves most of the repo
- targets the real bottlenecks
- lower rewrite risk than a full backend migration

Cons:

- interop complexity
- harder debugging
- maintenance split across stacks

When to choose:

- if profiling identifies one or two dominant hotspots where Apple-native code gives disproportionate benefit

### Option C: MLX-based training backend

Description:

- implement a second backend or reference trainer in MLX
- reuse tokenizer, data, and checkpoint ideas where practical
- focus on Apple Silicon training ergonomics and scale

Pros:

- best fit for Apple Silicon training research
- lower complexity than full Swift rewrite
- stronger long-term upside than incremental PyTorch-only tuning

Cons:

- introduces a second training stack
- requires model and optimizer re-expression
- checkpoint compatibility and evaluation parity need deliberate design

When to choose:

- if the project wants Apple Silicon to be a serious first-class training platform

### Option D: Swift-native inference or runtime path

Description:

- keep training mostly in Python
- add Swift-native runtime for inference, experimentation, or app integration

Pros:

- strong local deployment story
- good match for UI and app packaging
- lower cost than training rewrite

Cons:

- does not directly solve training throughput
- adds architecture split

When to choose:

- if product direction shifts toward on-device inference or a native macOS app

### Option E: Full Swift + Metal or MPSGraph training rewrite

Description:

- rebuild model, optimizer, checkpointing, and training loops in Apple-native stack

Pros:

- maximum control
- potentially best Apple-hardware utilization ceiling

Cons:

- highest cost by far
- highest risk
- longest time to first useful result
- substantial tooling and debugging rebuild required

When to choose:

- only if Apple Silicon becomes the strategic center of gravity for the project

## Recommended Strategic Direction

The recommended path for this phase is:

1. Keep PyTorch + MPS as the working baseline.
2. Evaluate MLX as the primary Apple-native training candidate.
3. Treat Core ML as an inference and export path, not a trainer replacement.
4. Defer any pure Swift or Metal rewrite until MLX and hybrid options have been experimentally ruled out.

In short:

- do not start with a full rewrite
- do not assume Core ML solves training
- do not abandon the existing PyTorch branch
- do use this phase to prove whether MLX or a hybrid path has enough upside to justify a second backend

## Deliverables Required To Make The Decision

### Research deliverables

- a framework comparison memo
- a backend decision matrix
- a profiling memo for current PyTorch MPS bottlenecks

### Prototype deliverables

- a minimal transformer forward and backward prototype in the selected Apple-native path
- a benchmark harness comparing that prototype against the PyTorch baseline on the same workload
- a checkpoint compatibility note describing what would or would not be portable

### Decision deliverables

- a go or no-go recommendation for MLX or hybrid acceleration
- an implementation plan for the chosen direction
- an explicit kill criterion for abandoning the Apple-native experiment if results are weak

## Success Metrics

This phase is successful if it answers the architecture question with evidence.

Minimum success criteria:

- one clear recommended direction
- one rejected direction with explicit reasons
- one quantified baseline-versus-prototype comparison
- one implementation roadmap that is realistic for this repo

Preferred success criteria:

- at least 20% meaningful improvement in one of:
  - tokens per second
  - stable model size at a fixed sequence length
  - inference latency
  - memory efficiency

If no option achieves meaningful improvement at acceptable complexity, success is still possible if the phase proves that staying with PyTorch + MPS is the correct choice.

## Decision Rules

The project should proceed beyond this phase only if all of the following are true:

1. The chosen Apple-native path shows a meaningful gain over the PyTorch MPS baseline.
2. The gain is large enough to justify the extra complexity.
3. The integration story is believable without destabilizing the current repo.
4. The chosen path fits the long-term identity of the project.

If any of those fail, the correct outcome is to stay with PyTorch + MPS and continue incremental optimization there.

## Final Planning Recommendation

For planning purposes only, the recommended sequence is:

1. Baseline and profile current PyTorch MPS path.
2. Research and compare MLX, Core ML, and Swift-native approaches.
3. Build a minimal MLX-based prototype first.
4. Reassess whether any Swift-native rewrite is justified after MLX results.

That is the most defensible path because it maximizes information gained per unit of engineering cost.

## Related Documents

- [APPLE_NATIVE_ACCELERATION_PHASE_SPEC.md](APPLE_NATIVE_ACCELERATION_PHASE_SPEC.md)
- [APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md](APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md)
- [APPLE_NATIVE_ACCELERATION_TASKLIST.md](APPLE_NATIVE_ACCELERATION_TASKLIST.md)
- [APPLE_SILICON_PORTABILITY.md](APPLE_SILICON_PORTABILITY.md)
- [M2_ULTRA_SCALING.md](M2_ULTRA_SCALING.md)