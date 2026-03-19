# Apple-Native Acceleration Research Notes (March 2026)

## Purpose

This note records March 2026 external research relevant to the `feature/apple-silicon-native-mps` branch. It is meant to answer a practical question:

> Given the current state of Apple tooling and upstream MLX/PyTorch guidance, what should this branch do next to improve the Apple-native path without wasting effort?

The repo already has strong local evidence that:

- Python MLX is the best current Apple-native training path.
- Explicit-state `mx.compile` is stable locally and materially faster than eager mode.
- The Swift MLX worker is useful as an inference seam, especially once output length is long enough for KV-cache reuse to dominate.

This document cross-checks those conclusions against current upstream guidance and adjacent ecosystem signals.

## Source Types

### Authoritative Apple Sources

1. Apple WWDC25: `Get started with MLX for Apple silicon`
   - https://developer.apple.com/videos/play/wwdc2025/315/
2. Apple Machine Learning Research: `Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU`
   - https://machinelearning.apple.com/research/exploring-llms-mlx-m5
3. Apple Metal guidance surfaced via search results on resource storage and memory-model best practices
   - search context pointed to Apple documentation on Metal resource storage modes and Apple silicon Mac architecture

### Upstream MLX Documentation Available in Workspace

These are vendored through the local `mlx-swift` checkout and are useful because they reflect upstream MLX behavior directly:

1. `swift/NanochatMLXStub/.build/checkouts/mlx-swift/Source/Cmlx/mlx/docs/src/usage/compile.rst`
2. `swift/NanochatMLXStub/.build/checkouts/mlx-swift/Source/Cmlx/mlx/docs/src/python/memory_management.rst`
3. `swift/NanochatMLXStub/.build/checkouts/mlx-swift/Source/MLX/Memory.swift`

### PyTorch Reference

1. `torch.mps.recommended_max_memory`
   - https://docs.pytorch.org/docs/stable/generated/torch.mps.recommended_max_memory.html

### Lower-Confidence Ecosystem Signals

These are useful for idea generation, not as primary evidence:

- MLX / mlx-lm GitHub discussions and PR activity around KV-cache and long-context memory behavior
- community writeups on Apple Silicon local inference behavior

## Findings

### 1. Apple is now explicitly positioning MLX as the primary Apple-native framework for local model work.

Apple's WWDC25 MLX session describes MLX as a framework for training, fine-tuning, and inference on Apple silicon, and calls out the core properties that match this branch's direction:

- unified memory
- lazy computation
- function transformations
- both Python and Swift front ends

Practical branch implication:

- The branch should keep treating MLX as the main Apple-native development surface.
- The current MLX-first strategy is aligned with Apple's own platform direction, not just an internal experiment.

### 2. The branch's `mx.compile` workaround is not a hack; it matches upstream MLX guidance.

The upstream MLX compile documentation explicitly describes the exact pattern this branch has already converged on:

- compiled functions treat non-argument state as constants
- mutable state must be passed explicitly or captured with `inputs=` and `outputs=`
- compiling a training step with `state = [model.state, optimizer.state]` and `@partial(mx.compile, inputs=state, outputs=state)` is an intended workflow

This is the single most important research confirmation for the branch because it means:

- the local success with explicit-state compilation is upstream-supported behavior
- the local failure of the implicit stateful closure path is not a reason to retreat from compiled MLX
- future compile-facing code should be written to make state capture explicit by design

Practical branch implication:

- Standardize on explicit-state compile wrappers for all long-lived MLX training hot paths.
- Avoid introducing new compiled closures that depend on hidden mutable state.

### 3. If stochastic layers are introduced, compile state must include RNG state.

The upstream MLX compile docs explicitly note that if a compiled function uses random sampling, such as dropout, `mx.random.state` should be included in captured compile state.

Practical branch implication:

- The current branch can ignore this if all hot paths remain deterministic.
- If dropout, stochastic regularization, sampling-based training features, or randomized augmentation are added later, compile wrappers need to include `mx.random.state` in the captured state list.
- This should be documented near compile utilities before future work accidentally introduces silent determinism or stale randomness.

### 4. `clear_cache` / `Memory.clearCache()` is an official allocator-management tool, not a local superstition.

Upstream MLX exposes memory-management APIs including:

- `get_active_memory`
- `get_peak_memory`
- `get_cache_memory`
- `clear_cache`

The Swift API also exposes `Memory.clearCache()` directly, with the implementation calling into `mlx_clear_cache()`.

This matters because the repo already found a real benefit from clearing cache between persistent-worker requests. External guidance does not say to spam cache clearing everywhere, but it validates the existence of the mechanism.

Practical branch implication:

- Keep `Memory.clearCache()` and Python `mx.clear_cache()` as request-boundary tools for long-lived inference workers.
- Do not treat cache clearing as a general hot-loop optimization for training; use it as an allocator-pressure relief valve where fragmentation or buffer-pool growth is observed.
- Add telemetry before broadening cache-clearing behavior.

### 5. Apple's latest MLX performance story reinforces the branch split between prompt/prefill optimization and decode optimization.

Apple's M5 MLX research note says:

- time to first token is primarily compute-bound
- subsequent token generation is primarily memory-bandwidth-bound

That is highly relevant to the branch because it matches the repo's measurements:

- Python MLX compiled training wins because the training hot path is compute-heavy and graph reuse helps.
- Swift persistent KV-cache inference improves as output length grows because decode cost becomes dominated by bandwidth and cache reuse.

Practical branch implication:

- Keep distinguishing prefill/TTFT work from decode work in all benchmarks.
- Short-output inference comparisons can mislead. Long-output benchmarks are necessary to justify KV-cache runtime work.
- The current Swift persistent-worker seam makes sense for chat responses of moderate or long length, not as a blanket replacement for every inference request.

### 6. Apple's unified-memory messaging is a strength signal, but Metal guidance still warns against over-simplified memory assumptions on macOS.

Apple's MLX material emphasizes unified memory as a major advantage. At the same time, Apple Metal guidance surfaced in search results continues to describe macOS resource management in terms of a discrete-model abstraction, even on integrated Apple GPUs, and emphasizes choosing correct storage modes and access patterns.

Practical branch implication:

- Staying at the MLX level is still the right move for this repo.
- There is no evidence that this branch should drop down into custom Metal resource management right now.
- If the repo ever adds custom Metal kernels, memory placement and buffer lifetime will need deliberate review rather than hand-wavy reliance on "unified memory solves it".

### 7. PyTorch MPS remains useful as a portability path, but not the preferred native optimization surface.

PyTorch's documented `torch.mps.recommended_max_memory()` returns the recommended Metal working-set size. That supports what this repo is already doing with MPS memory budgeting: treat it as a guardrail, not as an invitation to fill all available system memory.

Practical branch implication:

- Keep the MPS path healthy for portability and fallback.
- Keep memory headroom reporting and MPS safety checks.
- Continue treating MLX, not PyTorch MPS, as the main place to pursue new Apple-native performance work.

### 8. Swift MLX remains a runtime seam, not a training-engine priority.

Apple explicitly supports MLX Swift, and WWDC25 includes MLX Swift as part of the official MLX story. That validates the existence of the Swift path. But none of the current authoritative evidence suggests Swift should replace Python MLX as the main training surface for this repo.

Practical branch implication:

- Keep Swift focused on the inference/runtime boundary.
- Do not spend branch time rewriting the MLX training loop in Swift unless a later benchmark exposes a concrete gap that compiled Python MLX cannot close.

### 9. Long-context KV-cache design is an active optimization frontier in the MLX ecosystem.

Recent mlx-lm activity in early 2026 shows active work on reducing KV-cache memory overhead for long-context models through compressed or latent-cache approaches. This is not direct evidence for nanochat, but it is a useful forward-looking signal.

Practical branch implication:

- If the Swift worker eventually needs significantly longer context or multi-turn cache persistence, future work should consider compressed/latent KV-cache patterns before designing a bespoke cache format.
- This is a later-stage optimization, not a current blocker.

## Branch-Level Conclusions

### Conclusion 1

The branch is already on the correct Apple-native architecture:

- Python MLX for training and benchmark harnesses
- Swift MLX as an optional inference/runtime seam
- PyTorch MPS retained for fallback and compatibility

### Conclusion 2

Compiled MLX should be treated as the stable default direction for training work in this branch, provided all mutable state is captured explicitly.

### Conclusion 3

The best next inference work is not "more Swift" in general. It is more specific:

- persistent workers
- precise request-shape routing
- KV-cache-aware long-output workloads
- optional multi-turn cache reuse where semantics are exact

### Conclusion 4

The branch does not currently need custom Metal work. The right abstraction level remains MLX unless profiling later shows a clear kernel-level blocker.

## Recommended Guardrails

1. Never rely on implicit mutable state inside compiled MLX hot paths.
2. Keep training benchmarks split by eager vs compiled and repeated vs dataset-backed.
3. Keep inference benchmarks split by TTFT/prefill and steady-state decode.
4. Use cache clearing only at durable runtime boundaries where allocator growth is real.
5. Keep MPS memory budgeting conservative relative to `torch.mps.recommended_max_memory()`.
6. Treat Swift training rewrites as out-of-scope unless new measurements justify them.

## Follow-On Documents

- See `APPLE_NATIVE_ACCELERATION_PLAN_2026-03.md` for the action plan derived from these findings.
