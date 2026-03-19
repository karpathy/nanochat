# Apple-Native Acceleration Plan (March 2026)

## Goal

Use the current March 2026 research and existing branch evidence to define the next sequence of work for Apple-native acceleration.

This plan assumes:

- Python MLX remains the main Apple-native training backend.
- Swift MLX remains a runtime/inference seam.
- PyTorch + MPS remains a portability and fallback path.

## Planning Principles

1. Improve the path that already wins on evidence.
2. Do not rewrite working training code into Swift without a benchmark-driven reason.
3. Separate training, decode, and portability concerns instead of mixing them.
4. Prefer instrumentation and routing logic over speculative low-level rewrites.

## Priority 1: Harden Compiled MLX Training

### Why

The strongest validated performance win in the branch is compiled Python MLX with explicit state capture. This is now supported both by local benchmarks and by upstream MLX compile guidance.

### Actions

1. Audit all compiled MLX entrypoints and helpers so mutable state is always captured explicitly via `inputs=` and `outputs=`.
2. Add a focused regression test that would fail if a future refactor accidentally relies on implicit state in a compiled training step.
3. Document in the compile utility surface that stochastic layers must also capture `mx.random.state` if they are ever introduced.
4. Keep eager mode available, but treat it as a fallback and comparison path only.
5. Expand benchmark logging so compiled runs report compile warmup cost separately from steady-state throughput.

### Exit Criteria

- No compiled hot path depends on hidden mutable state.
- The branch has a test or probe that protects the explicit-state compile contract.
- Benchmark output clearly separates warmup, steady-state throughput, and memory telemetry.

## Priority 2: Productize the Swift Inference Seam

### Why

The repo has already established the real pattern:

- Python can still be better for short generations.
- Swift persistent KV-cache becomes attractive once output length crosses the measured crossover.

The next work is routing quality and runtime hardening, not proving Swift exists.

### Actions

1. Add explicit request routing heuristics around expected output length and request compatibility, instead of treating Swift as a simple binary on/off path.
2. Preserve the current safe fallbacks for unsupported sampling or state-machine features.
3. Add telemetry for:
   - TTFT
   - decode ms/token
   - cache memory / active memory at request boundaries
   - worker reuse count
4. Evaluate exact-prefix multi-turn KV-cache reuse only if prompt construction guarantees semantic correctness.
5. Add a benchmark mode that compares short, medium, and long outputs in one run so routing thresholds remain evidence-based.

### Exit Criteria

- Swift worker selection is based on measured workload characteristics.
- The runtime can explain why it chose Swift or Python for a request.
- Long-output requests reliably benefit without regressing short-output requests.

## Priority 3: Improve Memory Discipline and Telemetry

### Why

Allocator and unified-memory behavior remain central on Apple Silicon. The branch already has working mitigations, but they should become deliberate policy rather than scattered fixes.

### Actions

1. Standardize request-boundary cache clearing policy for long-lived inference workers.
2. Add memory telemetry in one place for MLX paths:
   - active memory
   - peak memory
   - cache memory
3. Keep PyTorch MPS reporting tied to `torch.mps.recommended_max_memory()` headroom.
4. Flag runs that exceed a configured percentage of recommended MPS working-set budget.
5. Avoid adding training-loop cache clears unless a measured regression specifically points to allocator fragmentation.

### Exit Criteria

- Memory policy is documented and consistent.
- Benchmark logs make allocator behavior visible instead of anecdotal.
- MPS fallback runs have clear headroom reporting and warnings.

## Priority 4: Keep the MPS Path Safe but Stop Investing Heavily in It

### Why

The branch goal has shifted from "make MPS work" to "use the best Apple-native path while keeping MPS functional." MPS still matters, but it is no longer the main optimization frontier.

### Actions

1. Preserve existing MPS fallback correctness tests.
2. Keep MPS-specific memory and compile guardrails in place.
3. Avoid new MPS-specific complexity unless it fixes correctness or portability.
4. Route Apple-native performance work toward MLX first.

### Exit Criteria

- MPS remains usable and well-bounded.
- New Apple-native development effort lands primarily in MLX codepaths.

## Priority 5: Defer the Right Things Explicitly

### Defer

1. Swift rewrite of training-session orchestration.
2. Custom Metal kernels or resource-management work.
3. Advanced KV-cache compression work unless long-context memory becomes a measured bottleneck.
4. Data-loader rewrites unless data time materially rises relative to step time.

### Reason

None of these are the best next move given current evidence. They are easy ways to spend time without improving the branch's main bottlenecks.

## Concrete Next Task Sequence

1. Add compile-state hardening tests and docs.
2. Add unified MLX memory telemetry and request-boundary cache metrics.
3. Add inference routing thresholds based on expected output length and compatibility.
4. Re-run the inference matrix with TTFT/decode split after routing changes.
5. Revisit multi-turn KV-cache reuse only after routing and telemetry are stable.

## Success Definition

The branch is in a good state when all of the following are true:

- compiled Python MLX is clearly the default Apple-native training path
- Swift persistent inference is used selectively where it wins
- the MPS fallback remains safe and measurable
- memory behavior is observable instead of inferred
- no major branch effort is going into speculative Swift training or custom Metal work
