# Apple-Native Acceleration Prototype Scope

## Purpose

This memo defines the minimum first prototype required to answer the Apple-native architecture question without slipping into a rewrite.

Selected backend for planning purposes:

- MLX

## Prototype Goal

The first prototype exists to answer one question:

- can an MLX-based training path beat or materially improve on the current PyTorch + MPS baseline on the reference M2 Ultra machine?

It is not intended to be a production backend.

## Minimum Required Model Surface

The first prototype should include only the minimum benchmarkable training slice:

- GPT block stack sufficient to represent the reference nanochat model shape
- forward pass
- backward pass
- one optimizer step
- representative input-batch loading compatible with the frozen benchmark workload

## Explicitly Out Of Scope

The first prototype should not include:

- a full repo-wide backend abstraction
- chat finetuning or RL pipelines
- web, CLI, or serving integration
- distributed or multi-machine training
- full production checkpoint portability on day one
- a Swift UI or Core ML deployment path
- broad feature parity with every existing training script

## Exact Prototype Benchmark Workload

Primary comparison workload:

- device class: reference M2 Ultra Mac Studio
- sequence length: `1024`
- device batch size: `2`
- depth: `32`
- benchmark shape: single representative training step on the frozen baseline tier

Comparison metrics:

1. tokens per second
2. MPS or Apple-native memory usage versus the current baseline budget
3. implementation complexity and missing features

## Acceptance Thresholds

The prototype is considered promising only if at least one of the following is true:

1. it improves throughput by roughly `20%` or more on the reference workload
2. it materially reduces memory pressure enough to expand the stable operating frontier on this machine
3. it produces a clear ergonomics or reliability improvement that plausibly justifies a second backend

Additional acceptance conditions:

- autodiff and optimizer behavior must be viable for the benchmarked training step
- the integration story must remain believable without destabilizing the existing PyTorch path

## Kill Criteria

Stop the first prototype track if any of the following becomes true:

1. throughput is flat or worse and memory behavior is not meaningfully improved
2. missing training primitives make the benchmark path unnatural or unrepresentative
3. checkpoint, tokenizer, or evaluation integration becomes more complex than the likely upside justifies
4. the prototype starts pulling the project toward a rewrite before benchmark evidence exists

## Deliverables From This Prototype Planning Step

- selected backend record: MLX
- prototype scope definition
- acceptance thresholds and kill criteria for the first benchmark gate

## Related Documents

- [APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md](APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md)
- [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md)
- [APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md](APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md)