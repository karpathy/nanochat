# Apple-Native Acceleration Final Recommendation

## Decision

Recommendation: go forward with a constrained Apple-native prototype.

Selected first implementation track:

- MLX-based training prototype

This is a go decision for prototype implementation only.

It is not a decision to replace the current PyTorch + MPS path.

## Why This Is The Right Next Step

The planning set now establishes four facts:

1. the current PyTorch + MPS baseline is stable, documented, and benchmarked
2. the dominant throughput bottleneck on the reference workload is optimizer and runtime overhead, not attention compute
3. the main memory pressure is a mix of optimizer-state residency, parameter and gradient residency, and a large activation spike
4. MLX is the strongest Apple-native candidate that can test those problems without paying the cost of a full Swift-native rewrite

That combination is enough to justify a prototype.

## Synthesis Of The Planning Evidence

### Baseline

The frozen reference tier is:

- depth `32`
- device batch size `2`
- max sequence length `1024`
- PyTorch + MPS on the reference M2 Ultra machine

This gives the project a stable comparison point for any Apple-native work.

### Profiling

The key profiling result is decisive:

- optimizer step time dominates the full training step at about `84.5%`

That weakens the case for starting with an attention-only hotspot replacement.

The memory picture is also not a narrow single-op problem:

- optimizer state is the largest persistent tensor block
- activations are the largest transient spike
- parameters and gradients remain large steady-state residents

So the next experiment should target backend-level training behavior, not just a small inference-side optimization.

### Framework Comparison

MLX remains the best first experiment because it is:

- Apple-native
- training-oriented
- materially lower cost than a Swift + Metal rewrite
- better aligned with the training question than Core ML

Core ML remains relevant for inference and deployment, but not as the answer to the current training bottleneck.

Hybrid acceleration remains a valid fallback if the MLX path stalls or if later evidence isolates a single hotspot with disproportionate payoff.

## Recommendation Details

Proceed with:

1. the MLX setup and model-prototype tasks already defined in the implementation section
2. a minimal benchmarkable training slice only
3. strict comparison against the frozen PyTorch + MPS baseline

Do not proceed with:

- a full Swift-native rewrite
- a repo-wide backend abstraction layer
- Core ML as the primary training answer
- a broad hybrid effort before the MLX gate is tested

## Success Gate For The MLX Prototype

The MLX track should continue only if it shows meaningful evidence in at least one of these areas:

1. materially improved optimizer execution cost
2. materially improved memory behavior
3. enough overall performance or ergonomics upside to justify a second backend

If it does not, the correct fallback is:

- stop the MLX expansion
- keep PyTorch + MPS as the main path
- revisit hybrid acceleration only if profiling suggests a sharply bounded hotspot worth isolating

## No-Go Alternative Considered

A no-go decision would have been reasonable only if one of the following were true:

- PyTorch + MPS profiling showed attention as the dominant issue and a narrow hybrid fix looked obviously better
- Apple-native training candidates appeared too immature to justify prototype effort
- the integration burden looked rewrite-sized even for a minimal experiment

The planning evidence does not support those conclusions.

## Approved Implementation Direction

Approved next implementation direction:

- Implementation Phase: MLX Prototype

Approved scope boundary:

- forward pass
- backward pass
- one optimizer step
- reference-workload benchmark comparison

Everything else remains gated behind the first benchmark result.

## Final Recommendation Summary

Go.

Build the smallest MLX training prototype that can answer the benchmark question.

Do not broaden the scope unless it materially improves optimizer cost or memory behavior over the PyTorch + MPS baseline.

## Related Documents

- [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md)
- [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md)
- [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md)
- [APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md](APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md)