# Apple-Native Acceleration Profiling Memo

## Purpose

This memo records Phase 1 profiling results for the frozen PyTorch + MPS reference workload on the reference M2 Ultra machine.

Profile date: 2026-03-16

## Profiled Workload

The workload matches the frozen planning baseline:

- backend: PyTorch + MPS
- depth: `32`
- device batch size: `2`
- max sequence length: `1024`
- window pattern: `L`
- parameter count: `2,818,575,424`

Profiling command:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/profile_mps_reference.py --depth 32 --device-batch-size 2 --max-seq-len 1024 --warmup-steps 1 --profile-steps 2
```

The profiler uses synthetic token inputs matching the frozen benchmark shape and splits one training step into:

- forward
- backward
- optimizer step
- zero_grad

## Timing Results

Mean per-step timings across two profiled steps:

- forward: `0.729 s`
- backward: `1.229 s`
- optimizer: `10.692 s`
- zero_grad: `0.00045 s`
- total: `12.650 s`
- effective throughput: `161.9 tokens/s`

Phase share of total step time:

- forward: `5.8%`
- backward: `9.7%`
- optimizer: `84.5%`
- zero_grad: effectively zero

## Throughput Conclusion

The dominant throughput bottleneck is not attention.

It is the optimizer step by a wide margin.

On this workload, the optimizer consumes roughly `8.7x` the forward time and `8.7x` the backward time, and about `84.5%` of the full step.

This means the current reference bottleneck is best classified as optimizer and runtime overhead, not model forward compute.

## Memory Results

MPS memory snapshots from the last profiled step:

- before forward:
  - allocated: `25.51 GB`
  - driver: `67.36 GB`
  - recommended: `107.52 GB`
  - driver fraction: `62.65%`
- after forward:
  - allocated: `39.89 GB`
  - driver: `67.49 GB`
  - recommended: `107.52 GB`
  - driver fraction: `62.77%`
- after backward:
  - allocated: `36.01 GB`
  - driver: `67.74 GB`
  - recommended: `107.52 GB`
  - driver fraction: `63.00%`
- after optimizer step:
  - allocated: `36.01 GB`
  - driver: `67.36 GB`
  - recommended: `107.52 GB`
  - driver fraction: `62.65%`
- after zero_grad:
  - allocated: `25.51 GB`
  - driver: `67.36 GB`
  - recommended: `107.52 GB`
  - driver fraction: `62.65%`

Estimated tensor ownership:

- parameters: `10.50 GB`
- gradients: `10.50 GB`
- optimizer state: `15.00 GB`
- forward activation delta in allocated memory: `14.38 GB`

## Memory Conclusion

There is no single memory category that fully explains the baseline limit.

Instead, memory pressure is split across:

1. optimizer state as the largest persistent tensor block
2. parameters and gradients as large steady-state residents
3. activations as the largest transient step-time spike

That gives two practical conclusions:

- the largest persistent contributor is optimizer state
- the largest transient contributor is activations during forward

So the current limit is mixed model-state plus activation pressure, not purely framework behavior.

## Ranked Bottleneck List

1. Optimizer step overhead
   Classification: primarily framework-specific and runtime-specific, with some model-size amplification
   Evidence: `84.5%` of step time is spent in the optimizer path

2. Activation spike during forward
   Classification: primarily model-specific, amplified by framework execution behavior
   Evidence: allocated memory rises by about `14.38 GB` during forward

3. Optimizer state residency
   Classification: mostly algorithm-specific and implementation-specific
   Evidence: optimizer state accounts for about `15.00 GB`, the largest persistent tensor bucket

4. Parameter and gradient residency
   Classification: model-specific
   Evidence: parameters and gradients each account for about `10.50 GB`

5. Attention compute
   Classification: model-specific, but not currently dominant on this workload
   Evidence: forward plus backward combined are still much smaller than optimizer time

## Framework-Specific Versus Model-Specific Readout

Likely framework-specific or runtime-specific:

- optimizer step dominance on MPS
- overhead from the current optimizer execution path versus raw model compute

Likely model-specific:

- parameter residency
- gradient residency
- activation growth with sequence length, width, and depth

Mixed:

- how activation pressure manifests inside MPS memory accounting
- whether optimizer-state layout and update style can be improved without changing the model

## Implication For The Apple-Native Decision

The current evidence strengthens the case for Apple-native work only if it can materially improve one of these two areas:

1. optimizer execution cost
2. persistent and transient memory footprint

This also weakens the case for starting with an attention-only hotspot replacement unless later profiling shows a more specific attention pathology.

At the current baseline, a backend or prototype that does not improve optimizer execution or memory behavior is unlikely to justify itself.

## Related Documents

- [APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md](APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md)
- [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md)
- [profile_mps_reference.py](profile_mps_reference.py)