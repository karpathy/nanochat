# Apple-Native Acceleration Baseline Benchmark

## Purpose

This note freezes the reference PyTorch + MPS baseline used for Apple-native acceleration planning.

It defines:

- the default benchmark workload
- the primary comparison metrics
- the planning scope for training versus inference
- the commands used to reproduce the baseline on the reference M2 Ultra machine

## Reference Workload

The default planning benchmark is the current stable M2 Ultra training tier:

- backend: PyTorch + MPS
- device: `mps`
- depth: `32`
- device batch size: `2`
- max sequence length: `1024`
- window pattern: `L`
- approximate parameter count: `2.82B`

This workload is the primary reference point for any Apple-native prototype comparison.

## Baseline Measurements

From the current M2 Ultra scaling note, the reference measurement for the frozen workload is:

- throughput: `139.3 tokens/s`
- driver memory: `65.4 GB`
- recommended memory budget: `107.5 GB`
- operating position: below recommended budget, suitable as the default starter tier

This is a single synthetic training-step probe, not a long-horizon end-to-end training run.

## Primary Comparison Metrics

The planning phase uses four primary metrics:

1. tokens per second
2. MPS driver memory
3. recommended memory fraction or headroom versus `torch.mps.recommended_max_memory()`
4. implementation complexity

Secondary interpretation metrics:

- whether a path expands the stable model-size frontier at fixed sequence length
- whether the engineering burden is small enough to justify ongoing maintenance

## In-Scope Workloads For This Planning Phase

Planning scope is both training and inference, but not equally:

- training is the primary benchmark and architecture decision driver
- inference remains in scope mainly for framework evaluation and deployment relevance
- Core ML and native-runtime questions should be evaluated, but they do not replace the training baseline in this phase

## Repeatable Baseline Commands

Refresh the machine ladder:

```bash
bash runs/runm2ultra.sh
```

Run the frozen default tier:

```bash
bash runs/runm2ultra_base32.sh
```

These scripts already encode the current baseline assumptions and auto-bootstrap missing tokenizer or dataset prerequisites.

## Repeatability Status

Baseline commands are considered repeatable on the reference machine because:

- the repo includes dedicated M2 Ultra benchmark scripts
- the scripts bootstrap missing prerequisites instead of assuming an already-warm machine
- the corresponding benchmark note records stable observed measurements for the reference tier

Current usage rule:

- use `runm2ultra_base32.sh` as the default direct-comparison command
- use `runm2ultra.sh` when refreshing the broader safe operating ladder

## Boundary Conditions

- `torch.compile` remains disabled by default on MPS in the current training path
- runs above the recommended MPS memory budget are not part of the frozen default baseline
- `depth=40`, `batch=1`, `seq=1024` remains a frontier experiment, not the comparison default

## Related Documents

- [APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md](APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md)
- [M2_ULTRA_SCALING.md](M2_ULTRA_SCALING.md)
- [archive/APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md](archive/APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md)