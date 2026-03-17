# Apple-Native Acceleration MLX Training Check

## Purpose

This note defines the MLX short-run training sanity check, the metrics to log, the success criteria, and the observed results.

The goal is not final model quality.

The goal is to answer a narrower question:

- does the MLX prototype behave like a healthy training path under repeated optimizer steps?

## Metrics To Test And Log

Per-step metrics:

- loss
- step time in seconds
- tokens per second
- gradient L2 norm
- gradient non-finite count
- parameter L2 norm
- parameter non-finite count
- active memory in GB
- peak memory in GB
- cache memory in GB

Aggregate metrics:

- initial loss
- final loss
- minimum loss
- loss drop percentage
- mean step time
- step-time coefficient of variation
- mean throughput
- mean gradient norm
- mean parameter norm
- active-memory span across the run
- peak memory across the run

## Success Criteria

The training check passes only if all of the following are true:

1. all losses are finite
2. all gradients are finite
3. all parameters remain finite
4. gradient signal is present on every step
5. final loss is at least `1%` lower than initial loss
6. throughput is positive on every step
7. step-time coefficient of variation is below `0.25`

These criteria are designed to detect:

- exploding or invalid training state
- dead optimization with zero gradients
- no actual learning progress on a repeated batch
- unstable runtime behavior during a short run

## Execution Commands

Small sanity run:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/mlx_training_check.py --depth 2 --device-batch-size 1 --max-seq-len 32 --steps 6 --warmup-steps 1 --init-from-pytorch-reference
```

Reference-tier sanity run:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/mlx_training_check.py --depth 32 --device-batch-size 2 --max-seq-len 1024 --steps 6 --warmup-steps 1 --init-from-pytorch-reference
```

Reference-tier sanity run with progress output and experimental Muon-style matrix optimizer:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/mlx_training_check.py --depth 32 --device-batch-size 2 --max-seq-len 1024 --steps 6 --warmup-steps 1 --init-from-pytorch-reference --matrix-optimizer muon --progress
```

Reference-tier sanity run using dataset-backed input batches when local parquet data is available:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/mlx_training_check.py --depth 32 --device-batch-size 2 --max-seq-len 1024 --steps 6 --warmup-steps 1 --init-from-pytorch-reference --input-mode dataset --progress
```

Longer Apple-native compiled training session:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/mlx_training_session.py --depth 32 --device-batch-size 2 --max-seq-len 1024 --steps 32 --warmup-steps 2 --init-from-pytorch-reference --matrix-optimizer adamw --progress-interval 4
```

## Results

### Small Run

Configuration:

- depth: `2`
- batch size: `1`
- sequence length: `32`
- steps: `6`
- initialization: translated PyTorch reference weights

Observed result:

- status: `PASS`
- initial loss: `9.276`
- final loss: `0.732`
- minimum loss: `0.732`
- loss drop: `92.1%`
- mean throughput: `3762.8 tok/s`
- steady-state step-time CV: `0.0596`
- peak memory: `0.487 GB`

### Reference-Tier Run

Configuration:

- depth: `32`
- batch size: `2`
- sequence length: `1024`
- steps: `6`
- initialization: translated PyTorch reference weights

Observed result:

- status: `PASS`
- initial loss: `9.349`
- final loss: `2.092`
- minimum loss: `2.092`
- loss drop: `77.6%`
- mean throughput: `803.2 tok/s`
- steady-state step-time CV: `0.0162`
- peak memory: `59.50 GB`

### Experimental Muon Matrix-Optimizer Run

Small configuration:

- status: `PASS`
- initial loss: `9.247`
- final loss: `0.154`
- loss drop: `98.3%`
- mean throughput: `2533.4 tok/s`
- steady-state step-time CV: `0.0565`
- peak memory: `0.462 GB`

Reference-tier configuration:

- status: `PASS`
- initial loss: `1.523`
- final loss: `0.0045`
- loss drop: `99.7%`
- mean throughput: `257.9 tok/s`
- steady-state step-time CV: `0.0043`
- peak memory: `58.93 GB`

Raw reference-tier benchmark with Muon-style matrix optimization:

- throughput: `269.1 tok/s`
- peak memory: `47.13 GB`

Interpretation:

- the experimental Muon-style matrix optimizer is numerically healthy in short runs
- it is currently much slower than the AdamW-based matrix path in this MLX prototype
- so it improves algorithmic resemblance to nanochat, but not yet wall-clock performance

### Dataset-Backed Input Progress

Status: `IMPLEMENTED BUT BLOCKED LOCALLY`

What is done:

- the MLX benchmark and training-check scripts now support `--input-mode dataset`
- dataset-backed batch construction is implemented in [mlx_input_batches.py](mlx_input_batches.py)

Current blocker on this machine:

- no local parquet shards were found under the nanochat dataset cache paths

Implication:

- dataset-backed MLX checks are ready to run once local shards exist
- until then, repeated synthetic batches remain the active fallback for MLX validation

## Interpretation

The MLX prototype currently passes the short-run training sanity check at both the small tier and the frozen reference tier.

What this means:

- repeated optimizer steps reduce loss on a fixed batch
- gradients and parameters remain finite
- there is visible gradient signal throughout the run
- step times are stable after the first measured iteration
- memory remains stable across the short run

Important caveat:

- throughput from the training-check script is lower than the raw benchmark script because the training check computes extra health metrics such as gradient norms and parameter norms on every step

So the training-check throughput should be used as an operational-health signal, not as the primary backend-comparison number.

## Step Status

Current progress on the immediate MLX follow-up steps:

1. closer optimizer parity: `PARTIALLY COMPLETE`
2. dataset-backed input path: `IMPLEMENTED, BLOCKED BY MISSING LOCAL DATA`
3. real checkpoint validation: `BLOCKED BY MISSING LOCAL CHECKPOINTS`