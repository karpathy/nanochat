# M2 Ultra Scaling

## Reality Check

- Apple M2 Ultra is exposed to PyTorch as one `mps` accelerator.
- The machine has a 60-core GPU, but not 60 separate GPUs.
- The practical scaling problem is therefore single-device unified-memory optimization, not distributed training.

## Objective

Move nanochat from "can run on Apple Silicon" to "can be intentionally scaled on a high-end Apple Silicon workstation".

## What Was Added

- `dev/benchmark_mps_scaling.py`: synthetic training-step benchmark for MPS.
- `runs/runm2ultra.sh`: setup + benchmark entrypoint for this class of machine.
- `runs/runm2ultra_base32.sh`: recommended first serious base-model run for this machine.
- MPS benchmark uses:
  - parameter count
  - tokens/sec
  - `torch.mps.current_allocated_memory()`
  - `torch.mps.driver_allocated_memory()`
  - `torch.mps.recommended_max_memory()`

## Measured Results On This Machine

Single synthetic train-step probes on this Mac Studio M2 Ultra, sequence length 1024:

| Depth | Batch | Params | Tokens/s | Driver Memory | Recommended Memory | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 24 | 2 | 1.38B | 83.3 | 33.9 GB | 107.5 GB | stable |
| 28 | 1 | 2.02B | 128.6 | 44.5 GB | 107.5 GB | stable |
| 32 | 1 | 2.82B | 64.7 | 60.5 GB | 107.5 GB | stable |
| 32 | 2 | 2.82B | 139.3 | 65.4 GB | 107.5 GB | best current starter tier |
| 40 | 1 | 4.99B | 17.7 | 111.3 GB | 107.5 GB | frontier only, above recommended memory |

These are not sustained end-to-end training throughput measurements. They are single-step capacity probes used to map a safe starting ladder on MPS.

## MPS Scaling Fixes Required

To make larger MPS training steps work, the project needed more than portability fixes:

- fused optimizer kernels are now left in eager mode by default on MPS
- AdamW scalar control tensors are moved onto the parameter device in eager mode
- Muon update scalars and final update dtype paths were aligned for MPSGraph compatibility

## Recommended Workflow

1. Run `bash runs/runm2ultra.sh` when you want to refresh the local ladder.
2. Start from `bash runs/runm2ultra_base32.sh` for the first meaningful beyond-GPT2 run on this machine.
3. Use `depth=32, batch=2, seq=1024` as the current default M2 Ultra base-model tier.
4. Treat `depth=40, batch=1, seq=1024` as an experimental frontier, not the default.
5. Only after a stable baseline should you experiment with larger sequence lengths or different attention patterns.

Both M2 Ultra scripts now auto-bootstrap missing prerequisites:

- missing dataset shards trigger `python -m nanochat.dataset -n $NANOCHAT_BOOTSTRAP_SHARDS`
- missing tokenizer artifacts trigger `python -m scripts.tok_train --max-chars=$NANOCHAT_TOKENIZER_MAX_CHARS`

Useful environment overrides:

- `NANOCHAT_BOOTSTRAP_SHARDS=32` for a smaller initial download
- `NANOCHAT_TOKENIZER_MAX_CHARS=500000000` for a faster tokenizer bootstrap

## Why This Matters

The original repo is tuned around CUDA speedruns and GPT-2-equivalent goals. On this machine the interesting axis is different:

- larger unified-memory experiments
- deeper local models
- longer sustained single-device training
- local exploration before renting H100-class hardware

## Immediate Target Direction

Good current candidates for this machine class are:

- default tier: depth 32, batch 2, sequence length 1024
- conservative tier: depth 24, batch 2, sequence length 1024
- exploration tier: depth 28 to 32 with longer training horizons
- frontier tier: depth 40, batch 1, sequence length 1024, only if you accept memory pressure above the recommended MPS budget

## Constraints To Keep In Mind

- `torch.compile` remains disabled by default on MPS in training entrypoints.
- MPS memory pressure is unified with system memory, so overcommitting hurts the whole machine.
- Throughput scaling on MPS will flatten earlier than on CUDA systems with tensor-core optimized kernels.

## Success Criteria For This Phase

- A repeatable benchmark-driven scaling workflow for M2 Ultra.
- A known stable depth and batch-size ladder on this machine.
- A documented path for training models meaningfully beyond the original "GPT-2 equivalent" framing.