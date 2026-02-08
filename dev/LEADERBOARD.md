# Leaderboard

Docs on participating in the "Time-to-GPT-2" leaderboard of nanochat.

The primary metric we care about is "time to GPT-2" - the wall clock time needed to outperform the GPT-2 (1.6B) CORE metric on an 8XH100 GPU node. Originally in 2019, GPT-2 was trained by OpenAI on 32 TPU v3 chips for 168 hours (7 days), with $8/hour/TPUv3 back then, for a total cost of approx. $43K. It achieves 0.256525 CORE score, which is an ensemble metric introduced in the DCLM paper over 22 evaluations like ARC/MMLU/etc.

## How to

The script [runs/speedrun.sh](runs/speedrun.sh) always implements the current state of the art on the leaderboard.

In practice, I tune the base_train command a little bit. For example, once all the setup is configured and a tokenizer is trained, I like to do something like:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --run="d26-feb2-fp8-ratio8.25" \
    --model-tag="d26_feb2_fp8_ratio8.25" \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=8.25 \
    --fp8
```

Note that:

- `depth` controls the size of the Transformer
- `run` is the wandb name
- `model-tag` is the location of the checkpoints on disk
- `device-batch-size` in the ideal world, you want this to be 32 because with sequence length of 2048 (the default) and 8 GPUs we get `32 X 2048 X 8 = 524,288`, which is the total desired batch size determined to work fairly well around this scale. However, for bigger (e.g. d26), 32 is too much and OOMs, so we decrease it by 2 to 16. The `base_train.py` script automatically compensates for this by calculating that it has to use gradient accumulation of 2 to meet the desired total batch size. Therefore, it will do forward+backward twice and then a single step. Long story short, the ideal value is 32. If that doesn't fit, you decrease it, e.g. 16, 8, etc., keeping it powers of two so that the gradient accumulation math works out neatly.
- `sample-every = -1` turns off periodic sampling
- `core-metric-max-per-task=-1` means we run the entire CORE eval
- `core-metric-every=999999` a bit of a hacky way to make the CORE eval only happen a single time at the very end of the run
- `target-param-data-ratio=8.25` controls the training horizon, which is determined in the script by taking the number of non-embedding model parameters and simply multiplying by this number. The current optimal Tokens:Params ratio can be seen in the defaults of the `base_train.py` script (it is 10.5). 10.5 would produce the *compute optimal* model given the currently measured scaling laws. However, GPT-2 capability is currently somewhere in between a d24 and d26. So to reach it exactly, we want to either overtrain d24 or undertrain d26. In this particular example, I am choosing to slightly undertrain a d26. Note that odd depths (e.g. d25) are not super recommended to use because the math around the transformer sizing and its head dimensions doesn't come out neatly.
- `--fp8` turns on fp8 training. If your GPU does not support fp8, you can leave this out and the code will simply train in bf16. bf16 is higher precision than fp8, so you can actually expect that you might be able to do fewer steps (lower the `target-param-data-ratio`) to achieve the same capability.

Once you kick off the run, you wait ~3 hours and then at the end you'll see something like:

```
wandb: Run summary:
wandb:          core_metric 0.25851
wandb:                 step 16704
wandb: total_training_flops 4.330784131228946e+19
wandb:  total_training_time 10949.46713
```

Your CORE metric must be greater than GPT-2 0.256525. Then you report the `total_training_time`, (e.g. 10949) which is the time of the training iterations alone, excluding all the evaluations and logging, in seconds. So here for example it is roughly 10949/60/60 ~= 3.04 hours. You should also note and report the validation bpb of your run because the CORE metric can be a little bit noisy.

If you outperform GPT-2 and the time is less than current SOTA in the Leaderboard, you get to make a PR. In addition to raw gains, there are some qualitative and aesthetic considerations that go into whether your improvement is merged. For example, if it is gnarly or it significantly bloats the code, or it seems too esoteric, then we will weigh those things against the improvement demonstrated. Additionally, nanochat cares not only about targeting a single model, but an entire miniseries of models. So your change must be principled enough that it can easily generalize to other model depths, so that we can sweep out a miniseries.

After you create the commit, to get the current short git commit hash:

```
git log -1 --format="%h"
```

## Run 1

Achieved Jan 29 2026 on commit `348fbb3`. The launch command was

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --run=d24-jan29 \
    --model-tag=d24_jan29 \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=3000 \
    --target-param-data-ratio=12
```

The result was:

```
wandb: Run summary:
wandb:          core_metric 0.25851
wandb:                 step 16704
wandb: total_training_flops 4.330784131228946e+19
wandb:  total_training_time 10949.46713
```

The validation bpb was 0.74833.

Detailed writeup: [Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)

## Run 2

Achieved Feb 2 2026 on commit `a67eba3`. The launch command was

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --run="d26-feb2-fp8-ratio8.5" \
    --model-tag="d26_feb2_fp8_ratio8.5" \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=8.5 \
    --fp8
```

The result was:

```
core_metric 0.2578
step 14889
total_training_time 10493
Minimum validation bpb: 0.745036
```

The big change in this run is `--fp8`, which causes all Linear layers (other than the gates) to be switched to fp8 training using `torchao` with tensorwise fp8 scaling. Each step is of slightly lower quality, but we are taking them a lot faster, coming out net ahead. Anyone who does not have fp8 (e.g. using a GPU without it) can simply leave out the `--fp8` flag to train in bfloat16. This will work just fine but it will produce a slightly stronger model than GPT-2 because of the fp8 -> bf16 precision upgrade. It's possible that one can further tune which layers to include in the fp8 conversion and that e.g. some of the smaller matmuls should be just kept in bf16 etc.

Previous record was 3.04 hours, so 2.91 hours is `(3.04 - 2.91)/3.04*100` ~= 4.3% speed improvement.

## Run 3

Achieved Feb 5 2026 on commit `2c062aa`. Launch command:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --run="d26_feb4_double_batch_ratio8.25" \
    --model-tag="d26_feb4_double_batch_ratio8.25" \
    --device-batch-size=16 \
    --total-batch-size=1048576 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=8.25 \
    --fp8
```

Result:

```
core_metric 0.26024
step 7226
total_training_time 9922
Minimum validation bpb: 0.74645
```

The big change here is that the batch size was doubled from 0.5M to 1M, which works better for a d26 model and allowed me to decrease the number of optimization steps a bit via `--target-param-data-ratio` from 8.5 to 8.25. The TLDR is that the original batch size of 0.5M was tuned for d12, but bigger models (e.g. d26) prefer larger total batch size. I determined in experiments that d26 prefers 1M. Then I implemented and merged a principled way to calculate the optimal batch size given depth so that all nanochat models of all depths benefit. See [dev/LOG.md](dev/LOG.md) entry "2026-02-05: Auto Batch Size Scaling" for more detail.
