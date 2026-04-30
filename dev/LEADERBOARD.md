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

Once you kick off the run, you wait ~1.5 hours and then at the end you'll see something like:

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

## Run 4

Achived Mar 3 2026 on commit `324e69c`. The big change is the switch from HuggingFace FineWeb-EDU to NVIDIA ClimbMix dataset. `@karpathy` has tried to swap the dataset many times, each time with a negative result (FineWeb, DCLM, Olmo), but ClimbMix produced clear and immediate gains. Credit to `@ddudek` for originally discovering ClimbMix for nanochat and reporting the improvements, which kicked off the followup investigation.

To reproduce, use the commit above, download at least 150 data shards, train the tokenizer:

```
python -m nanochat.dataset -n 150
python -m scripts.tok_train
```

Then kick off the run in the typical way, using a slightly lower than compute optimal ratio of 9.5 (vs compute optimal 10.5), meaning the d24 is slightly undertrained.

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --run="d24-climbmix" \
    --model-tag="d24-climbmix" \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=9.5 \
    --device-batch-size=16 \
    --fp8
```

I ran this command 7 individual times. Because our training is mildly non-deterministic, we get a spread of CORE scores, e.g.:

```
0.25373
0.2584
0.25489
0.2568
0.25732
0.26765
0.25119
```

Mean is 0.25714 (higher than the GPT-2 threshold needed), max-min is 0.01646. Something to investigate in the future is that even slightly better results can be obtained by randomly shuffling the the data shards (i.e. just going in a different order). This is unexpected because the documents were completely fully shuffled during data construction, so one would expect a relatively uniform data distribution. Indeed, the current default order is unfortunately among the worse ("unlucky") ones you can obtain with different shuffle seeds, but it suffices to beat GPT-2 for now so I am merging. TODO investing a bit more later.

NOTE: The `val_bpb` is as of this run *NOT* comparable due to the data distribution change to the previous 3 runs. This run happens to be at `0.71854` validation bpb. If the dataset is not changed, the `val_bpb` number is a great, smooth metric to track relative performance w.r.t. and has less noise than CORE.

## Run 5

Achieved Mar 9, 2026 on commit `6ed7d1d`. Exactly the same launch command as Run 4 except `--target-param-data-ratio=8.7`. I ran 5 identical runs, the average CORE was 0.2690, which is quite a bit above the needed threshold of 0.2565. But the reason I didn't decrease the ratio further (i.e. train shorter) is that while the CORE "safety gap" is large, the val_loss safety gap is smaller - 0.71808, which we want to be below the Run 4 val loss of 0.71854. It's likely that we could have reduced the ratio even lower, possibly to 8.6, but it's not worth splitting hairs at this point.

This commit is special because all of the improvements that went into [this commit](https://github.com/karpathy/nanochat/commit/6ed7d1d82cee16c2e26f45d559ad3338447a6c1b) came from fully autonomous "research" done by a private version of [autoresearch](https://github.com/karpathy/autoresearch) run on a d12 model. I wrote more about this in [this tweet](https://x.com/karpathy/status/2031135152349524125). The changes easily translated from d12 to d24, hence new leaderboard record, taking us from 2.02 hours "time to GPT-2" to 1.80 hours.

## Run 6

Achieved Mar 14, 2026 on commit `a825e63`. Exactly the same launch command as Run 4 except `--target-param-data-ratio=8`. Improvements in the architecture are allowing us to train shorter and shorter time. Instead of an undertrained d24 I attempted to train an overtrained d22 but it was worse. This set of changes came from autoresearch round 2, where I asked it to reference the modded-nanogpt repo for inspiration. So the exploration tried out a number of ideas and in particular found a way to incorporate the backout and smear in such a way that they are helpful (I had previously tried them manually a long time ago and they caused regressions). The smear idea in particular is a little bit heavier and bloaty because it is essentially an "early fusion" of context across tokens, producing a kind of a bigram input into the network and allowing it to focus on higher ngrams earlier. But for this reason the code gets a bit more complex and required some changes to inference. I verified with a unit test that the Engine inference is correct compared to the naive inference of `GPT.generate()`. The average of 5 runs was CORE 0.262634 and each of them lasted 1.65 hours (99 minutes).
