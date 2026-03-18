## BackLite (experimental)

[BackLite](https://github.com/moonmath-ai/BackLite) is an experimental drop-in kernel that can accelerate pretraining by modifying the backward pass. It is currently Hopper-only (H100/H200).

### Install BackLite

Clone the BackLite repo into the project root and build the Hopper kernel:

```bash
git clone https://github.com/moonmath-ai/BackLite.git
uv pip install --no-build-isolation BackLite/hopper/
```

### Launch a BackLite training run

Pass `--backlite-negl-prob` and `--negl-prob-warmup-steps` to `base_train`:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --run="d24-backlite" \
    --model-tag="d24_backlite" \
    --fp8 \
    --backlite-negl-prob=0.1 \
    --negl-prob-warmup-steps=100
```

You should see `✓ BackLite enabled, negl_prob=0.1` in the output.
