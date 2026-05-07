# Minimal Bigram Speedrun PR Changes

This branch is based on upstream nanochat master at `dc54a1a`. The goal is to
keep the submission patch limited to the changes needed to reproduce the
best-performing speedrun recipe:

```bash
--fp8
--bigram-embed-factor=5
--muon-plus
--muon-eq=row
--scalar-lr=0.3
--train-log-every=50
--compile-mode=max-autotune-no-cudagraphs
```

It does not include the experimental branches that were tested and rejected:
sparse architecture changes, MoE/TOP auxiliary losses, train-time logit-bias
losses, post-hoc calibration, NorMuon variants, checkpoint merging, or d22/d24
run-management scripts.

## `nanochat/gpt.py`

### Hashed Bigram Residual Embedding

Adds two config fields:

- `bigram_embed_factor`, default `0`
- `bigram_lambda_init`, default `0.05`

When `bigram_embed_factor > 0`, the model creates a separate bigram embedding
table with `vocab_size * bigram_embed_factor` entries. For each token position,
the current token id and previous token id are hashed into that table. The
resulting embedding is added as a residual input before every transformer block:

```python
x = x + bigram_lambdas[i] * x0_bigram
```

The first token in each sequence uses a sentinel bucket because it has no
previous token. During KV-cache decoding, the previous token is read from the
cache so generation matches the training-time bigram definition.

Why this helps: it gives the model a cheap, direct representation of adjacent
token pairs without adding attention or MLP compute. The bigram table is
zero-initialized, so the model starts from the original network function, while
the per-layer `bigram_lambdas` start at `0.05` to let the residual learn quickly.

### Parameter Counting and FLOP Accounting

The bigram embedding table and bigram lambdas are excluded from the main matmul
FLOP/scaling parameter count. They are not transformer matrix weights, and
including them would distort the target param/data ratio logic.

### Optimizer Groups

Adds dedicated optimizer groups for:

- `bigram_embed`
- `bigram_lambdas`

The bigram embedding uses AdamW with a configurable multiplier relative to the
main embedding LR. The layer lambdas use a small AdamW LR. This keeps the bigram
residual trainable without mixing it into the Muon-managed transformer matrices.

### Muon Options Plumbed Through

`setup_optimizer()` accepts:

- `muon_plus`
- `muon_eq_axis`

These are forwarded into the Muon parameter groups so the optimizer can apply
the selected Muon variants to matrix weights.

## `nanochat/optim.py`

### Muon+ Renormalization

After Newton-Schulz orthogonalization, Muon+ rescales the update by its
Frobenius norm. This is a small post-processing step on the Muon update and was
the strongest optimizer-side change in the experiments.

Why this helps: it stabilizes update scale after orthogonalization without
changing the model architecture or adding optimizer state.

### Row/Column Equilibration

Adds optional row or column norm equilibration before orthogonalization:

- `muon_eq_axis=1`: row equilibration
- `muon_eq_axis=2`: column equilibration
- `muon_eq_axis=0`: disabled

The speedrun recipe uses row equilibration. It normalizes rows toward a common
target norm before the polar/Newton-Schulz step, then continues through the
existing Muon update path.

Why this helps: row equilibration was a small but positive companion to Muon+ in
the winning recipe, with minimal extra code and no extra persistent optimizer
state.

## `nanochat/engine.py`

### Previous Token in KV Cache

Adds `prev_token` to `KVCache`, resets it with the rest of the cache, and copies
it during prefill expansion.

Why this is needed: full-sequence training can compute bigram hashes from
`idx[:, :-1]`, but one-token decode does not have the previous token in the
current input tensor. Keeping `prev_token` in the cache makes generation use the
same bigram feature as training.

## `scripts/base_train.py`

### Bigram CLI Flags

Adds:

- `--bigram-embed-factor`
- `--bigram-lambda-init`
- `--bigram-embedding-lr-mult`
- `--bigram-lambda-lr`

These configure the bigram residual and its optimizer treatment from the
training script without changing defaults. With default values, upstream
behavior is unchanged because `--bigram-embed-factor` defaults to `0`.

### Muon Variant Flags

Adds:

- `--muon-plus`
- `--muon-eq`

These expose the optimizer variants used in the recipe. Defaults preserve the
original optimizer behavior.

### Train Logging Cadence

Adds `--train-log-every`. Values greater than 1 avoid converting the loss tensor
to a Python scalar every step.

Why this helps: per-step logging creates extra synchronization overhead. The
speedrun uses `--train-log-every=50`, which keeps useful progress reporting
while reducing logging overhead.

### Compile Mode

Adds `--compile-mode` so the speedrun can request:

```bash
--compile-mode=max-autotune-no-cudagraphs
```

Why this helps: on the d16 probe, this compile mode was about 2.5% faster than
default `torch.compile` for the candidate recipe.

### Skip Initial Eval

Adds `--skip-initial-eval`. This avoids spending benchmark wall time on the
step-0 validation pass when it is not needed for a speedrun submission.

## `runs/speedrun.sh`

Updates the default speedrun command to use the winning recipe flags:

- FP8
- total batch size `1048576`
- Muon+
- row equilibration
- bigram factor 5
- scalar LR `0.3`
- log every 50 training steps
- `max-autotune-no-cudagraphs` compile mode

This script is the intended entry point for reproducing the submitted run.

## `tests/test_engine.py`

Adds coverage for preserving `prev_token` through KV-cache prefill/expansion.

Why this matters: the bigram feature must behave consistently during generation.
The test guards the cache state required for single-token decode.

## `dev/bigram_speedrun_results.md`

Records the validation and throughput evidence used to justify the recipe:

- minimal branch sanity check against the prior candidate branch
- full d16 comparison against upstream dense
- controlled d16 throughput comparison
- compile-mode probe
- test status

This is supporting documentation for the PR, not code required at runtime.

## Submission Readiness

Completed checks:

- `python -m pytest tests/test_engine.py -q`
- `python -m py_compile nanochat/gpt.py nanochat/optim.py scripts/base_train.py nanochat/engine.py`
- `git diff --check`

The remaining work is operational: run the final benchmark on the 8xH100 system
from this branch and include the measured result in the submission PR.
