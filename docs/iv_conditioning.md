# Clarinet: source-conditioned generation, end to end

This note explains how Clarinet conditions an autoregressive language model on
the *source* of a document, how that conditioning is installed during training,
and how it is exploited at inference — including the L1 content-adaptive
guidance schedule. It is written to be read top to bottom: each layer builds on
the one before it.

Clarinet is a research fork of nanochat. Everything below concerns the files in
`clarinet/` (`dataset.py`, `dataloader.py`, `engine.py`) plus the special-token
registration in `nanochat/tokenizer.py` and the loss path in `nanochat/gpt.py`.
The technique is a classifier-free-guidance (CFG) adaptation of autoregressive
text generation, dressed in instrumental-variable (IV) vocabulary; we keep the
IV names because the code uses them, but the honest description is "CFG with a
source marker."

---

## 1. The core idea

A plain language model learns one next-token distribution `p(x_t | x_<t)`. We
would like to *steer* generation toward a particular kind of text — here,
"reasoning"-style (math, step-by-step) continuations versus generic web text —
without training a separate model or a reward model.

The mechanism is a single extra token, the **source marker**, placed at the
start of every document. During training the model sees the marker that
describes where the document came from, so it learns a *conditional*
distribution `p(x_t | x_<t, marker)`. At inference we run the model twice —
once with the "reasoning" marker, once with a neutral "unknown" marker — and
push the output distribution in the direction the marker induces. That push is
the whole game; sections 5–6 make it precise.

The reason a *single* model can produce both the conditional and the neutral
distribution is a training-time dropout trick (section 4). The reason we are
allowed to set the marker ourselves at inference rather than sampling it is a
target-masking trick (section 4). Both are small, and both are load-bearing.

---

## 2. The special tokens

Three markers are registered as special tokens in
`nanochat/tokenizer.py` (`SPECIAL_TOKENS`, lines 28–30):

```
<|src_reasoning|>   # document came from the reasoning corpus
<|src_general|>     # document came from the general corpus
<|src_unknown|>     # source deliberately hidden (the "neutral" marker)
```

They get fixed integer ids at the top of the vocabulary, alongside the existing
chat/tool specials (`<|bos|>`, `<|assistant_end|>`, `<|python_start|>`, …). The
tokenizer exposes `encode_special(name) -> id` and `get_bos_token_id()`, which
both the dataloader and the engine use to look these ids up by name rather than
hard-coding integers.

`<|bos|>` (beginning-of-sequence) is not new, but it is central: it marks the
start of every document and doubles as the document separator inside a packed
training row. Every document, in training and inference, has the layout

```
[ <|bos|> , <marker> , ...document tokens... ]
   pos 0      pos 1        pos 2 onward
```

The marker always sits at **position 1**, immediately after BOS.

---

## 3. Datasets: two corpora with a source flag

`clarinet/dataset.py` combines two parquet-backed corpora:

- **climbmix** — the general corpus inherited from upstream nanochat
  (`nanochat.dataset.list_parquet_files`).
- **reasoning** — math/reasoning shards written by
  `clarinet/prepare_reasoning_data.py`. This is currently FineMath
  (`HuggingFaceTB/finemath`, `finemath-4plus`); the original plan named
  `proof-pile-2`, but its HF hosting 404s as of 2026-05, and FineMath is the
  closest parquet-native, actively-maintained replacement.

The public entry point is:

```python
list_parquet_files_with_source(split)  ->  [(path, is_reasoning), ...]
```

It returns every shard for the requested split, each tagged with a boolean
`is_reasoning`. The train/val convention matches upstream exactly: within each
source directory the **last shard is validation**, all earlier shards are
training. So `split="train"` drops the last shard of each corpus and returns the
rest; `split="val"` keeps only the last shard of each. The function asserts that
reasoning shards exist, with a message pointing at
`python -m clarinet.prepare_reasoning_data`, because a silent empty reasoning
set would quietly collapse Clarinet back into plain nanochat.

Note what this function does *not* do: it does not interleave or order the two
sources. It just labels them. Ordering is the dataloader's job, because ordering
has to be consistent across distributed-training ranks.

---

## 4. The dataloader: where conditioning is installed

`clarinet/dataloader.py` is a modified copy of upstream's BOS-aligned best-fit
loader. It does five things on top of the original; the first three are what
make conditioning work.

### 4.1 Deterministic interleaving of the two sources

`_interleave_sources(paths_with_source, reasoning_mix_ratio)` merges the climbmix
and reasoning shard lists into one ordered list that hits the target mix ratio
(default `0.3`) as evenly as possible. The schedule is purely arithmetic — at
step `k` the desired cumulative count of reasoning files is `round(k * ratio)`,
and a reasoning file is emitted whenever that target ticks up:

```python
want_reasoning = round(step * reasoning_mix_ratio)
if want_reasoning > emitted_reasoning and reasoning_left:
    emit reasoning file
else:
    emit climbmix file
```

This spreads the reasoning shards evenly (no clumping) and, crucially, uses **no
RNG**. Every distributed rank computes the identical ordering from the identical
shard list, so the per-rank row-group sharding downstream is well-defined
without a shared seed.

### 4.2 BOS prepend, marker splice, and unconditional dropout

`_document_batches` walks the interleaved shard list, reads parquet row groups,
and yields batches of raw document strings tagged with their `is_reasoning`
flag. `refill_buffer` (inside `clarinet_data_loader`) then tokenizes each batch
with a BOS prepended:

```python
token_lists = tokenizer.encode(doc_batch, prepend=bos_token, ...)
```

so every document already starts `[BOS, ...]`. The marker is spliced in right
after BOS:

```python
true_marker = src_reasoning_id if is_reasoning else src_general_id
for tokens in token_lists:
    marker = src_unknown_id if rng.random() < p_uncond else true_marker
    tokens.insert(1, marker)            # -> [BOS, marker, ...doc tokens]
    doc_buffer.append(tokens)
```

The `p_uncond` line (default `0.1`) is the **classifier-free-guidance
unconditional dropout**: 10% of documents have their true marker replaced by
`<|src_unknown|>`. This is what teaches the *one* set of weights to model the
neutral distribution `p(x | unknown)` in addition to the source-conditional
distributions. Without it, `<|src_unknown|>` would be an out-of-distribution
token at inference and the unconditional pass (section 5) would be meaningless.

The dropout RNG is seeded per rank (`seed + ddp_rank`), so ranks make
independent dropout choices — fine, since each rank owns disjoint row groups.

### 4.3 Best-fit packing and target masking

Documents are packed into rows of width `T + 1` using a best-fit heuristic: for
each position, take the longest buffered document that still fits; if none fits,
truncate the shortest to fill the remainder. This is upstream's packing,
unchanged, and it is why a single row contains several documents each starting
with its own BOS.

After packing, inputs and targets are the usual shifted pair, plus one new line:

```python
cpu_inputs.copy_(row_buffer[:, :-1])    # tokens 0 .. T-1
cpu_targets.copy_(row_buffer[:, 1:])    # tokens 1 .. T
cpu_targets[cpu_inputs == bos_token] = -1
```

That last line is the **marker target mask**. The token that follows a BOS is
always the source marker; setting those target positions to `-1` means the model
is *never trained to predict the marker*. Combined with
`F.cross_entropy(..., ignore_index=-1)` in `nanochat/gpt.py:477`, the marker
contributes zero loss as an output. It is purely an input that conditions the
distribution — which is exactly why, at inference, we are free to set it
ourselves rather than letting the model emit it.

### 4.4 The plumbing that did not change

Everything else mirrors upstream: DDP sharding at the row-group level
(`rg_idx += ddp_world_size`), a pinned CPU staging buffer, a single
host-to-device copy per batch, and a resumable `state_dict` of
`{pq_idx, rg_idx, epoch}`. `pq_idx` now indexes the *interleaved* shard list, so
resume points stay valid across the combined corpus.

### 4.5 What training therefore learns

`scripts/clarinet_train.py` drives this loader with `p_uncond` from its CLI for
the train split and `p_uncond=0.0` for the validation split (we always evaluate
with the true marker). The net effect: a model whose next-token logits are a
function of the marker placed at position 1 — call them `ℓ(x | context, marker)`
— and which has seen enough `<|src_unknown|>` examples to give a sensible neutral
distribution `ℓ(x | context, unknown)`.

---

## 5. The inference engine: dual-pass guidance

`clarinet/engine.py` defines `ClarinetEngine(Engine)`. The upstream engine runs
the model once per decode step; Clarinet runs it **twice**, once per
conditioning, and combines the two logit vectors.

### 5.1 Building the two prompts

`_prefix_with_marker` reproduces the training layout for the prompt, once with
the reasoning marker (the *conditional* prompt) and once with the unknown marker
(the *unconditional* prompt):

```
cond   = [BOS, <|src_reasoning|>, ...prompt]
uncond = [BOS, <|src_unknown|>,  ...prompt]
```

They differ only at position 1, so they are guaranteed equal length — the decode
loop relies on this, and `generate` asserts it. (If a prompt does not already
begin with BOS, the helper prepends one; chat rendering always does.)

### 5.2 Two KV caches in lock-step

Each prompt gets its own prefill and its own `KVCache`. After prefill, both
caches are expanded to the decode width and advanced together: every step, the
*same* chosen token is appended to both caches. The two passes therefore share
an identical suffix and differ only by the single marker token baked into their
respective prefixes. This doubles compute and KV memory — the real cost of the
method — but keeps the two distributions exactly comparable at every step.

### 5.3 Combining the logits

The combine is a static method (extracted for unit-testing):

```python
combine_logits(cond, uncond, w, s) = uncond + w * s * (cond - uncond)
```

with `w = iv_weight` and `s` the scale factor. The vector `(cond - uncond)` is
*the direction in logit space that the reasoning marker induces*; we walk `w·s`
steps along it starting from the unconditional logits:

- `w = 0` → unconditional distribution.
- `w = 1, s = 1` → conditional distribution exactly.
- `w > 1` (useful range ~1.5–2.5) → **extrapolation beyond** the conditional —
  push harder toward reasoning-mode tokens than the conditional model alone.

Because softmax normalizes away additive constants, this is multiplicative in
probability space:

```
p_iv(x)  ∝  p_uncond(x)^(1 - ws) · p_cond(x)^(ws)
```

a product-of-experts / geometric extrapolation: tokens the marker makes
*relatively* more likely are boosted super-linearly, tokens it makes less likely
are suppressed, and tokens it is neutral about (`p_cond ≈ p_uncond`) are
untouched. One important subtlety: this all happens on the **softcapped** logits
(`15·tanh(raw/15)`, `nanochat/gpt.py:472`), the same space the model emits — not
on raw pre-softcap logits.

Sampling, temperature/top-k, the calculator tool-use state machine, forced
tokens, and completion detection are all inherited unchanged from upstream and
run on the *combined* logits.

---

## 6. The L1 content-adaptive scale

Until recently `s` was a constant (`wald_scale`, default `1.0` ≡ vanilla CFG).
The new feature makes `s` optionally **vary per decode step** based on how much
the marker actually changes the prediction at that step.

### 6.1 The signal and the schedule

At each step we already hold the two distributions. Their **L1 / total-variation
distance** is a single scalar summary of how far apart they are:

```
d = 0.5 * Σ_v | softmax(cond)_v - softmax(uncond)_v |        # d in [0, 1]
```

`d ≈ 0` means the marker barely matters here (context already pins the next
token); `d ≈ 1` means the reasoning and neutral continuations sharply disagree.
The scale is a linear interpolation keyed on `d`:

```python
s = base_scale * (scale_lo + (scale_hi - scale_lo) * d)
```

implemented in `l1_adaptive_scale` (a static method, per-row, returning shape
`(rows, 1)` so it broadcasts over the vocab axis in `combine_logits`). The
intent is to *spend guidance budget where the marker is decisive and back off
where it is irrelevant.*

### 6.2 Properties and the backward-compatible default

- **Floor / ceiling.** At `d = 0`, `s = base·scale_lo`; at `d = 1`,
  `s = base·scale_hi`. The schedule is monotone in `d` and bounded to
  `[base·scale_lo, base·scale_hi]`.
- **Exact CFG default.** `scale_lo == scale_hi` short-circuits to the constant
  `base·scale_lo` *without even computing the softmax*. The shipped default
  `scale_lo = scale_hi = 1.0` therefore reproduces vanilla CFG bit-for-bit, so
  the feature is inert until explicitly enabled.
- **Recommended opt-in.** `scale_lo = 0.5, scale_hi = 2.0` is the suggested
  starting point.

The CLI exposes this in `scripts/clarinet_cli.py`:

```bash
# vanilla CFG (default)
python -m scripts.clarinet_cli --iv-weight 2.0 --wald-scale 1.0
# adaptive
python -m scripts.clarinet_cli --iv-weight 2.0 --scale-lo 0.5 --scale-hi 2.0
```

### 6.3 What this is, and what it is not

This is an **adaptive-CFG schedule**, not a causal estimator. An earlier design
tried to derive `s` as a Wald/IV ratio from pre-`lm_head` hidden states; review
found that incoherent (it divided a vocab-length logit vector by a scalar
"first-stage," pointed the modulation the wrong way, and relied on an exclusion
restriction the transformer does not satisfy). The L1 version keeps only the
defensible core: a cheap, output-space measure of how much the marker moves the
prediction, computed from logits we already have — no probe, no hidden-state
plumbing, no third forward pass.

Crucially, the *direction* of the modulation ("more guidance where `d` is high")
is a **hypothesis**, not a theorem. It must be validated against held-out
reasoning likelihood or a downstream metric like GSM8K, with constant `s` as the
control it has to beat — which is exactly why the default is the control. Note
also that generation behavior depends only on the **product `w·s`**, so `w` and
`s` are not separately identifiable from samples alone; tune them as a pair.

---

## 7. End to end, in one breath

1. **Prepare data.** `prepare_reasoning_data.py` writes FineMath shards;
   `dataset.list_parquet_files_with_source` labels every shard reasoning/general.
2. **Train.** The dataloader interleaves the two corpora deterministically,
   prepends `[BOS, marker]` to each document (dropping the marker to
   `<|src_unknown|>` 10% of the time), best-fit-packs them into rows, and masks
   the marker-prediction targets to `-1`. The model learns
   `p(x | context, marker)` and a neutral `p(x | context, unknown)`.
3. **Generate.** `ClarinetEngine` runs two lock-step passes (reasoning marker
   vs. unknown marker), combines their logits as
   `uncond + w·s·(cond − uncond)`, optionally lets `s` adapt per step via the L1
   distance between the two distributions, and samples from the result —
   feeding the chosen token back into both KV caches.

The dataloader installs a conditioning channel; the engine exploits it; the L1
schedule is an opt-in, still-to-be-validated refinement of *how hard* to lean on
that channel at each step.
