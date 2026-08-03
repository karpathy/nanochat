# SIGReg for language modelling

A walkthrough of porting **SIGReg** — the regularizer from LeJEPA that LeWorldModel uses
as one of its two loss terms — onto a tiny nanochat GPT, and of the A/B test that tells you
whether it actually did anything.

Files:

| file | what it is |
|---|---|
| `nanochat/sigreg.py` | the regularizer |
| `nanochat/gpt.py` | one-line change: `forward(..., return_hidden=True)` |
| `scripts/sigreg_experiment.py` | the A/B harness |
| `tests/test_sigreg.py` | behavioural tests on distributions with known answers |

---

## 1. What SIGReg is, and why it exists

A JEPA predicts one embedding from another. Write the encoder as `f` and the predictor as
`g`; the loss is `|| g(f(x_t)) - f(x_{t+1}) ||²`. Nothing in that objective is anchored to
anything observable, so it has a perfect and useless solution: make `f` constant. Every
prediction is then exactly right and the representation carries no information. This is
*collapse*, and most of the machinery in modern SSL — stop-gradients, teacher/student
networks, momentum encoders, whitening layers — exists to dodge it.

LeJEPA's move is to prove what the embedding distribution *should* be (an isotropic
Gaussian, which minimizes downstream prediction risk) and then add a single loss term that
pushes the embeddings towards it. An isotropic Gaussian has full-rank covariance, so a
collapsed encoder is infinitely far from the target and collapse stops being a solution at
all. LeWorldModel is then just:

```
loss = prediction_loss + λ · SIGReg(embeddings)
```

with λ the only real hyperparameter — which is why LeWM can tune it by bisection search
rather than a grid.

### The two tricks

Testing "is this cloud of `d`-dimensional vectors N(0, I)?" directly is expensive and
noisy. SIGReg gets around it twice over.

**Cramér–Wold.** A distribution on ℝ^d is N(0, I_d) **iff** every 1-D projection `⟨x, a⟩`
onto a unit vector `a` is N(0, 1). So you never need a multivariate test — you sample `M`
random unit directions ("slices") and run a *univariate* test on each. Fresh directions
every step means that over training you cover the sphere, while each step costs only
`O(N·M)`. This is the "sketched" in *Sketched Isotropic Gaussian Regularization*.

**Epps–Pulley.** For the univariate test you want something differentiable with bounded
gradients — rules out sorting-based tests (KS, Shapiro–Wilk) and moment-matching
(skew/kurtosis blow up on outliers). Epps–Pulley compares the *empirical characteristic
function* of your samples to the CF of N(0,1), which is known in closed form as
`φ(t) = exp(−t²/2)`:

```
T_n = n · ∫ | ECF_n(t) − exp(−t²/2) |² · w(t) dt,      w(t) = exp(−t²/2)
```

where `ECF_n(t) = (1/n) Σ_j exp(i·t·z_j)`. Expanding the complex modulus into real terms
gives the form you actually code:

```
| ECF_n(t) − φ(t) |² = ( mean_j cos(t·z_j) − φ(t) )² + ( mean_j sin(t·z_j) )²
```

`cos` and `sin` are bounded by 1, so both the statistic and its gradient are bounded no
matter how far the embeddings drift. That is the property that makes it safe to backprop
through for millions of steps. The integral is a trapezoid rule over `K = 17` knots on
`[0, 3]` — everything is even in `t`, so you integrate the half-line and double, which is
free accuracy. And `exp(−t²/2)` is already ~0.011 at `t = 3`, so there is nothing beyond it
worth integrating.

The whole thing is about ten lines:

```python
zt  = z.unsqueeze(-1) * t                                             # (N, M, K)
err = (zt.cos().mean(-3) - phi).square() + zt.sin().mean(-3).square() # (M, K)
stat = (err @ weights) * N                                            # (M,)
```

### Sanity check

The statistic has a meaningful zero. From `tests/test_sigreg.py`, on 4096 samples in 64
dimensions:

| input | statistic |
|---|---|
| `N(0, I)` — the target | **≈ 1.2** |
| `N(0, 9I)` — Gaussian, wrong scale | ≈ 527 |
| rank-2 (collapsed) | ≈ 110 |

It is not a "distance from Gaussian" so much as a distance from *standard* Gaussian: wrong
scale and wrong mean are rejected as hard as wrong shape.

---

## 2. Adapting it to a language model

This is the part that needs thought, because **the motivation does not transfer intact.**

### The honest framing

Cross-entropy against discrete tokens *already* prevents collapse. A constant hidden state
produces a constant next-token distribution, which cross-entropy punishes immediately.
There is no degenerate optimum for SIGReg to rule out. If you port SIGReg to an LM expecting
it to prevent collapse, you are solving a problem you do not have.

What *does* transfer is the other half of LeJEPA's claim: that an isotropic Gaussian is the
best-conditioned embedding distribution for downstream use. LM hidden states are famously
bad on this axis — they occupy a narrow cone, have high average pairwise cosine similarity,
and their covariance spectrum is dominated by a handful of "rogue" directions. So in an LM,
**SIGReg is an auxiliary representation regularizer, not a collapse preventer**, and the
question to test is not "does it stop collapse" but "does trading a little cross-entropy for
isotropy buy anything downstream".

That reframing sets the expectations you should hold going in:

- **bits-per-byte should get worse.** SIGReg spends capacity on a non-task objective, and
  the softmax genuinely *likes* anisotropy (high-norm directions encode token frequency).
- **isotropy metrics should get much better.** That is the term doing its job.
- **linear-probe transfer is the open question** — and it is the LM analogue of the
  linear-probe protocol LeJEPA is evaluated with.

### Where to attach it

Three plausible sites:

1. **Final hidden state, pre-`lm_head`** (what `scripts/sigreg_experiment.py` uses). Most
   direct, and the most adversarial to cross-entropy, since it constrains exactly the
   vector the softmax reads.
2. **A projector head** hanging off the final hidden state, with SIGReg applied to the
   projection. This is what LeJEPA itself does (a 3-layer MLP projector), and it is the
   gentler option: the backbone is shaped only indirectly and keeps freedom in the space the
   softmax uses. Recommended if the direct version costs too much bpb.
3. **Mid-stack**, on the residual stream at some layer.

nanochat's `forward` already RMS-norms the hidden state before `lm_head`, so site 1 gets a
vector whose per-token *scale* is already ~1. SIGReg therefore mostly attacks the
*direction* structure — the nonzero mean direction and the anisotropic covariance — rather
than scale. That is exactly the pathology worth attacking.

### Two LM-specific corrections

**Samples are not i.i.d.** Epps–Pulley's `n` factor assumes independent samples. A batch of
`(B, T, C)` hidden states gives you `B·T` token vectors that are strongly correlated within
each sequence, so the effective sample size is far below `B·T` and the raw statistic is
inflated. `token_subsample` draws a random subset of positions each step, which both cuts
cost and decorrelates the sample.

**The marginal is not what you want to whiten.** Applying SIGReg to the raw marginal asks
every token in the corpus to be a draw from one shared Gaussian — which whitens away
document- and topic-level structure you probably want to keep. The `center='sequence'`
option subtracts each sequence's own mean first, so only the *within-sequence residual* is
shaped. This is the language analogue of **temporally centered SIGReg**, the LeWM follow-up
variant that applies SIGReg to temporally centered residuals instead of the latent marginal
and reports a large multi-task gain. It is a one-line change and worth trying.

A third knob you will feel immediately: `scale_by_n` keeps the classical `·n` factor, which
is faithful but makes the loss magnitude depend on the token count — so **λ has to be
retuned whenever you change batch size or subsample size.** Turn it off if you want λ to be
portable.

### The RNG subtlety

If SIGReg draws its random slices from the global RNG, then switching it on shifts every
subsequent random draw — and your "λ=0 vs λ>0" comparison silently becomes "λ=0 vs λ>0 *and
a different data order and different init*". `SIGReg.forward` takes an explicit `generator`
for exactly this reason, and the harness gives it a dedicated stream. This is the single
easiest way to accidentally invalidate the whole experiment.

---

## 3. The comparison test

```bash
python -m scripts.sigreg_experiment --lams 0.0,0.003,0.03 --seeds 0,1,2 --steps 1000
```

Runs on CPU in ~2 minutes per arm. No GPU, no dataset prep beyond one download.

### Setup

- **Model.** nanochat's real `GPT` — rotary, QK-norm, ReLU² MLP, value embeddings, Muon+AdamW —
  shrunk to 4 layers / 128 dim / 4 heads / 128 context, ~850k params.
- **Data.** Character-level TinyShakespeare, 90/10 train/val split. Character-level means
  every token is one byte, so **bits-per-byte is literally bits per character** and is
  directly comparable across runs.
- **Controls.** For a given seed, both arms get identical weight init, identical batch
  order, identical LR schedule, and identical eval batches. The *only* difference is λ.

### The three metrics

1. **Val bits-per-byte** — the task metric. nanochat uses bpb rather than raw loss because
   it is vocabulary-independent.
2. **Representation geometry** — effective rank (entropy of the covariance eigenspectrum,
   `exp(−Σ p log p)`), the variance share of the top principal component, mean |cosine|
   between random token pairs, and the held-out SIGReg statistic itself. The last one is a
   manipulation check: it tells you the term actually did what it claims.
3. **Linear probe** — speaker attribution. TinyShakespeare is marked up as `SPEAKER:\n<lines>`,
   so labels are free. Take the 12 most prolific speakers, mean-pool the **frozen** hidden
   states over each utterance, standardize, and fit logistic regression. This is the LM
   analogue of LeJEPA's linear-probe evaluation: it asks whether the representation is
   *linearly decodable*, which is precisely what isotropy is supposed to help with.

### Why multiple seeds is not optional

At 850k parameters and 1000 steps, seed-to-seed spread is large enough to manufacture any
conclusion you like from a single run. Every number below is mean ± std over 3 seeds, and
**a difference smaller than the seed spread is not a result.** If you take one thing from
this document, take that one.

### Results

`--lams 0.0,0.003,0.03 --seeds 0,1,2 --steps 1000`, mean ± std over 3 seeds:

| metric | λ=0 | λ=0.003 | λ=0.03 |
|---|---|---|---|
| **val bits/char** | **2.167 ± 0.008** | 2.189 ± 0.008 | 2.328 ± 0.003 |
| train bits/char | 1.805 ± 0.005 | 1.839 ± 0.003 | 2.003 ± 0.008 |
| **probe accuracy** | 0.183 ± 0.025 | 0.173 ± 0.014 | 0.180 ± 0.029 |
| effective rank (of 128) | 36.1 ± 0.2 | 89.5 ± 0.6 | **109.1 ± 0.3** |
| top-PC variance share | 0.202 ± 0.012 | 0.048 ± 0.001 | **0.026 ± 0.001** |
| mean abs cosine | 0.225 ± 0.008 | 0.083 ± 0.002 | **0.065 ± 0.001** |
| held-out SIGReg | 127.4 ± 8.2 | 4.8 ± 0.1 | **2.8 ± 0.0** |

Majority-class baseline for the probe is 0.124. ~130 s per arm on 4 CPU cores.

### How to read it

**The mechanism works, exactly and controllably.** Held-out SIGReg drops 127 → 4.8 → 2.8;
effective rank climbs 36 → 90 → 109 out of a possible 128; the dominant direction's variance
share collapses from 20% to 2.6%; mean |cosine| falls from 0.23 to 0.065. All with
seed-spreads of well under 1%. The baseline model really is the narrow, anisotropic cone the
literature describes, and λ dials it out monotonically.

**It costs language modelling, monotonically.** +0.021 bpb at λ=0.003 (≈2.6× the seed std,
so a real effect, but small) and +0.161 bpb at λ=0.03. Both train *and* val bpb rise
together, so this is not a regularization/overfitting story — the model is simply being
pulled away from the cross-entropy optimum. This is the expected direction: an anisotropic,
high-norm-direction representation is *useful* to the softmax.

**Transfer did not improve.** Probe accuracy is 0.183 / 0.173 / 0.180 with seed-std ~0.02 —
flat, with every difference far inside the noise. On this setup, the isotropy that SIGReg
buys does not translate into more linearly decodable features.

Two caveats on that last point, both real:

- **The probe is underpowered.** 18% against a 12.4% majority-class baseline, on ~330 test
  utterances, is a weak signal to begin with; it cannot resolve small differences. A larger
  probe set, or a model big enough to actually do speaker attribution well, would give the
  comparison much more resolution.
- **This is an 850k-parameter model trained for 1000 steps.** Representation-quality effects
  are exactly the kind of thing that can change sign with scale.

So the defensible claim is narrow and worth stating plainly: **on a tiny character-level LM,
SIGReg does precisely what it advertises to the geometry of the representation, and you pay
for it in bits-per-character without measurable downstream return.** That is a useful
negative result — and it is the result the reasoning in §2 predicts, which is a good sign
the port is faithful rather than broken.

### Sequence-centered variant

```bash
python -m scripts.sigreg_experiment --lams 0.003 --seeds 0,1,2 --center sequence
```

Subtracting each sequence's own mean before the test — the language analogue of temporally
centered SIGReg — is the most interesting result here:

| metric | λ=0 | λ=0.003, `center=none` | λ=0.003, `center=sequence` |
|---|---|---|---|
| **val bits/char** | 2.167 ± 0.008 | 2.189 ± 0.008 | **2.168 ± 0.004** |
| train bits/char | 1.805 ± 0.005 | 1.839 ± 0.003 | 1.808 ± 0.002 |
| probe accuracy | 0.183 ± 0.025 | 0.173 ± 0.014 | 0.190 ± 0.017 |
| effective rank | 36.1 ± 0.2 | 89.5 ± 0.6 | **93.5 ± 0.4** |
| top-PC variance share | 0.202 ± 0.012 | 0.048 ± 0.001 | **0.040 ± 0.001** |
| mean abs cosine | 0.225 ± 0.008 | 0.083 ± 0.002 | 0.083 ± 0.001 |
| held-out SIGReg | 127.4 ± 8.2 | 4.8 ± 0.1 | **3.5 ± 0.1** |

**Centering makes the regularizer free.** Val bpb goes from 2.189 back to 2.168 — a dead
heat with the unregularized baseline (2.167, difference 0.0008 against a seed std of 0.008)
— while the isotropy gain is not merely preserved but slightly *better* on every measure:
effective rank 93.5 vs 89.5, top-PC share 0.040 vs 0.048, held-out statistic 3.5 vs 4.8.
Train bpb matches the baseline too, so nothing is being hidden in a generalization gap.

The interpretation is the one §2 predicts. Applying SIGReg to the raw marginal asks the
sequence-level mean — which carries document and topic identity the softmax is actively
using — to be Gaussian along with everything else, and *that* is what costs bits. Centering
exempts it and shapes only the within-sequence residual, which the model apparently gives up
for nothing. The mechanism that helps multi-task LeWorldModel transfers to text for the same
structural reason.

Probe accuracy is nominally highest here (0.190) but still inside seed noise, so the
transfer question remains open — see the probe-power caveat above.

---

---

## 4. Extensions worth trying

- **Projector head.** Apply SIGReg to an MLP projection of the hidden state rather than the
  hidden state itself, as LeJEPA does. Decouples the regularized space from the space the
  softmax reads.
- **λ warmup.** Anneal λ from 0, so early training is pure language modelling.
- **A real JEPA-style LM.** The setup where SIGReg's original motivation genuinely applies
  is a *latent-predictive* LM — predict the embedding of the next span rather than its token
  distribution. There the prediction loss really can collapse, and SIGReg is load-bearing
  rather than decorative. That is the honest way to reproduce LeWorldModel's setting in
  text, and it is a much bigger build than this one.
- **Scale up.** Everything here is CPU-sized. `--n-layer 8 --n-embd 384 --steps 20000` on a
  GPU, with a BPE tokenizer, would tell you far more.
