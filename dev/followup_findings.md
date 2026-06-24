## Follow-up: a simpler d22 recipe (alpha stack, default matrix-lr) modestly beats Run 7 on CORE and clearly on loss + wall-clock

Since Run 7 we asked: is the MuonClip + long-warmdown machinery necessary, or can a plainer recipe match it? We tested a plainer recipe — the alpha stack (z-loss 1e-4, multi-hash bigram, muon-plus Frobenius renorm, value-residual) at **default matrix-lr 0.020**, total-batch-size **786 K**, **no QK-clip / no warmdown tuning** — head-to-head against Run 7 **on an identical tokenizer and eval protocol**, multi-seeded.

### Method (controls that matter)

CORE is single-shot-noisy (σ ≈ 0.004–0.005) and, we found, **tokenizer-sensitive**: the same recipe scores ~0.01–0.018 higher CORE on a tokenizer trained over the wrong shard scope, with **identical val_bpb**. So both recipes were run with a **fresh 8-shard tokenizer** (matching `runs/speedrun.sh`) and CORE measured at **full eval** (`--core-metric-max-per-task=-1`, the `base_eval` convention) — not the limited (max-per-task=500) base_train default, which inflates CORE ~0.01.

### Results (fresh 8-shard tok, full eval, identical harness)

| recipe | CORE (full) | val_bpb (eval) | time | seeds |
|---|---:|---:|---:|---|
| **this follow-up** (alpha stack, default mlr, TBS 786K) | **0.2539** | **0.7196** | **85.6 min** | N=5 |
| Run 7 (MuonClip + warmdown=0.85 + final-lr=0) | 0.2462 | 0.7206 | 88.4 min | N=4 |

- **CORE: +0.0077** (Welch t ≈ 3.0, p ≈ 0.01; 95% CIs non-overlapping: [0.2499, 0.2579] vs [0.2432, 0.2492]).
- **val_bpb: −0.0010** (tokenizer-invariant, near-deterministic axis).
- **wall-clock: −2.8 min** (TBS 786K → ~7,733 Muon steps vs Run 7's ~6,000 at TBS 1M but more total tokens; net faster).
- **best single run:** CORE 0.2584.

The driver is the alpha stack (multi-hash bigram + value-residual + z-loss), which Run 7 lacks; the MuonClip/warmdown tuning Run 7 adds is then unnecessary. Simpler recipe (matrix-lr at default, no QK-clip flag), and the stack is depth-portable (verified to transfer d22→d24→d26).

### Honest caveats

- **The edge is modest, not dominant.** An earlier internal measurement on a cached tokenizer showed +0.013; controlling the tokenizer corrected it to **+0.008**. We report the controlled number.
- **Absolute GPT-2 clearance is setup-sensitive.** In our harness both recipes land ~0.005–0.01 below their official-`speedrun.sh` absolutes (val_bpb also ~0.001 off), so we make the **relative** claim (this recipe > Run 7, same harness) rather than re-asserting absolute GPT-2 thresholds.

### Axes tried that did NOT improve it (each multi-seeded)

| lever | result | verdict |
|---|---|---|
| batch-size warmup (GLM-4.5: ramp grad-accum 1→full) | N=3 mean ≈ baseline, val worse | neutral (single 0.2691 draw was +1.5σ noise) |
| total-batch-size 1 M + warmup | CORE 0.245 | too aggressive — speed cut overshoots |
| matrix-lr 0.025 / 0.022 @ 786 K | no CORE gain, val worse | rejected |
| GLM-5.2 survey (IndexShare, DSA, MTP, Muon-split, World-Knowledge reweight) | inference-only / too-slow / infra-blocked | none iso-wall-clock-viable |

**Takeaway:** at d22/≤87 min the CORE axis is noise- and tokenizer-limited; a simpler default-LR recipe modestly but significantly beats Run 7's tuned recipe on CORE while clearly winning loss and wall-clock. Run 7's single new flag (`--muon-qk-clip-tau`) remains available but isn't required here.

### Position vs the current leader (Muon+/eq-row + bigram, 87 min)

This recipe is a **superset** of the current leader's (same `--muon-plus --muon-eq=row --bigram-embed-factor=5` base, plus TBS=786K, z-loss, multi-hash bigram, scalar-lr). Comparing the two is only clean on **wall-clock**, which is setup-invariant: **85.6 min vs the leader's 86.98 min** (TBS786 → fewer Muon steps). On **CORE** and **val_bpb** the only available comparison crosses harnesses (our measurements run non-uniformly below the leader's official absolutes — lower CORE *and* lower val), so this recipe is *likely* ahead on CORE (~+0.001–0.006 after a setup-gap correction) and ~tied on val, but **neither is proven without running the leader's recipe in this harness** (not done). The rigorously-proven claim here is the same-harness comparison vs Run 7 above.
