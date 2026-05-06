# Bigram Speedrun Verification Notes

This branch is based on upstream nanochat master at `dc54a1a` and keeps the
submission implementation focused on the winning recipe:

- per-layer hashed bigram residual embeddings
- Muon+ post-orthogonalization normalization
- row equilibration before Muon orthogonalization
- lower scalar LR (`--scalar-lr=0.3`)
- batched training logging (`--train-log-every=50`)
- `torch.compile(..., mode="max-autotune-no-cudagraphs")` for the speedrun script

It intentionally excludes the experimental branches that were not part of the
final candidate: sparse layers, MoE/TOP losses, train-time logit bias losses,
post-hoc fitting, NorMuon, and checkpoint merging.

## Reproduction Sanity Check

Minimal branch d4/20 matched the prior experimental branch:

| Run | Step 0 BPB | Step 10 BPB | Final BPB |
| --- | ---: | ---: | ---: |
| Prior candidate branch | `3.237224` | `3.234722` | `3.223259` |
| Minimal PR branch | `3.237224` | `3.234722` | `3.223286` |

The final difference is `0.000027` BPB on a tiny run, consistent with small
compile/graph differences after removing unused experimental code.

## Full d16 Verification

Both runs used d16, FP8, target param/data ratio 8, total batch `524288`, and
device batch `32` on the same machine.

| Run | Final BPB | Train time | Avg logged tok/s, excluding first | Avg logged step time, excluding first |
| --- | ---: | ---: | ---: | ---: |
| Upstream master dense | `0.800673` | `94.64m` | `329,904` | `1589.232ms` |
| Bigram/Muon+ candidate | `0.798000` | `93.61m` | `333,507` | `1572.058ms` |

Candidate delta versus upstream master dense:

- BPB: `-0.002673`
- train time: `-1.03m` (`1.09%` faster)
- logged throughput: `+3,603 tok/s` (`1.09%` higher)

Important caveat: this is a full recipe comparison, not an architecture-only
comparison. The candidate also uses `--train-log-every=50` and
`--compile-mode=max-autotune-no-cudagraphs`, while upstream master logs every
step and uses the default compile mode.

## Controlled d16 Throughput

A denser control run with the same log50/compile-control style is the better
way to estimate the per-step overhead of the bigram path.

| Run | Final BPB | Train time | Avg logged tok/s, excluding first | Avg logged step time, excluding first |
| --- | ---: | ---: | ---: | ---: |
| Dense log50 compile control | `0.800604` | `92.85m` | `336,247` | `1559.258ms` |
| Bigram/Muon+ candidate, full 3584 | `0.798000` | `93.61m` | `333,507` | `1572.058ms` |

Against this controlled dense run, the bigram candidate is about `0.81%` slower
per step, but `0.002604` BPB better at the same horizon.

A shortened bigram run at 3400 steps landed at `0.800232` BPB in `88.92m`,
which is `0.000372` BPB better than the dense log50 compile control while using
about `4.23%` less training time.

## Compile Mode Probe

Short d16/40 throughput probes on the minimal branch:

| Compile mode | Avg logged tok/s, excluding first | Avg logged step time, excluding first | Total time |
| --- | ---: | ---: | ---: |
| default `torch.compile` | `324,995` | `1613.250ms` | `0.78m` |
| `max-autotune-no-cudagraphs` | `333,261` | `1573.250ms` | `0.76m` |

On this d16 probe, `max-autotune-no-cudagraphs` was about `2.5%` faster than
the default compile mode. The speedrun script keeps this compile mode for that
reason.

## Test Status

- `python -m pytest tests/test_engine.py -q`: `9 passed`
- `python -m py_compile nanochat/gpt.py nanochat/optim.py scripts/base_train.py nanochat/engine.py`: passed
