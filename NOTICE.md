# Clarinet

Clarinet is a research project by Vemund Rundberget that imports the econometric notion of **instrumental variables** (IV) into the auto-regressive token generation of an LLM. The core idea: use the data source (scientific/formal-reasoning corpora vs. general web text) as an instrument to isolate the reasoning-induced component of next-token transitions.

## Upstream

Clarinet is a fork of [`nanochat`](https://github.com/karpathy/nanochat) by Andrej Karpathy. The original `nanochat/` package is retained largely unchanged so that upstream improvements can be rebased cleanly. Clarinet's contributions live in the sibling `clarinet/` package and a handful of new scripts under `scripts/` and `runs/`.

Both copyrights are preserved in `LICENSE` per the terms of the MIT License.
