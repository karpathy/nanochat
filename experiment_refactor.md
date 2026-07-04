# The Experiment Refactor

*Design doc, July 2026. Context and proposal for making "an experiment" the first-class
unit of nanochat. This is a large refactor that will touch most of the code; this doc
exists so the design can be considered carefully before any code is written, and so that
anyone (human or LLM) can pick up the work with full context.*

## Where nanochat is going (context)

nanochat is repositioning from "train your own ChatGPT clone for $100" (tinkerer
audience) to **the simplest solid baseline / forking point for end-to-end academic LLM
research** (researcher audience with real compute allocations). Spiritual guiding light:
SQLite — small, boring dependencies, obsessively verified, a reference implementation
people fork rather than a framework people configure.

Recent work in this direction (already landed):
- Deleted the web UI (fastapi/uvicorn), identity conversations, spellingbee, the
  HuggingFace tokenizer/model comparison paths, and the `datasets` library (replaced
  with ~75 lines of direct parquet-over-HTTP + pyarrow in `tasks/common.py`).
  Repo went from ~9.8K to ~7.5K lines; uv.lock from ~3.4K to ~2.5K lines.
- Unified the single-GPU and distributed optimizers into one `MuonAdamW` (the
  distributed code degenerates cleanly at world_size=1).
- Simplified `nanochat/execution.py` (350 -> 134 lines, subprocess-based sandbox).
- Added unit tests (tokenizer round-trips, conversation masks, task slicing,
  sandbox properties, optimizer vs torch reference): `python -m pytest tests/`.
- Added `scripts/infer_bench.py`: inference as an *eval* (TTFT, TPOT, tok/s, MBU, MFU,
  VRAM, batch-size sweep). Established the **JSON-tail stdout contract**: human-readable
  tables, and the last line of stdout is one compact JSON document for scripts.
  Also added `GPT.num_matmul_params/estimate_decode_flops/estimate_prefill_flops/`
  `kv_bytes_per_token/kv_read_bytes` — cost accounting lives ON the model; scripts
  never reach into module internals.

**The product is not a single model.** It is a *miniseries*: reliably producing a ladder
of compute-optimal models across a cost-performance curve, with `--depth` as the single
dial that determines everything else. "Time-to-GPT-2" (the leaderboard) is one famous
point on that curve, not the product itself.

## The problem

The repo currently has no notion of an experiment. `runs/miniseries.sh` and
`runs/speedrun.sh` compensate with duct tape:

1. **Log scraping.** miniseries.sh extracts params/bpb/CORE by grepping pretty stdout
   (`grep "Validation bpb:" | grep -oP '[\d.]+$'`). Any cosmetic print change silently
   corrupts the results CSV (there is already a fallback-to-"0.0" hack for CORE).
2. **Checkpoints are divorced from experiments.** (Correction of an earlier claim:
   miniseries.sh does keep final checkpoints — `--save-every=-1` means "save at end
   only" and the last step always saves.) But they land in a global
   `base_checkpoints/<tag>/` pool shared by all runs, where tags collide across
   experiments and nothing records which code produced which checkpoint.
3. **No provenance.** No commit hash, no environment, no record of what code produced a
   result. `~/.cache/nanochat/` is a graveyard of `jan5_/jan7_/.../mar17_miniseries_results`
   dirs whose provenance is unrecoverable. (Related: every checkpoint currently on disk
   predates the current architecture and fails to load — silent drift.)
4. **Hyperparameters leaking into bash.** The device-batch-size if-ladder in
   miniseries.sh is a function-of-depth living outside `base_train.py`, violating the
   single-dial rule.
5. **speedrun.sh conflates provisioning with experimentation.** Installing uv / creating
   the venv and running a training pipeline have different lifecycles.
6. **No resume.** A crash at d20 means re-training d12-d18.

## The proposal

### An experiment is (name, git commit, depth ladder, dataset)

Every hyperparameter is derived from `--depth` by design, so there is no config file and
never should be: **the code is the config.** To test an idea: edit the code, name an
experiment, run the ladder. Reproducing an experiment means checking out its commit.

The dataset is a **named, immutable input** that the experiment references (recorded in
meta.json), never contains. Datasets live in a shared store and are trivially reused by
name; the default is the canonical one (climbmix), which the leaderboard and golden
numbers pin. The data contract is deliberately narrow: *a dataset is a directory of
parquet text shards, last shard = validation*. `nanochat.dataset` is the materializer
for the canonical dataset; a data researcher can drop in their own directory satisfying
the same contract and name it in their experiment — data experimentation with everything
downstream held fixed.

Comparability note: when the dataset varies, `val_bpb` (measured on the dataset's own
val shard) is no longer comparable across experiments, but CORE / ChatCORE / the
inference bench are external to the training data and remain a shared yardstick. This
is exactly the affordance data research needs.

### What happens in a single experiment

```
create experiment <name>
  -> record meta.json: name, git commit (+ dirty flag), date, GPU/node info, ladder
1. materialize/verify the experiment's dataset by name (shared store; default: climbmix)
2. train tokenizer                          (fast; per-experiment for now, see open questions)
3. for each depth d in ladder:
     train base model                       -> d{d}/base checkpoint + base_train.log
     eval base model (CORE, bpb, samples)   -> d{d}/base_eval.log
     inference bench                        -> d{d}/infer_bench.log
     SFT                                    -> d{d}/sft checkpoint + sft.log
     chat eval (ChatCORE)                   -> d{d}/chat_eval.log
4. aggregate                                -> curve.log  (one row per depth: the product)
```

### Directory layout

```
$NANOCHAT_BASE_DIR/            # shared immutable caches ONLY
  datasets/<dataset_name>/     # named datasets (parquet shards, last = val); default: climbmix
  eval_bundle/                 # CORE eval data
  experiments/<name>/          # everything one experiment produces (RESOLVED: lives in
                               # the base dir, not the repo, to keep the source tree lean)
  meta.json                    # commit, dirty, date, env, ladder, dataset name
  tokenizer/
  d12/
    base checkpoint(s), base_train.log, base_eval.log, infer_bench.log,
    sft checkpoint(s), sft.log, chat_eval.log
  d16/ ...
  curve.log
```

### The stage contract

Every stage script (`base_train`, `base_eval`, `infer_bench`, `chat_sft`, `chat_eval`):

1. **Reads** what it needs from the experiment dir (and shared caches).
2. **Prints** to stdout only (records + prose); the runner (runs/run.sh) tees
   stdout+stderr into the stage's .log in the experiment dir. Scripts contain no
   logging machinery; the runner owns record-keeping. (Consequence: standalone ad-hoc
   invocations are not recorded — correct, they are not experiment events.) The log is
   the single record: human-first plain text, but lines that carry data follow a
   stable key=value grammar (see below, and nanochat/logfmt.py). No JSON sidecars; no
   wandb dependency — wandb stays optional for live monitoring; logs are the source of
   truth. Runner details that make tee correct: `set -o pipefail` (a stage failure
   must not be masked by tee's exit code) and PYTHONUNBUFFERED=1 (keep console live).
3. **Is idempotent**: if its output already exists, it skips (make-like semantics).
   This gives crash-resume for free: rerun the same command, completed stages skip.
4. Honest precision: round measured floats to what the measurement supports.

### The log grammar (logfmt-style): one channel for humans AND machines

Design tension: JSON summary files are machine-friendly but (a) write-once at the end —
a crash loses everything, (b) not really for human eyes, (c) a second channel beside the
log that can drift from it. Plain logs are appendable and pleasant but grep-scraping
prose is brittle (the current miniseries.sh failure). Resolution: **the log is the only
artifact, and the machine contract is a line grammar, not the prose.**

Convention (logfmt-style): lines that begin with a record tag consist of
space-separated `key=value` pairs. Values with spaces are quoted (parse with shlex).
Everything else in the log is free-form decoration with no stability promise.

```
step step=4990 loss=2.311 lrm=0.45 dt_ms=124.3 tok_s=4212000 mfu=40.2
bench batch=32 ttft=0.0103 tpot=0.0061 tok_s=5247.0 mbu=9.1 mfu=0.14 vram_bytes=3583744000
summary params=286261730 val_bpb=0.8213 core=0.2231 train_time_sec=3721 gpu="NVIDIA H100 80GB HBM3"
```

Properties: appendable (a crash at step 4000 leaves 4000 durable step records — the
"summary" is just more appended lines); greppable/awkable directly; one ~10-line stdlib
parser (scan lines, shlex.split, split on `=`, infer int/float) is the only schema
machinery in the repo. The domain is genuinely flat (steps, evals, bench rows,
summaries), so the no-nesting limitation costs nothing and keeps the aggregator trivial.

Notes: infer_bench.py emits `prefill`/`bench`/`summary` records in this grammar (its
earlier JSON-tail output was converted). meta.json stays JSON (tiny, write-once,
stdlib) — appendability is irrelevant there. YAML was considered and
rejected: write-once like JSON, parsing footguns, and an extra dependency.

With this contract the orchestrator becomes a trivially readable bash loop (no parsing,
no state), and aggregation is a small Python join over `summary` records into curve.log.
`base_train.py` already computes every value the current grep-scraping extracts; the
change is emitting them as one `summary` line instead of prose.

### curve.log (the product)

One row per depth. Roughly:
`depth, model_dim, num_params, num_matmul_params, num_iterations, tokens_trained,
param_data_ratio, train_time, val_bpb, core, ttft, tpot_bs1, tok_s_bs{1,8,32,128},
mbu, mfu_prefill, peak_vram, kv_bytes_per_token, chatcore, sft_time, ...`
(Exact columns TBD during implementation; everything sourced from `summary` records in the stage logs.)

## Decisions made (with reasoning)

- **Dataset is a named experiment input, canonical by default.** Datasets are immutable
  named directories of parquet shards in the shared store
  (`$NANOCHAT_BASE_DIR/datasets/<name>/`); experiments reference one by name in
  meta.json (default: climbmix). The leaderboard and golden numbers pin the canonical
  dataset, preserving comparability; naming a different dataset opens data research with
  everything downstream held fixed. (This revises an earlier "dataset is fixed, out of
  scope" decision — the reference-by-name model gets both properties.) An optional
  symlink `experiments/<name>/dataset -> <store>` may be added as navigation sugar, but
  the name in meta.json is the identity. Consequence: the tokenizer is trained on the
  experiment's dataset, so per-experiment tokenizers are required, not just convenient.
- **SFT all depths, not just the largest.** SFT costs minutes vs pretraining hours;
  ChatCORE-vs-depth extends the curve into post-training and answers questions like
  "at what scale does instruction-following click." A single SFT'd model collapses the
  curve back to a point.
- **RL is out of the canonical experiment.** SFT does the overwhelming share of "makes
  it a chat model" at these scales; RL is an incremental gain on rewardable tasks; there
  is no RLHF setup or environment suite, so it cannot be reference-grade.
  `scripts/chat_rl.py` remains in the repo as a documented optional appendix
  (good pedagogy: rollouts, masking, rewards) but does not feed curve.log.
- **speedrun and miniseries become the same runner.** speedrun = the famous named
  experiment with ladder (24,) (the leaderboard entry); miniseries = full ladder.
  One code path, so the leaderboard run exercises exactly the machinery researchers use.
- **Provisioning splits out** (uv install, venv creation -> setup script or README).
- **No config objects, no YAML, no experiment "framework".** The experiment machinery
  should itself pass the repo's minimalism bar.

## Open questions (to resolve before/while implementing)

1. **Where does `experiments/` live?** RESOLVED: `$NANOCHAT_BASE_DIR/experiments/<name>`
   (overridable via NANOCHAT_EXPERIMENTS_DIR). Repo-local was tried first and rejected:
   large artifact trees in the source dir slow down editors/tooling.
2. **Tokenizer identity.** Per-experiment (simple, costs ~minutes + small disk) vs
   shared/content-addressed (dedup across experiments that don't touch tokenizer code).
   Start per-experiment, optimize later?
3. **Checkpoint retention.** Full ladder + SFT for d12..d26 is tens of GB per experiment.
   Keep everything? Keep final steps only? A `nanochat experiments gc` story?
4. **Orchestrator language.** Thin bash (readable, pedagogical, matches runs/ tradition)
   vs Python (better error handling). Current lean: bash for sequencing, Python only for
   the aggregator. The stage contract makes this choice low-stakes.
5. **Dirty-tree policy.** Refuse to run? Warn and record `git diff` into the experiment
   dir? (Recording the diff makes even dirty runs reproducible — attractive.)
6. **Seeds / noise floor.** Run-to-run variance matters for research claims (data-order
   shuffling alone moves CORE ~0.016 per LEADERBOARD.md). Should an experiment support
   N seeds per depth, with curve.log reporting mean +/- std? Probably a later phase, but
   the layout should not preclude it (e.g. `d12-seed1/`).
7. **wandb naming.** If used, runs should be named `<experiment>_d<depth>` automatically.
8. **Existing flags.** `--model-tag`, `--run`, `--save-every` etc. partially overlap with
   experiment semantics; decide what subsumes what. Also move the device-batch-size
   derivation into base_train (function of depth + VRAM).
9. **Audit chat_sft hyperparameters** — RESOLVED: chat_sft inherits max_seq_len, batch
   sizes and LRs from the pretrained checkpoint's meta, so they derive from depth
   transitively. (Same audit eventually for RL if it ever rejoins.)
10. **Dataset integrity.** "Named and immutable" is a convention, not yet a guarantee.
    Cheap first steps: dataset.py validates shard sizes (not just existence) and prints
    one quiet summary line when cached; meta.json records shard count + total bytes as a
    weak fingerprint. Real checksums require the hosting side to publish a manifest.
11. **Midtraining slot.** A context-extension stage (YaRN) is planned for the future and
    would slot between base and SFT in the per-depth pipeline. Design the stage list to
    be extensible in the obvious way (it already is, if stages are just scripts + JSONs).

## Suggested implementation order

STATUS (jul 4, branch experiment_refactor): phases 1-5 are implemented. runs/run.sh
runs the full pipeline (dataset -> tokenizer -> per-depth: base_train, infer_bench,
sft, chat_eval -> curve.log); speedrun.sh/miniseries.sh are deleted (the leaderboard
speedrun is: DEPTHS="24" BASE_TRAIN_FLAGS="--target-param-data-ratio=8 --fp8"
bash runs/run.sh speedrun). Remaining: open questions above, runcpu.sh/scaling_laws.sh
not yet ported to the experiment world, full README repositioning.

Each phase lands independently and is useful on its own:

1. **The log grammar.** Add the logfmt-style parser helper (one function) and emit
   `summary` (and where natural, `step`/`bench`) records from base_train / base_eval /
   chat_sft / chat_eval / infer_bench. Immediately kills prose-scraping even before the
   experiment concept exists.
2. **Experiment identity.** The experiment dir convention + meta.json (commit, dirty,
   env, ladder). Stage scripts learn one new thing: an experiment name/dir to write
   their artifacts + JSONs into (env var e.g. `NANOCHAT_EXPERIMENT`, or a flag).
   Checkpoint manager reads/writes within the experiment dir.
3. **Idempotence.** Stages skip when their output exists. Crash-resume falls out.
4. **The runner + aggregator.** Rewrite runs/ as: `setup.sh` (provisioning),
   one experiment runner (name + ladder -> full pipeline), aggregator -> curve.log.
   speedrun.sh and miniseries.sh become thin invocations of the runner.
5. **Docs.** README reframes around experiments; leaderboard section points at the
   runner with ladder (24,).

## Constraints to preserve (repo values)

- Single dial: everything derives from `--depth`; a change must hold across the ladder.
- CLI-native, stdout-first; wandb optional; no report/dashboard machinery.
- Minimal deps (torch is the "libc"; everything else fights for its life).
- Code readable in one sitting (~85K tokens total budget — fits an LLM context window).
- No config objects, model factories, or if-then-else architecture zoos.
- Style: one thing per line; comments explain constraints, not narration.
- Any new machinery must itself be depth-parameterized and sweep-tested at >=3 depths.

## Related threads (not this refactor, but adjacent)

- **Architecture simplification debate**: exotic features (value embeddings, smear gate,
  backout) exist for the speedrun but dilute the clean-baseline story and distort
  param counts (at d12, VE tables are ~151M params vs ~110M matmul params). Planned
  approach: define an admission bar — a feature stays only if its benefit holds across
  the miniseries — and settle by ablation at 3 depths. The experiment machinery in this
  doc is exactly the tool for running that ablation.
- **Golden numbers / noise floor**: once experiments are cheap and reproducible, publish
  expected curve.log values (mean +/- std) per depth so forks can verify themselves.
- **Quantization + speculative decoding**: future inference-side stages; speculative
  decoding is natural here because the miniseries provides draft models for free.
