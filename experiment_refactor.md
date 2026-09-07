# The Experiment Refactor

*The memory of the `experiment_refactor` branch (PR #800), July to September 2026. It
retires at merge: a dev/LOG.md entry and the PR body carry the story from then on. The
design detail that used to live here (the shard format, the SFT recipe, the deletion
table, the A/Bs) is in this file's git history up to a514c54.*

## Motivation

nanochat is repositioning from "train your own ChatGPT clone for $100" (a tinkerer
audience) to **the reference pretraining harness for small LLMs**: the simplest solid
baseline that a researcher with a compute allocation forks. Spiritual guiding light:
SQLite. Small, boring dependencies, obsessively verified, a reference people fork rather
than a framework people configure. Code is cheap now, anyone can ask an LLM for a
feature, so the repo only has to contain the hard essence, and it should fit in one
context window (the code is about 90K tokens at 3 chars per token).

**The product is the miniseries curve.** `--depth` is the single dial; sweeping it traces
a ladder of compute-optimal models, and `curve.log` is the deliverable: quality, speed
and cost of every model in one file, comparable across experiments by compute
multiplier. "Chat" is the payoff at the end of the pipeline (the way `sample.py` is the
payoff in nanoGPT), not a second research surface: at these scales SFT does all the work
that matters, RL is an appendage, and nobody was training their own chatbots anyway.
Speedrunning one model is a different thing from sweeping a ladder, so the GPT-2
leaderboard is gone too.

What the branch did, in the order it landed:

- **An experiment is `(name, git commit, depth ladder, dataset)`.** `run.sh <name>` runs
  one end to end; every stage is idempotent (done = a `summary` record in its log) and a
  running ladder is steered by touching files. Experiments live in
  `$NANOCHAT_BASE_DIR/experiments/<name>/`, datasets by name in `datasets/<name>/`,
  `meta.json` records commit, dirty diff and dataset. There is no config file: the code
  is the config.
- **The logs are the source of truth.** Stage scripts print to stdout, `run.sh` tees;
  lines of the form `<tag> key=value ...` are the machine contract, everything else is
  prose. `harness/experiment.py` is the one CLI for experiment things: the grammar,
  `init`, `curve` (summaries -> `curve.log`), `compare` (compute multipliers at matched
  quality). Nothing on the CLI path imports matplotlib.
- **All data work happens once, up front.** `base_prepare` downloads the raw parquet
  shards, trains the tokenizer (it lives with the dataset), and packs BOS-aligned rows
  into self-describing `.bin` shards (256-byte header, uint16 inputs and targets, 65535
  = ignore); `chat_prepare` renders the SFT conversations into the same format. The
  loader is a memmap with a one-integer resume state. Training scripts never touch text.
- **SFT is base_train continued on conversations, nothing more.** SmolTalk plus MMLU's
  auxiliary_train as the multiple-choice format primer; ChatCORE is the three
  categorical tasks. RL, tool use, the sandbox, GSM8K, HumanEval, fp16, report.py and
  the generative evals were deleted; each deletion was measured or had no caller.
- **The model went functional** (`init_params` + `forward`, a thin `GPT` shell), the
  optimizer flat-tape; smear gate deleted by ablation; muP LR width scaling adopted.
- **A layout a reader can hold.** `nanochat/` is what a model is and how it trains and
  never imports the rest (a test enforces it); `harness/` is the experiment around it
  (experiment, checkpoint, dataset = raw text, tasks = raw conversations, runtime);
  `evals/` measures; `scripts/` is the grid `{base,chat}_{prepare,train,eval,inference}`,
  and script = stage = log = curve prefix, with `base/` and `chat/` the two checkpoints
  per depth. `base_eval` scores CORE (the full 22-task eval; its sampling noise is 0.006
  regardless of the per-task cap), `base_inference` makes inference an eval.
- **Verified as a twin, then tightened.** v2rc1 (Sep 2, the full ladder on the new
  pipeline) matches July's v1rc1 rung for rung: same minutes and MFU, val bpb a constant
  +0.004 (the val packing plus 4 BPE merges), CORE and chat within noise, batch-1
  inference 5-10% faster. It is the baseline every later experiment compares against.
  The repo is ~5,600 lines of code and ~650 of tests, from ~9,750 in July.

This is a breaking change and that is fine: old checkpoints do not load, the on-disk
layout changed, scripts were deleted. Tag master before merging so the old "$100
chatbot" stays pinnable.

Constraints to preserve: single dial; CLI-native and stdout-first; minimal deps (torch is
the libc, everything else fights for its life); readable in one sitting; no
general-purpose machinery for situations this codebase does not create; one thing per
line, name repeated predicates, comments explain constraints not narration.

## TODOs

What is still needed to ship the merge, in order:

1. Andrej: a manual pass over README.md (the voice is Claude's; judgment calls: the
   example conversation, the citation title).
2. A fresh-clone check. pyproject fails the audit: `jinja2` is imported but not declared
   (transitive only), `psutil` and `python-dotenv` declared but never imported. Fix,
   `uv lock`, clone to a temp dir, `uv sync --extra gpu --group dev`, run the tests.
3. The migration paragraph: this is a breaking change. One paragraph in the README quick
   start and in the PR body saying so and what to do (old download -> one `mv`, the
   dataset banner already says it; old checkpoints are simply not found).
4. The dev/LOG.md entry summarizing the whole refactor (phases, A/Bs, deletions); retire
   this doc into it and the PR body.
5. Tag master before merging, so the old "$100 chatbot" stays pinnable.
6. Push the branch (one squashed commit ahead of origin) and rewrite PR #800: it still describes
   July. Above all the deletions people will hit: RL, tool use, fp16, report.py, the
   leaderboard, the tokenizer change, the layout.
