"""
An experiment is (name, git commit, depth ladder, dataset). Everything about one
lives here: the log line grammar its stage scripts report in, creating it
(meta.json), reading it back (the stage logs written by run.sh), its product
(curve.log), and comparing two of them (compute multipliers). Three commands:

    python -m harness.experiment init        # create/resume $NANOCHAT_EXPERIMENT, write meta.json
    python -m harness.experiment curve       # stage summaries -> curve.log, print the headline table
    python -m harness.experiment compare <baseline> <variant>   # cm records: compute multipliers

run.sh calls the first two. Nothing here imports matplotlib: the records are the
product, pictures are a notebook away.

Because every hyperparameter derives from --depth and the dataset is referenced by
name, there is no config file: the code is the config, and meta.json records which
code (the commit). If the working tree is dirty, the diff is saved alongside so that
even dirty runs are reproducible.
"""

import os
import re
import sys
import json
import math
import shlex
import argparse
import subprocess
from datetime import datetime

from nanochat.common import get_experiment_name, get_experiment_dir
from nanochat.common import get_dataset_name

# -----------------------------------------------------------------------------
# The log line grammar: how the stage scripts report data to the experiment record.
# Stage scripts print to stdout only; run.sh tees stdout+stderr into per-stage
# .log files in the experiment directory. Most of a log is free-form prose for
# humans with no stability promise. Lines that carry data follow this grammar and
# are the machine-readable contract:
#
#     <tag> key=value key=value ...
#
#     eval step=4990 eflops=0.9 val_bpb=0.8213
#     summary params=286261730 val_bpb=0.8213 gpu="NVIDIA H100 80GB HBM3"
#
# Values with spaces are quoted; parsing infers int/float and leaves the rest as
# strings. This is the only schema machinery in the repo: scripts format records
# with format_record(), and the readers below get them back with parse_records().

def format_record(tag, **fields):
    """Format a record line, e.g. format_record("summary", loss=2.3) -> 'summary loss=2.3'."""
    parts = [tag]
    for key, value in fields.items():
        value_str = shlex.quote(str(value))
        parts.append(f"{key}={value_str}")
    line = " ".join(parts)
    return line


def format_invocation(args):
    """
    The record pair every stage script prints at startup, so that a run is
    reproducible from its log alone: `argv` carries the verbatim command line,
    `config` the fully resolved arguments, i.e. including all the defaults
    (which argv does not show, and which change over time as the code evolves).
    args is the argparse Namespace of the calling script.
    """
    main_spec = getattr(sys.modules["__main__"], "__spec__", None)
    script = main_spec.name if main_spec is not None else sys.argv[0]
    argv_str = shlex.join(sys.argv[1:])
    argv_line = format_record("argv", script=script, args=argv_str)
    config_line = format_record("config", **vars(args))
    lines = argv_line + "\n" + config_line
    return lines


def _parse_value(value_str):
    """Infer the type of a value: int, then float, else string."""
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def parse_record(line, tag=None):
    """
    Parse a single record line into a dict (with the tag under the "tag" key),
    or return None if the line is not a well-formed record (or has a different tag).
    """
    line = line.strip()
    # fast path: when filtering by tag, reject non-matching lines before the
    # (comparatively slow) shlex split -- a log is mostly step records and prose,
    # and analysis tools scan many multi-MB logs
    if tag is not None and not line.startswith(tag + " "):
        return None
    try:
        tokens = shlex.split(line)
    except ValueError:
        return None # e.g. unbalanced quotes: prose, not a record
    if not tokens:
        return None
    line_tag = tokens[0]
    if tag is not None and line_tag != tag:
        return None
    record = {"tag": line_tag}
    for token in tokens[1:]:
        if "=" not in token:
            return None # every field must be key=value, otherwise this is prose
        key, value_str = token.split("=", 1)
        record[key] = _parse_value(value_str)
    if len(record) == 1:
        return None # a lone word is prose, not a record
    return record


def parse_records(path, tag=None):
    """Scan a log file and return all record dicts (optionally only those with a given tag)."""
    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            record = parse_record(line, tag=tag)
            if record is not None:
                records.append(record)
    return records


# -----------------------------------------------------------------------------
# Creating an experiment

def git(*args):
    """Run a git command and return its stdout, or empty string on failure."""
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    return result.stdout.strip()


def init_experiment():
    """Create (or resume) the active experiment. Returns the experiment directory."""
    name = get_experiment_name()
    experiment_dir = get_experiment_dir(name)
    meta_path = os.path.join(experiment_dir, "meta.json")

    # Resuming: the experiment already exists, just sanity check the code identity
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"experiment {name}: resuming (created {meta['created']}, commit {meta['git_commit'][:7]})")
        current_commit = git("rev-parse", "HEAD")
        if current_commit != meta["git_commit"]:
            print(f"experiment {name}: WARNING: current commit {current_commit[:7]} differs from the recorded one")
        return experiment_dir

    # Creating: record the experiment's identity
    os.makedirs(experiment_dir, exist_ok=True)
    dirty_diff = git("diff", "HEAD") # tracked, uncommitted changes
    meta = {
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty_diff),
        "dataset": get_dataset_name(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    # a dirty tree would make the commit hash a lie, so record the diff for reproducibility
    if dirty_diff:
        diff_path = os.path.join(experiment_dir, "code_diff.patch")
        with open(diff_path, "w") as f:
            f.write(dirty_diff + "\n")
    dirty_suffix = " (dirty tree, diff saved to code_diff.patch)" if dirty_diff else ""
    print(f"experiment {name}: created at {experiment_dir}, commit {meta['git_commit'][:7]}{dirty_suffix}")
    return experiment_dir


# -----------------------------------------------------------------------------
# Reading an experiment back

def read_meta(experiment_dir):
    """The experiment's meta.json as a dict ({} if absent)."""
    meta_path = os.path.join(experiment_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def model_sort_key(model_tag):
    """Sort d<depth> tags numerically, anything else after them alphabetically."""
    match = re.fullmatch(r"d(\d+)", model_tag)
    if match:
        return (0, int(match.group(1)), model_tag)
    return (1, 0, model_tag)


def list_model_tags(experiment_dir):
    """All model directories of an experiment (subdirs with at least one stage log), sorted."""
    tags = []
    for name in os.listdir(experiment_dir):
        model_dir = os.path.join(experiment_dir, name)
        if not os.path.isdir(model_dir):
            continue
        has_log = any(f.endswith(".log") for f in os.listdir(model_dir))
        if has_log:
            tags.append(name)
    tags.sort(key=model_sort_key)
    return tags


def read_stage_summary(log_path):
    """The last `summary` record of a stage log, without the tag key (None if absent)."""
    if not os.path.exists(log_path):
        return None
    records = parse_records(log_path, tag="summary")
    if not records:
        return None
    summary = records[-1]
    summary.pop("tag", None)
    return summary


def read_base_summary(experiment_dir, model_tag):
    """The final summary of one model's pretraining, or None if incomplete."""
    log_path = os.path.join(experiment_dir, model_tag, "base_train.log")
    summary = read_stage_summary(log_path)
    if summary is None or "val_bpb" not in summary:
        return None
    return summary


def list_completed_tags(experiment_dir):
    """Model tags whose pretraining finished (a summary record exists), sorted."""
    tags = list_model_tags(experiment_dir)
    completed = [tag for tag in tags if read_base_summary(experiment_dir, tag) is not None]
    return completed


def read_bench_sweep(experiment_dir, model_tag):
    """The per-batch-size `bench` records of one model's inference bench log."""
    log_path = os.path.join(experiment_dir, model_tag, "base_inference.log")
    if not os.path.exists(log_path):
        return []
    records = list(parse_records(log_path, tag="bench"))
    records.sort(key=lambda r: r["batch"])
    return records


# -----------------------------------------------------------------------------
# The curve: the product of an experiment. Quality, speed and cost of every model
# in the ladder, in one file: curve.log holds one `model` record per model, fields
# namespaced by stage (base_train.val_bpb, chat_eval.chatcore, ...). To get a dataframe:
#
#     import pandas as pd
#     from harness.experiment import parse_records
#     df = pd.DataFrame(parse_records("curve.log", tag="model"))

# stage logs to join, in curve column order
STAGE_LOGS = ["base_train", "base_eval", "base_inference", "chat_train", "chat_eval"] # <stage>.log each, in curve column order

# columns for the compact stdout table (the full set goes to curve.log)
TABLE_COLUMNS = [
    "model_tag", "base_train.num_params", "base_train.tokens_trained", "base_train.eflops", "base_train.train_time_sec",
    "base_train.val_bpb", "base_eval.core", "base_inference.tok_per_sec_bs1", "chat_train.val_bpb", "chat_eval.chatcore",
]

def build_row(experiment_dir, model_tag):
    """Join the stage summaries of one model into a flat row."""
    row = {"model_tag": model_tag}
    model_dir = os.path.join(experiment_dir, model_tag)
    for stage in STAGE_LOGS:
        if stage == "base_train":
            # read_base_summary additionally checks the run completed (has val_bpb)
            summary = read_base_summary(experiment_dir, model_tag) or {}
        else:
            summary = read_stage_summary(os.path.join(model_dir, f"{stage}.log")) or {}
        summary.pop("model_tag", None) # identity, already in the row
        if stage == "base_train" and "depth" in summary:
            row["depth"] = summary.pop("depth")
        for key, value in summary.items():
            row[f"{stage}.{key}"] = value
    # a few headline throughput numbers from the inference bench sweep
    bench = read_bench_sweep(experiment_dir, model_tag)
    if bench:
        row["base_inference.tok_per_sec_bs1"] = bench[0]["tok_per_sec"]
        row[f"base_inference.tok_per_sec_bs{bench[-1]['batch']}"] = bench[-1]["tok_per_sec"]
    return row


def write_curve(experiment_dir):
    """Aggregate every model's stage records into curve.log and print the headline table."""
    model_tags = list_model_tags(experiment_dir)
    assert model_tags, f"No model directories with stage logs found in {experiment_dir}"
    rows = [build_row(experiment_dir, tag) for tag in model_tags]

    # write the full curve as `model` records in the log grammar, one line per model
    curve_path = os.path.join(experiment_dir, "curve.log")
    with open(curve_path, "w") as f:
        for row in rows:
            f.write(format_record("model", **row) + "\n")

    # print the compact table
    table_columns = [c for c in TABLE_COLUMNS if any(c in row for row in rows)]
    widths = {c: max(len(c), *(len(str(row.get(c, ""))) for row in rows)) for c in table_columns}
    print(f"experiment {os.path.basename(experiment_dir)}: {len(rows)} models")
    print("  ".join(c.ljust(widths[c]) for c in table_columns))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in table_columns))
    print(f"full curve written to {curve_path}")


# -----------------------------------------------------------------------------
# Comparing two experiments by compute multiplier (CM). For a target quality,
#
#     CM = baseline EFLOPs to reach the target / variant EFLOPs to reach the target
#
# so CM > 1 means the variant reaches the same quality with less compute (a win)
# and CM < 1 means it needs more (a regression).
#
# Only fully-annealed endpoints participate. Each experiment's ladder traces a
# *frontier*: the final (EFLOPs, quality) point of every completed model, and CMs
# are read off by inverting the baseline's frontier at each variant endpoint.
# Mid-training curve points are never compared against: a mid-schedule model has
# not had its LR anneal yet, so its loss sits above what a run *scheduled* to end
# there would achieve, which systematically flatters the variant. (This bias bites
# exactly when the experiments derive different horizons, e.g. an ablation that
# changes the parameter count and thereby the token budget.)
#
# Conventions:
# - The anchor is each variant endpoint's quality: CM answers "how much compute
#   does the baseline need to match what this variant model achieved".
# - Frontier inversion is piecewise-linear in (log EFLOPs, metric). No parametric
#   form (e.g. a power law) is assumed.
# - Matched depths are not required: an anchor only needs to land within the
#   baseline's frontier range. If it falls outside, the boundary segment is
#   extrapolated linearly and the reported CM is marked with a trailing "?": a
#   low-confidence result, but a stated one.
#
# Three metrics are compared, each where it applies:
# - val_bpb: the precise instrument (~±0.5% of compute), but only meaningful when
#   both experiments trained on the same dataset.
# - CORE: dataset-independent, so it also referees data ablations, but noisy
#   (~±0.003 at d16+, which the ~0.08/decade slope turns into ~±10% of compute);
#   alongside the piecewise CM its rows carry cm_fit: both frontiers are
#   least-squares fit as y = a + b*log(eflops) and the fits are inverted instead,
#   smoothing the noise at the cost of assuming a shared shape. (val_bpb gets no
#   cm_fit: it is precise, and its frontier is visibly curved in (log eflops, bpb)
#   space, so a single global line would misfit it.)
# - ChatCORE: the sft stage's aggregate score, same treatment as CORE; noisier
#   still, so read it mostly through its cm_fit.
#
# The compare command prints one `cm` record per variant model and metric and
# writes them to compare_vs_<baseline>.log in the variant's experiment directory.

def read_frontier(experiment_dir, metric):
    """The (eflops, metric, tag) endpoints of all completed models, sorted by
    eflops. Every point is a fully-annealed model: comparisons only ever invert
    this frontier, never mid-training curves (see above). val_bpb comes from the
    base_train summary, core and chatcore from the base_eval and chat_eval
    summaries. The x axis is always *pretraining* eflops: the eval and chat stages'
    compute is negligible next to it."""
    metric_stage = {"val_bpb": "base_train", "core": "base_eval", "chatcore": "chat_eval"}[metric]
    points = []
    for tag in list_completed_tags(experiment_dir):
        eflops = read_base_summary(experiment_dir, tag)["eflops"]
        stage_log = os.path.join(experiment_dir, tag, f"{metric_stage}.log")
        summary = read_stage_summary(stage_log) or {}
        if metric in summary:
            points.append((eflops, summary[metric], tag))
    points.sort()
    return points


def solve_segment(p0, p1, target):
    """The eflops where the straight line through p0,p1 in (log eflops, y) space
    reaches y=target. Works for interpolation and extrapolation alike."""
    (f0, y0), (f1, y1) = p0, p1
    w = (target - y0) / (y1 - y0)
    log_f = (1 - w) * math.log(f0) + w * math.log(f1)
    return math.exp(log_f)


def eflops_at_metric(points, target):
    """
    Invert a monotone (eflops, y) frontier: the eflops where it reaches y=target.
    Piecewise-linear in (log eflops, y); increasing (CORE) and decreasing (bpb)
    frontiers alike. If the target lies outside the measured range, the boundary
    segment is extrapolated. Returns (eflops, extrapolated) where extrapolated=True
    flags the low-confidence case, or None if the frontier is degenerate.
    """
    if len(points) < 2:
        return None
    # interpolation: find the first measured segment that crosses the target
    for p0, p1 in zip(points, points[1:]):
        (_, y0), (_, y1) = p0, p1
        if min(y0, y1) <= target <= max(y0, y1) and y0 != y1:
            eflops = solve_segment(p0, p1, target)
            return eflops, False
    # extrapolation: continue the boundary segment past the measured range
    increasing = points[-1][1] > points[0][1]
    before_start = (target < points[0][1]) == increasing # target precedes the smallest model
    segment = (points[0], points[1]) if before_start else (points[-2], points[-1])
    (_, y0), (_, y1) = segment
    if y0 == y1:
        return None # flat boundary segment: no slope to extrapolate along
    eflops = solve_segment(*segment, target)
    return eflops, True


def fit_frontier(points):
    """Least-squares fit y = a + b*log10(eflops) through a frontier. Returns (a, b)."""
    xs = [math.log10(f) for f, y in points]
    ys = [y for f, y in points]
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum((x - xbar) ** 2 for x in xs)
    a = ybar - b * xbar
    return a, b


def invert_fit(fit, target):
    """The eflops where a fitted frontier y = a + b*log10(eflops) reaches y=target."""
    a, b = fit
    return 10 ** ((target - a) / b)


def compute_comparison(baseline_dir, variant_dir, metric, fit=False):
    """
    One row per variant endpoint with the metric: frontier-inverted CMs.
    Returns (rows, baseline_frontier, variant_frontier, baseline_fit, variant_fit),
    or None when either side has fewer than the 2 endpoints a frontier needs.
    """
    baseline_frontier = read_frontier(baseline_dir, metric)
    variant_frontier = read_frontier(variant_dir, metric)
    if len(baseline_frontier) < 2 or len(variant_frontier) < 2:
        print(f"{metric} comparison needs at least 2 completed models with {metric} on each side")
        return None
    baseline_points = [(f, y) for f, y, _ in baseline_frontier]
    variant_points = [(f, y) for f, y, _ in variant_frontier]
    baseline_fit = fit_frontier(baseline_points) if fit else None
    variant_fit = fit_frontier(variant_points) if fit else None
    rows = []
    for variant_eflops, anchor, tag in variant_frontier:
        baseline_result = eflops_at_metric(baseline_points, anchor)
        if baseline_result is None:
            print(f"{tag}: skipped, baseline frontier is degenerate at anchor {metric} {anchor:.6f}")
            continue
        baseline_eflops, extrapolated = baseline_result
        row = dict(
            model_tag=tag,
            anchor=anchor,
            baseline_eflops=baseline_eflops,
            variant_eflops=variant_eflops,
            cm=baseline_eflops / variant_eflops,
            extrapolated=extrapolated,
        )
        if fit:
            row["cm_fit"] = invert_fit(baseline_fit, anchor) / invert_fit(variant_fit, anchor)
        rows.append(row)
    return rows, baseline_frontier, variant_frontier, baseline_fit, variant_fit


def datasets_match(baseline_dir, variant_dir):
    """val_bpb is only comparable when both experiments trained on the same dataset."""
    baseline_dataset = read_meta(baseline_dir).get("dataset")
    variant_dataset = read_meta(variant_dir).get("dataset")
    return baseline_dataset == variant_dataset


def compare_experiments(baseline, variant):
    """Print the cm records of variant over baseline and write them next to the variant's curve."""
    baseline_dir = get_experiment_dir(baseline)
    variant_dir = get_experiment_dir(variant)
    assert list_completed_tags(baseline_dir), f"No completed pretraining runs found in {baseline_dir}"
    assert list_completed_tags(variant_dir), f"No completed pretraining runs found in {variant_dir}"
    lines = []

    # the val_bpb comparison: only meaningful when both experiments trained on the same data
    bpb_result = None
    if datasets_match(baseline_dir, variant_dir):
        bpb_result = compute_comparison(baseline_dir, variant_dir, "val_bpb")
    else:
        print("datasets differ: skipping the val_bpb comparison")
    if bpb_result is not None:
        for row in bpb_result[0]:
            cm_str = f"{row['cm']:.4f}?" if row["extrapolated"] else f"{row['cm']:.4f}"
            record = format_record(
                "cm",
                metric="bpb",
                model_tag=row["model_tag"],
                anchor_bpb=round(row["anchor"], 6),
                baseline_eflops=round(row["baseline_eflops"], 4),
                variant_eflops=round(row["variant_eflops"], 4),
                cm=cm_str,
            )
            lines.append(record)

    # the CORE comparison: dataset-independent, works whenever both ladders have 2+ scores
    # the ChatCORE comparison: same treatment for the sft stage's aggregate score
    core_result = compute_comparison(baseline_dir, variant_dir, "core", fit=True)
    chatcore_result = compute_comparison(baseline_dir, variant_dir, "chatcore", fit=True)
    for metric, result in [("core", core_result), ("chatcore", chatcore_result)]:
        if result is None:
            continue
        for row in result[0]:
            cm_str = f"{row['cm']:.4f}?" if row["extrapolated"] else f"{row['cm']:.4f}"
            record = format_record(
                "cm",
                metric=metric,
                model_tag=row["model_tag"],
                **{f"anchor_{metric}": round(row["anchor"], 6)},
                baseline_eflops=round(row["baseline_eflops"], 4),
                variant_eflops=round(row["variant_eflops"], 4),
                cm=cm_str,
                cm_fit=round(row["cm_fit"], 4),
            )
            lines.append(record)

    # report to stdout and persist next to the experiment's other artifacts
    log_file = os.path.join(variant_dir, f"compare_vs_{baseline}.log")
    for line in lines:
        print(line)
    with open(log_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved records to {log_file}")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create, aggregate and compare experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="create or resume $NANOCHAT_EXPERIMENT, recording its identity in meta.json")
    commands.add_parser("curve", help="join the stage summaries of $NANOCHAT_EXPERIMENT into curve.log")
    compare_parser = commands.add_parser("compare", help="compute multipliers of a variant experiment over a baseline")
    compare_parser.add_argument("baseline", type=str, help="Baseline experiment name")
    compare_parser.add_argument("variant", type=str, help="Variant experiment name")
    args = parser.parse_args()

    if args.command == "init":
        init_experiment()
    elif args.command == "curve":
        write_curve(get_experiment_dir())
    elif args.command == "compare":
        compare_experiments(args.baseline, args.variant)
