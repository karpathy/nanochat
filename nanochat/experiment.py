"""
An experiment is (name, git commit, depth ladder, dataset). This module has two halves:
- initializing an experiment: create the directory, record identity in meta.json
- reading an experiment back: shared helpers for analysis tools (scripts/curve.py,
  scripts/compare.py) that parse the stage logs written by runs/run.sh
See experiment_refactor.md for the design.

Because every hyperparameter derives from --depth and the dataset is referenced by
name, there is no config file: the code is the config, and meta.json records which
code (the commit). If the working tree is dirty, the diff is saved alongside so that
even dirty runs are reproducible.

Usage (normally called by runs/run.sh, reads $NANOCHAT_EXPERIMENT and $NANOCHAT_DATASET):
    python -m nanochat.experiment
"""

import os
import re
import json
import subprocess
from datetime import datetime

from nanochat.common import get_experiment_name, get_experiment_dir
from nanochat.dataset import get_dataset_name
from nanochat.logfmt import parse_records


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
    if "eflops" not in summary:
        # legacy fallback: logs written before the eflops summary field (Jul 2026)
        # carry FLOPs/token only as prose; reconstruct total eflops from it
        flops_per_token = None
        with open(log_path) as f:
            for line in f:
                if line.startswith("Estimated FLOPs per token:"):
                    flops_per_token = float(line.rsplit(":", 1)[1])
                    break
        if flops_per_token is None:
            return None
        summary["eflops"] = flops_per_token * summary["tokens_trained"] / 1e18
    return summary


def read_training_curve(experiment_dir, model_tag):
    """
    The (eflops, val_bpb) points traced by one model's pretraining, from its eval
    records. Newer logs carry eflops on each eval record; for older ones the mapping
    is reconstructed from the summary (so an old incomplete run yields no curve).
    """
    log_path = os.path.join(experiment_dir, model_tag, "base_train.log")
    records = parse_records(log_path, tag="eval")
    records = [r for r in records if r["step"] > 0] # step 0 is before training: zero flops
    if any("eflops" not in r for r in records):
        summary = read_base_summary(experiment_dir, model_tag)
        if summary is None:
            return []
        eflops_per_step = summary["eflops"] / summary["num_iterations"]
        for record in records:
            record.setdefault("eflops", record["step"] * eflops_per_step)
    points = [(r["eflops"], r["val_bpb"]) for r in records]
    points = [p for p in points if p[0] > 0] # tiny debug runs can round to eflops=0.0, unusable in log space
    return points


if __name__ == "__main__":
    init_experiment()
