"""
An experiment is (name, git commit, depth ladder, dataset). This module initializes
the experiment directory and records the experiment's identity in meta.json.
See experiment_refactor.md for the design.

Because every hyperparameter derives from --depth and the dataset is referenced by
name, there is no config file: the code is the config, and meta.json records which
code (the commit). If the working tree is dirty, the diff is saved alongside so that
even dirty runs are reproducible.

Usage (normally called by runs/run.sh, reads $NANOCHAT_EXPERIMENT and $NANOCHAT_DATASET):
    python -m nanochat.experiment
"""

import os
import json
import subprocess
from datetime import datetime

from nanochat.common import get_experiment_name, get_experiment_dir
from nanochat.dataset import get_dataset_name


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


if __name__ == "__main__":
    init_experiment()
