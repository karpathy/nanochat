"""
Compare two experiments by compute multiplier (CM).

For a target quality (val_bpb), the compute multiplier is the ratio

    CM = baseline EFLOPs to reach the target / variant EFLOPs to reach the target

so CM > 1 means the variant reaches the same quality with less compute (a win) and
CM < 1 means it needs more (a regression). In the infinite-data regime the training
curves (val_bpb vs FLOPs) are monotone decreasing, so the intersection with a
horizontal line at the target bpb is well defined; we read FLOPs off each curve
with piecewise-linear interpolation in (log FLOPs, bpb) space.

Conventions:
- The anchor bpb for a pair of runs is the *worse* of the two final val_bpbs, so
  both curves reach it without extrapolation (the better run passed through it
  mid-training; the worse run contributes its endpoint exactly).
- If the baseline never trained a model tag (e.g. the variant's ladder has extra
  depths), we fall back to the baseline *frontier*: the final (EFLOPs, val_bpb)
  points of all baseline models, interpolated the same way. This compares the
  variant model against the baseline model family rather than a single run,
  and is marked mode=frontier in the output.

Usage:

    python -m scripts.compare <baseline_experiment> <variant_experiment>

Prints one `cm` record per variant model (the log grammar, see nanochat/logfmt.py).
"""

import os
import json
import math
import argparse

from nanochat.common import get_experiment_dir
from nanochat.logfmt import parse_records, format_record
from scripts.curve import model_sort_key


def read_base_summary(experiment_dir, model_tag):
    """The final summary record of one model's pretraining, or None if incomplete."""
    log_path = os.path.join(experiment_dir, model_tag, "base_train.log")
    if not os.path.exists(log_path):
        return None
    summaries = parse_records(log_path, tag="summary")
    if not summaries:
        return None
    summary = summaries[-1]
    if "val_bpb" not in summary:
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


def list_model_tags(experiment_dir):
    """All model tags in an experiment with completed pretraining, sorted."""
    tags = []
    for name in os.listdir(experiment_dir):
        if os.path.isdir(os.path.join(experiment_dir, name)) and read_base_summary(experiment_dir, name) is not None:
            tags.append(name)
    tags.sort(key=model_sort_key)
    return tags


def read_training_curve(experiment_dir, model_tag):
    """The (eflops, val_bpb) points traced by one training run, from its eval records."""
    summary = read_base_summary(experiment_dir, model_tag)
    eflops_per_step = summary["eflops"] / summary["num_iterations"]
    log_path = os.path.join(experiment_dir, model_tag, "base_train.log")
    points = []
    for record in parse_records(log_path, tag="eval"):
        if record["step"] == 0:
            continue # zero flops: undefined in log space (and not interesting)
        points.append((record["step"] * eflops_per_step, record["val_bpb"]))
    return points


def read_frontier(experiment_dir, model_tags):
    """The (eflops, val_bpb) frontier traced by an experiment's final models."""
    points = []
    for tag in model_tags:
        summary = read_base_summary(experiment_dir, tag)
        points.append((summary["eflops"], summary["val_bpb"]))
    points.sort()
    return points


def eflops_at_bpb(points, target_bpb):
    """
    Invert a monotone (eflops, bpb) curve: the eflops where it first reaches target_bpb,
    interpolating linearly in (log eflops, bpb). Returns None if the curve never gets
    there, or if the curve is already below the target at its first point.
    """
    if not points:
        return None
    if points[0][1] <= target_bpb:
        return None # already better than target at the first point: true crossing unknown
    for (f0, b0), (f1, b1) in zip(points, points[1:]):
        if b1 <= target_bpb:
            w = (b0 - target_bpb) / (b0 - b1)
            log_f = (1 - w) * math.log(f0) + w * math.log(f1)
            return math.exp(log_f)
    return None


def read_dataset_name(experiment_dir):
    meta_path = os.path.join(experiment_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute multipliers of a variant experiment over a baseline")
    parser.add_argument("baseline", type=str, help="Baseline experiment name")
    parser.add_argument("variant", type=str, help="Variant experiment name")
    args = parser.parse_args()

    baseline_dir = get_experiment_dir(args.baseline)
    variant_dir = get_experiment_dir(args.variant)

    # bpb is only comparable when both experiments trained on the same data
    baseline_dataset = read_dataset_name(baseline_dir)
    variant_dataset = read_dataset_name(variant_dir)
    if baseline_dataset != variant_dataset:
        print(f"WARNING: datasets differ ({baseline_dataset} vs {variant_dataset}), val_bpb is not comparable")

    baseline_tags = list_model_tags(baseline_dir)
    variant_tags = list_model_tags(variant_dir)
    assert baseline_tags, f"No completed pretraining runs found in {baseline_dir}"
    assert variant_tags, f"No completed pretraining runs found in {variant_dir}"
    frontier = read_frontier(baseline_dir, baseline_tags)

    for tag in variant_tags:
        variant_summary = read_base_summary(variant_dir, tag)
        variant_final_bpb = variant_summary["val_bpb"]
        if tag in baseline_tags:
            # primary path: same-depth training curve vs training curve
            mode = "curve"
            baseline_summary = read_base_summary(baseline_dir, tag)
            anchor_bpb = max(variant_final_bpb, baseline_summary["val_bpb"])
            baseline_eflops = eflops_at_bpb(read_training_curve(baseline_dir, tag), anchor_bpb)
            variant_eflops = eflops_at_bpb(read_training_curve(variant_dir, tag), anchor_bpb)
        else:
            # fallback: variant's final point vs the baseline model family frontier
            mode = "frontier"
            anchor_bpb = variant_final_bpb
            baseline_eflops = eflops_at_bpb(frontier, anchor_bpb)
            variant_eflops = variant_summary["eflops"]
        if baseline_eflops is None or variant_eflops is None:
            print(f"{tag}: cm undefined, anchor bpb {anchor_bpb:.6f} is outside the baseline {mode} (needs extrapolation)")
            continue
        cm = baseline_eflops / variant_eflops
        record = format_record(
            "cm",
            model_tag=tag,
            mode=mode,
            anchor_bpb=round(anchor_bpb, 6),
            baseline_eflops=round(baseline_eflops, 4),
            variant_eflops=round(variant_eflops, 4),
            cm=round(cm, 4),
        )
        print(record)
