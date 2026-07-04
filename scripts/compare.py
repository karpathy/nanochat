"""
Compare two experiments by compute multiplier (CM).

For a target quality (val_bpb), the compute multiplier is the ratio

    CM = baseline EFLOPs to reach the target / variant EFLOPs to reach the target

so CM > 1 means the variant reaches the same quality with less compute (a win) and
CM < 1 means it needs more (a regression). In the infinite-data regime the training
curves (val_bpb vs FLOPs) are monotone decreasing, so the intersection with a
horizontal line at the target bpb is well defined; we read FLOPs off each curve
with piecewise-linear interpolation in (log FLOPs, log bpb) space.

Conventions:
- The anchor is the variant's final val_bpb: CM answers "how much compute does the
  baseline need to match what the variant achieved".
- Interpolation is piecewise-linear in (log EFLOPs, log val_bpb): these curves are
  approximately power laws, so log-log is the natural space for straight segments.
- A model tag is only compared when the same tag was trained in both experiments.
- If the anchor lies beyond a curve's measured range (e.g. the variant reached a
  bpb the baseline never did), the boundary segment is extrapolated in the same
  log-log space and the reported CM is marked with a trailing "?": a low-confidence
  result, but a stated one.

Usage:

    python -m scripts.compare <baseline_experiment> <variant_experiment>

Prints one `cm` record per comparable model (the log grammar, see nanochat/logfmt.py).
The records are also written to compare_vs_<baseline>.log in the variant's experiment
directory and, unless --plot=0, a two-panel png (training curves + CM vs compute) is
saved next to it.
"""

import os
import math
import argparse

from nanochat.common import get_experiment_dir
from nanochat.logfmt import format_record
from nanochat.experiment import list_model_tags, read_base_summary, read_training_curve, read_meta


def list_completed_tags(experiment_dir):
    """Model tags whose pretraining finished (a summary record exists), sorted."""
    tags = list_model_tags(experiment_dir)
    completed = [tag for tag in tags if read_base_summary(experiment_dir, tag) is not None]
    return completed

# -----------------------------------------------------------------------------
# The compute multiplier calculation

def solve_segment(p0, p1, target_bpb):
    """The eflops where the straight line through p0,p1 in (log eflops, log bpb)
    space reaches target_bpb. Works for interpolation and extrapolation alike."""
    (f0, b0), (f1, b1) = p0, p1
    w = (math.log(b0) - math.log(target_bpb)) / (math.log(b0) - math.log(b1))
    log_f = (1 - w) * math.log(f0) + w * math.log(f1)
    return math.exp(log_f)


def eflops_at_bpb(points, target_bpb):
    """
    Invert a monotone (eflops, bpb) curve: the eflops where it first reaches target_bpb.
    Piecewise-linear in (log eflops, log bpb). If the target lies outside the measured
    range, the boundary segment is extrapolated. Returns (eflops, extrapolated) where
    extrapolated=True flags the low-confidence case, or None if the curve is degenerate.
    """
    if len(points) < 2:
        return None
    # interpolation: find the first measured segment that crosses the target
    for (f0, b0), (f1, b1) in zip(points, points[1:]):
        if b1 <= target_bpb <= b0 and b0 != b1:
            eflops = solve_segment((f0, b0), (f1, b1), target_bpb)
            return eflops, False
    # extrapolation: continue the boundary segment past the measured range
    if target_bpb > points[0][1]:
        segment = (points[0], points[1]) # target is worse than the first eval: extend backward
    else:
        segment = (points[-2], points[-1]) # target is better than the final eval: extend forward
    (_, b0), (_, b1) = segment
    if b0 == b1:
        return None # flat boundary segment: no slope to extrapolate along
    eflops = solve_segment(*segment, target_bpb)
    return eflops, True


def compute_comparison(baseline_dir, variant_dir, common_tags):
    """One row per comparable model tag; the CM machinery in one place."""
    rows = []
    for tag in common_tags:
        variant_summary = read_base_summary(variant_dir, tag)
        # the anchor: the quality the variant reached at the end of its training
        anchor_bpb = variant_summary["val_bpb"]
        variant_eflops = variant_summary["eflops"]
        baseline_result = eflops_at_bpb(read_training_curve(baseline_dir, tag), anchor_bpb)
        if baseline_result is None:
            print(f"{tag}: skipped, baseline curve is degenerate at anchor bpb {anchor_bpb:.6f}")
            continue
        baseline_eflops, extrapolated = baseline_result
        rows.append(dict(
            model_tag=tag,
            anchor_bpb=anchor_bpb,
            baseline_eflops=baseline_eflops,
            variant_eflops=variant_eflops,
            cm=baseline_eflops / variant_eflops,
            extrapolated=extrapolated,
        ))
    return rows

# -----------------------------------------------------------------------------
# Plotting

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
BASELINE_COLOR = "#2a78d6"
VARIANT_COLOR = "#1baf7a"
EFLOP = 1e18 # records store EFLOPs; the plot axes are raw FLOPs

def make_plot(baseline, variant, baseline_dir, variant_dir, common_tags, rows, out_path):
    """Two panels: training curves + final points (log-log), and CM vs compute."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, NullFormatter, LogLocator

    fig, (ax_curves, ax_cm) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    # left panel: training curves + final points, log-log
    ax = ax_curves
    seen_tags = {}
    styles = [(baseline_dir, BASELINE_COLOR, f"{baseline} (baseline)"), (variant_dir, VARIANT_COLOR, f"{variant} (variant)")]
    for experiment_dir, color, label in styles:
        for tag in common_tags:
            points = read_training_curve(experiment_dir, tag)
            xs = [p[0] * EFLOP for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, color=color, lw=1.2, marker="o", ms=2.5, alpha=0.25)
        finals = []
        for tag in common_tags:
            summary = read_base_summary(experiment_dir, tag)
            finals.append((summary["eflops"] * EFLOP, summary["val_bpb"], tag))
        finals.sort()
        xs = [p[0] for p in finals]
        ys = [p[1] for p in finals]
        ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=6, label=label)
        for x, y, tag in finals:
            if tag not in seen_tags:
                seen_tags[tag] = (x, y)
    for tag, (x, y) in seen_tags.items():
        ax.annotate(tag, (x, y), xytext=(6, -2), textcoords="offset points",
                    fontsize=9, color=TEXT_SECONDARY, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pretraining compute (FLOPs)", color=TEXT_PRIMARY)
    ax.set_ylabel("val bpb", color=TEXT_PRIMARY)
    ax.set_title("final val_bpb per depth (faded: training curves)", color=TEXT_PRIMARY, pad=12)
    # bpb spans well under a decade: label every 0.1-ish step, hide minor labels
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=tuple(range(1, 10))))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    # right panel: compute multiplier per depth vs variant training FLOPs
    ax = ax_cm
    xs = [row["variant_eflops"] * EFLOP for row in rows]
    cms = [row["cm"] for row in rows]
    ax.axhline(1.0, color=TEXT_SECONDARY, lw=1.2, ls="--", alpha=0.7)
    ax.annotate("baseline (CM=1)", (xs[-1], 1.0), xytext=(0, 6), textcoords="offset points",
                fontsize=9, color=TEXT_SECONDARY, ha="right")
    ax.plot(xs, cms, color=VARIANT_COLOR, lw=2.2, zorder=2)
    for x, row in zip(xs, rows):
        facecolor = SURFACE if row["extrapolated"] else VARIANT_COLOR # open marker = extrapolated
        ax.plot([x], [row["cm"]], marker="o", ms=7, color=VARIANT_COLOR,
                markerfacecolor=facecolor, zorder=3)
        tag_label = f"{row['model_tag']}?" if row["extrapolated"] else row["model_tag"]
        ax.annotate(tag_label, (x, row["cm"]), xytext=(6, -10), textcoords="offset points",
                    fontsize=9, color=TEXT_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlabel("pretraining compute (FLOPs)", color=TEXT_PRIMARY)
    ax.set_ylabel("compute multiplier", color=TEXT_PRIMARY)
    ax.set_title(f"CM of {variant} vs {baseline}", color=TEXT_PRIMARY, pad=12)
    lo = min(cms + [1.0])
    hi = max(cms + [1.0])
    pad = 0.25 * (hi - lo) + 0.005
    ax.set_ylim(lo - pad, hi + pad)

    # shared cosmetics
    for ax in [ax_curves, ax_cm]:
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", alpha=0.28, lw=0.6)
        ax.grid(True, which="minor", alpha=0.12, lw=0.5)
        ax.tick_params(colors=TEXT_SECONDARY)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved plot to {out_path}")

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute multipliers of a variant experiment over a baseline")
    parser.add_argument("baseline", type=str, help="Baseline experiment name")
    parser.add_argument("variant", type=str, help="Variant experiment name")
    parser.add_argument("--plot", type=int, default=1, help="save the comparison png (1) or not (0)")
    parser.add_argument("--plot-file", type=str, default=None, help="png path (default: <variant experiment dir>/compare_vs_<baseline>.png)")
    args = parser.parse_args()

    baseline_dir = get_experiment_dir(args.baseline)
    variant_dir = get_experiment_dir(args.variant)

    # bpb is only comparable when both experiments trained on the same data
    baseline_dataset = read_meta(baseline_dir).get("dataset")
    variant_dataset = read_meta(variant_dir).get("dataset")
    if baseline_dataset != variant_dataset:
        print(f"WARNING: datasets differ ({baseline_dataset} vs {variant_dataset}), val_bpb is not comparable")

    baseline_tags = list_completed_tags(baseline_dir)
    variant_tags = list_completed_tags(variant_dir)
    assert baseline_tags, f"No completed pretraining runs found in {baseline_dir}"
    assert variant_tags, f"No completed pretraining runs found in {variant_dir}"
    lines = []
    for tag in variant_tags:
        if tag not in baseline_tags:
            lines.append(f"{tag}: skipped, the baseline never trained this model tag")
    common_tags = [tag for tag in variant_tags if tag in baseline_tags]

    rows = compute_comparison(baseline_dir, variant_dir, common_tags)
    for row in rows:
        cm_str = f"{row['cm']:.4f}?" if row["extrapolated"] else f"{row['cm']:.4f}"
        record = format_record(
            "cm",
            model_tag=row["model_tag"],
            anchor_bpb=round(row["anchor_bpb"], 6),
            baseline_eflops=round(row["baseline_eflops"], 4),
            variant_eflops=round(row["variant_eflops"], 4),
            cm=cm_str,
        )
        lines.append(record)

    # report to stdout and persist next to the experiment's other artifacts
    for line in lines:
        print(line)
    log_file = os.path.join(variant_dir, f"compare_vs_{args.baseline}.log")
    with open(log_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved records to {log_file}")

    if args.plot == 1 and rows:
        plot_file = args.plot_file
        if plot_file is None:
            plot_file = os.path.join(variant_dir, f"compare_vs_{args.baseline}.png")
        make_plot(args.baseline, args.variant, baseline_dir, variant_dir, common_tags, rows, plot_file)
