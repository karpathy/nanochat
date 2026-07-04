"""
Aggregate an experiment's stage records into the cost-performance curve.

Walks the model directories of the active experiment, parses the `summary`
records out of each stage's log (see nanochat/logfmt.py), and joins them into
one row per model:
- prints a compact human-readable table of the headline columns
- writes the full join to the experiment's curve.log: one `model` record per
  model, fields namespaced by stage (e.g. base.val_bpb, chat.chatcore)

The curve is the product of an experiment: quality, speed and cost of every
model in the ladder, in one file. Compare two experiments by comparing their
curves. To get a dataframe:

    import pandas as pd
    from nanochat.logfmt import parse_records
    df = pd.DataFrame(parse_records("curve.log", tag="model"))

Usage: python -m scripts.curve    (reads $NANOCHAT_EXPERIMENT)
"""

import os
import re

from nanochat.common import get_experiment_dir, get_experiment_name
from nanochat.logfmt import format_record, parse_records

# stage logs to join, in curve column order
STAGE_LOGS = [
    ("base", "base_train.log"),
    ("infer", "infer_bench.log"),
    ("sft", "sft.log"),
    ("chat", "chat_eval.log"),
]

def read_summary(log_path):
    """The last `summary` record of a stage log (or empty dict if absent)."""
    if not os.path.exists(log_path):
        return {}
    records = parse_records(log_path, tag="summary")
    if not records:
        return {}
    summary = records[-1]
    summary.pop("tag", None)
    return summary

def model_sort_key(model_tag):
    """Sort d<depth> tags numerically, anything else after them alphabetically."""
    match = re.fullmatch(r"d(\d+)", model_tag)
    if match:
        return (0, int(match.group(1)), model_tag)
    return (1, 0, model_tag)

def build_row(model_dir, model_tag):
    """Join the stage summaries of one model directory into a flat row."""
    row = {"model_tag": model_tag}
    for stage, log_name in STAGE_LOGS:
        summary = read_summary(os.path.join(model_dir, log_name))
        summary.pop("model_tag", None) # identity, already in the row
        if stage == "base" and "depth" in summary:
            row["depth"] = summary.pop("depth")
        for key, value in summary.items():
            row[f"{stage}.{key}"] = value
    # a few headline throughput numbers from the inference bench sweep
    bench = parse_records(os.path.join(model_dir, STAGE_LOGS[1][1]), tag="bench") \
        if os.path.exists(os.path.join(model_dir, STAGE_LOGS[1][1])) else []
    if bench:
        bench = sorted(bench, key=lambda r: r["batch"])
        row["infer.tok_per_sec_bs1"] = bench[0]["tok_per_sec"]
        row[f"infer.tok_per_sec_bs{bench[-1]['batch']}"] = bench[-1]["tok_per_sec"]
    return row

# columns for the compact stdout table (the full set goes to curve.log)
TABLE_COLUMNS = [
    "model_tag", "base.num_params", "base.tokens_trained", "base.eflops", "base.train_time_sec",
    "base.val_bpb", "base.core", "infer.tok_per_sec_bs1", "sft.val_bpb", "chat.chatcore",
]

if __name__ == "__main__":
    experiment_dir = get_experiment_dir()
    model_tags = sorted(
        (d for d in os.listdir(experiment_dir)
         if os.path.isdir(os.path.join(experiment_dir, d))
         and any(os.path.exists(os.path.join(experiment_dir, d, log)) for _, log in STAGE_LOGS)),
        key=model_sort_key,
    )
    assert model_tags, f"No model directories with stage logs found in {experiment_dir}"
    rows = [build_row(os.path.join(experiment_dir, tag), tag) for tag in model_tags]

    # write the full curve as `model` records in the log grammar, one line per model
    curve_path = os.path.join(experiment_dir, "curve.log")
    with open(curve_path, "w") as f:
        for row in rows:
            f.write(format_record("model", **row) + "\n")

    # print the compact table
    table_columns = [c for c in TABLE_COLUMNS if any(c in row for row in rows)]
    widths = {c: max(len(c), *(len(str(row.get(c, ""))) for row in rows)) for c in table_columns}
    print(f"experiment {get_experiment_name()}: {len(rows)} models")
    print("  ".join(c.ljust(widths[c]) for c in table_columns))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in table_columns))
    print(f"full curve written to {curve_path}")
