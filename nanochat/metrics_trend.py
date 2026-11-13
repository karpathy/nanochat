"""Trend crumb helpers for val_bpb / CORE vs last_metrics.json.

Pure stdlib — no torch. Callers print the line (rank0 / print0) then record.
"""
from __future__ import annotations

import json
import os


def load_metrics(path: str) -> dict:
    """Load metrics JSON; empty dict if missing or unreadable."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def format_trend(
    name: str,
    value: float,
    prior: float | None,
    *,
    lower_is_better: bool,
) -> str | None:
    """One trend line, or None when there is no prior to compare."""
    if prior is None:
        return None
    if value == prior:
        relation = "same as"
    elif lower_is_better:
        relation = "better than" if value < prior else "worse than"
    else:
        relation = "better than" if value > prior else "worse than"
    return f"trend: {name} {value:.6f} → {relation} prior {prior:.6f}"


def record_metric(path: str, name: str, value: float) -> None:
    """Merge one metric into the JSON file at path (create parent dirs as needed)."""
    metrics = load_metrics(path)
    metrics[name] = float(value)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
