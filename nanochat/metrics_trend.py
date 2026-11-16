"""Trend crumb helpers for val_bpb / CORE vs last_metrics.json.

Pure stdlib — no torch. Prefer ``emit_metric_trend`` from train/eval so
enrichment I/O can never abort a run.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable


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


def _coerce_float(value: object) -> float | None:
    """Best-effort float; None if missing or not numeric."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def format_trend(
    name: str,
    value: float,
    prior: float | None,
    *,
    lower_is_better: bool,
) -> str | None:
    """One trend line, or None when there is no usable prior to compare."""
    prior_f = _coerce_float(prior)
    if prior_f is None:
        return None
    value_f = float(value)
    if value_f == prior_f:
        relation = "same as"
    elif lower_is_better:
        relation = "better than" if value_f < prior_f else "worse than"
    else:
        relation = "better than" if value_f > prior_f else "worse than"
    return f"trend: {name} {value_f:.6f} → {relation} prior {prior_f:.6f}"


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


def emit_metric_trend(
    path: str,
    name: str,
    value: float,
    *,
    lower_is_better: bool,
    print_fn: Callable[[str], None],
) -> None:
    """Print trend (if prior) then record — never raises (enrichment fail-soft)."""
    try:
        prior = load_metrics(path).get(name)
        line = format_trend(name, float(value), prior, lower_is_better=lower_is_better)
        if line is not None:
            print_fn(line)
        record_metric(path, name, float(value))
    except Exception:
        # Disk full, permission, corrupt prior coercion edge — never kill train/eval.
        return
