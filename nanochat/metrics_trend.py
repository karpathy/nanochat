"""
Trend crumb vs last_metrics.json — pure helpers (no torch).

When a new val_bpb / CORE value is available and a prior exists, print one line
like ``trend: val_bpb 7.470000 → better than prior 8.100000``.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_metrics(path: str | Path) -> dict:
    """Return metrics dict; empty dict if the file is missing/unreadable."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
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
    """
    One enrichment line comparing value to prior.

    Returns None when prior is missing (no trend line).
    """
    if prior is None:
        return None

    if value == prior:
        verdict = "same as prior"
    else:
        improved = value < prior if lower_is_better else value > prior
        verdict = "better than prior" if improved else "worse than prior"

    return f"trend: {name} {value:.6f} → {verdict} {prior:.6f}"


def record_metric(path: str | Path, name: str, value: float) -> None:
    """Merge ``name: value`` into last_metrics.json (creates parent dirs)."""
    p = Path(path)
    data = load_metrics(p)
    data[name] = value
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
