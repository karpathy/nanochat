from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path


DEFAULT_MLX_LOG_DIR = "runs/mlx_logs"


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-dir",
        type=str,
        default=DEFAULT_MLX_LOG_DIR,
        help=f"Directory for timestamped JSON summaries (default: {DEFAULT_MLX_LOG_DIR}). Use '' to disable.",
    )
    parser.add_argument(
        "--log-prefix",
        type=str,
        default=None,
        help="Optional filename prefix for the JSON summary.",
    )


def write_summary_log(summary: dict[str, object], *, log_dir: str | None, script_name: str, log_prefix: str | None, depth: int, input_mode: str) -> str | None:
    if log_dir is None:
        return None
    normalized_log_dir = log_dir.strip()
    if normalized_log_dir == "":
        return None

    output_dir = Path(normalized_log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = log_prefix.strip() if log_prefix is not None and log_prefix.strip() else script_name
    filename = f"{prefix}_d{depth}_{input_mode}_{timestamp}.json"
    log_path = output_dir / filename

    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return os.path.abspath(log_path)