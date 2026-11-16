"""
last_run.json helpers for Phase 1.5 knob-diff enrichment.

Pure Python (no torch). Callable as:
    python -m nanochat.last_run diff  --path ... --profile ... --depth ... \\
        --device-batch-size ... --max-seq-len ...
    python -m nanochat.last_run write --path ... --profile ... --depth ... \\
        --device-batch-size ... --max-seq-len ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Stable compare order for the enrichment line (matches plan AC1 examples).
KNOB_KEYS = ("profile", "depth", "device_batch_size", "max_seq_len")


def load_last_run(path: str | Path) -> dict | None:
    """Load last_run.json; return None if missing or unreadable."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def format_knob_diff(current: dict, previous: dict | None) -> str | None:
    """
    One enrichment line comparing current knobs to previous.

    Returns None when there is no previous run (first run / missing file).
    Otherwise a single line that starts with ``changed:`` and/or contains ``same:``,
    e.g. ``changed: device_batch_size 32→8; same: depth=6,profile=runcpu``.
    """
    if previous is None:
        return None

    changed_parts: list[str] = []
    same_parts: list[str] = []
    for key in KNOB_KEYS:
        cur = _as_str(current.get(key))
        prev = _as_str(previous.get(key))
        if cur == prev:
            same_parts.append(f"{key}={cur}")
        else:
            changed_parts.append(f"{key} {prev}→{cur}")

    pieces: list[str] = []
    if changed_parts:
        pieces.append("changed: " + ", ".join(changed_parts))
    if same_parts:
        pieces.append("same: " + ",".join(same_parts))
    if not pieces:
        # Both empty only if KNOB_KEYS is empty; keep a safe fallback.
        return "same:"
    return "; ".join(pieces)


def write_last_run(path: str | Path, data: dict) -> None:
    """Write last_run.json (creates parent dirs). Adds git_sha + ts if absent."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(data)
    if "git_sha" not in out:
        out["git_sha"] = _git_sha()
    if "ts" not in out:
        out["ts"] = _now_ts()
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or "unknown"
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _knobs_from_args(args: argparse.Namespace) -> dict:
    return {
        "profile": args.profile,
        "depth": args.depth,
        "device_batch_size": args.device_batch_size,
        "max_seq_len": args.max_seq_len,
    }


def _add_knob_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--path", required=True, help="Path to last_run.json")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--device-batch-size", required=True, dest="device_batch_size")
    parser.add_argument("--max-seq-len", required=True, dest="max_seq_len")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m nanochat.last_run")
    sub = parser.add_subparsers(dest="cmd", required=True)

    diff_p = sub.add_parser("diff", help="Print knob-diff line if previous run exists")
    _add_knob_args(diff_p)

    write_p = sub.add_parser("write", help="Write last_run.json with git_sha + ts")
    _add_knob_args(write_p)

    args = parser.parse_args(argv)
    current = _knobs_from_args(args)

    if args.cmd == "diff":
        line = format_knob_diff(current, load_last_run(args.path))
        if line:
            print(line)
        return 0

    if args.cmd == "write":
        write_last_run(args.path, current)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
