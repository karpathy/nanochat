"""
Knob diff vs last_run.json — pure helpers (no torch).

Stores the last successful script run knobs so the next invocation can print
one enrichment line: `changed: ...; same: ...` (or skip on first run).

CLI (for shell):
  python -m nanochat.last_run diff --path ... --profile ... --depth ... \\
      --device-batch-size ... --max-seq-len ...
  python -m nanochat.last_run write --path ... --profile ... --depth ... \\
      --device-batch-size ... --max-seq-len ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

# Knobs compared in the enrichment line (git_sha / ts are metadata only).
KNOB_KEYS = ("profile", "depth", "device_batch_size", "max_seq_len")


def load_last_run(path: str | Path) -> dict | None:
    """Return the previous run dict, or None if the file is missing/unreadable."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def format_knob_diff(current: dict, previous: dict | None) -> str | None:
    """
    One enrichment line comparing current knobs to previous.

    Returns None when previous is missing (first run — skip silently).
    Otherwise e.g. ``changed: device_batch_size 32→8; same: depth=6,profile=runcpu``.
    """
    if previous is None:
        return None

    changed_parts: list[str] = []
    same_parts: list[str] = []
    for key in KNOB_KEYS:
        if key not in current:
            continue
        cur = current[key]
        if key not in previous or previous[key] != cur:
            prev_val = previous.get(key, "?")
            changed_parts.append(f"{key} {prev_val}→{cur}")
        else:
            same_parts.append(f"{key}={cur}")

    segments: list[str] = []
    if changed_parts:
        segments.append("changed: " + ",".join(changed_parts))
    if same_parts:
        segments.append("same: " + ",".join(same_parts))
    if not segments:
        return None
    return "; ".join(segments)


def write_last_run(path: str | Path, data: dict) -> None:
    """Write last_run.json (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _knobs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "profile": args.profile,
        "depth": args.depth,
        "device_batch_size": args.device_batch_size,
        "max_seq_len": args.max_seq_len,
    }


def _add_knob_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--path", required=True, help="Path to last_run.json")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--depth", required=True, type=int)
    parser.add_argument("--device-batch-size", required=True, type=int, dest="device_batch_size")
    parser.add_argument("--max-seq-len", required=True, type=int, dest="max_seq_len")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nanochat.last_run")
    sub = parser.add_subparsers(dest="cmd", required=True)

    diff_p = sub.add_parser("diff", help="Print knob diff line or nothing")
    _add_knob_args(diff_p)

    write_p = sub.add_parser("write", help="Write last_run.json with knobs + git_sha + ts")
    _add_knob_args(write_p)

    args = parser.parse_args(argv)
    knobs = _knobs_from_args(args)

    if args.cmd == "diff":
        line = format_knob_diff(knobs, load_last_run(args.path))
        if line:
            print(line)
        return 0

    if args.cmd == "write":
        payload = dict(knobs)
        payload["git_sha"] = _git_sha()
        payload["ts"] = int(time.time())
        write_last_run(args.path, payload)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
