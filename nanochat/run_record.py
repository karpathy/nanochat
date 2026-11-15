"""Run record: one folder per training run, the same shape locally and on Modal.

    <run-dir>/
      run.json      identity, config, code revision, host, resources, status, start/end times
      events.jsonl  stage spans (written by runs/_stage_lib.sh) and milestone metrics
                    (written by the trainers through wandb_run.log)

Nothing else is stored. Summaries (durations, throughput, quality) are computed from these two
files when someone asks, so they never go stale.

Standard library only, so the shell scripts can call it without loading torch.

CLI:
    python3 -m nanochat.run_record start <run-dir> --mode local-cpu   # write run.json at launch
    python3 -m nanochat.run_record compare <root>                     # table of every run under root
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from typing import Any

RUN_JSON = "run.json"
EVENTS_JSONL = "events.jsonl"

# Metric points are kept only at milestones: one per SAMPLE_EVERY steps, plus every CORE and
# ChatCORE evaluation (both are rare). A 5000-step run keeps about 10 metric lines.
SAMPLE_EVERY = 500
ALWAYS_KEEP_KEYS = ("core_metric", "chatcore_metric")


def is_milestone(data: dict[str, Any]) -> bool:
    """True when a metric point is worth writing to disk."""
    step = data.get("step")
    if isinstance(step, int) and step % SAMPLE_EVERY == 0:
        return True
    return any(k in data for k in ALWAYS_KEEP_KEYS)


# ─────────────────────────────────────────────────────────────────────────────
# Files


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def append_event(run_dir: str, event: dict[str, Any]) -> None:
    """Append one event as a JSON line."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, EVENTS_JSONL), "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def read_events(run_dir: str) -> list[dict[str, Any]]:
    """Read all events. A partly written last line (run still going, or killed) is skipped."""
    events: list[dict[str, Any]] = []
    try:
        with open(os.path.join(run_dir, EVENTS_JSONL), encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return events


# ─────────────────────────────────────────────────────────────────────────────
# run.json: identity, config, resources, status


def git_revision(repo_dir: str = ".") -> dict[str, Any]:
    """Short commit and whether the working tree has uncommitted changes."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"code_rev": "unknown", "code_dirty": None}
    return {"code_rev": rev, "code_dirty": dirty}


def host_info() -> dict[str, Any]:
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"machine": platform.machine(), "cores": cores}


def start_run(run_dir: str, mode: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write run.json at launch. Calling again keeps identity and merges config."""
    existing = _read_json(os.path.join(run_dir, RUN_JSON))
    code = (
        {"code_rev": os.environ["NANOCHAT_CODE_REV"]}
        if os.environ.get("NANOCHAT_CODE_REV")
        else git_revision()
    )
    run = {
        "run_id": os.path.basename(os.path.normpath(run_dir)),
        "mode": mode,
        "status": "running",
        "started_at": time.time(),
        **code,
        "host": host_info(),
        **existing,
    }
    run["config"] = {**existing.get("config", {}), **(config or {})}
    _write_json(os.path.join(run_dir, RUN_JSON), run)
    return run


def record_span(run_dir: str, stage: str, status: str, dur_s: int, hook_s: int, start: int) -> None:
    """One span per stage: duration, status, and how long its post-stage hook (Modal commit) took."""
    append_event(run_dir, {
        "type": "span", "name": f"stage:{stage}", "status": status,
        "dur_s": dur_s, "hook_s": hook_s, "start": start, "t": time.time(),
    })


def update_run(run_dir: str, config: dict[str, Any] | None = None, **fields: Any) -> None:
    """Merge config values and top-level fields (for example resources) into run.json."""
    path = os.path.join(run_dir, RUN_JSON)
    run = _read_json(path)
    run["config"] = {**run.get("config", {}), **(config or {})}
    run.update(fields)
    _write_json(path, run)


# ─────────────────────────────────────────────────────────────────────────────
# Recording from the trainers (same .log / .finish shape as wandb)


class RunRecorder:
    """Drop-in for DummyWandb / wandb: .log(dict) records a milestone, .finish() closes the run."""

    def __init__(self, run_dir: str, config: dict[str, Any] | None = None) -> None:
        self.run_dir = run_dir
        start_run(run_dir, mode=os.environ.get("NANOCHAT_RUN_MODE", "local"), config=config)

    def log(self, data: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        if is_milestone(data):
            append_event(self.run_dir, {"type": "metric", "t": time.time(), "data": data})

    def finish(self, status: str = "ok") -> None:
        update_run(self.run_dir, status=status, ended_at=time.time())


class TeeLogger:
    """Send each call to the recorder and to the existing backend (wandb or dummy)."""

    def __init__(self, backends: Iterable[Any]) -> None:
        self.backends = list(backends)

    def log(self, data: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        for backend in self.backends:
            backend.log(data, *args, **kwargs)

    def finish(self, *args: Any, **kwargs: Any) -> None:
        for backend in self.backends:
            backend.finish(*args, **kwargs)


def make_run_logger(backend: Any, config: dict[str, Any] | None = None) -> Any:
    """Return the logger a trainer should use.

    NANOCHAT_RUN_DIR set (local runs and Modal runs set it) → record to that folder as well.
    Unset → the backend unchanged, so nothing new is written.
    """
    run_dir = os.environ.get("NANOCHAT_RUN_DIR")
    if not run_dir:
        return backend
    return TeeLogger([RunRecorder(run_dir, config=config), backend])


# ─────────────────────────────────────────────────────────────────────────────
# Summary, computed on demand


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, q in [0, 100]. None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _last(values: list[float]) -> float | None:
    return values[-1] if values else None


def summarize(run_dir: str, now: float | None = None) -> dict[str, Any]:
    """Summary from run.json and events.jsonl. Works on running and killed runs too."""
    run = _read_json(os.path.join(run_dir, RUN_JSON))
    events = read_events(run_dir)
    now = now or time.time()

    spans = [e for e in events if e.get("type") == "span"]
    metrics = [e.get("data", {}) for e in events if e.get("type") == "metric"]

    started = run.get("started_at")
    ended = run.get("ended_at")
    wall = (ended or now) - started if started is not None else None

    def values(key: str) -> list[float]:
        return [m[key] for m in metrics if isinstance(m.get(key), (int, float))]

    steps = [int(m["step"]) for m in metrics if isinstance(m.get("step"), (int, float))]
    dts = values("train/dt")
    val = values("val/bpb")

    return {
        "run_id": run.get("run_id", os.path.basename(os.path.normpath(run_dir))),
        "mode": run.get("mode"),
        "status": run.get("status", "unknown"),
        "code_rev": run.get("code_rev"),
        "code_dirty": run.get("code_dirty"),
        "device": run.get("config", {}).get("device_type") or None,
        "wall_s": round(wall, 1) if wall is not None else None,
        "stages_s": {s["name"].removeprefix("stage:"): s["dur_s"] for s in spans if s["name"].startswith("stage:")},
        "hook_s": round(sum(s.get("hook_s", 0) for s in spans), 1),
        "steps_done": max(steps) if steps else 0,
        "step_time_s_p50": percentile(dts, 50),
        "step_time_s_p95": percentile(dts, 95),
        "tok_per_sec_last": _last(values("train/tok_per_sec")),
        "train_loss_last": _last(values("train/loss")),
        "val_bpb_last": _last(val),
        "val_bpb_min": min(val) if val else None,
        "core_metric_last": _last(values("core_metric")),
        "chatcore_last": _last(values("chatcore_metric")),
    }


def compare_table(run_dirs: list[str]) -> str:
    """One aligned text table of runs, newest first."""
    header = ["run", "mode", "status", "device", "code", "wall", "steps", "s/step p50", "tok/s", "val bpb", "core", "hook"]
    rows = [header]
    for d in run_dirs:
        s = summarize(d)
        rows.append([
            str(s["run_id"]),
            str(s["mode"] or "-"),
            str(s["status"]),
            str(s["device"] or "-"),
            str(s["code_rev"] or "-") + ("*" if s["code_dirty"] else ""),
            _hms(s["wall_s"]) if s["wall_s"] is not None else "-",
            str(s["steps_done"]),
            _fmt(s["step_time_s_p50"], "{:.2f}"),
            _fmt(s["tok_per_sec_last"], "{:,.0f}"),
            _fmt(s["val_bpb_min"], "{:.4f}"),
            _fmt(s["core_metric_last"], "{:.4f}"),
            _hms(s["hook_s"]),
        ])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


def _fmt(value: float | None, pattern: str) -> str:
    return "-" if value is None else pattern.format(value)


def _hms(seconds: float | None) -> str:
    total = max(0, round(seconds or 0))
    return f"{total // 3600}h{total % 3600 // 60:02d}m{total % 60:02d}s"


def find_run_dirs(root: str) -> list[str]:
    """Run folders under root (any folder holding a run.json), newest run id first."""
    if not os.path.isdir(root):
        return []
    dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, RUN_JSON))]
    return sorted(dirs, key=os.path.basename, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI


def replay_commands(run_dir: str) -> list[str]:
    """Return the shell commands needed to replay a run.

    For every run:
      git checkout <code_rev>
      MODE=<mode> bash runs/runmodal.sh

    For local runs (mode contains "local") an extra line is inserted:
      uv sync --extra cpu
    """
    run = _read_json(os.path.join(run_dir, RUN_JSON))
    code_rev = run.get("code_rev") or "unknown"
    mode = run.get("mode") or "local"
    lines: list[str] = [f"git checkout {code_rev}"]
    if "local" in mode:
        lines.append("uv sync --extra cpu")
    lines.append(f"MODE={mode} bash runs/runmodal.sh")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m nanochat.run_record")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_start = sub.add_parser("start", help="write run.json at launch")
    p_start.add_argument("run_dir")
    p_start.add_argument("--mode", required=True)
    p_span = sub.add_parser("span", help="record one stage span (called by runs/_stage_lib.sh)")
    p_span.add_argument("run_dir")
    p_span.add_argument("stage")
    p_span.add_argument("status")
    p_span.add_argument("dur_s", type=int)
    p_span.add_argument("hook_s", type=int)
    p_span.add_argument("start", type=int)
    p_cmp = sub.add_parser("compare", help="table of every run under a root")
    p_cmp.add_argument("root")
    p_replay = sub.add_parser("replay", help="print commands to reproduce a run")
    p_replay.add_argument("run_dir")
    args = parser.parse_args(argv)

    if args.cmd == "start":
        start_run(args.run_dir, mode=args.mode)
        return 0
    if args.cmd == "span":
        record_span(args.run_dir, args.stage, args.status, args.dur_s, args.hook_s, args.start)
        return 0
    if args.cmd == "replay":
        for line in replay_commands(args.run_dir):
            print(line)
        return 0
    dirs = find_run_dirs(args.root)
    if not dirs:
        print(f"no runs under {args.root}", file=sys.stderr)
        return 1
    print(compare_table(dirs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
