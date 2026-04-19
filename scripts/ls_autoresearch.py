#!/usr/bin/env python3
"""Restartable LS-only autoresearch harness for the LS-enabled nanochat fork.

This script adapts the tiny upstream autoresearch workflow to this repo:

- the agent can edit multiple LS-related files instead of a single train.py
- experiments run remotely on the 4090 box over ssh/rsync
- run state, logs, and best-known metrics are persisted in autoresearch_runs/<tag>/
- the search can be resumed later with the same tag without losing context

The script deliberately does not assume one fixed benchmark. A run tag stores a
JSON config with a small suite of named experiments. Each `run` executes the
suite for the current git commit, logs every result, and updates the best-known
primary metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
RUNS_ROOT = REPO_ROOT / "autoresearch_runs"

STATUS_KEEP = "keep"
STATUS_DISCARD = "discard"
STATUS_CRASH = "crash"

TSV_HEADER = [
    "timestamp",
    "commit",
    "experiment",
    "metric_name",
    "metric_value",
    "memory_gb",
    "status",
    "description",
    "log_path",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def expand_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def run_cmd(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        check=check,
    )


def git(args: list[str], cwd: Path = REPO_ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_cmd(["git", *args], cwd=cwd, check=check)


def repo_root() -> Path:
    out = git(["rev-parse", "--show-toplevel"]).stdout.strip()
    return Path(out).resolve()


def current_branch() -> str:
    return git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def current_commit(short: bool = False) -> str:
    args = ["rev-parse", "--short", "HEAD"] if short else ["rev-parse", "HEAD"]
    return git(args).stdout.strip()


def working_tree_clean() -> bool:
    return git(["status", "--porcelain"]).stdout.strip() == ""


def ensure_branch(tag: str) -> str:
    branch = f"autoresearch/{tag}"
    existing = git(["branch", "--list", branch]).stdout.strip()
    if existing:
        git(["checkout", branch])
    else:
        git(["checkout", "-b", branch])
    return branch


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def results_tsv_path(run_dir: Path) -> Path:
    return run_dir / "results.tsv"


def config_path(run_dir: Path) -> Path:
    return run_dir / "config.json"


def state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def summary_path(run_dir: Path, run_id: str) -> Path:
    return run_dir / "runs" / f"{run_id}.json"


def latest_summary_path(run_dir: Path) -> Path:
    return run_dir / "latest_summary.json"


def local_remote_runs_dir(run_dir: Path) -> Path:
    return run_dir / "remote_runs"


def local_remote_run_dir(run_dir: Path, run_id: str) -> Path:
    return local_remote_runs_dir(run_dir) / run_id


def notebook_index_path(run_dir: Path) -> Path:
    return run_dir / "notebook_index.json"


def ensure_results_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(TSV_HEADER)


def append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([row.get(field, "") for field in TSV_HEADER])


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def safe_tag(tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
        raise SystemExit(f"Invalid tag {tag!r}. Use only letters, numbers, '.', '_', '-'.")
    return tag


def default_remote_repo(tag: str) -> str:
    return f"/tmp/nanochat-ls-autoresearch/{tag}/repo"


def default_remote_cache(tag: str) -> str:
    return f"/tmp/nanochat-ls-autoresearch/{tag}/cache"


def default_remote_results(tag: str) -> str:
    return f"/tmp/nanochat-ls-autoresearch/{tag}/results"


def preset_smoke(tag: str) -> dict[str, Any]:
    return {
        "primary_experiment": "lsres_k4_curve",
        "experiments": [
            {
                "name": "lslinear_curve",
                "kind": "compare_loss_curves",
                "description": "LSLinear deterministic loss curve smoke",
                "metric_name": "final_loss",
                "extract": {
                    "comparison": "transformer_dense_vs_lslinear",
                    "variant": "lslinear",
                },
                "command": [
                    "{python}",
                    "-m",
                    "scripts.compare_loss_curves",
                    "--device",
                    "cuda",
                    "--depth",
                    "12",
                    "--model-dim",
                    "768",
                    "--head-dim",
                    "64",
                    "--seq-len",
                    "256",
                    "--batch-size",
                    "2",
                    "--vocab-size",
                    "32768",
                    "--steps",
                    "100",
                    "--log-every",
                    "20",
                    "--lr",
                    "0.0005",
                    "--ls-num-blocks",
                    "16",
                    "--ls-rank",
                    "128",
                    "--lsrec-n-iter",
                    "4",
                    "--lsrec-n-mem",
                    "128",
                    "--lsrec-log-dt-init",
                    "-3.5",
                    "--data-mode",
                    "increment",
                ],
            },
            {
                "name": "lsres_k4_curve",
                "kind": "compare_loss_curves",
                "description": "LSRes K4/nmem128 deterministic loss curve",
                "metric_name": "final_loss",
                "extract": {
                    "comparison": "transformer_vs_lsrecurrent_scan_driven",
                    "variant": "lsrecurrent-scan-driven",
                },
                "command": [
                    "{python}",
                    "-m",
                    "scripts.compare_loss_curves",
                    "--device",
                    "cuda",
                    "--depth",
                    "12",
                    "--model-dim",
                    "768",
                    "--head-dim",
                    "64",
                    "--seq-len",
                    "256",
                    "--batch-size",
                    "2",
                    "--vocab-size",
                    "32768",
                    "--steps",
                    "100",
                    "--log-every",
                    "20",
                    "--lr",
                    "0.0005",
                    "--ls-num-blocks",
                    "16",
                    "--ls-rank",
                    "128",
                    "--lsrec-n-iter",
                    "4",
                    "--lsrec-n-mem",
                    "128",
                    "--lsrec-log-dt-init",
                    "-3.5",
                    "--data-mode",
                    "increment",
                ],
            },
            {
                "name": "lsres_k8_curve",
                "kind": "compare_loss_curves",
                "description": "LSRes K8/nmem64 deterministic loss curve",
                "metric_name": "final_loss",
                "extract": {
                    "comparison": "transformer_vs_lsrecurrent_scan_driven",
                    "variant": "lsrecurrent-scan-driven",
                },
                "command": [
                    "{python}",
                    "-m",
                    "scripts.compare_loss_curves",
                    "--device",
                    "cuda",
                    "--depth",
                    "12",
                    "--model-dim",
                    "768",
                    "--head-dim",
                    "64",
                    "--seq-len",
                    "256",
                    "--batch-size",
                    "2",
                    "--vocab-size",
                    "32768",
                    "--steps",
                    "100",
                    "--log-every",
                    "20",
                    "--lr",
                    "0.0005",
                    "--ls-num-blocks",
                    "16",
                    "--ls-rank",
                    "128",
                    "--lsrec-n-iter",
                    "8",
                    "--lsrec-n-mem",
                    "64",
                    "--lsrec-log-dt-init",
                    "-3.5",
                    "--data-mode",
                    "increment",
                ],
            },
        ],
        "notes": f"Smoke preset for LS-only autoresearch tag {tag}.",
    }


def preset_short_train(tag: str) -> dict[str, Any]:
    common = [
        "{python}",
        "-m",
        "scripts.base_train",
        "--run",
        "dummy",
        "--device-type",
        "cuda",
        "--depth",
        "12",
        "--aspect-ratio",
        "64",
        "--head-dim",
        "64",
        "--max-seq-len",
        "512",
        "--window-pattern",
        "L",
        "--device-batch-size",
        "2",
        "--total-batch-size",
        "65536",
        "--num-iterations",
        "100",
        "--eval-every",
        "50",
        "--eval-tokens",
        "65536",
        "--core-metric-every",
        "-1",
        "--sample-every",
        "-1",
        "--save-every",
        "-1",
        "--model-tag",
        f"autoresearch-{tag}",
    ]
    return {
        "primary_experiment": "lsres_k4_train",
        "experiments": [
            {
                "name": "lsres_k4_train",
                "kind": "base_train",
                "description": "Short real-data LSRes K4/nmem128 dt-3.25 train",
                "metric_name": "min_val_bpb",
                "command": common
                + [
                    "--architecture",
                    "lsrecurrent-scan-driven",
                    "--ls-num-blocks",
                    "16",
                    "--ls-rank",
                    "128",
                    "--lsrec-n-iter",
                    "4",
                    "--lsrec-n-mem",
                    "128",
                    "--lsrec-log-dt-init",
                    "-3.25",
                ],
            },
            {
                "name": "lsres_k8_train",
                "kind": "base_train",
                "description": "Short real-data LSRes K8/nmem64 train",
                "metric_name": "min_val_bpb",
                "command": common
                + [
                    "--architecture",
                    "lsrecurrent-scan-driven",
                    "--ls-num-blocks",
                    "16",
                    "--ls-rank",
                    "128",
                    "--lsrec-n-iter",
                    "8",
                    "--lsrec-n-mem",
                    "64",
                    "--lsrec-log-dt-init",
                    "-3.5",
                ],
            },
        ],
        "notes": f"Short-train preset for LS-only autoresearch tag {tag}.",
    }


PRESET_FACTORIES = {
    "smoke": preset_smoke,
    "short_train": preset_short_train,
}


def make_config(tag: str, preset: str) -> dict[str, Any]:
    if preset not in PRESET_FACTORIES:
        raise SystemExit(f"Unknown preset {preset!r}.")
    config = PRESET_FACTORIES[preset](tag)
    config.update(
        {
            "tag": tag,
            "preset": preset,
            "repo_root": str(repo_root()),
            "remote_host": "jrmolnia@paffenroth23-1.dyn.wpi.edu",
            "ssh_key": "~/.ssh/keys/id_WPI_rsa",
            "remote_repo_dir": default_remote_repo(tag),
            "remote_cache_dir": default_remote_cache(tag),
            "remote_results_dir": default_remote_results(tag),
            "remote_command_prefix": ["/home/jrmolnia/.local/bin/uv", "run", "--extra", "gpu"],
            "remote_python": "python",
            "bootstrap_dataset_train_shards": 1,
            "branch": current_branch(),
            "sync_excludes": [
                ".git",
                ".venv",
                "__pycache__",
                "autoresearch_runs",
                "wandb",
                "report.md",
            ],
            "artifact_paths": [
                "{remote_repo_dir}/results",
                "{remote_cache_dir}",
            ],
        }
    )
    return config


def make_state(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "tag": config["tag"],
        "branch": config["branch"],
        "created_at": utc_now(),
        "baseline_commit": current_commit(),
        "best_commit": None,
        "best_metrics": {},
        "last_run_id": None,
        "run_count": 0,
    }


def init_run(args: argparse.Namespace) -> None:
    tag = safe_tag(args.tag)
    if args.create_branch:
        branch = ensure_branch(tag)
    else:
        branch = current_branch()
    run_dir = RUNS_ROOT / tag
    if run_dir.exists() and not args.force:
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "runs").mkdir(exist_ok=True)
    config = make_config(tag, args.preset)
    config["branch"] = branch
    state = make_state(config)
    dump_json(config_path(run_dir), config)
    dump_json(state_path(run_dir), state)
    ensure_results_file(results_tsv_path(run_dir))
    print(f"Initialized LS autoresearch tag {tag} at {run_dir}")
    print(f"Branch: {branch}")
    print(f"Preset: {args.preset}")
    print(f"Primary experiment: {config['primary_experiment']}")


def load_run(tag: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    run_dir = RUNS_ROOT / safe_tag(tag)
    if not run_dir.exists():
        raise SystemExit(f"Unknown tag {tag!r}. Run init first.")
    return run_dir, load_json(config_path(run_dir)), load_json(state_path(run_dir))


def replace_placeholders(parts: list[str], config: dict[str, Any], run_id: str, experiment_name: str) -> list[str]:
    out = []
    for part in parts:
        if part == "{python}":
            out.extend(config.get("remote_command_prefix", []))
            out.append(config["remote_python"])
            continue
        value = (
            part.replace("{tag}", config["tag"])
            .replace("{run_id}", run_id)
            .replace("{experiment}", experiment_name)
        )
        out.append(value)
    return out


def replace_template(value: str, config: dict[str, Any], run_id: str, experiment_name: str) -> str:
    replacements = {
        "{python}": config["remote_python"],
        "{tag}": config["tag"],
        "{run_id}": run_id,
        "{experiment}": experiment_name,
        "{remote_repo_dir}": config["remote_repo_dir"],
        "{remote_cache_dir}": config["remote_cache_dir"],
        "{remote_results_dir}": config.get("remote_results_dir", ""),
    }
    out = value
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def rsync_repo(config: dict[str, Any]) -> None:
    excludes: list[str] = config.get("sync_excludes", [])
    ssh_key = expand_path(config["ssh_key"])
    remote_repo_dir = config["remote_repo_dir"].rstrip("/")
    remote_parent = str(Path(remote_repo_dir).parent)
    subprocess.run(
        [
            "ssh",
            "-i",
            ssh_key,
            config["remote_host"],
            f"mkdir -p {shlex.quote(remote_parent)}",
        ],
        check=True,
    )
    cmd = ["rsync", "-az", "--delete"]
    for entry in excludes:
        cmd.extend(["--exclude", entry])
    cmd.extend(
        [
            "-e",
            f"ssh -i {ssh_key}",
            f"{config['repo_root'].rstrip('/')}/",
            f"{config['remote_host']}:{remote_repo_dir}/",
        ]
    )
    subprocess.run(cmd, check=True)


def ssh_capture(config: dict[str, Any], remote_command: str, log_path: Path) -> int:
    ssh_key = expand_path(config["ssh_key"])
    ssh_cmd = [
        "ssh",
        "-i",
        ssh_key,
        config["remote_host"],
        remote_command,
    ]
    with log_path.open("w") as handle:
        proc = subprocess.run(ssh_cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    return proc.returncode


def ssh_run(config: dict[str, Any], remote_command: str) -> subprocess.CompletedProcess[str]:
    ssh_key = expand_path(config["ssh_key"])
    return subprocess.run(
        [
            "ssh",
            "-i",
            ssh_key,
            config["remote_host"],
            remote_command,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def ensure_remote_dataset(config: dict[str, Any]) -> None:
    train_shards = int(config.get("bootstrap_dataset_train_shards", 0))
    if train_shards <= 0:
        return
    remote_base = shlex.quote(config["remote_cache_dir"])
    remote_repo = shlex.quote(config["remote_repo_dir"])
    prefix = shell_join(config.get("remote_command_prefix", []))
    remote_cmd = (
        f"cd {remote_repo} && "
        f"export PYTHONPATH=. && "
        f"export NANOCHAT_BASE_DIR={remote_base} && "
        f"if [ ! -f {remote_base}/base_data_climbmix/shard_00000.parquet ] || "
        f"[ ! -f {remote_base}/base_data_climbmix/shard_06542.parquet ]; then "
        f"{prefix} {shlex.quote(config['remote_python'])} -m nanochat.dataset -n {train_shards} -w 1; "
        f"fi"
    )
    proc = ssh_run(config, remote_cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "failed to prepare remote dataset")


def rsync_remote_paths(
    config: dict[str, Any],
    remote_paths: list[str],
    local_dest: Path,
    extra_excludes: list[str] | None = None,
) -> list[str]:
    ssh_key = expand_path(config["ssh_key"])
    local_dest.mkdir(parents=True, exist_ok=True)
    fetched: list[str] = []
    excludes = extra_excludes or []
    for remote_path in remote_paths:
        relative_name = Path(remote_path.rstrip("/")).name or "artifacts"
        target = local_dest / relative_name
        cmd = ["rsync", "-az"]
        for entry in excludes:
            cmd.extend(["--exclude", entry])
        cmd.extend(
            [
                "-e",
                f"ssh -i {ssh_key}",
                f"{config['remote_host']}:{remote_path.rstrip('/')}/",
                f"{str(target)}/",
            ]
        )
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode == 0:
            fetched.append(str(target))
            continue
        missing_markers = (
            "No such file or directory",
            "link_stat",
            "No such file",
            "failed to connect",
        )
        if any(marker in proc.stderr for marker in missing_markers):
            continue
        raise RuntimeError(f"Failed to rsync {remote_path}: {proc.stderr.strip() or proc.stdout.strip()}")
    return fetched


def refresh_latest_links(run_dir: Path, run_id: str) -> dict[str, str]:
    latest_dir = run_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "summary": summary_path(run_dir, run_id),
        "remote_run_dir": local_remote_run_dir(run_dir, run_id),
    }
    resolved: dict[str, str] = {}
    for name, source in targets.items():
        destination = latest_dir / name
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if source.exists():
            destination.symlink_to(os.path.relpath(source, latest_dir))
            resolved[name] = str(destination)
    latest_summary = latest_summary_path(run_dir)
    if latest_summary.exists() or latest_summary.is_symlink():
        latest_summary.unlink()
    latest_summary.symlink_to(os.path.relpath(summary_path(run_dir, run_id), run_dir))
    resolved["latest_summary"] = str(latest_summary)
    return resolved


def update_notebook_index(
    run_dir: Path,
    run_id: str,
    commit_short: str,
    suite_results: list[dict[str, Any]],
    fetched_artifacts: list[str],
) -> dict[str, Any]:
    index = {
        "tag": run_dir.name,
        "results_tsv": str(results_tsv_path(run_dir)),
        "latest_summary": str(latest_summary_path(run_dir)),
        "latest_remote_run_dir": str(local_remote_run_dir(run_dir, run_id)),
        "runs_dir": str(run_dir / "runs"),
        "logs_dir": str(run_dir / "logs"),
        "remote_runs_dir": str(local_remote_runs_dir(run_dir)),
        "last_run": {
            "run_id": run_id,
            "commit": commit_short,
            "summary_path": str(summary_path(run_dir, run_id)),
            "fetched_artifacts": fetched_artifacts,
            "logs": [row["log_path"] for row in suite_results],
        },
    }
    dump_json(notebook_index_path(run_dir), index)
    return index


def parse_json_from_log(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in log")
    return json.loads(text[start : end + 1])


def parse_compare_loss_curves(log_text: str, experiment: dict[str, Any]) -> dict[str, Any]:
    payload = parse_json_from_log(log_text)
    target_comparison = experiment["extract"]["comparison"]
    target_variant = experiment["extract"]["variant"]
    for comparison in payload["comparisons"]:
        if comparison["comparison"] != target_comparison:
            continue
        for result in comparison["results"]:
            if result["name"] == target_variant:
                return {
                    "metric_name": experiment["metric_name"],
                    "metric_value": float(result["final_loss"]),
                    "memory_gb": "",
                    "status": STATUS_KEEP,
                }
    raise ValueError(f"Could not find {target_comparison}/{target_variant} in compare_loss_curves output")


MIN_VAL_RE = re.compile(r"Minimum validation bpb:\s*([0-9.]+)")
PEAK_MEM_RE = re.compile(r"Peak memory usage:\s*([0-9.]+)MiB")


def parse_base_train(log_text: str, experiment: dict[str, Any]) -> dict[str, Any]:
    min_val = MIN_VAL_RE.search(log_text)
    peak_mem = PEAK_MEM_RE.search(log_text)
    if min_val is None:
        raise ValueError("Could not find minimum validation bpb in base_train log")
    metric_value = float(min_val.group(1))
    memory_gb = ""
    if peak_mem is not None:
        memory_gb = f"{float(peak_mem.group(1)) / 1024.0:.1f}"
    return {
        "metric_name": experiment["metric_name"],
        "metric_value": metric_value,
        "memory_gb": memory_gb,
        "status": STATUS_KEEP,
    }


PARSERS = {
    "compare_loss_curves": parse_compare_loss_curves,
    "base_train": parse_base_train,
}


def metric_better(kind: str, metric_name: str, new_value: float, old_value: float | None) -> bool:
    if old_value is None:
        return True
    # Current metrics are all loss-like: lower is better.
    return new_value < old_value


def run_suite(args: argparse.Namespace) -> None:
    run_dir, config, state = load_run(args.tag)
    if not args.allow_dirty and not working_tree_clean():
        raise SystemExit("Working tree is dirty. Commit or pass --allow-dirty.")
    commit_full = current_commit()
    commit_short = current_commit(short=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rsync_repo(config)
    ensure_remote_dataset(config)
    remote_run_dir = local_remote_run_dir(run_dir, run_id)
    remote_run_dir.mkdir(parents=True, exist_ok=True)

    suite_results: list[dict[str, Any]] = []
    logs_dir = run_dir / "logs"
    for experiment in config["experiments"]:
        exp_name = experiment["name"]
        log_path = logs_dir / f"{run_id}_{commit_short}_{exp_name}.log"
        remote_cmd_parts = replace_placeholders(experiment["command"], config, run_id, exp_name)
        remote_env = (
            f"cd {shlex.quote(config['remote_repo_dir'])} && "
            f"export PYTHONPATH=. && "
            f"export NANOCHAT_BASE_DIR={shlex.quote(config['remote_cache_dir'])} && "
            f"export AUTORESEARCH_TAG={shlex.quote(config['tag'])} && "
            f"export AUTORESEARCH_RUN_ID={shlex.quote(run_id)} && "
            f"export AUTORESEARCH_EXPERIMENT={shlex.quote(exp_name)} && "
            f"export AUTORESEARCH_REMOTE_RESULTS_DIR={shlex.quote(config.get('remote_results_dir', ''))} && "
            f"{shell_join(remote_cmd_parts)}"
        )
        returncode = ssh_capture(config, remote_env, log_path)
        result = {
            "timestamp": utc_now(),
            "commit": commit_short,
            "experiment": exp_name,
            "description": args.description or experiment["description"],
            "log_path": str(log_path),
            "returncode": returncode,
        }
        try:
            log_text = log_path.read_text()
            if returncode != 0:
                raise ValueError(f"Remote command failed with exit code {returncode}")
            parser = PARSERS[experiment["kind"]]
            parsed = parser(log_text, experiment)
            result.update(parsed)
        except Exception as exc:  # noqa: BLE001
            result.update(
                {
                    "metric_name": experiment["metric_name"],
                    "metric_value": "",
                    "memory_gb": "",
                    "status": STATUS_CRASH,
                    "error": str(exc),
                }
            )
        suite_results.append(result)

    artifact_templates: list[str] = list(config.get("artifact_paths", []))
    artifact_templates.extend(
        [
            "{remote_results_dir}",
            "{remote_repo_dir}/results",
        ]
    )
    artifact_paths = []
    for template in artifact_templates:
        rendered = replace_template(template, config, run_id, "suite").rstrip("/")
        if rendered and rendered not in artifact_paths:
            artifact_paths.append(rendered)
    fetched_artifacts = rsync_remote_paths(
        config,
        artifact_paths,
        remote_run_dir,
        extra_excludes=["*.pt", "*.bin", "*.safetensors", "*.ckpt"],
    )

    primary_name = config["primary_experiment"]
    primary_result = next((row for row in suite_results if row["experiment"] == primary_name), None)
    primary_experiment = next((row for row in config["experiments"] if row["name"] == primary_name), None)
    if primary_result is None:
        raise SystemExit(f"Primary experiment {primary_name!r} missing from suite.")
    if primary_experiment is None:
        raise SystemExit(f"Primary experiment {primary_name!r} missing from config.")

    best_so_far = state["best_metrics"].get(primary_name)
    improved = False
    if primary_result["status"] != STATUS_CRASH and primary_result["metric_value"] != "":
        improved = metric_better(
            primary_experiment["kind"],
            str(primary_result["metric_name"]),
            float(primary_result["metric_value"]),
            None if best_so_far is None else float(best_so_far),
        )

    suite_status = STATUS_KEEP if improved else STATUS_DISCARD
    if primary_result["status"] == STATUS_CRASH:
        suite_status = STATUS_CRASH

    for row in suite_results:
        row["status"] = STATUS_CRASH if row["status"] == STATUS_CRASH else suite_status
        append_result(results_tsv_path(run_dir), row)

    if improved:
        state["best_commit"] = commit_full
        for row in suite_results:
            if row["status"] != STATUS_CRASH and row["metric_value"] != "":
                state["best_metrics"][row["experiment"]] = row["metric_value"]

    state["last_run_id"] = run_id
    state["run_count"] += 1
    dump_json(state_path(run_dir), state)
    summary = {
        "tag": config["tag"],
        "run_id": run_id,
        "commit": commit_full,
        "commit_short": commit_short,
        "improved": improved,
        "suite_status": suite_status,
        "results": suite_results,
        "local_paths": {
            "results_tsv": str(results_tsv_path(run_dir)),
            "logs_dir": str(logs_dir),
            "remote_run_dir": str(remote_run_dir),
        },
        "remote_sync": {
            "remote_host": config["remote_host"],
            "artifact_paths": artifact_paths,
            "fetched_artifacts": fetched_artifacts,
        },
    }
    dump_json(summary_path(run_dir, run_id), summary)
    latest_links = refresh_latest_links(run_dir, run_id)
    notebook_index = update_notebook_index(run_dir, run_id, commit_short, suite_results, fetched_artifacts)

    print(f"Run {run_id} for {args.tag}: {suite_status}")
    print(f"Commit: {commit_short}")
    for row in suite_results:
        metric = row["metric_value"] if row["metric_value"] != "" else "n/a"
        mem = row["memory_gb"] if row["memory_gb"] != "" else "n/a"
        print(f"  {row['experiment']}: {row['status']} metric={metric} mem_gb={mem}")
    if improved:
        print(f"Primary experiment improved: {primary_name}")
    else:
        print(f"No primary improvement over current best for {primary_name}")
    print(f"Local summary: {summary_path(run_dir, run_id)}")
    print(f"Notebook index: {notebook_index_path(run_dir)}")
    print(f"Latest summary link: {latest_links.get('latest_summary', 'n/a')}")
    print(f"Fetched artifact roots: {', '.join(fetched_artifacts) if fetched_artifacts else 'none'}")


def show_status(args: argparse.Namespace) -> None:
    run_dir, config, state = load_run(args.tag)
    print(f"Tag: {config['tag']}")
    print(f"Preset: {config['preset']}")
    print(f"Branch: {config['branch']}")
    print(f"Run dir: {run_dir}")
    print(f"Primary experiment: {config['primary_experiment']}")
    print(f"Baseline commit: {state['baseline_commit']}")
    print(f"Best commit: {state['best_commit'] or 'none'}")
    print(f"Runs completed: {state['run_count']}")
    print(f"Results TSV: {results_tsv_path(run_dir)}")
    print(f"Notebook index: {notebook_index_path(run_dir)}")
    print(f"Latest summary: {latest_summary_path(run_dir)}")
    if state["last_run_id"]:
        print(f"Latest remote artifacts: {local_remote_run_dir(run_dir, state['last_run_id'])}")
    print("Best metrics:")
    for experiment in config["experiments"]:
        name = experiment["name"]
        value = state["best_metrics"].get(name, "none")
        print(f"  {name}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restartable LS-only autoresearch harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init", help="initialize a new restartable run tag")
    p_init.add_argument("--tag", required=True)
    p_init.add_argument("--preset", default="smoke", choices=sorted(PRESET_FACTORIES))
    p_init.add_argument("--create-branch", action="store_true", help="create/switch to autoresearch/<tag>")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    p_init.set_defaults(func=init_run)

    p_run = subparsers.add_parser("run", help="run the configured remote experiment suite for the current commit")
    p_run.add_argument("--tag", required=True)
    p_run.add_argument("--description", default="", help="short note attached to this suite run")
    p_run.add_argument("--allow-dirty", action="store_true", help="allow running without a clean git tree")
    p_run.set_defaults(func=run_suite)

    p_status = subparsers.add_parser("status", help="show persisted state for a run tag")
    p_status.add_argument("--tag", required=True)
    p_status.set_defaults(func=show_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
