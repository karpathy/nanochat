"""
Author: nanochat run-record slice (Claude)
Created: 2026-11-14
Scope: nanochat/run_record.py. run.json and events.jsonl for local and Modal runs; summaries on demand.
"""

import json
from pathlib import Path

import pytest

from nanochat.run_record import (
    EVENTS_JSONL,
    RUN_JSON,
    RunRecorder,
    compare_table,
    find_run_dirs,
    is_milestone,
    make_run_logger,
    percentile,
    read_events,
    record_span,
    replay_commands,
    summarize,
    update_run,
)


def _run_json(run_dir: Path, **fields) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_JSON).write_text(json.dumps({"run_id": run_dir.name, "mode": "local-cpu", **fields}))


def test_percentile_uses_nearest_rank_and_handles_empty() -> None:
    """
    Scenario: p50 and p95 of step times; empty input gives None.
    Slice: latency percentiles.
    """
    # ### Given / ### When / ### Then
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 95) == 4
    assert percentile([], 50) is None


def test_milestones_are_every_500_steps_and_every_evaluation() -> None:
    """
    Scenario: a step on a 500 boundary, a plain step, and a CORE evaluation off the boundary.
    Slice: only important points are written.
    """
    # ### Given / ### When / ### Then
    assert is_milestone({"step": 500, "train/loss": 1.0})
    assert not is_milestone({"step": 510, "train/loss": 1.0})
    assert is_milestone({"step": 530, "core_metric": 0.1})


def test_recorder_keeps_only_milestones_and_marks_finished(tmp_path: Path) -> None:
    """
    Scenario: the trainer logs every 10 steps, then finishes.
    Slice: small folder, and a finished run is marked ok with an end time.
    """
    # ### Given
    recorder = RunRecorder(str(tmp_path), config={"depth": 6})

    # ### When
    for step in range(10, 1001, 10):
        recorder.log({"step": step, "train/loss": 1.0})
    recorder.finish()

    # ### Then
    assert [e["data"]["step"] for e in read_events(str(tmp_path))] == [500, 1000]
    run = json.loads((tmp_path / RUN_JSON).read_text())
    assert run["status"] == "ok"
    assert run["config"]["depth"] == 6
    assert "ended_at" in run


def test_make_run_logger_is_unchanged_without_a_run_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Scenario: NANOCHAT_RUN_DIR is unset. The backend is returned as-is.
    Slice: nothing new is written unless asked for.
    """
    # ### Given
    monkeypatch.delenv("NANOCHAT_RUN_DIR", raising=False)
    backend = object()

    # ### When / ### Then
    assert make_run_logger(backend) is backend


def test_make_run_logger_tees_to_recorder_and_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Scenario: NANOCHAT_RUN_DIR is set. Logs reach the recorder and the existing backend.
    Slice: wandb keeps working alongside the run record.
    """
    # ### Given
    monkeypatch.setenv("NANOCHAT_RUN_DIR", str(tmp_path))
    seen: list[dict] = []

    class Backend:
        def log(self, data, *args, **kwargs):
            seen.append(data)

        def finish(self, *args, **kwargs):
            pass

    logger = make_run_logger(Backend())

    # ### When
    logger.log({"step": 500})

    # ### Then
    assert seen == [{"step": 500}]
    assert read_events(str(tmp_path))[0]["data"] == {"step": 500}


def test_record_span_writes_one_stage_span_with_hook_time(tmp_path: Path) -> None:
    """
    Scenario: one stage ran for 90 s and its hook took 4 s.
    Slice: one span per stage, with the hook time on it (no separate hook span).
    """
    # ### Given / ### When
    record_span(str(tmp_path), "base_train", "ok", 90, 4, 1000)

    # ### Then
    (event,) = read_events(str(tmp_path))
    assert event["name"] == "stage:base_train"
    assert event["dur_s"] == 90
    assert event["hook_s"] == 4


def test_summarize_derives_durations_throughput_and_quality(tmp_path: Path) -> None:
    """
    Scenario: a finished run with two stages, metrics and a validation score.
    Slice: summary is computed from run.json and events.jsonl, not stored.
    """
    # ### Given
    _run_json(tmp_path, started_at=1000.0, ended_at=4600.0, status="ok", code_rev="abc1234",
              code_dirty=False, config={"device_type": "mps"})
    record_span(str(tmp_path), "tok_train", "ok", 60, 0, 1000)
    record_span(str(tmp_path), "base_train", "ok", 3000, 2, 1060)
    for step, dt, tok, val in [(500, 1.0, 9000, 1.2), (1000, 3.0, 8000, 1.1)]:
        append_metric = {"type": "metric", "t": 1.0, "data": {
            "step": step, "train/dt": dt, "train/tok_per_sec": tok, "val/bpb": val}}
        with (tmp_path / EVENTS_JSONL).open("a") as f:
            f.write(json.dumps(append_metric) + "\n")

    # ### When
    s = summarize(str(tmp_path))

    # ### Then
    assert s["status"] == "ok"
    assert s["device"] == "mps"
    assert s["wall_s"] == 3600.0
    assert s["stages_s"] == {"tok_train": 60, "base_train": 3000}
    assert s["hook_s"] == 2
    assert s["steps_done"] == 1000
    assert s["step_time_s_p50"] == 1.0
    assert s["step_time_s_p95"] == 3.0
    assert s["tok_per_sec_last"] == 8000
    assert s["val_bpb_min"] == 1.1


def test_running_or_killed_run_is_never_reported_ok(tmp_path: Path) -> None:
    """
    Scenario: run.json says running and there is no end time (still going, or killed).
    Slice: a run that did not finish is not reported as ok.
    """
    # ### Given
    _run_json(tmp_path, status="running", started_at=0.0)

    # ### When
    s = summarize(str(tmp_path), now=60.0)

    # ### Then
    assert s["status"] == "running"
    assert s["wall_s"] == 60.0


def test_read_events_skips_a_partly_written_last_line(tmp_path: Path) -> None:
    """
    Scenario: a run was killed mid-write, leaving a truncated JSON line.
    Slice: summaries of killed runs still work.
    """
    # ### Given
    record_span(str(tmp_path), "tok_train", "ok", 1, 0, 0)
    (tmp_path / EVENTS_JSONL).open("a").write('{"type": "metric", "t": 2')

    # ### When / ### Then
    assert len(read_events(str(tmp_path))) == 1


def test_update_run_merges_config_and_resources(tmp_path: Path) -> None:
    """
    Scenario: launcher settings are merged first, then the resource report; neither erases the other.
    Slice: run.json is the single place for config and resources.
    """
    # ### Given
    _run_json(tmp_path, config={"device_type": "cpu"})

    # ### When
    update_run(str(tmp_path), config={"cores": 64}, resources={"peak_rss_gib": 3.2})

    # ### Then
    run = json.loads((tmp_path / RUN_JSON).read_text())
    assert run["config"] == {"device_type": "cpu", "cores": 64}
    assert run["resources"] == {"peak_rss_gib": 3.2}


def test_compare_table_lists_each_run_with_key_columns(tmp_path: Path) -> None:
    """
    Scenario: two runs under one root; the table has one row per run and marks uncommitted code.
    Slice: compare shows the important numbers at a glance.
    """
    # ### Given
    for name, dirty in [("20261114-010000-local-cpu", False), ("20261114-020000-modal-cpu", True)]:
        _run_json(tmp_path / name, mode=name.split("-", 2)[2], status="ok", started_at=0.0,
                  ended_at=60.0, code_rev="abc", code_dirty=dirty)

    # ### When
    lines = compare_table(find_run_dirs(str(tmp_path))).splitlines()

    # ### Then
    assert lines[0].split()[0] == "run"
    assert len(lines) == 3
    assert "modal-cpu" in lines[1] and "local-cpu" in lines[2]
    assert "abc*" in lines[1]


def test_replay_local_cpu_uses_runcpu_not_runmodal(tmp_path: Path) -> None:
    """
    Scenario: replay of a local-cpu run uses runcpu.sh, not runmodal.sh.
    runmodal.sh only accepts cpu|gpu modes; local runs must go to the dedicated scripts.
    Slice: nanochat/run_record.py replay_commands local-cpu path (FIX 3).
    ### Given: a run.json for a local-cpu run with a known code_rev
    ### When: replay_commands is called
    ### Then: output has git checkout, uv sync --extra cpu, and bash runs/runcpu.sh
    """
    # ### Given
    _run_json(tmp_path, mode="local-cpu", code_rev="abc1234", code_dirty=False)
    # ### When
    cmds = replay_commands(str(tmp_path))
    # ### Then
    assert cmds[0] == "git checkout abc1234"
    assert cmds[1] == "uv sync --extra cpu"
    assert cmds[2] == "bash runs/runcpu.sh"
    assert all("runmodal" not in c for c in cmds), "local-cpu must not reference runmodal.sh"


def test_replay_modal_run_omits_uv_sync(tmp_path: Path) -> None:
    """
    Scenario: replay of a modal-cpu run omits uv sync (not a local run).
    Slice: nanochat/run_record.py replay_commands modal path.
    ### Given: a run.json for a modal-cpu run
    ### When: replay_commands is called
    ### Then: output has git checkout and runmodal.sh but no uv sync
    """
    # ### Given
    _run_json(tmp_path, mode="modal-cpu", code_rev="def5678", code_dirty=False)
    # ### When
    cmds = replay_commands(str(tmp_path))
    # ### Then
    assert cmds[0] == "git checkout def5678"
    assert all("uv sync" not in c for c in cmds)
    assert cmds[1] == "MODE=modal-cpu bash runs/runmodal.sh"


def test_replay_cli_prints_commands(tmp_path: Path) -> None:
    """
    Scenario: the CLI replay subcommand exits 0 and prints one line per command.
    Slice: nanochat/run_record.py main() replay dispatch.
    ### Given: a local-cpu run directory
    ### When: main(["replay", run_dir]) is invoked
    ### Then: returns 0
    """
    # ### Given
    from nanochat.run_record import main as rr_main
    _run_json(tmp_path, mode="local-cpu", code_rev="abc1234", code_dirty=False)
    # ### When / Then
    assert rr_main(["replay", str(tmp_path)]) == 0
