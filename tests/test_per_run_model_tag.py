"""
Author: nanochat per-run model tag slice (Claude)
Created: 2026-11-16
Scope: a run's model tag is its run id; the Modal chat picker accepts run-id folders and
       includes last_run.json; last_run.json records and returns the tag of the newest run.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
if str(RUNS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNS_DIR))

import modal_run

from nanochat.last_run import last_run_model_tag, resolve_chat_model_tag


def _listing(*paths: str) -> dict[str, int]:
    return {p: 1 for p in paths}


# Scenario: a new CPU run stores checkpoints under its run id, not under d<depth>.
def test_modal_picker_selects_run_id_folder():
    # Given a Volume listing with a run-id tag folder holding a complete pair
    tag = "20261116-114442-cpu-d6"
    sizes = _listing(
        f"cpu/chatsft_checkpoints/{tag}/model_000100.pt",
        f"cpu/chatsft_checkpoints/{tag}/meta_000100.json",
    )
    # When the picker selects chat files
    picked = modal_run._select_chat_files(sizes, "cpu")
    # Then the run-id folder's pair is chosen
    assert f"cpu/chatsft_checkpoints/{tag}/model_000100.pt" in picked
    assert f"cpu/chatsft_checkpoints/{tag}/meta_000100.json" in picked


# Scenario: last_run.json must come down with the checkpoints so chat without -g works locally.
def test_modal_picker_includes_last_run_json():
    # Given a listing that contains last_run.json and a complete pair
    sizes = _listing(
        "cpu/last_run.json",
        "cpu/chatsft_checkpoints/.train_ok",
        "cpu/chatsft_checkpoints/d6/model_000001.pt",
        "cpu/chatsft_checkpoints/d6/meta_000001.json",
    )
    # When the picker selects chat files
    picked = modal_run._select_chat_files(sizes, "cpu")
    # Then last_run.json is picked
    assert "cpu/last_run.json" in picked


# Scenario: writing last_run.json with a model tag makes that tag readable; no tag means none.
def test_last_run_model_tag_round_trip(tmp_path: Path):
    # Given a last_run.json written by the CLI with a model tag
    path = tmp_path / "last_run.json"
    subprocess.run(
        [sys.executable, "-m", "nanochat.last_run", "write", "--path", str(path),
         "--profile", "runcpu", "--depth", "6", "--device-batch-size", "8",
         "--max-seq-len", "512", "--model-tag", "20261116-114442-cpu-d6"],
        check=True, cwd=ROOT,
    )
    # When the tag is read back
    # Then it is the run id
    assert last_run_model_tag(path) == "20261116-114442-cpu-d6"
    assert json.loads(path.read_text())["model_tag"] == "20261116-114442-cpu-d6"


def test_last_run_model_tag_missing_returns_none(tmp_path: Path):
    # Given no last_run.json
    # When the tag is read
    # Then nothing is returned
    assert last_run_model_tag(tmp_path / "absent.json") is None


@pytest.mark.parametrize("payload", ['{"profile": "runcpu"}', "[]"])
def test_last_run_model_tag_without_tag_returns_none(tmp_path: Path, payload: str):
    # Given a last_run.json that records no model tag (legacy or malformed shape)
    path = tmp_path / "last_run.json"
    path.write_text(payload)
    # When the tag is read
    # Then nothing is returned
    assert last_run_model_tag(path) is None


# Scenario: several completed CPU runs on the Volume; chat downloads only the newest run's pair.
def test_modal_picker_keeps_only_newest_run_folder():
    # Given three run-id folders, each with a complete pair
    old, mid, new = "20261101-000000-cpu-d6", "20261108-000000-cpu-d6", "20261116-114442-cpu-d6"
    sizes = _listing(*[
        f"cpu/chatsft_checkpoints/{tag}/{kind}_000100.{ext}"
        for tag in (old, mid, new)
        for kind, ext in (("model", "pt"), ("meta", "json"))
    ])
    # When the picker selects chat files
    picked = modal_run._select_chat_files(sizes, "cpu")
    # Then only the newest run's pair is picked
    assert picked == sorted([f"cpu/chatsft_checkpoints/{new}/model_000100.pt",
                             f"cpu/chatsft_checkpoints/{new}/meta_000100.json"])


# Scenario: a CPU run archives only its own checkpoint folder, not every past run's.
def test_archive_copies_only_this_runs_checkpoint_folder(tmp_path: Path, monkeypatch):
    # Given a Volume with two past runs' checkpoints and the current run's folder
    base = tmp_path / "vol"
    current = "20261116-114442-cpu-d6"
    for tag in ("20261101-000000-cpu-d6", current):
        (base / "cpu" / "base_checkpoints" / tag).mkdir(parents=True)
        (base / "cpu" / "base_checkpoints" / tag / "model_000100.pt").write_text("w")
    monkeypatch.setattr(modal_run, "BASE_DIR", str(base))
    monkeypatch.setattr(modal_run, "vol", MagicMock())
    # When the archive runs for the current run
    modal_run._archive_artifacts(current, "cpu")
    # Then only the current run's checkpoint folder is archived
    archived = base / "runs" / current / "artifacts" / "base_checkpoints"
    assert (archived / current / "model_000100.pt").is_file()
    assert not (archived / "20261101-000000-cpu-d6").exists()


# Scenario: chat with no -g uses the recorded run's tag only while its folder still exists.
def test_resolve_chat_tag_uses_recorded_run_when_folder_exists(tmp_path: Path):
    # Given last_run.json naming a run whose chatsft folder exists
    tag = "20261116-114442-cpu-d6"
    (tmp_path / "chatsft_checkpoints" / tag).mkdir(parents=True)
    (tmp_path / "last_run.json").write_text(json.dumps({"model_tag": tag}))
    # When no tag is requested for the sft source
    # Then the recorded run's tag is used
    assert resolve_chat_model_tag(tmp_path, "sft", None) == tag


def test_resolve_chat_tag_ignores_deleted_recorded_run(tmp_path: Path):
    # Given last_run.json naming a run whose folder was deleted
    (tmp_path / "last_run.json").write_text(json.dumps({"model_tag": "gone-cpu-d6"}))
    # When no tag is requested
    # Then the loader is left to choose (None)
    assert resolve_chat_model_tag(tmp_path, "sft", None) is None


def test_resolve_chat_tag_respects_explicit_request_and_other_sources(tmp_path: Path):
    # Given a recorded run
    (tmp_path / "last_run.json").write_text(json.dumps({"model_tag": "r-cpu-d6"}))
    # When a tag is explicitly requested, or the source is not sft
    # Then the request is returned unchanged
    assert resolve_chat_model_tag(tmp_path, "sft", "explicit") == "explicit"
    assert resolve_chat_model_tag(tmp_path, "rl", None) is None


# Scenario: a CPU relaunch continues the newest interrupted run of the same depth, refuses while
# another run is live, and starts a new run when nothing is unfinished.
def test_relaunch_continues_newest_interrupted_run_of_same_depth():
    # Given two depth-6 runs, the newer one interrupted (no .done), and a depth-12 run
    picked = modal_run.resolve_cpu_run_id(
        ["20261101-000000-cpu-d6", "20261116-114442-cpu-d6", "20261117-000000-cpu-d12"],
        {"20261101-000000-cpu-d6"}, None, 1_000_000.0, "-cpu-d6",
    )
    # Then the interrupted depth-6 run's id is reused; the depth-12 run is never continued
    assert picked == "20261116-114442-cpu-d6"


def test_relaunch_refuses_while_another_cpu_run_holds_fresh_lock():
    # Given a heartbeat written 60 s ago
    # When a relaunch is resolved
    # Then it is refused rather than starting a second run
    with pytest.raises(SystemExit):
        modal_run.resolve_cpu_run_id(["20261116-114442-cpu-d6"], set(), 999_940.0, 1_000_000.0, "-cpu-d6")


def test_relaunch_continues_when_lock_is_stale():
    # Given a heartbeat older than the staleness threshold (holder is gone)
    picked = modal_run.resolve_cpu_run_id(["20261116-114442-cpu-d6"], set(), 1.0, 1_000_000.0, "-cpu-d6")
    # Then the interrupted run continues
    assert picked == "20261116-114442-cpu-d6"


def test_relaunch_starts_new_run_when_nothing_unfinished():
    # Given the only depth-6 run is finished, and a GPU run is unfinished (ignored for CPU)
    picked = modal_run.resolve_cpu_run_id(
        ["20261116-114442-cpu-d6", "20261117-000000-gpu"], {"20261116-114442-cpu-d6"}, None, 1_000_000.0, "-cpu-d6",
    )
    # Then None: the caller mints a new id
    assert picked is None


# Scenario: an old unfinished run is never resumed automatically when a newer run has finished.
def test_relaunch_never_resumes_older_unfinished_run():
    # Given an old unfinished run and a newer run that finished
    picked = modal_run.resolve_cpu_run_id(
        ["20261101-000000-cpu-d6", "20261116-114442-cpu-d6"], {"20261116-114442-cpu-d6"}, None, 1_000_000.0, "-cpu-d6",
    )
    # Then nothing is resumed: a new run starts
    assert picked is None


# Scenario: NANOCHAT_LOCK_STALE_SECS shortens the threshold, so the same heartbeat no longer blocks.
def test_relaunch_honours_configured_stale_threshold():
    # Given a heartbeat 60 s old, with a 30 s staleness threshold
    picked = modal_run.resolve_cpu_run_id(
        ["20261116-114442-cpu-d6"], set(), 999_940.0, 1_000_000.0, "-cpu-d6", stale_secs=30.0,
    )
    # Then the interrupted run is continued (the lock is considered dead)
    assert picked == "20261116-114442-cpu-d6"


# Scenario: on Modal, runcpu.sh writes .done into the Volume run folder that resolve_run probes.
# Both sides must agree, or a finished run looks interrupted and is resumed forever.
def test_done_marker_written_where_modal_resolver_looks(tmp_path: Path, monkeypatch):
    # Given the Modal layout: the Volume root, the CPU base dir under it, and the run record folder
    volume = tmp_path / "volume"
    run_id = "20261116-114442-cpu-d6"
    run_dir = volume / "runs" / run_id
    monkeypatch.setattr(modal_run, "BASE_DIR", str(volume))
    # When runcpu's end-of-run marking runs with those environment values
    env = {
        "PATH": "/usr/bin:/bin",
        "NANOCHAT_BASE_DIR": str(volume / "cpu"),
        "NANOCHAT_RUN_DIR": str(run_dir),
        "NANOCHAT_RUN_ID": run_id,
        "PROFILE": "runcpu",
        "DEPTH": "6",
    }
    subprocess.run(
        ["bash", "-c", f'source "{ROOT}/runs/_stage_lib.sh"; mark_run_done "$NANOCHAT_RUN_DIR"'],
        check=True, env=env, cwd=ROOT,
    )
    # Then the marker is at runs/<id>/.done on the Volume, which is the path resolve_run checks
    assert (run_dir / ".done").is_file()
    assert modal_run._run_dir(run_id) == str(run_dir)
    assert f"runs/{run_id}/.done" == os.path.relpath(run_dir / ".done", volume).replace(os.sep, "/")


# Scenario: a named run is resumed only if it is a timestamped id of this depth, exists, is not
# finished, and no other run is live.
def test_forced_resume_refuses_other_depth():
    with pytest.raises(SystemExit):
        modal_run.check_forced_cpu_run_id(
            "20261116-000000-cpu-d12", "-cpu-d6", ["20261116-000000-cpu-d12"], set(), None, 1.0, 600.0,
        )


def test_forced_resume_refuses_finished_run():
    rid = "20261116-000000-cpu-d6"
    with pytest.raises(SystemExit):
        modal_run.check_forced_cpu_run_id(rid, "-cpu-d6", [rid], {rid}, None, 1.0, 600.0)


def test_forced_resume_refuses_while_lock_is_fresh():
    rid = "20261116-000000-cpu-d6"
    with pytest.raises(SystemExit):
        modal_run.check_forced_cpu_run_id(rid, "-cpu-d6", [rid], set(), 999_940.0, 1_000_000.0, 600.0)


def test_forced_resume_refuses_id_with_no_folder():
    # A typo'd id must not start a paid run that quietly trains from step 0.
    with pytest.raises(SystemExit):
        modal_run.check_forced_cpu_run_id("20261116-14442-cpu-d6", "-cpu-d6", ["20261116-114442-cpu-d6"], set(), None, 1.0, 600.0)


def test_forced_resume_refuses_name_that_is_not_a_timestamp():
    rid = "mytest-cpu-d6"
    with pytest.raises(SystemExit):
        modal_run.check_forced_cpu_run_id(rid, "-cpu-d6", [rid], set(), None, 1.0, 600.0)


def test_forced_resume_accepts_interrupted_same_depth_run_of_any_age():
    rid = "20260101-000000-cpu-d6"
    picked = modal_run.check_forced_cpu_run_id(rid, "-cpu-d6", [rid], set(), None, 1_000_000.0, 600.0)
    assert picked == rid


# Scenario: an alphabetic folder must never be treated as the "most recent" run.
def test_alphabetic_run_name_does_not_shadow_timestamped_runs():
    picked = modal_run.resolve_cpu_run_id(
        ["mytest-cpu-d6", "20261116-114442-cpu-d6"], set(), None, 1_000_000.0, "-cpu-d6",
    )
    assert picked == "20261116-114442-cpu-d6"
