"""
Author: nanochat Modal slice (Claude)
Created: 2026-11-14
Scope: runs/runmodal.sh MODE selection. Uses a fake `modal` CLI on PATH, so nothing reaches Modal.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

RUNMODAL = Path(__file__).resolve().parents[1] / "runs" / "runmodal.sh"


@pytest.fixture
def fake_modal(tmp_path: Path) -> tuple[Path, Path]:
    """Fake `modal` that records its argv to a file. Returns (bin_dir, record_file)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "argv.txt"
    script = bin_dir / "modal"
    script.write_text(f'#!/bin/bash\necho "$@" > "{record}"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return bin_dir, record


def _run(fake_bin: Path, mode: str | None) -> subprocess.CompletedProcess[str]:
    """Run runmodal.sh non-interactively with the fake modal first on PATH."""
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RUNMODAL_YES": "1",
        "NANOCHAT_MODAL_ARCHIVE": str(fake_bin.parent / "archive"),
    }
    env.pop("MODE", None)
    if mode is not None:
        env["MODE"] = mode
    return subprocess.run(["bash", str(RUNMODAL)], env=env, capture_output=True, text=True, check=False)


def test_rejects_unknown_mode(fake_modal: tuple[Path, Path]) -> None:
    """
    Scenario: an unknown MODE value is refused before any Modal call.
    Slice: MODE validation in runmodal.sh.
    """
    # ### Given
    fake_bin, record = fake_modal

    # ### When
    result = _run(fake_bin, "tpu")

    # ### Then
    assert result.returncode != 0
    assert "MODE must be cpu or gpu" in result.stderr
    assert not record.exists()


def test_defaults_to_cpu_mode(fake_modal: tuple[Path, Path]) -> None:
    """
    Scenario: no MODE set, so the run is CPU and modal gets --mode cpu.
    Slice: default mode selection.
    """
    # ### Given
    fake_bin, record = fake_modal

    # ### When
    result = _run(fake_bin, None)

    # ### Then
    assert result.returncode == 0, result.stderr
    argv = record.read_text().split()
    assert argv[:4] == ["run", "runs/modal_run.py", "--mode", "cpu"]
    assert "--run-id" in argv and "--code-rev" in argv
    assert "CPU mode" in result.stdout


def test_gpu_mode_passes_gpu_and_shows_gpu_cost(fake_modal: tuple[Path, Path]) -> None:
    """
    Scenario: MODE=gpu passes --mode gpu and prints the H100 cost banner.
    Slice: GPU path stays selectable for later.
    """
    # ### Given
    fake_bin, record = fake_modal

    # ### When
    result = _run(fake_bin, "gpu")

    # ### Then
    assert result.returncode == 0, result.stderr
    argv = record.read_text().split()
    assert argv[:4] == ["run", "runs/modal_run.py", "--mode", "gpu"]
    assert "GPU mode" in result.stdout
    assert "8×H100" in result.stdout


def test_given_modal_env_when_modal_run_then_no_color_set() -> None:
    """
    Scenario: _modal_env() includes NO_COLOR=1 to disable color in container.
    Slice: NO_COLOR ensures plain text in archived logs.
    """
    # ### Given
    from runs.modal_run import _modal_env

    # ### When
    env = _modal_env("/base/dir")

    # ### Then
    assert env["NO_COLOR"] == "1"
    assert env["NANOCHAT_BASE_DIR"] == "/base/dir"
    assert env["PYTHONPATH"] == "/root/nanochat"
