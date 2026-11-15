"""
Author: nanochat Modal slice (Claude)
Created: 2026-11-14
Scope: runs/runmodal.sh MODE selection. Uses a fake `modal` CLI on PATH, so nothing reaches Modal.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

# Add runs dir to path for imports
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
if str(RUNS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNS_DIR))

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


def _run(
    fake_bin: Path,
    mode: str | None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
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
    if extra_env:
        env.update(extra_env)
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
    Scenario: no MODE set, so the run is CPU and modal gets --mode cpu and --detach.
    Slice: default mode selection, detached by default.
    """
    # ### Given
    fake_bin, record = fake_modal

    # ### When
    result = _run(fake_bin, None)

    # ### Then
    assert result.returncode == 0, result.stderr
    argv = record.read_text().split()
    assert argv[:4] == ["run", "runs/modal_run.py", "--mode", "cpu"]
    assert "--detach" in argv
    assert "--run-id" in argv and "--code-rev" in argv
    assert "CPU mode" in result.stdout


def test_gpu_mode_passes_gpu_and_shows_gpu_cost(fake_modal: tuple[Path, Path]) -> None:
    """
    Scenario: MODE=gpu passes --mode gpu --detach and prints the H100 cost banner.
    Slice: GPU path stays selectable for later, detached by default.
    """
    # ### Given
    fake_bin, record = fake_modal

    # ### When
    result = _run(fake_bin, "gpu")

    # ### Then
    assert result.returncode == 0, result.stderr
    argv = record.read_text().split()
    assert argv[:4] == ["run", "runs/modal_run.py", "--mode", "gpu"]
    assert "--detach" in argv
    assert "GPU mode" in result.stdout
    assert "8×H100" in result.stdout


@pytest.fixture
def fake_modal_running_app(tmp_path: Path) -> tuple[Path, Path]:
    """Fake `modal` that reports one live nanochat app via `app list --json`.

    Returns (bin_dir, record_file); record_file is written only when `run` is called.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "argv.txt"
    running_json = '[{"app_id":"ap-running","description":"nanochat-speedrun","stopped_at":null}]'
    script = bin_dir / "modal"
    script.write_text(
        f'#!/bin/bash\n'
        f'if [ "$1" = "app" ] && [ "$2" = "list" ]; then\n'
        f'    echo \'{running_json}\'\n'
        f'elif [ "$1" = "run" ]; then\n'
        f'    echo "$@" > "{record}"\n'
        f'fi\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return bin_dir, record


@pytest.fixture
def fake_modal_unavailable(tmp_path: Path) -> tuple[Path, Path]:
    """Fake `modal` whose `app list --json` exits non-zero (simulates CLI failure).

    Returns (bin_dir, record_file); `run` still records argv.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "argv.txt"
    script = bin_dir / "modal"
    script.write_text(
        f'#!/bin/bash\n'
        f'if [ "$1" = "app" ] && [ "$2" = "list" ]; then\n'
        f'    exit 1\n'
        f'elif [ "$1" = "run" ]; then\n'
        f'    echo "$@" > "{record}"\n'
        f'fi\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return bin_dir, record


def test_guard_refuses_when_app_running(fake_modal_running_app: tuple[Path, Path]) -> None:
    """
    Scenario: A nanochat Modal app is already running; a second launch must be refused.
    Slice: concurrency guard in runmodal.sh.
    ### Given: fake modal reports one live nanochat-speedrun app
    ### When: runmodal.sh is invoked
    ### Then: exits 1, prints ERROR, never calls modal run
    """
    # ### Given
    fake_bin, record = fake_modal_running_app

    # ### When
    result = _run(fake_bin, None)

    # ### Then
    assert result.returncode == 1
    assert "ERROR" in result.stderr
    assert "nanochat Modal app appears to be running" in result.stderr
    assert not record.exists()


def test_guard_bypassed_with_allow_concurrent(fake_modal_running_app: tuple[Path, Path]) -> None:
    """
    Scenario: RUNMODAL_ALLOW_CONCURRENT=1 overrides the concurrency guard.
    Slice: RUNMODAL_ALLOW_CONCURRENT override in runmodal.sh.
    ### Given: fake modal reports a running app AND RUNMODAL_ALLOW_CONCURRENT=1 is set
    ### When: runmodal.sh is invoked
    ### Then: exits 0 and modal run is called
    """
    # ### Given
    fake_bin, record = fake_modal_running_app

    # ### When
    result = _run(fake_bin, None, extra_env={"RUNMODAL_ALLOW_CONCURRENT": "1"})

    # ### Then
    assert result.returncode == 0, result.stderr
    assert record.exists()


def test_guard_warns_and_proceeds_when_modal_unavailable(
    fake_modal_unavailable: tuple[Path, Path],
) -> None:
    """
    Scenario: modal app list fails (CLI not configured); runmodal.sh warns and proceeds.
    Slice: guard fallback / WARNING path in runmodal.sh.
    ### Given: fake modal exits 1 for app list --json
    ### When: runmodal.sh is invoked
    ### Then: exits 0, prints WARNING in stderr, modal run is still called
    """
    # ### Given
    fake_bin, record = fake_modal_unavailable

    # ### When
    result = _run(fake_bin, None)

    # ### Then
    assert result.returncode == 0, result.stderr
    assert "WARNING" in result.stderr
    assert "concurrency guard" in result.stderr
    assert record.exists()


@pytest.fixture
def fake_modal_app_id(tmp_path: Path) -> Path:
    """Fake `modal`: `app list` returns no apps; `run` prints the app id then 200 lines, like a real detached run."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "modal"
    script.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "app" ]; then echo "[]"; exit 0; fi\n'
        'echo "Modal app id: ap-abc123XYZ"\n'
        'for i in $(seq 1 200); do echo "line $i"; done\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def test_detached_launch_succeeds_and_records_app_id(fake_modal_app_id: Path) -> None:
    """
    Scenario: a detached launch prints the app id then keeps logging; the launcher must not fail.
    Slice: stopping the reader early (SIGPIPE) turned a successful launch into exit 141 and cut the log.
    ### Given: fake modal that prints the app id and 200 more lines
    ### When: runmodal.sh runs detached (default)
    ### Then: exit 0, app.id records the app and mode, PENDING_DOWNLOAD is set, full run.log kept
    """
    result = _run(fake_modal_app_id, None)

    assert result.returncode == 0, result.stderr
    run_dirs = list((fake_modal_app_id.parent / "archive").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "app.id").read_text().strip() == "ap-abc123XYZ cpu"
    assert (run_dir / "PENDING_DOWNLOAD").exists()
    assert "line 200" in (run_dir / "run.log").read_text()

