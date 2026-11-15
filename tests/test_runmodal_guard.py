"""
Author: nanochat Modal slice (Claude)
Created: 2026-11-15
Scope: runs/runmodal.sh launch guard edge cases: stopped apps, different descriptions, invalid JSON, multiple apps, bypass variations.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

RUNMODAL = Path(__file__).resolve().parents[1] / "runs" / "runmodal.sh"


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


@pytest.fixture
def fake_modal_stopped_app(tmp_path: Path) -> tuple[Path, Path]:
    """Fake `modal` that reports a stopped nanochat app (stopped_at is not None).

    Returns (bin_dir, record_file); only `run` is recorded.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "argv.txt"
    stopped_json = '[{"app_id":"ap-stopped","description":"nanochat-speedrun","stopped_at":"2026-11-15T10:00:00Z"}]'
    script = bin_dir / "modal"
    script.write_text(
        f'#!/bin/bash\n'
        f'if [ "$1" = "app" ] && [ "$2" = "list" ]; then\n'
        f'    echo \'{stopped_json}\'\n'
        f'elif [ "$1" = "run" ]; then\n'
        f'    echo "$@" > "{record}"\n'
        f'fi\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return bin_dir, record


@pytest.fixture
def fake_modal_different_description(tmp_path: Path) -> tuple[Path, Path]:
    """Fake `modal` that reports a live app with a different description.

    Returns (bin_dir, record_file); only `run` is recorded.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "argv.txt"
    different_json = '[{"app_id":"ap-other","description":"some-other-app","stopped_at":null}]'
    script = bin_dir / "modal"
    script.write_text(
        f'#!/bin/bash\n'
        f'if [ "$1" = "app" ] && [ "$2" = "list" ]; then\n'
        f'    echo \'{different_json}\'\n'
        f'elif [ "$1" = "run" ]; then\n'
        f'    echo "$@" > "{record}"\n'
        f'fi\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return bin_dir, record


@pytest.fixture
def fake_modal_invalid_json(tmp_path: Path) -> tuple[Path, Path]:
    """Fake `modal` that returns invalid JSON from app list --json.

    Returns (bin_dir, record_file); only `run` is recorded.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "argv.txt"
    script = bin_dir / "modal"
    script.write_text(
        f'#!/bin/bash\n'
        f'if [ "$1" = "app" ] && [ "$2" = "list" ]; then\n'
        f'    echo "{{invalid json"\n'
        f'elif [ "$1" = "run" ]; then\n'
        f'    echo "$@" > "{record}"\n'
        f'fi\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return bin_dir, record


@pytest.fixture
def fake_modal_multiple_apps(tmp_path: Path) -> tuple[Path, Path]:
    """Fake `modal` that reports multiple apps, one of which is a live nanochat-speedrun.

    Returns (bin_dir, record_file); only `run` is recorded.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "argv.txt"
    multiple_json = (
        '[{"app_id":"ap-other1","description":"other-app","stopped_at":null},'
        '{"app_id":"ap-running","description":"nanochat-speedrun","stopped_at":null},'
        '{"app_id":"ap-other2","description":"another-app","stopped_at":"2026-11-15T09:00:00Z"}]'
    )
    script = bin_dir / "modal"
    script.write_text(
        f'#!/bin/bash\n'
        f'if [ "$1" = "app" ] && [ "$2" = "list" ]; then\n'
        f'    echo \'{multiple_json}\'\n'
        f'elif [ "$1" = "run" ]; then\n'
        f'    echo "$@" > "{record}"\n'
        f'fi\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return bin_dir, record


def test_guard_allows_stopped_app_with_matching_description(fake_modal_stopped_app: tuple[Path, Path]) -> None:
    """
    Scenario: A nanochat-speedrun app exists but is stopped (stopped_at is set).
    Slice: guard filter on stopped_at is None.
    ### Given: fake modal reports stopped nanochat-speedrun app
    ### When: runmodal.sh is invoked
    ### Then: exits 0, modal run is called (not blocked)
    """
    fake_bin, record = fake_modal_stopped_app

    result = _run(fake_bin, None)

    assert result.returncode == 0, result.stderr
    assert record.exists()


def test_guard_allows_live_app_with_different_description(fake_modal_different_description: tuple[Path, Path]) -> None:
    """
    Scenario: A live app exists but with a different description (not nanochat-speedrun).
    Slice: guard filter on description.
    ### Given: fake modal reports a live app with description != "nanochat-speedrun"
    ### When: runmodal.sh is invoked
    ### Then: exits 0, modal run is called (not blocked)
    """
    fake_bin, record = fake_modal_different_description

    result = _run(fake_bin, None)

    assert result.returncode == 0, result.stderr
    assert record.exists()


def test_guard_warns_and_proceeds_on_invalid_json(fake_modal_invalid_json: tuple[Path, Path]) -> None:
    """
    Scenario: modal app list returns invalid JSON; guard treats this as unavailable modal.
    Slice: Python JSON parse error handling in runmodal.sh.
    ### Given: fake modal returns malformed JSON from app list
    ### When: runmodal.sh is invoked
    ### Then: exits 0, prints WARNING, modal run is still called
    """
    fake_bin, record = fake_modal_invalid_json

    result = _run(fake_bin, None)

    assert result.returncode == 0, result.stderr
    assert "WARNING" in result.stderr
    assert "concurrency guard" in result.stderr
    assert record.exists()


def test_guard_refuses_multiple_apps_with_one_live_matching(fake_modal_multiple_apps: tuple[Path, Path]) -> None:
    """
    Scenario: Multiple apps exist; one matches nanochat-speedrun and is live; others don't.
    Slice: guard correctly identifies the matching app in a list.
    ### Given: fake modal reports multiple apps, one matching and live
    ### When: runmodal.sh is invoked
    ### Then: exits 1, prints ERROR, never calls modal run
    """
    fake_bin, record = fake_modal_multiple_apps

    result = _run(fake_bin, None)

    assert result.returncode == 1
    assert "ERROR" in result.stderr
    assert "nanochat Modal app appears to be running" in result.stderr
    assert not record.exists()


def test_guard_bypassed_with_any_nonempty_value(fake_modal_multiple_apps: tuple[Path, Path]) -> None:
    """
    Scenario: RUNMODAL_ALLOW_CONCURRENT with any non-empty value bypasses the guard.
    Slice: guard check uses [ -z ] on RUNMODAL_ALLOW_CONCURRENT, so any non-empty value is truthy.
    ### Given: a running nanochat-speedrun app and RUNMODAL_ALLOW_CONCURRENT=0
    ### When: runmodal.sh is invoked
    ### Then: exits 0, modal run is called (guard is bypassed)
    """
    fake_bin, record = fake_modal_multiple_apps

    result = _run(fake_bin, None, extra_env={"RUNMODAL_ALLOW_CONCURRENT": "0"})

    assert result.returncode == 0, result.stderr
    assert record.exists()


def test_guard_not_bypassed_with_empty_string(fake_modal_multiple_apps: tuple[Path, Path]) -> None:
    """
    Scenario: RUNMODAL_ALLOW_CONCURRENT="" (empty string) does not bypass the guard.
    Slice: guard check only recognizes exactly "1".
    ### Given: a running app and RUNMODAL_ALLOW_CONCURRENT set to empty string
    ### When: runmodal.sh is invoked
    ### Then: exits 1, ERROR printed, guard is not bypassed
    """
    fake_bin, record = fake_modal_multiple_apps

    result = _run(fake_bin, None, extra_env={"RUNMODAL_ALLOW_CONCURRENT": ""})

    assert result.returncode == 1
    assert "ERROR" in result.stderr
    assert not record.exists()


def test_guard_blocked_only_when_var_unset(fake_modal_multiple_apps: tuple[Path, Path]) -> None:
    """
    Scenario: Guard blocks only when RUNMODAL_ALLOW_CONCURRENT is unset or empty.
    Slice: guard check [ -z "${RUNMODAL_ALLOW_CONCURRENT:-}" ] requires variable to be absent or empty.
    ### Given: a running app and RUNMODAL_ALLOW_CONCURRENT unset
    ### When: runmodal.sh is invoked
    ### Then: exits 1, ERROR printed, guard blocks the run
    """
    fake_bin, record = fake_modal_multiple_apps

    # Test with the variable explicitly not set (pop to ensure removal)
    env = os.environ.copy()
    env.pop("RUNMODAL_ALLOW_CONCURRENT", None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["RUNMODAL_YES"] = "1"
    env["NANOCHAT_MODAL_ARCHIVE"] = str(fake_bin.parent / "archive")

    result = subprocess.run(["bash", str(RUNMODAL)], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 1
    assert "ERROR" in result.stderr
    assert not record.exists()
