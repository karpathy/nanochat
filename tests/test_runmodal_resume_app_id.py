"""
Author: nanochat Modal slice (Claude)
Created: 2026-11-17
Scope: runs/runmodal.sh records the Modal app id of each launch. A resumed CPU run must not keep
       the app id of its first launch, because `runmodal.sh finish` looks that app up.
"""

import os
import stat
import subprocess
from pathlib import Path

RUNMODAL = Path(__file__).resolve().parents[1] / "runs" / "runmodal.sh"
RUN_ID = "20261116-154817-cpu-d6"
NEW_URL = "https://modal.com/apps/neomatrix369/main/ap-new123"


def _fake_modal(bin_dir: Path, launch_output: str) -> None:
    """`modal app list` reports nothing running; `modal run resolve_run` returns RUN_ID;
    any other `modal run` prints launch_output (which may contain an app URL)."""
    script = bin_dir / "modal"
    script.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "app" ] && [ "$2" = "list" ]; then echo "[]"; exit 0; fi\n'
        'if [ "$1" = "run" ]; then\n'
        '    if [[ "$*" == *resolve_run* ]]; then echo "RUN_ID=' + RUN_ID + '"; exit 0; fi\n'
        f"    printf '%s\\n' '{launch_output}'\n"
        "    exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def _launch(tmp_path: Path, launch_output: str) -> Path:
    """Run runmodal.sh for a CPU resume (detached, no prompts). Returns the archive run folder."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_modal(bin_dir, launch_output)
    archive = tmp_path / "archive"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "RUNMODAL_YES": "1",
        "MODE": "cpu",
        "NANOCHAT_MODAL_ARCHIVE": str(archive),
        "NANOCHAT_RESUME_RUN_ID": RUN_ID,
    }
    env.pop("ATTACHED", None)
    subprocess.run(["bash", str(RUNMODAL)], env=env, capture_output=True, text=True, check=False)
    return archive / RUN_ID


def test_resumed_launch_records_the_new_app_not_the_old_one(tmp_path: Path) -> None:
    """
    Scenario: a CPU run that already has an app.id from its first launch is resumed.
    Slice: the resumed launch overwrites app.id with the app it started.
    ### Given: app.id holds the first launch's app id
    ### When: runmodal.sh resumes the run and Modal prints the new app URL
    ### Then: app.id names the new app, not the old one
    """
    # ### Given
    old_run = tmp_path / "archive" / RUN_ID
    old_run.mkdir(parents=True)
    (old_run / "app.id").write_text("ap-old999 cpu\n")

    # ### When
    run_dir = _launch(tmp_path, f"Created app. View run at {NEW_URL}")

    # ### Then
    assert (run_dir / "app.id").read_text().strip() == "ap-new123 cpu"


def test_launch_without_an_app_url_leaves_no_stale_app_id(tmp_path: Path) -> None:
    """
    Scenario: a resumed launch prints no app URL (for example, it failed early).
    Slice: the previous launch's app id is not kept, so finish cannot chase a dead app.
    ### Given: app.id holds the first launch's app id
    ### When: the resumed launch prints no app URL
    ### Then: app.id is gone
    """
    # ### Given
    old_run = tmp_path / "archive" / RUN_ID
    old_run.mkdir(parents=True)
    (old_run / "app.id").write_text("ap-old999 cpu\n")

    # ### When
    run_dir = _launch(tmp_path, "no app link here")

    # ### Then
    assert not (run_dir / "app.id").exists()


def test_first_launch_records_the_app_id(tmp_path: Path) -> None:
    """
    Scenario: a brand-new CPU run prints its app URL.
    Slice: unchanged behaviour — the first launch still records app.id.
    ### Given: no archive folder for the run yet
    ### When: the launch prints an app URL
    ### Then: app.id names that app
    """
    # ### When
    run_dir = _launch(tmp_path, f"Created app. View run at {NEW_URL}")

    # ### Then
    assert (run_dir / "app.id").read_text().strip() == "ap-new123 cpu"
