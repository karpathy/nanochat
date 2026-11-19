"""
Author: nanochat Modal slice (Claude)
Created: 2026-11-16
Scope: Modal CPU caps the ChatCORE categorical sample (runs/modal_run.py -> runs/runcpu.sh ->
       scripts/chat_sft.py --chatcore-max-cat). Local runs keep the full set (-1).
"""

import sys
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
if str(RUNS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNS_DIR))

import modal_run  # noqa: E402

RUNCPU = RUNS_DIR / "runcpu.sh"


def test_modal_cpu_env_caps_chatcore_categorical_eval():
    """
    Scenario: The Modal CPU container builds its runcpu.sh environment.
    Slice: ChatCORE categorical cap is passed to runcpu.sh on Modal
    ### Given: a 4-core container, a run id, and the default iterations and save interval
    ### When: the CPU train env is built
    ### Then: NANOCHAT_RUNCPU_CHATCORE_MAX_CAT is the Modal cap (500)
    """
    env = modal_run._cpu_train_env(
        cores=4, run_id="20260101-000000-cpu-d6", code_rev="abc1234",
        num_iterations=5000, save_every_secs=300,
    )

    assert env["NANOCHAT_RUNCPU_CHATCORE_MAX_CAT"] == str(modal_run.CPU_CHATCORE_MAX_CAT) == "500"


def test_runcpu_forwards_chatcore_cap_and_keeps_local_default_full_set():
    """
    Scenario: runcpu.sh is read as the local and Modal entry point.
    Slice: the cap is forwarded to chat_sft; local default stays -1 (full sets)
    ### Given: runs/runcpu.sh
    ### When: the script text is inspected
    ### Then: the chat_sft call forwards the variable and the unset default is -1
    """
    text = RUNCPU.read_text(encoding="utf-8")

    assert 'NANOCHAT_RUNCPU_CHATCORE_MAX_CAT:--1' in text
    assert '--chatcore-max-cat="$RUNCPU_CHATCORE_MAX_CAT"' in text
