"""Guard: the Modal image Python must match the project's pinned Python.

Scenario: the image once hard-coded 3.11 while .python-version said 3.12, so
uv sync failed inside the image and runs stalled. This test fails on any drift.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_modal_image_python_matches_project_pin() -> None:
    # Given the project's pinned Python version
    pinned = (ROOT / ".python-version").read_text().strip()
    # When reading the version the Modal image is built with
    source = (ROOT / "runs" / "modal_run.py").read_text()
    match = re.search(r'^PYTHON_VERSION = "([^"]+)"', source, re.MULTILINE)
    # Then they agree
    assert match is not None, "runs/modal_run.py must define PYTHON_VERSION"
    assert match.group(1) == pinned
