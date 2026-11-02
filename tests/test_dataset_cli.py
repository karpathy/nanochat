"""The downloader must propagate worker failures to shell pipelines."""
import runpy
import sys
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("results, expected", [([True, True], 0), ([True, False], 1), ([False, False], 1)])
def test_download_exit_status(tmp_path, monkeypatch, results, expected):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["nanochat.dataset", "-n", "1", "-w", "1"])
    with patch("multiprocessing.Pool") as pool:
        pool.return_value.__enter__.return_value.map.return_value = results
        try:
            runpy.run_module("nanochat.dataset", run_name="__main__")
        except SystemExit as error:
            code = error.code
        else:
            code = 0
    assert code == expected
