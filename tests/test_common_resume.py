"""
Test resume visibility and config changes.

Author: Claude Haiku 4.5
Created: 2026-11-15
Scope: Test compute_eta_str helper, get_base_dir require_explicit, and resume banner logic
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from nanochat.common import compute_eta_str, get_base_dir


class TestComputeEtaStr:
    """Test ETA calculation helper."""

    def test_eta_str_base_case(self):
        # Scenario: Normal case with known steps done and iterations
        # Given: 1000 total training time, 100 steps done, 1000 iterations
        # When: We compute ETA
        # Then: Should return formatted ETA string
        eta = compute_eta_str(total_training_time=1000, steps_done=100, num_iterations=1000)
        assert "eta:" in eta
        assert "h" in eta or "m" in eta or "s" in eta

    def test_eta_str_zero_steps(self):
        # Scenario: No steps completed yet
        # Given: 1000 total training time but only 0 steps done
        # When: We compute ETA
        # Then: Should return empty string (not ready)
        eta = compute_eta_str(total_training_time=1000, steps_done=0, num_iterations=1000)
        assert eta == ""

    def test_eta_str_rounding(self):
        # Scenario: ETA calculation with rounding
        # Given: 600s training time, 10 steps done, 100 iterations
        # When: We compute ETA
        # Then: Should correctly round and format
        eta = compute_eta_str(total_training_time=600, steps_done=10, num_iterations=100)
        # Should compute average: 600/10=60s per step, remaining ~80 steps, so ~4800s = 1h 20m
        assert eta == " | eta: 1h 20m 0s"

    def test_eta_str_near_end(self):
        # Scenario: Training nearly complete
        # Given: Few remaining steps
        # When: We compute ETA
        # Then: Should handle gracefully (return empty if remaining_steps <= 0)
        eta = compute_eta_str(total_training_time=100, steps_done=90, num_iterations=100)
        # remaining = 100 - 90 - 10 = 0, so no ETA is shown
        assert eta == ""


class TestGetBaseDirRequireExplicit:
    """Test get_base_dir with require_explicit flag."""

    def test_get_base_dir_explicit_unset_raises(self):
        # Scenario: NANOCHAT_BASE_DIR not set, require_explicit=True
        # Given: Environment has NANOCHAT_BASE_DIR unset
        # When: We call get_base_dir(require_explicit=True)
        # Then: Should raise RuntimeError with helpful message
        with patch.dict(os.environ, {}, clear=False):
            if "NANOCHAT_BASE_DIR" in os.environ:
                del os.environ["NANOCHAT_BASE_DIR"]
            with pytest.raises(RuntimeError) as exc_info:
                get_base_dir(require_explicit=True)
            assert "NANOCHAT_BASE_DIR not set" in str(exc_info.value)
            assert ".cache/nanochat" in str(exc_info.value)

    def test_get_base_dir_explicit_set_succeeds(self):
        # Scenario: NANOCHAT_BASE_DIR is set, require_explicit=True
        # Given: Environment has NANOCHAT_BASE_DIR=/tmp/nanochat_test
        # When: We call get_base_dir(require_explicit=True)
        # Then: Should succeed and return the path
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": tmpdir}):
                result = get_base_dir(require_explicit=True)
                assert result == tmpdir

    def test_get_base_dir_default_no_explicit(self):
        # Scenario: require_explicit=False (default)
        # Given: NANOCHAT_BASE_DIR unset
        # When: We call get_base_dir()
        # Then: Should use default ~/.cache/nanochat
        with patch.dict(os.environ, {}, clear=False):
            if "NANOCHAT_BASE_DIR" in os.environ:
                del os.environ["NANOCHAT_BASE_DIR"]
            with patch("os.makedirs"):
                result = get_base_dir(require_explicit=False)
                assert result == os.path.expanduser("~/.cache/nanochat")
