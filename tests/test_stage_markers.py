"""
Author: Mani Sarkar
Created: 2026-11-15
Scope: runs/_stage_lib.sh — profile+depth stage markers (B2 fix).
       Cases: profile+depth match skips, profile mismatch runs, depth mismatch runs,
              old-style marker (no profile/depth fields) is treated as mismatch and runs.

Run: uv run pytest tests/test_stage_markers.py -q
"""
import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_LIB = os.path.join(REPO_ROOT, "runs", "_stage_lib.sh")


def _run_bash(script: str, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        check=False, capture_output=True, text=True, timeout=timeout,
    )


def _stage_lib_preamble(tmp_dir: str, profile: str, depth: int) -> str:
    """Return bash preamble that sources _stage_lib.sh with the given profile+depth."""
    return f"""
set -euo pipefail
export NANOCHAT_BASE_DIR="{tmp_dir}"
PROFILE={profile}
DEPTH={depth}
source "{STAGE_LIB}"
mkdir -p "{tmp_dir}/base_checkpoints" "{tmp_dir}/chatsft_checkpoints"
"""


class TestStageMarkers:
    """
    Scenario: Stage markers carry profile+depth so GPU and CPU runs do not share skip state.
    Slice: runs/_stage_lib.sh stage_done() / _stage_marker_path() / run_stage() (B2 fix).
    """

    def test_matching_profile_and_depth_causes_skip(self, tmp_path):
        """
        Scenario: Profile and depth match the existing marker → stage is skipped.
        Slice: stage_done returns true when profile+depth marker is present
        ### Given: a marker written for profile=runcpu depth=6 (the current profile+depth)
        ### When: stage_done is checked for base_train with the same profile+depth
        ### Then: returns true (stage is considered done → SKIP in run_stage)
        """
        script = _stage_lib_preamble(str(tmp_path), "runcpu", 6) + f"""
marker="$(_stage_marker_path base_train)"
touch "$marker"
stage_done base_train && echo "MATCH_DONE" || echo "MATCH_NOT_DONE"
"""
        r = _run_bash(script)
        assert r.returncode == 0, r.stderr
        assert "MATCH_DONE" in r.stdout

    def test_profile_mismatch_causes_run_with_reason(self, tmp_path):
        """
        Scenario: Existing marker is for profile=runcpu; current profile=speedrun → must RUN.
        Slice: stage_done returns false on profile mismatch; run_stage logs the reason
        ### Given: a marker written for profile=runcpu depth=6
        ### When: run_stage is called with profile=speedrun depth=24
        ### Then: stage_done returns false; run_stage prints a mismatch reason line
        """
        # Write a marker for the wrong profile
        script_write = _stage_lib_preamble(str(tmp_path), "runcpu", 6) + """
marker="$(_stage_marker_path base_train)"
touch "$marker"
"""
        r = _run_bash(script_write)
        assert r.returncode == 0, r.stderr

        # Now check with the correct profile (speedrun/24): stage must NOT be done
        script_check = _stage_lib_preamble(str(tmp_path), "speedrun", 24) + """
stage_done base_train && echo "PROFILE_DONE" || echo "PROFILE_NOT_DONE"
"""
        r = _run_bash(script_check)
        assert r.returncode == 0, r.stderr
        assert "PROFILE_NOT_DONE" in r.stdout, (
            "Profile mismatch: stage_done should return false when marker is for a different profile"
        )

        # run_stage must log a mismatch reason line
        script_run = _stage_lib_preamble(str(tmp_path), "speedrun", 24) + """
out="$(NANOCHAT_PLAIN=1 NO_COLOR=1 run_stage base_train "" "next" -- echo dummy 2>&1)"
echo "$out"
"""
        r = _run_bash(script_run)
        assert "mismatch" in r.stdout.lower(), (
            f"Expected mismatch reason in output, got:\n{r.stdout}"
        )

    def test_depth_mismatch_causes_run_with_reason(self, tmp_path):
        """
        Scenario: Existing marker is for depth=6; current depth=24 → must RUN.
        Slice: stage_done returns false on depth mismatch; run_stage logs the reason
        ### Given: a marker written for profile=runcpu depth=6
        ### When: run_stage is called with the same profile but depth=24
        ### Then: stage_done returns false; run_stage prints a mismatch reason line
        """
        # Write a marker for depth=6
        script_write = _stage_lib_preamble(str(tmp_path), "runcpu", 6) + """
marker="$(_stage_marker_path base_train)"
touch "$marker"
"""
        r = _run_bash(script_write)
        assert r.returncode == 0, r.stderr

        # Check with depth=24: must not be done
        script_check = _stage_lib_preamble(str(tmp_path), "runcpu", 24) + """
stage_done base_train && echo "DEPTH_DONE" || echo "DEPTH_NOT_DONE"
"""
        r = _run_bash(script_check)
        assert r.returncode == 0, r.stderr
        assert "DEPTH_NOT_DONE" in r.stdout, (
            "Depth mismatch: stage_done should return false when marker is for a different depth"
        )

        # run_stage must log a mismatch reason line
        script_run = _stage_lib_preamble(str(tmp_path), "runcpu", 24) + """
out="$(NANOCHAT_PLAIN=1 NO_COLOR=1 run_stage base_train "" "next" -- echo dummy 2>&1)"
echo "$out"
"""
        r = _run_bash(script_run)
        assert "mismatch" in r.stdout.lower(), (
            f"Expected mismatch reason in output, got:\n{r.stdout}"
        )

    def test_old_style_marker_without_profile_depth_causes_run(self, tmp_path):
        """
        Scenario: Only the old .train_ok exists (no profile+depth marker) → must RUN.
        Slice: backward compatibility — older marker is treated as mismatch
        ### Given: only the old-style .train_ok marker exists (from a pre-B2-fix run)
        ### When: stage_done is checked for base_train with any profile+depth
        ### Then: returns false (stage reruns to produce a correct profile+depth marker)
        """
        script = _stage_lib_preamble(str(tmp_path), "runcpu", 6) + f"""
# Write only the old generic marker, not the new profile+depth one
touch "{tmp_path}/base_checkpoints/.train_ok"
stage_done base_train && echo "OLD_DONE" || echo "OLD_NOT_DONE"
"""
        r = _run_bash(script)
        assert r.returncode == 0, r.stderr
        assert "OLD_NOT_DONE" in r.stdout, (
            "Old-style .train_ok alone must not cause a SKIP (treated as mismatch)"
        )

    def test_tok_train_uses_legacy_path_no_profile_depth(self, tmp_path):
        """
        Scenario: tok_train is depth-independent and uses the original artifact path.
        Slice: _stage_marker_path returns empty for tok_train; stage_done uses tokenizer.pkl
        ### Given: tokenizer.pkl exists
        ### When: stage_done is checked for tok_train
        ### Then: returns true (tokenizer is profile-independent)
        """
        tok_dir = tmp_path / "tokenizer"
        tok_dir.mkdir()
        (tok_dir / "tokenizer.pkl").touch()

        script = _stage_lib_preamble(str(tmp_path), "runcpu", 6) + """
stage_done tok_train && echo "TOK_DONE" || echo "TOK_NOT_DONE"
"""
        r = _run_bash(script)
        assert r.returncode == 0, r.stderr
        assert "TOK_DONE" in r.stdout, (
            "tok_train should still use legacy artifact path (tokenizer is depth-independent)"
        )


class TestStageLibDepthGuard:
    """
    Scenario: _stage_lib.sh must fail loudly when DEPTH is unset for depth-sensitive stages.
    Slice: runs/_stage_lib.sh FIX 2 — DEPTH:? guard in _stage_marker_path
    """

    def test_depth_unset_fails_for_depth_sensitive_stage(self, tmp_path):
        """
        Scenario: Caller forgot to set DEPTH; calling _stage_marker_path on base_train must fail.
        Slice: FIX 2 — ${DEPTH:?} guard in _stage_marker_path depth branch
        ### Given: _stage_lib.sh sourced without DEPTH set
        ### When: _stage_marker_path is called for base_train
        ### Then: exits non-zero with a message containing "DEPTH"
        """
        script = f"""
set -euo pipefail
export NANOCHAT_BASE_DIR="{tmp_path}"
PROFILE=runcpu
# Deliberately do NOT set DEPTH
source "{STAGE_LIB}"
mkdir -p "{tmp_path}/base_checkpoints"
_stage_marker_path base_train
"""
        r = _run_bash(script)
        assert r.returncode != 0, "Expected failure when DEPTH is unset for a depth-sensitive stage"
        assert "DEPTH" in r.stderr, (
            f"Expected 'DEPTH' in error output; got:\n{r.stderr}"
        )

    def test_depth_unset_does_not_fail_for_tok_train(self, tmp_path):
        """
        Scenario: tok_train is depth-independent; calling _stage_marker_path must succeed even
                  when DEPTH is unset.
        Slice: FIX 2 — depth-free stages return empty string without touching DEPTH
        ### Given: _stage_lib.sh sourced without DEPTH set
        ### When: _stage_marker_path is called for tok_train
        ### Then: exits 0 and outputs an empty string (no marker path for depth-free stages)
        """
        script = f"""
set -euo pipefail
export NANOCHAT_BASE_DIR="{tmp_path}"
PROFILE=runcpu
# Deliberately do NOT set DEPTH
source "{STAGE_LIB}"
result="$(_stage_marker_path tok_train)"
[ -z "$result" ] && echo "EMPTY_OK" || echo "NOT_EMPTY:$result"
"""
        r = _run_bash(script)
        assert r.returncode == 0, r.stderr
        assert "EMPTY_OK" in r.stdout, (
            "tok_train _stage_marker_path should return empty string without needing DEPTH"
        )
