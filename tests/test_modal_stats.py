"""
Author: nanochat Modal slice (Claude)
Created: 2026-11-14
Scope: runs/modal_stats.py resource verdicts (over/under-provisioned cores and memory),
       the memory-limit source, and unknown values for runs where nothing was measured.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs"))

from modal_stats import (
    core_verdict,
    memory_verdict,
    parse_memory_max,
    resolve_memory_limit,
    summarize_run,
)

GIB = 2**30


def test_parse_memory_max_reads_bytes_and_treats_max_as_no_limit() -> None:
    """
    Scenario: cgroup memory.max is a byte count, or 'max' when unlimited.
    Slice: reading the container memory limit.
    """
    # ### Given / ### When / ### Then
    assert parse_memory_max("137438953472\n") == 128 * GIB
    assert parse_memory_max("max\n") is None
    assert parse_memory_max("") is None


def test_core_verdict_judges_busy_cores_against_configured_cores() -> None:
    """
    Scenario: 64 cores configured; busy cores decide the verdict.
    Slice: core right-sizing against what the container really has.
    """
    # ### Given
    configured = 64

    # ### When / ### Then
    assert core_verdict(10.0, configured).startswith("over-provisioned cores")
    assert core_verdict(40.0, configured).startswith("cores ok")
    assert core_verdict(58.48, configured) == "cores saturated (58.5/64); more cores may help"


def test_core_verdict_reports_unknown_when_nothing_was_measured() -> None:
    """
    Scenario: the run crashed before any child process ran, so there is no CPU sample.
    Slice: a crash must not look like over-provisioning.
    """
    # ### Given / ### When / ### Then
    assert core_verdict(None, 64) == "cores unknown (no child process measured)"


def test_memory_verdict_flags_tight_loose_and_ok() -> None:
    """
    Scenario: 128 GiB limit; peak RSS decides tight, loose, or ok.
    Slice: memory right-sizing.
    """
    # ### Given
    limit = 128 * GIB

    # ### When / ### Then
    assert memory_verdict(120 * GIB, limit).startswith("memory tight")
    assert memory_verdict(10 * GIB, limit).startswith("memory loose")
    assert memory_verdict(60 * GIB, limit).startswith("memory ok")


def test_memory_verdict_without_any_limit_reports_peak_only() -> None:
    # ### Given / ### When / ### Then
    assert memory_verdict(10 * GIB, None) == "memory peak 10.0 GiB (no memory limit known)"


def test_memory_verdict_reports_unknown_when_nothing_was_measured() -> None:
    # ### Given / ### When / ### Then
    assert memory_verdict(None, 128 * GIB) == "memory unknown (no child process measured)"


def test_resolve_memory_limit_prefers_cgroup_then_configured_then_none() -> None:
    """
    Scenario: the cgroup limit is visible, or only the configured 128 GiB is known, or neither.
    Slice: the verdict always has a limit when one is known, and records which source it came from.
    """
    # ### Given / ### When / ### Then
    assert resolve_memory_limit(64 * GIB, 128 * GIB) == (64 * GIB, "cgroup")
    assert resolve_memory_limit(None, 128 * GIB) == (128 * GIB, "configured")
    assert resolve_memory_limit(None, None) == (None, "none")


def test_summarize_run_reports_timing_and_both_verdicts() -> None:
    """
    Scenario: a 1 h run on 64 configured cores using 6 cores on average, peak 20 GiB of 128 GiB.
    Slice: full run report that tells the user what to change.
    """
    # ### Given / ### When
    report = summarize_run(
        wall_s=3600.0,
        child_cpu_s=6 * 3600.0,
        cores_configured=64,
        peak_rss_kb=20 * 1024 * 1024,
        cgroup_memory_limit_bytes=None,
        configured_memory_bytes=128 * GIB,
        timeout_s=3 * 3600,
    )

    # ### Then
    assert report["avg_cores_busy"] == 6.0
    assert report["peak_rss_gib"] == 20.0
    assert report["timeout_share"] == round(1 / 3, 3)
    assert report["verdicts"][0].startswith("over-provisioned cores")
    assert report["verdicts"][1].startswith("memory loose")


def test_summarize_run_uses_configured_limit_and_records_source() -> None:
    """
    Scenario: run 20261116-003831-cpu. 58.48 busy cores of 64 configured; 25.1 GiB peak; no cgroup limit visible.
    Slice: the memory verdict gets a real limit (configured 128 GiB) instead of null.
    """
    # ### Given
    wall_s = 3600.0

    # ### When
    report = summarize_run(
        wall_s=wall_s,
        child_cpu_s=58.48 * wall_s,
        cores_configured=64,
        peak_rss_kb=int(25.1 * 1024 * 1024),
        cgroup_memory_limit_bytes=None,
        configured_memory_bytes=128 * GIB,
        timeout_s=3 * 3600,
    )

    # ### Then
    assert report["cores_configured"] == 64
    assert report["avg_cores_busy"] == 58.48
    assert report["verdicts"][0] == "cores saturated (58.5/64); more cores may help"
    assert report["memory_limit_gib"] == 128.0
    assert report["memory_limit_source"] == "configured"
    assert report["verdicts"][1].startswith("memory loose")


def test_summarize_run_marks_crashed_run_as_unknown_not_over_provisioned() -> None:
    """
    Scenario: run 20261114-125440-cpu crashed before training; no child CPU time, no peak RSS.
    Slice: unmeasured values are None and the verdicts say unknown.
    """
    # ### Given / ### When
    report = summarize_run(
        wall_s=120.0,
        child_cpu_s=0.0,
        cores_configured=64,
        peak_rss_kb=0,
        cgroup_memory_limit_bytes=None,
        configured_memory_bytes=128 * GIB,
        timeout_s=3 * 3600,
    )

    # ### Then
    assert report["avg_cores_busy"] is None
    assert report["peak_rss_gib"] is None
    assert report["verdicts"] == [
        "cores unknown (no child process measured)",
        "memory unknown (no child process measured)",
    ]
