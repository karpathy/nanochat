"""Resource report for a Modal run: what was configured vs what the job actually used.

Pure functions only (no modal import) so they can be unit-tested locally.
Used by runs/modal_run.py after a training run to say whether cores and memory
were over- or under-provisioned.
"""

from __future__ import annotations

from typing import Any

# Thresholds for the verdicts. Tune after the first real measurements.
CORE_UNDERUSE = 0.5  # below this share of configured cores busy → over-provisioned cores
CORE_SATURATED = 0.9  # above this → more cores may help
MEMORY_TIGHT = 0.85  # peak RSS above this share of the limit → raise memory
MEMORY_LOOSE = 0.25  # peak RSS below this share → lower memory


def parse_memory_max(text: str) -> int | None:
    """Parse a cgroup v2 memory.max value: a byte count, or 'max' (no limit) → None."""
    value = text.strip()
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def resolve_memory_limit(
    cgroup_limit_bytes: int | None, configured_bytes: int | None
) -> tuple[int | None, str]:
    """Pick the memory limit to judge against, and say where it came from."""
    if cgroup_limit_bytes:
        return cgroup_limit_bytes, "cgroup"
    if configured_bytes:
        return configured_bytes, "configured"
    return None, "none"


def core_verdict(avg_cores_busy: float | None, cores_configured: int) -> str:
    """Say whether the configured cores were used. None means no child process was measured."""
    if avg_cores_busy is None:
        return "cores unknown (no child process measured)"
    share = avg_cores_busy / cores_configured if cores_configured else 0.0
    if share < CORE_UNDERUSE:
        return f"over-provisioned cores: only {avg_cores_busy:.1f} of {cores_configured} busy; try fewer"
    if share > CORE_SATURATED:
        return f"cores saturated ({avg_cores_busy:.1f}/{cores_configured}); more cores may help"
    return f"cores ok ({avg_cores_busy:.1f}/{cores_configured} busy)"


def memory_verdict(peak_bytes: int | None, limit_bytes: int | None) -> str:
    """Say whether the memory request was right. None peak means nothing was measured."""
    if peak_bytes is None:
        return "memory unknown (no child process measured)"
    if not limit_bytes:
        return f"memory peak {peak_bytes / 2**30:.1f} GiB (no memory limit known)"
    share = peak_bytes / limit_bytes
    if share > MEMORY_TIGHT:
        return f"memory tight: peak {share:.0%} of limit; raise memory"
    if share < MEMORY_LOOSE:
        return f"memory loose: peak {share:.0%} of limit; lower memory"
    return f"memory ok: peak {share:.0%} of limit"


def summarize_run(
    *,
    wall_s: float,
    child_cpu_s: float,
    cores_configured: int,
    peak_rss_kb: int,
    cgroup_memory_limit_bytes: int | None,
    configured_memory_bytes: int | None,
    timeout_s: int,
) -> dict[str, Any]:
    """Build the run report: timing, core use, memory use and verdicts.

    Unmeasured values (no child process ran, e.g. the run crashed before training) are
    None, so a crash does not read as over-provisioning.
    """
    avg_cores_busy = child_cpu_s / wall_s if child_cpu_s > 0 and wall_s > 0 else None
    peak_bytes = peak_rss_kb * 1024 if peak_rss_kb > 0 else None
    limit_bytes, limit_source = resolve_memory_limit(cgroup_memory_limit_bytes, configured_memory_bytes)
    return {
        "wall_s": round(wall_s, 1),
        "timeout_s": timeout_s,
        "timeout_share": round(wall_s / timeout_s, 3) if timeout_s else None,
        "cores_configured": cores_configured,
        "avg_cores_busy": round(avg_cores_busy, 2) if avg_cores_busy is not None else None,
        "peak_rss_gib": round(peak_bytes / 2**30, 2) if peak_bytes is not None else None,
        "memory_limit_gib": round(limit_bytes / 2**30, 2) if limit_bytes else None,
        "memory_limit_source": limit_source,
        "verdicts": [
            core_verdict(avg_cores_busy, cores_configured),
            memory_verdict(peak_bytes, limit_bytes),
        ],
    }
