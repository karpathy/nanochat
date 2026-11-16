"""Educational tip text for Phase 1.5 (one tip per script run).

Pure stdlib — no torch. Shell prints via ``python -m nanochat.run_tip``.
"""

from __future__ import annotations

# Fixed tip for now (rotation can come later without changing the call site).
RUN_TIP = (
    "tip: smaller --device-batch-size uses grad accumulation to keep the same total batch"
)


def run_tip() -> str:
    """Return the single tip enrichment line (always starts with ``tip:``)."""
    return RUN_TIP


def main() -> int:
    print(run_tip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
