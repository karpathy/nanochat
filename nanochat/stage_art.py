"""
Module docstring for nanochat.stage_art.

Author: Claude Haiku 4.5
Created: 2026-11-14
Scope: ASCII art plaque renderer for stage boundaries.
  - render_plaque(stage, kind, subtitle) → ASCII box with stage name, kind, and subtitle
  - Respects NANOCHAT_PLAIN=1 to suppress output
  - Pure function; I/O in CLI main
  - CLI: plaque --stage X --kind start|done|fail --subtitle "..."
"""

import argparse
import os
import sys
from typing import Literal

from nanochat import term_style


def render_plaque(
    stage: str,
    kind: Literal["start", "done", "fail"],
    subtitle: str,
) -> str:
    """
    Render an ASCII art plaque for a stage boundary.

    Args:
        stage: Stage name (e.g., 'base_train').
        kind: One of 'start', 'done', 'fail'.
        subtitle: Subtitle text (e.g., stage purpose).

    Returns:
        str: Multi-line ASCII plaque or empty string if NANOCHAT_PLAIN=1.
             Borders are ASCII-only (+, -, |).
             The kind label is colorized (if term_style.enabled()) but not the borders.
    """
    # Check NANOCHAT_PLAIN: if set to 1, return empty (no plaque).
    if os.environ.get("NANOCHAT_PLAIN") == "1":
        return ""

    kind_upper = kind.upper()

    # Compute internal width: space for content between the | borders
    # Need to fit: "stage_name KIND" (e.g., "base_train START")
    min_width = max(len(stage) + 1 + len(kind_upper), len(subtitle))
    # Add padding
    content_width = max(min_width + 2, 30)
    # Clamp to reasonable max
    content_width = min(content_width, 76)

    # Build the plaque with borders
    border = "+" + "-" * content_width + "+"

    # Line 1: "stage_name KIND"
    line1_text = f"{stage} {kind_upper}"
    if len(line1_text) > content_width:
        # Truncate with ".." (ASCII only)
        line1_text = line1_text[: content_width - 2] + ".."
    line1_text = line1_text.ljust(content_width)
    # Paint the line with the kind color
    line1 = f"| {term_style.paint(kind, line1_text)} |"

    # Line 2: subtitle
    line2_text = subtitle
    if len(line2_text) > content_width:
        # Truncate with ".." (ASCII only)
        line2_text = line2_text[: content_width - 2] + ".."
    line2_text = line2_text.ljust(content_width)
    line2 = f"| {line2_text} |"

    return "\n".join([border, line1, line2, border])


def main() -> None:
    """CLI entry point for stage_art.plaque."""
    parser = argparse.ArgumentParser(
        description="Stage art module CLI: render ASCII plaques."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    plaque_parser = subparsers.add_parser("plaque", help="Render a stage plaque")
    plaque_parser.add_argument(
        "--stage",
        required=True,
        help="Stage name (e.g., base_train)",
    )
    plaque_parser.add_argument(
        "--kind",
        required=True,
        choices=["start", "done", "fail"],
        help="Plaque kind",
    )
    plaque_parser.add_argument(
        "--subtitle",
        required=True,
        help="Subtitle text (e.g., stage purpose)",
    )

    args = parser.parse_args()

    if args.command == "plaque":
        plaque = render_plaque(args.stage, args.kind, args.subtitle)
        if plaque:
            print(plaque)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
