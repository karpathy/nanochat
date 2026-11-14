"""
Module docstring for nanochat.term_style.

Author: Claude Haiku 4.5
Created: 2026-11-14
Scope: ANSI color helper for direct print output (not logging records).
  - enabled() → True only when stdout is a TTY and NO_COLOR is unset
  - paint(role, text) → wraps text in ANSI codes or returns unchanged
"""

import os
import sys

# ANSI color codes
_ANSI_CODES = {
    "bold_cyan": "\033[1;36m",
    "bold_green": "\033[1;32m",
    "dim_yellow": "\033[2;33m",
    "bold_red": "\033[1;31m",
    "reset": "\033[0m",
}


def enabled() -> bool:
    """
    Check if ANSI color output is enabled.

    Returns True only when:
    - stdout is a TTY (interactive terminal)
    - NO_COLOR environment variable is not set

    Returns:
        bool: True if color output should be enabled.
    """
    if not sys.stdout.isatty():
        return False

    return "NO_COLOR" not in os.environ


def paint(role: str, text: str) -> str:
    """
    Wrap text in ANSI color codes based on role.

    If color is not enabled, returns text unchanged.
    If role is unknown, returns text unchanged.

    Args:
        role: One of "start", "force", "done", "skip", "fail".
              "start" and "force" → bold cyan
              "done" → bold green
              "skip" → dim yellow
              "fail" → bold red
        text: The text to wrap.

    Returns:
        str: Text wrapped in ANSI codes (if enabled) or unchanged text.
    """
    if not enabled():
        return text

    # Map roles to ANSI codes
    role_map = {
        "start": "bold_cyan",
        "force": "bold_cyan",
        "done": "bold_green",
        "skip": "dim_yellow",
        "fail": "bold_red",
    }

    code_name = role_map.get(role)
    if code_name is None:
        return text

    code = _ANSI_CODES[code_name]
    reset = _ANSI_CODES["reset"]
    return f"{code}{text}{reset}"
