"""
Author: nanochat Modal slice (Claude)
Created: 2026-11-14
Scope: nanochat.common.format_duration, the 'Hh Mm Ss' form used in training logs.
"""

import pytest

from nanochat.common import format_duration


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0h 0m 0s"),
        (42, "0h 0m 42s"),
        (125, "0h 2m 5s"),
        (3725, "1h 2m 5s"),
        (489.8 * 60, "8h 9m 48s"),
        (-5, "0h 0m 0s"),
    ],
)
def test_format_duration_shows_hours_minutes_seconds(seconds: float, expected: str) -> None:
    """
    Scenario: durations of seconds, minutes, hours, and a long ETA are shown as Hh Mm Ss.
    Slice: readable time in training logs.
    """
    # ### Given / ### When
    text = format_duration(seconds)

    # ### Then
    assert text == expected
