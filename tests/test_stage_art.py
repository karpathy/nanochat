"""
Test module for nanochat.stage_art.

Author: Claude Haiku 4.5
Created: 2026-11-14
Scope: Unit tests for ASCII art plaque renderer and CLI.
"""

import pytest

from nanochat import stage_art


class TestRenderPlaque:
    """Test render_plaque function."""

    def test_given_nanochat_plain_set_when_render_plaque_called_then_returns_empty_string(self, monkeypatch):
        """
        Scenario: render_plaque returns empty string when NANOCHAT_PLAIN=1.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN=1.

        ### When
        We call render_plaque("base_train", "start", "some subtitle").

        ### Then
        It returns empty string.
        """
        monkeypatch.setenv("NANOCHAT_PLAIN", "1")

        result = stage_art.render_plaque("base_train", "start", "some subtitle")
        assert result == ""

    def test_given_nanochat_plain_not_set_when_render_plaque_called_then_returns_non_empty_string(self, monkeypatch):
        """
        Scenario: render_plaque returns plaque text when NANOCHAT_PLAIN is not 1.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set or not 1.

        ### When
        We call render_plaque("base_train", "start", "some subtitle").

        ### Then
        It returns a non-empty string.
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        result = stage_art.render_plaque("base_train", "start", "some subtitle")
        assert result != ""
        assert len(result) > 0

    def test_given_base_train_stage_when_render_plaque_called_then_result_contains_stage_name(self, monkeypatch):
        """
        Scenario: Plaque contains the stage name.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set.

        ### When
        We call render_plaque("base_train", "start", "subtitle").

        ### Then
        The plaque contains "base_train".
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        result = stage_art.render_plaque("base_train", "start", "subtitle")
        assert "base_train" in result

    def test_given_start_kind_when_render_plaque_called_then_result_contains_start(self, monkeypatch):
        """
        Scenario: Plaque for 'start' kind contains START.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set, kind is 'start'.

        ### When
        We call render_plaque.

        ### Then
        The plaque contains "START".
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        result = stage_art.render_plaque("base_train", "start", "subtitle")
        assert "START" in result

    def test_given_done_kind_when_render_plaque_called_then_result_contains_done(self, monkeypatch):
        """
        Scenario: Plaque for 'done' kind contains DONE.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set, kind is 'done'.

        ### When
        We call render_plaque.

        ### Then
        The plaque contains "DONE".
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        result = stage_art.render_plaque("base_train", "done", "subtitle")
        assert "DONE" in result

    def test_given_fail_kind_when_render_plaque_called_then_result_contains_fail(self, monkeypatch):
        """
        Scenario: Plaque for 'fail' kind contains FAIL.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set, kind is 'fail'.

        ### When
        We call render_plaque.

        ### Then
        The plaque contains "FAIL".
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        result = stage_art.render_plaque("base_train", "fail", "subtitle")
        assert "FAIL" in result

    def test_given_subtitle_when_render_plaque_called_then_result_contains_subtitle_text(self, monkeypatch):
        """
        Scenario: Plaque contains the subtitle text.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set.

        ### When
        We call render_plaque with subtitle "test subtitle".

        ### Then
        The plaque contains "test subtitle".
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        result = stage_art.render_plaque("base_train", "start", "test subtitle")
        assert "test subtitle" in result

    def test_given_render_plaque_when_called_then_uses_only_ascii_characters_for_borders(self, monkeypatch):
        """
        Scenario: Plaque uses only ASCII characters for borders.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set.

        ### When
        We call render_plaque.

        ### Then
        The plaque contains only ASCII characters (ord < 128) except for ANSI codes.
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        result = stage_art.render_plaque("base_train", "start", "subtitle")

        # Remove ANSI escape sequences for checking
        import re
        clean = re.sub(r"\033\[[0-9;]*m", "", result)

        # Check that all characters are ASCII (ord < 128)
        for char in clean:
            assert ord(char) < 128, f"Non-ASCII character found: {repr(char)} (ord={ord(char)})"

    def test_given_render_plaque_when_called_then_contains_border_characters(self, monkeypatch):
        """
        Scenario: Plaque borders use +, -, and | characters.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set.

        ### When
        We call render_plaque.

        ### Then
        The plaque contains +, -, and | for borders.
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        result = stage_art.render_plaque("base_train", "start", "subtitle")

        # Remove ANSI codes
        import re
        clean = re.sub(r"\033\[[0-9;]*m", "", result)

        # Check for border characters
        assert "+" in clean, "Plaque missing + corners"
        assert "-" in clean, "Plaque missing - horizontal border"
        assert "|" in clean, "Plaque missing | vertical border"

    def test_given_render_plaque_when_called_then_result_is_multiline(self, monkeypatch):
        """
        Scenario: Plaque is multi-line (contains newlines).
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set.

        ### When
        We call render_plaque.

        ### Then
        The result contains at least 3 newlines (top border, 2 content lines, bottom border).
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        result = stage_art.render_plaque("base_train", "start", "subtitle")
        newline_count = result.count("\n")
        assert newline_count >= 3, f"Plaque has {newline_count} newlines (expected >=3)"

    def test_given_very_long_subtitle_when_render_plaque_called_then_renders_without_error(self, monkeypatch):
        """
        Scenario: Long subtitle is truncated with ellipsis.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set.

        ### When
        We call render_plaque with a very long subtitle (100+ chars).

        ### Then
        The plaque still renders without error and contains ellipsis or truncation.
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        long_subtitle = "x" * 150
        result = stage_art.render_plaque("base_train", "start", long_subtitle)
        assert result != ""
        # Should either truncate or handle gracefully
        assert len(result) < len(long_subtitle) * 3  # Rough sanity check

    def test_given_all_kinds_when_render_plaque_called_then_each_returns_non_empty_string(self, monkeypatch):
        """
        Scenario: All three kinds (start, done, fail) render without error.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set.

        ### When
        We call render_plaque for each kind.

        ### Then
        Each returns a non-empty string.
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        for kind in ["start", "done", "fail"]:
            result = stage_art.render_plaque("base_train", kind, "subtitle")
            assert result != "", f"Kind '{kind}' returned empty string"
            assert len(result) > 0

    def test_given_empty_subtitle_when_render_plaque_called_then_returns_valid_plaque(self, monkeypatch):
        """
        Scenario: Plaque handles empty subtitle.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_PLAIN is not set.

        ### When
        We call render_plaque with empty subtitle.

        ### Then
        It returns a valid plaque (no error).
        """
        monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)

        result = stage_art.render_plaque("base_train", "start", "")
        assert result != ""


def test_given_any_stage_and_kind_when_plaque_rendered_then_every_line_has_equal_width(monkeypatch) -> None:
    """
    Scenario: every plaque line lines up so the right-hand border is straight.
    Slice: cli-learn-color-ux
    """
    # given
    monkeypatch.delenv("NANOCHAT_PLAIN", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    # when
    rendered = stage_art.render_plaque("base_train", "start", "teach raw next-token prediction on web text")

    # then
    widths = {len(line) for line in rendered.splitlines()}
    assert len(widths) == 1, f"plaque lines have mixed widths {sorted(widths)}"
