"""
Test module for nanochat.term_style.

Author: Claude Haiku 4.5
Created: 2026-11-14
Scope: Unit tests for ANSI color helpers enabled() and paint().
"""


from nanochat import term_style


class TestEnabled:
    """Test enabled() function."""

    def test_given_non_tty_stdout_when_enabled_called_then_returns_false(self, monkeypatch):
        """
        Scenario: enabled() returns False when stdout is not a TTY.
        Slice: cli-learn-color-ux

        ### Given
        stdout is not a TTY (simulated by monkeypatch).

        ### When
        We call enabled().

        ### Then
        It returns False.
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        monkeypatch.delenv("NO_COLOR", raising=False)

        assert term_style.enabled() is False

    def test_given_tty_and_no_color_set_when_enabled_called_then_returns_false(self, monkeypatch):
        """
        Scenario: enabled() returns False when NO_COLOR is set.
        Slice: cli-learn-color-ux

        ### Given
        stdout is a TTY but NO_COLOR=1.

        ### When
        We call enabled().

        ### Then
        It returns False.
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setenv("NO_COLOR", "1")

        assert term_style.enabled() is False

    def test_given_tty_and_no_color_empty_when_enabled_called_then_returns_false(self, monkeypatch):
        """
        Scenario: enabled() returns False when NO_COLOR is empty string.
        Slice: cli-learn-color-ux

        ### Given
        stdout is a TTY but NO_COLOR="".

        ### When
        We call enabled().

        ### Then
        It returns False (empty string is falsy).
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setenv("NO_COLOR", "")

        assert term_style.enabled() is False

    def test_given_tty_and_no_color_unset_when_enabled_called_then_returns_true(self, monkeypatch):
        """
        Scenario: enabled() returns True when stdout is TTY and NO_COLOR is unset.
        Slice: cli-learn-color-ux

        ### Given
        stdout is a TTY and NO_COLOR is not set.

        ### When
        We call enabled().

        ### Then
        It returns True.
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)

        assert term_style.enabled() is True


class TestPaint:
    """Test paint() function."""

    def test_given_color_disabled_when_paint_called_then_returns_text_unchanged(self, monkeypatch):
        """
        Scenario: paint() returns text unchanged when color is disabled.
        Slice: cli-learn-color-ux

        ### Given
        NO_COLOR is set (color disabled).

        ### When
        We call paint("start", "text").

        ### Then
        It returns "text" unchanged.
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        monkeypatch.setenv("NO_COLOR", "1")

        result = term_style.paint("start", "text")
        assert result == "text"

    def test_given_start_role_and_color_enabled_when_paint_called_then_wraps_in_bold_cyan(self, monkeypatch):
        """
        Scenario: paint() wraps text in bold cyan for 'start' role when enabled.
        Slice: cli-learn-color-ux

        ### Given
        Color is enabled and role is 'start'.

        ### When
        We call paint("start", "START").

        ### Then
        It returns text wrapped in bold cyan ANSI codes.
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)

        result = term_style.paint("start", "START")
        assert "\033[1;36m" in result  # bold cyan
        assert "START" in result
        assert "\033[0m" in result  # reset

    def test_given_force_role_and_color_enabled_when_paint_called_then_wraps_in_bold_cyan(self, monkeypatch):
        """
        Scenario: paint() wraps text in bold cyan for 'force' role when enabled.
        Slice: cli-learn-color-ux

        ### Given
        Color is enabled and role is 'force'.

        ### When
        We call paint("force", "FORCE").

        ### Then
        It returns text wrapped in bold cyan ANSI codes (same as start).
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)

        result = term_style.paint("force", "FORCE")
        assert "\033[1;36m" in result  # bold cyan
        assert "FORCE" in result
        assert "\033[0m" in result

    def test_given_done_role_and_color_enabled_when_paint_called_then_wraps_in_bold_green(self, monkeypatch):
        """
        Scenario: paint() wraps text in bold green for 'done' role when enabled.
        Slice: cli-learn-color-ux

        ### Given
        Color is enabled and role is 'done'.

        ### When
        We call paint("done", "DONE").

        ### Then
        It returns text wrapped in bold green ANSI codes.
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)

        result = term_style.paint("done", "DONE")
        assert "\033[1;32m" in result  # bold green
        assert "DONE" in result
        assert "\033[0m" in result

    def test_given_skip_role_and_color_enabled_when_paint_called_then_wraps_in_dim_yellow(self, monkeypatch):
        """
        Scenario: paint() wraps text in dim yellow for 'skip' role when enabled.
        Slice: cli-learn-color-ux

        ### Given
        Color is enabled and role is 'skip'.

        ### When
        We call paint("skip", "SKIP").

        ### Then
        It returns text wrapped in dim yellow ANSI codes.
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)

        result = term_style.paint("skip", "SKIP")
        assert "\033[2;33m" in result  # dim yellow
        assert "SKIP" in result
        assert "\033[0m" in result

    def test_given_fail_role_and_color_enabled_when_paint_called_then_wraps_in_bold_red(self, monkeypatch):
        """
        Scenario: paint() wraps text in bold red for 'fail' role when enabled.
        Slice: cli-learn-color-ux

        ### Given
        Color is enabled and role is 'fail'.

        ### When
        We call paint("fail", "FAIL").

        ### Then
        It returns text wrapped in bold red ANSI codes.
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)

        result = term_style.paint("fail", "FAIL")
        assert "\033[1;31m" in result  # bold red
        assert "FAIL" in result
        assert "\033[0m" in result

    def test_given_unknown_role_and_color_enabled_when_paint_called_then_returns_text_unchanged(self, monkeypatch):
        """
        Scenario: paint() returns text unchanged for unknown role.
        Slice: cli-learn-color-ux

        ### Given
        Color is enabled but role is unknown.

        ### When
        We call paint("unknown", "text").

        ### Then
        It returns "text" unchanged.
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)

        result = term_style.paint("unknown", "text")
        assert result == "text"

    def test_given_all_known_roles_and_color_enabled_when_paint_called_then_all_results_end_with_reset_code(self, monkeypatch):
        """
        Scenario: All painted text includes the reset code.
        Slice: cli-learn-color-ux

        ### Given
        Color is enabled.

        ### When
        We call paint for each known role.

        ### Then
        Each result ends with reset code.
        """
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)

        for role in ["start", "force", "done", "skip", "fail"]:
            result = term_style.paint(role, "text")
            assert result.endswith("\033[0m"), f"Role '{role}' missing reset code"
