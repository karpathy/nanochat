"""
Phase 1.5 enrichments: knob diff (last_run) and trend crumb (last_metrics).

Pure helpers only — no torch. Covers ACs 1, 3, 6, 7 from
docs/plan/2026-11-13-fix-forward-mac-phase15.md.

Run: python -m pytest tests/test_phase15_enrichments.py -q
"""

from pathlib import Path

from nanochat.last_run import format_knob_diff, load_last_run, write_last_run
from nanochat.metrics_trend import (
    emit_metric_trend,
    format_trend,
    load_metrics,
    record_metric,
)
from nanochat.run_tip import run_tip

# ----------------------------------------------------------------------------- last_run / knob diff

def test_format_knob_diff_returns_none_when_previous_is_missing():
    # Given: first run (no previous last_run)
    current = {"profile": "runcpu", "depth": 6, "device_batch_size": 8, "max_seq_len": 512}
    # When: formatting the knob diff
    line = format_knob_diff(current, None)
    # Then: no enrichment line
    assert line is None


def test_load_last_run_returns_none_when_file_is_missing(tmp_path):
    # Given: no last_run.json on disk
    path = tmp_path / "last_run.json"
    # When: loading
    # Then: None (first run)
    assert load_last_run(path) is None


def test_format_knob_diff_includes_changed_when_a_key_differs():
    # Given: device_batch_size changed; other knobs same
    previous = {"profile": "runcpu", "depth": 6, "device_batch_size": 32, "max_seq_len": 512}
    current = {"profile": "runcpu", "depth": 6, "device_batch_size": 8, "max_seq_len": 512}
    # When: formatting the knob diff
    line = format_knob_diff(current, previous)
    # Then: one line with changed: and the old→new pair
    assert line is not None
    assert "changed:" in line
    assert "device_batch_size 32→8" in line
    assert "same:" in line


def test_format_knob_diff_includes_same_when_keys_match():
    # Given: all compared knobs identical
    previous = {"profile": "runcpu", "depth": 6, "device_batch_size": 8, "max_seq_len": 512}
    current = dict(previous)
    # When: formatting the knob diff
    line = format_knob_diff(current, previous)
    # Then: one line with same: and no changed:
    assert line is not None
    assert "same:" in line
    assert "changed:" not in line
    assert "depth=6" in line
    assert "profile=runcpu" in line


def test_write_last_run_roundtrips_through_load(tmp_path):
    # Given: knobs to persist
    path = tmp_path / "last_run.json"
    data = {
        "profile": "runcpu",
        "depth": 6,
        "device_batch_size": 8,
        "max_seq_len": 512,
        "git_sha": "abc1234",
        "ts": 1_700_000_000,
    }
    # When: writing then loading
    write_last_run(path, data)
    loaded = load_last_run(path)
    # Then: same payload
    assert loaded == data


# ----------------------------------------------------------------------------- metrics trend

def test_format_trend_returns_none_when_prior_is_absent():
    # Given: no prior metric
    # When: formatting a trend crumb
    line = format_trend("val_bpb", 7.47, None, lower_is_better=True)
    # Then: no enrichment line
    assert line is None


def test_format_trend_reports_better_when_lower_is_better_and_value_drops():
    # Given: val_bpb improved (lower is better)
    # When: formatting
    line = format_trend("val_bpb", 7.47, 8.1, lower_is_better=True)
    # Then: better-than-prior crumb
    assert line is not None
    assert line.startswith("trend:")
    assert "val_bpb" in line
    assert "better than prior" in line
    assert "7.470000" in line
    assert "8.100000" in line


def test_format_trend_reports_worse_when_lower_is_better_and_value_rises():
    # Given: val_bpb regressed
    line = format_trend("val_bpb", 8.5, 7.0, lower_is_better=True)
    # Then: worse-than-prior crumb
    assert line is not None
    assert "worse than prior" in line


def test_format_trend_reports_better_when_higher_is_better_and_value_rises():
    # Given: CORE improved (higher is better)
    line = format_trend("CORE", 0.22, 0.18, lower_is_better=False)
    # Then: better-than-prior crumb
    assert line is not None
    assert line.startswith("trend:")
    assert "CORE" in line
    assert "better than prior" in line


def test_format_trend_reports_worse_when_higher_is_better_and_value_drops():
    # Given: CORE regressed
    line = format_trend("CORE", 0.10, 0.20, lower_is_better=False)
    # Then: worse-than-prior crumb
    assert line is not None
    assert "worse than prior" in line


def test_format_trend_reports_same_when_value_equals_prior():
    # Given: unchanged metric
    line = format_trend("val_bpb", 7.5, 7.5, lower_is_better=True)
    # Then: same-as-prior crumb
    assert line is not None
    assert "same as prior" in line


def test_record_metric_roundtrips_through_load(tmp_path):
    # Given: empty metrics store
    path = tmp_path / "last_metrics.json"
    assert load_metrics(path) == {}
    # When: recording a metric
    record_metric(path, "val_bpb", 7.47)
    # Then: load returns the stored value
    loaded = load_metrics(path)
    assert loaded["val_bpb"] == 7.47


def test_record_metric_merges_without_clobbering_other_keys(tmp_path):
    # Given: an existing metric
    path = tmp_path / "subdir" / "last_metrics.json"
    record_metric(path, "val_bpb", 8.0)
    # When: recording another
    record_metric(path, "CORE", 0.2)
    # Then: both keys present
    loaded = load_metrics(path)
    assert loaded["val_bpb"] == 8.0
    assert loaded["CORE"] == 0.2


def test_format_trend_returns_none_when_prior_is_non_numeric():
    # Given: corrupt prior stored as a non-numeric string
    # When: formatting
    line = format_trend("val_bpb", 7.47, "not-a-number", lower_is_better=True)
    # Then: treat as absent (no crash, no line)
    assert line is None


def test_emit_metric_trend_is_fail_soft_when_path_is_unwritable(tmp_path):
    # Given: a path whose parent cannot be created as a file-as-dir trap
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    path = blocker / "last_metrics.json"
    printed: list[str] = []
    # When: emit would fail on write
    emit_metric_trend(
        str(path),
        "val_bpb",
        7.47,
        lower_is_better=True,
        print_fn=printed.append,
    )
    # Then: no exception escapes (printed may be empty)
    assert printed == [] or all(isinstance(x, str) for x in printed)


def test_emit_metric_trend_prints_then_records(tmp_path):
    # Given: a prior metric on disk
    path = tmp_path / "last_metrics.json"
    record_metric(path, "val_bpb", 8.1)
    printed: list[str] = []
    # When: emitting a better value
    emit_metric_trend(
        str(path),
        "val_bpb",
        7.47,
        lower_is_better=True,
        print_fn=printed.append,
    )
    # Then: one trend line and updated store
    assert len(printed) == 1
    assert printed[0].startswith("trend:")
    assert load_metrics(path)["val_bpb"] == 7.47


# ----------------------------------------------------------------------------- tip once

def test_run_tip_is_one_line_mentioning_device_batch_size():
    # Given: the tip SSOT helper
    # When: fetching the tip
    tip = run_tip()
    # Then: exactly one tip: line mentioning the real knob
    assert tip.startswith("tip:")
    assert "--device-batch-size" in tip
    assert tip.count("\n") == 0


# ----------------------------------------------------------------------------- glossary (learn/term)

def test_given_glossary_learn_stage_when_format_learn_line_then_includes_all_terms():
    # Scenario: Stage has 4 terms (base_train max); format_learn_line returns one line.
    from nanochat.glossary import STAGE_TERMS, format_learn_line

    # given: base_train stage with 4 terms
    stage = "base_train"
    # when: formatting the learn line
    line = format_learn_line(stage)
    # then: one line with all terms and URLs
    assert line.startswith("learn:")
    for term in STAGE_TERMS[stage]:
        assert term in line
    assert "http" in line  # has URLs


def test_given_unknown_stage_when_format_learn_line_then_returns_empty():
    # Scenario: Unknown stage returns empty string.
    from nanochat.glossary import format_learn_line

    # given: unknown stage
    stage = "nonexistent_stage"
    # when: formatting the learn line
    line = format_learn_line(stage)
    # then: empty string (no crash)
    assert line == ""


def test_given_nanochat_stage_briefed_when_emit_define_then_no_output():
    # Scenario: emit_define is a no-op when NANOCHAT_STAGE_BRIEFED=1.
    import os

    from nanochat.glossary import emit_define

    # given: env var set to suppress output
    old_val = os.environ.get("NANOCHAT_STAGE_BRIEFED")
    os.environ["NANOCHAT_STAGE_BRIEFED"] = "1"
    captured: list[str] = []
    # when: emit_define is called
    emit_define("BPB", print_fn=captured.append)
    # then: no output
    assert captured == []
    # cleanup
    if old_val is None:
        del os.environ["NANOCHAT_STAGE_BRIEFED"]
    else:
        os.environ["NANOCHAT_STAGE_BRIEFED"] = old_val


def test_given_briefed_env_unset_when_emit_define_then_prints_term():
    # Scenario: emit_define prints when NANOCHAT_STAGE_BRIEFED is unset.
    import os

    from nanochat.glossary import emit_define

    # given: env var unset (normal case)
    old_val = os.environ.pop("NANOCHAT_STAGE_BRIEFED", None)
    captured: list[str] = []
    # when: emit_define is called
    emit_define("BPB", print_fn=captured.append)
    # then: one definition line
    assert len(captured) == 1
    assert "define:" in captured[0]
    assert "BPB" in captured[0]
    # cleanup
    if old_val is not None:
        os.environ["NANOCHAT_STAGE_BRIEFED"] = old_val


# ----------------------------------------------------------------------------- term_style

def test_given_no_color_env_when_term_enabled_then_false():
    # Scenario: NO_COLOR=1 disables color.
    import os

    from nanochat.term_style import enabled

    # given: NO_COLOR set
    old_val = os.environ.get("NO_COLOR")
    os.environ["NO_COLOR"] = "1"
    # when: checking if color is enabled
    result = enabled()
    # then: false
    assert result is False
    # cleanup
    if old_val is None:
        del os.environ["NO_COLOR"]
    else:
        os.environ["NO_COLOR"] = old_val


def test_given_plain_when_paint_then_returns_plain_text():
    # Scenario: paint returns plain text when disabled.
    import os

    from nanochat.term_style import paint

    # given: NANOCHAT_PLAIN set
    old_val = os.environ.get("NANOCHAT_PLAIN")
    os.environ["NANOCHAT_PLAIN"] = "1"
    # when: painting text
    result = paint("START", "test message")
    # then: plain text (no ANSI codes)
    assert result == "test message"
    assert "\033" not in result
    # cleanup
    if old_val is None:
        del os.environ["NANOCHAT_PLAIN"]
    else:
        os.environ["NANOCHAT_PLAIN"] = old_val


# ----------------------------------------------------------------------------- stage_art

def test_given_nanochat_plain_when_plaque_then_returns_empty():
    # Scenario: plaque returns empty string when NANOCHAT_PLAIN=1.
    import os

    from nanochat.stage_art import render_plaque as plaque

    # given: NANOCHAT_PLAIN set
    old_val = os.environ.get("NANOCHAT_PLAIN")
    os.environ["NANOCHAT_PLAIN"] = "1"
    # when: generating a plaque
    result = plaque("base_train", "start", "teach raw next-token prediction")
    # then: empty string
    assert result == ""
    # cleanup
    if old_val is None:
        del os.environ["NANOCHAT_PLAIN"]
    else:
        os.environ["NANOCHAT_PLAIN"] = old_val


def test_given_plaque_when_nanochat_plain_unset_then_returns_multiline():
    # Scenario: plaque returns ASCII art when not suppressed.
    import os

    from nanochat.stage_art import render_plaque as plaque

    # given: NANOCHAT_PLAIN unset
    os.environ.pop("NANOCHAT_PLAIN", None)
    # when: generating a plaque
    result = plaque("base_train", "start", "teach raw next-token prediction")
    # then: multi-line with box-drawing characters
    assert result != ""
    assert "base_train" in result
    assert "START" in result
    assert "+" in result and "|" in result  # ASCII-safe borders (plan decision)


def test_given_shell_tip_when_printed_then_exactly_one_tip_token(tmp_path: Path) -> None:
    """
    Scenario: print_run_tip prints one tip line with a single 'tip:' token.
    Slice: cli-learn-color-ux (plan goal 4: existing contract lines unchanged)
    """
    import subprocess

    # ### Given
    root = Path(__file__).resolve().parents[1]
    script = f"""
set -euo pipefail
export NANOCHAT_BASE_DIR={tmp_path} PROFILE=cpu NANOCHAT_RUN_DIR= NO_COLOR=1
source {root / "runs" / "_stage_lib.sh"}
print_run_tip
"""

    # ### When
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout

    # ### Then
    assert out.count("tip:") == 1, f"expected one 'tip:' token, got: {out!r}"
    assert out.startswith("  tip: "), f"tip line must keep master's shape, got: {out!r}"
