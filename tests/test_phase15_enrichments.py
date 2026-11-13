"""
Phase 1.5 enrichments: knob diff (last_run) and trend crumb (last_metrics).

Pure helpers only — no torch. Covers ACs 1, 3, 6, 7 from
docs/plan/2026-11-13-fix-forward-mac-phase15.md.

Run: python -m pytest tests/test_phase15_enrichments.py -q
"""
from pathlib import Path

from nanochat.last_run import format_knob_diff, load_last_run, write_last_run
from nanochat.metrics_trend import format_trend, load_metrics, record_metric


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
