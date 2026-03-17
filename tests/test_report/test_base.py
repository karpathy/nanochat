"""Tests for report base classes."""

import datetime
import os
import pytest

from nanochat.report.base import BaseReport, Report
from nanochat.report.utils import slugify, extract, extract_timestamp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def report_dir(tmp_path):
    return str(tmp_path / "report")


@pytest.fixture
def report(report_dir):
    return Report(report_dir)


# ---------------------------------------------------------------------------
# BaseReport
# ---------------------------------------------------------------------------

def test_base_report_creates_dir(report_dir):
    BaseReport(report_dir)
    assert os.path.isdir(report_dir)


def test_base_report_log_path(report_dir):
    r = BaseReport(report_dir)
    assert r._log_path("My Section") == os.path.join(report_dir, "my-section.md")


def test_base_report_report_path(report_dir):
    r = BaseReport(report_dir)
    assert r._report_path() == os.path.join(report_dir, "report.md")


def test_base_report_log_returns_path(report_dir):
    r = BaseReport(report_dir)
    assert r.log("My Section", []) == r._log_path("My Section")


def test_base_report_generate_returns_path(report_dir):
    r = BaseReport(report_dir)
    assert r.generate() == r._report_path()


# ---------------------------------------------------------------------------
# Report.log
# ---------------------------------------------------------------------------

def test_report_log_creates_file(report):
    path = report.log("Base Model Training", [])
    assert os.path.exists(path)


def test_report_log_section_header(report):
    path = report.log("Base Model Training", [])
    content = open(path).read()
    assert "## Base Model Training\n" in content


def test_report_log_string_item(report):
    path = report.log("Notes", ["hello world"])
    assert "hello world" in open(path).read()


def test_report_log_dict_item(report):
    path = report.log("Stats", [{"loss": 0.1234, "steps": 50000}])
    content = open(path).read()
    assert "- loss: 0.1234" in content
    assert "- steps: 50,000" in content


def test_report_log_skips_falsy(report):
    path = report.log("Stats", [None, {}, "real"])
    assert "real" in open(path).read()


def test_report_log_float_formatting(report):
    path = report.log("Stats", [{"val": 1.23456789}])
    assert "- val: 1.2346" in open(path).read()


# ---------------------------------------------------------------------------
# Report._read_header
# ---------------------------------------------------------------------------

def test_read_header_missing_file(report):
    start_time, bloat_data, content = report._read_header(report.report_dir)
    assert start_time is None
    assert bloat_data == "[bloat data missing]"
    assert content == ""


def test_read_header_extracts_start_time(report):
    header_file = os.path.join(report.report_dir, "header.md")
    with open(header_file, "w") as f:
        f.write("Run started: 2026-03-14 10:00:00\n\n")
    start_time, _, _ = report._read_header(report.report_dir)
    assert start_time == datetime.datetime(2026, 3, 14, 10, 0, 0)


def test_read_header_extracts_bloat_data(report):
    header_file = os.path.join(report.report_dir, "header.md")
    with open(header_file, "w") as f:
        f.write("### Bloat\n- Lines: 1,234\n\nRun started: 2026-03-14 10:00:00\n")
    _, bloat_data, _ = report._read_header(report.report_dir)
    assert "Lines: 1,234" in bloat_data


def test_read_header_returns_content(report):
    header_file = os.path.join(report.report_dir, "header.md")
    with open(header_file, "w") as f:
        f.write("# nanochat\nRun started: 2026-03-14 10:00:00\n")
    _, _, content = report._read_header(report.report_dir)
    assert "# nanochat" in content


# ---------------------------------------------------------------------------
# Report._write_summary
# ---------------------------------------------------------------------------

def test_write_summary_table_structure(report, tmp_path):
    out_file = open(tmp_path / "out.md", "w")
    final_metrics = {"base": {"CORE": "0.42"}, "sft": {"CORE": "0.55"}}
    report._write_summary(out_file, final_metrics, "", None, None)
    out_file.close()
    content = open(tmp_path / "out.md").read()
    assert "## Summary" in content
    assert "CORE" in content
    assert "BASE" in content
    assert "SFT" in content


def test_write_summary_core_first(report, tmp_path):
    out_file = open(tmp_path / "out.md", "w")
    final_metrics = {"base": {"CORE": "0.42", "ARC-Easy": "0.70", "ChatCORE": "0.50"}}
    report._write_summary(out_file, final_metrics, "", None, None)
    out_file.close()
    content = open(tmp_path / "out.md").read()
    lines = [l for l in content.splitlines() if "CORE" in l or "ARC" in l]
    assert lines[0].startswith("| CORE")
    assert lines[-1].startswith("| ChatCORE")


def test_write_summary_wall_clock_time(report, tmp_path):
    out_file = open(tmp_path / "out.md", "w")
    start = datetime.datetime(2026, 3, 14, 10, 0, 0)
    end = datetime.datetime(2026, 3, 14, 12, 30, 0)
    report._write_summary(out_file, {}, "", start, end)
    out_file.close()
    assert "2h30m" in open(tmp_path / "out.md").read()


def test_write_summary_unknown_wall_clock(report, tmp_path):
    out_file = open(tmp_path / "out.md", "w")
    report._write_summary(out_file, {}, "", None, None)
    out_file.close()
    assert "unknown" in open(tmp_path / "out.md").read()


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def test_slugify():
    assert slugify("Base Model Training") == "base-model-training"


def test_extract_single_key():
    section = "## Stats\n- CORE: 0.42\n- loss: 1.23\n"
    assert extract(section, "CORE") == {"CORE": "0.42"}


def test_extract_multiple_keys():
    section = "- ARC-Easy: 0.70\n- GSM8K: 0.55\n"
    result = extract(section, ["ARC-Easy", "GSM8K"])
    assert result == {"ARC-Easy": "0.70", "GSM8K": "0.55"}


def test_extract_timestamp():
    content = "timestamp: 2026-03-14 10:00:00\n"
    ts = extract_timestamp(content, "timestamp:")
    assert ts == datetime.datetime(2026, 3, 14, 10, 0, 0)


def test_extract_timestamp_missing():
    assert extract_timestamp("no timestamp here", "timestamp:") is None
