"""
Test the log line grammar: the machine-readable contract inside stage logs.

python -m pytest tests/test_logfmt.py -v
"""

from nanochat.logfmt import format_record, parse_record, parse_records


def test_roundtrip_types():
    line = format_record("summary", vocab_size=32768, ratio=4.69, name="climbmix")
    record = parse_record(line)
    assert record == {"tag": "summary", "vocab_size": 32768, "ratio": 4.69, "name": "climbmix"}
    # types survive: int stays int, float stays float
    assert isinstance(record["vocab_size"], int)
    assert isinstance(record["ratio"], float)


def test_values_with_spaces_are_quoted():
    line = format_record("summary", gpu="NVIDIA H100 80GB HBM3", n=8)
    record = parse_record(line)
    assert record["gpu"] == "NVIDIA H100 80GB HBM3"
    assert record["n"] == 8


def test_prose_is_not_a_record():
    prose = [
        "Training time: 0.50s",
        "dataset climbmix: 6542 train shards + 1 val shard, 558.6 GiB",
        "Downloading shard_00001.parquet...",
        "==============================",
        "",
        "summary",  # a lone tag with no fields is prose too
        "half record=1 then prose words",  # any non key=value token disqualifies the line
    ]
    for line in prose:
        assert parse_record(line) is None, f"should not parse: {line!r}"


def test_negative_and_scientific_numbers():
    line = format_record("step", loss=-2.5, lr=1e-4)
    record = parse_record(line)
    assert record["loss"] == -2.5
    assert record["lr"] == 1e-4


def test_parse_records_scans_mixed_file(tmp_path):
    log = tmp_path / "stage.log"
    log.write_text(
        "Some banner prose\n"
        "step step=1 loss=3.0\n"
        "a warning: something happened\n"
        "step step=2 loss=2.5\n"
        "summary val_bpb=0.82 core=0.26\n"
    )
    steps = parse_records(log, tag="step")
    assert [r["loss"] for r in steps] == [3.0, 2.5]
    summary = parse_records(log, tag="summary")
    assert len(summary) == 1 and summary[0]["val_bpb"] == 0.82
    everything = parse_records(log)
    assert len(everything) == 3
