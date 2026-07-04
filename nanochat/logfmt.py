"""
The log line grammar: how nanochat scripts report data to the experiment record.

Stage scripts print to stdout only; runs/run.sh tees stdout+stderr into per-stage
.log files in the experiment directory. Most of a log is free-form prose for humans
with no stability promise. Lines that carry data follow this grammar and are the
machine-readable contract (see experiment_refactor.md):

    <tag> key=value key=value ...

    step step=4990 loss=2.311 mfu=40.2
    summary vocab_size=32768 train_time_sec=38.2
    summary params=286261730 val_bpb=0.8213 gpu="NVIDIA H100 80GB HBM3"

Values with spaces are quoted; parsing infers int/float and leaves the rest as
strings. This one small module is the only schema machinery in the repo: scripts
format records with format_record(), and the aggregator reads them back with
parse_records().
"""

import shlex

def format_record(tag, **fields):
    """Format a record line, e.g. format_record("summary", loss=2.3) -> 'summary loss=2.3'."""
    parts = [tag]
    for key, value in fields.items():
        value_str = shlex.quote(str(value))
        parts.append(f"{key}={value_str}")
    line = " ".join(parts)
    return line

def _parse_value(value_str):
    """Infer the type of a value: int, then float, else string."""
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str

def parse_record(line, tag=None):
    """
    Parse a single record line into a dict (with the tag under the "tag" key),
    or return None if the line is not a well-formed record (or has a different tag).
    """
    line = line.strip()
    try:
        tokens = shlex.split(line)
    except ValueError:
        return None # e.g. unbalanced quotes: prose, not a record
    if not tokens:
        return None
    line_tag = tokens[0]
    if tag is not None and line_tag != tag:
        return None
    record = {"tag": line_tag}
    for token in tokens[1:]:
        if "=" not in token:
            return None # every field must be key=value, otherwise this is prose
        key, value_str = token.split("=", 1)
        record[key] = _parse_value(value_str)
    if len(record) == 1:
        return None # a lone word is prose, not a record
    return record

def parse_records(path, tag=None):
    """Scan a log file and return all record dicts (optionally only those with a given tag)."""
    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            record = parse_record(line, tag=tag)
            if record is not None:
                records.append(record)
    return records
