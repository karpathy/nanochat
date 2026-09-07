"""
Test the hub dataset wrapper and the multiple choice prompt (in-memory, no network).

python -m pytest tests/test_tasks.py -v
"""

import numpy as np
import pyarrow as pa
from harness.tasks import HubDataset, render_mc


def test_hub_dataset_rows():
    table = pa.table({"x": list(range(100)), "y": [str(i) for i in range(100)]})
    ds = HubDataset(table)
    assert len(ds) == 100
    assert ds[7] == {"x": 7, "y": "7"}


def test_hub_dataset_shuffle_matches_numpy():
    # the shuffle must reproduce datasets.Dataset.shuffle(seed) exactly,
    # which is a np.random.default_rng(seed) permutation
    table = pa.table({"x": list(range(100))})
    ds = HubDataset(table).shuffle(seed=42)
    perm = np.random.default_rng(42).permutation(100)
    assert [ds[i]["x"] for i in range(100)] == [int(p) for p in perm]
    # shuffling returns a view; the original order is untouched
    assert HubDataset(table)[0] == {"x": 0}


def test_render_mc_letter_binding():
    query = render_mc("What is 1+1?", ("A", "B"), ("1", "2"))
    # the letter must directly follow '=' with no whitespace, so that the
    # prompt token for "A" matches the assistant's bare "A" response token
    assert "=A\n" in query and "=B\n" in query
