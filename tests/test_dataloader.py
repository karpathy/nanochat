"""
Test the shard format, the packer, and the loader that trains on the shards.

python -m pytest tests/test_dataloader.py -v
"""

import os

import numpy as np
import pytest
import torch

from nanochat.dataloader import write_shard, read_shard, read_header, pack_rows, data_loader, IGNORE, HEADER_BYTES
from nanochat.tokenizer import tokenizer_id


TOY_TOKENIZER_ID = 12345 # replaced by the fixture with the fingerprint of its stand-in tokenizer

def make_shard(path, num_rows, row_len, seed, vocab_size=1000, tokenizer_id=None):
    tokenizer_id = TOY_TOKENIZER_ID if tokenizer_id is None else tokenizer_id
    rng = np.random.default_rng(seed)
    inputs = rng.integers(0, vocab_size, size=(num_rows, row_len), dtype=np.uint16)
    targets = rng.integers(0, vocab_size, size=(num_rows, row_len), dtype=np.uint16)
    write_shard(path, inputs, targets, vocab_size, tokenizer_id)
    return inputs, targets


@pytest.fixture
def dataset_dir(tmp_path, monkeypatch):
    """A dataset directory that the loader resolves as the active dataset, holding a
    stand-in tokenizer whose fingerprint the shards must carry."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("NANOCHAT_DATASET", "toy")
    path = tmp_path / "datasets" / "toy"
    (path / "tokenizer").mkdir(parents=True)
    (path / "tokenizer" / "tokenizer.pkl").write_bytes(b"not a real tokenizer, but it has a checksum")
    global TOY_TOKENIZER_ID
    TOY_TOKENIZER_ID = tokenizer_id(str(path / "tokenizer"))
    return path


def test_shard_roundtrip(tmp_path):
    path = str(tmp_path / "train_00000.bin")
    inputs, targets = make_shard(path, num_rows=5, row_len=8, seed=0)
    assert read_header(path) == (8, 5, 1000, TOY_TOKENIZER_ID)
    got_inputs, got_targets = read_shard(path)
    assert np.array_equal(got_inputs, inputs)
    assert np.array_equal(got_targets, targets)
    # header + two uint16 sections, and the write was atomic (no tmp file left behind)
    assert os.path.getsize(path) == HEADER_BYTES + 2 * inputs.nbytes
    assert not os.path.exists(path + ".tmp")


def test_pack_rows_is_bos_aligned_bestfit():
    bos = 7
    capacity = 10
    # doc lengths 4, 6, 3, 10 (BOS included). Row 1: the largest doc that fits is the 10.
    # Row 2: the 6, then the 4. The 3 is left over and cannot fill a row, so it is dropped.
    docs = [[bos] + [1] * 3, [bos] + [2] * 5, [bos] + [3] * 2, [bos] + [4] * 9]
    rows = list(pack_rows(iter(docs), capacity))
    assert rows == [[bos] + [4] * 9, [bos] + [2] * 5 + [bos] + [1] * 3]
    # doc lengths 7, 8: the 8 fits, then nothing fits the 2 remaining, so the shortest
    # (the 7) is cropped to fill the row exactly
    docs = [[bos] + [1] * 6, [bos] + [2] * 7]
    rows = list(pack_rows(iter(docs), capacity))
    assert rows == [[bos] + [2] * 7 + [bos] + [1]]
    assert all(len(row) == capacity and row[0] == bos for row in rows)


def test_pack_rows_pads_when_cropping_is_off():
    bos = 7
    # doc lengths 6, 8 with capacity 10: the 8 fits, then the 6 does not and must not be cut,
    # so the row ends short (the caller pads); the 6 gets its own row, also short
    docs = [[bos] + [1] * 5, [bos] + [2] * 7]
    rows = list(pack_rows(iter(docs), 10, crop=False))
    assert rows == [[bos] + [2] * 7, [bos] + [1] * 5]


def test_loader_walks_shards_in_order_and_wraps(dataset_dir):
    a_inputs, a_targets = make_shard(str(dataset_dir / "train_00000.bin"), num_rows=3, row_len=4, seed=1)
    b_inputs, b_targets = make_shard(str(dataset_dir / "train_00001.bin"), num_rows=2, row_len=4, seed=2)
    inputs = np.concatenate([a_inputs, b_inputs]).astype(np.int64)
    targets = np.concatenate([a_targets, b_targets]).astype(np.int64)
    loader = data_loader(B=2, T=4, split="train", device="cpu")
    x, y, row = next(loader)
    assert row == 0
    assert torch.equal(x, torch.from_numpy(inputs[0:2]))
    assert torch.equal(y, torch.from_numpy(targets[0:2]))
    x, y, row = next(loader) # rows 2, 3 straddle the shard boundary
    assert row == 2
    assert torch.equal(x, torch.from_numpy(inputs[2:4]))
    x, y, row = next(loader) # rows 4, 0: wraps around the 5 rows of the split
    assert row == 4
    assert torch.equal(x, torch.from_numpy(inputs[[4, 0]]))
    assert torch.equal(y, torch.from_numpy(targets[[4, 0]]))


def test_loader_resumes_and_slices_a_prefix(dataset_dir):
    inputs, targets = make_shard(str(dataset_dir / "train_00000.bin"), num_rows=6, row_len=8, seed=3)
    # training at T=5 on rows of length 8 uses the first 5 columns; resume_row picks up mid-split
    loader = data_loader(B=2, T=5, split="train", device="cpu", resume_row=2)
    x, y, row = next(loader)
    assert row == 2
    assert torch.equal(x, torch.from_numpy(inputs[2:4, :5].astype(np.int64)))
    assert torch.equal(y, torch.from_numpy(targets[2:4, :5].astype(np.int64)))
    with pytest.raises(AssertionError):
        next(data_loader(B=1, T=9, split="train", device="cpu")) # longer than the rows


def test_loader_maps_ignore_targets(dataset_dir):
    path = str(dataset_dir / "train_00000.bin")
    inputs = np.ones((2, 4), dtype=np.uint16)
    targets = np.ones((2, 4), dtype=np.uint16)
    targets[0, 2] = IGNORE
    targets[1, :] = IGNORE
    write_shard(path, inputs, targets, 1000, TOY_TOKENIZER_ID)
    x, y, row = next(data_loader(B=2, T=4, split="train", device="cpu"))
    expected = torch.tensor([[1, 1, -1, 1], [-1, -1, -1, -1]])
    assert torch.equal(y, expected)


def test_loader_refuses_shards_from_another_tokenizer(dataset_dir):
    make_shard(str(dataset_dir / "train_00000.bin"), num_rows=2, row_len=4, seed=0, tokenizer_id=TOY_TOKENIZER_ID + 1)
    with pytest.raises(AssertionError, match="different tokenizer"):
        next(data_loader(B=1, T=4, split="train", device="cpu"))
