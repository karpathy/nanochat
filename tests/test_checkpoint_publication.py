"""Checkpoint writes must finish serialization before replacing prior files."""
import json
import os
from pathlib import Path
import stat

import pytest
import torch

from nanochat import checkpoint_manager as checkpoints


def save(directory, step=1, value=1, rank=0):
    checkpoints.save_checkpoint(str(directory), step, {"weight": torch.tensor([value])},
                                {"moment": torch.tensor([value])}, {"value": value}, rank=rank)


def test_checkpoint_round_trip(tmp_path):
    save(tmp_path)
    model, optimizer, metadata = checkpoints.load_checkpoint(str(tmp_path), 1, "cpu", load_optimizer=True)
    assert model["weight"].item() == optimizer["moment"].item() == metadata["value"] == 1


@pytest.mark.parametrize("failed_writer", ["model", "metadata", "optimizer"])
def test_failed_serialization_preserves_existing_checkpoint(tmp_path, monkeypatch, failed_writer):
    save(tmp_path)
    previous = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    original_save = checkpoints.torch.save
    original_dump = checkpoints.json.dump

    def save_tensor(value, path):
        name = Path(path).name
        if ((failed_writer == "model" and name.startswith("model_")) or
                (failed_writer == "optimizer" and name.startswith("optim_"))):
            Path(path).write_bytes(b"partial")
            raise OSError("interrupted serialization")
        return original_save(value, path)

    def save_metadata(value, stream, **kwargs):
        if failed_writer == "metadata":
            stream.write("{")
            raise OSError("interrupted serialization")
        return original_dump(value, stream, **kwargs)

    monkeypatch.setattr(checkpoints.torch, "save", save_tensor)
    monkeypatch.setattr(checkpoints.json, "dump", save_metadata)
    with pytest.raises(OSError, match="interrupted serialization"):
        save(tmp_path, value=2)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == previous


def test_new_step_is_not_discoverable_during_serialization(tmp_path, monkeypatch):
    save(tmp_path)
    original_dump = checkpoints.json.dump

    def paused_metadata(value, stream, **kwargs):
        assert checkpoints.find_last_step(str(tmp_path)) == 1
        return original_dump(value, stream, **kwargs)

    monkeypatch.setattr(checkpoints.json, "dump", paused_metadata)
    save(tmp_path, step=2)
    assert checkpoints.find_last_step(str(tmp_path)) == 2
    assert json.loads((tmp_path / "meta_000002.json").read_text())["value"] == 1


def test_nonzero_rank_only_publishes_its_optimizer(tmp_path):
    save(tmp_path, rank=1)
    assert [path.name for path in tmp_path.iterdir()] == ["optim_000001_rank1.pt"]
    assert torch.load(tmp_path / "optim_000001_rank1.pt", weights_only=True)["moment"].item() == 1


def test_overwrite_preserves_file_permissions(tmp_path):
    save(tmp_path)
    for path in tmp_path.iterdir():
        path.chmod(0o600)
    save(tmp_path, value=2)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in tmp_path.iterdir())


def test_existing_checkpoint_symlinks_and_targets_are_preserved(tmp_path):
    checkpoints_dir = tmp_path / "links"
    targets = tmp_path / "targets"
    save(targets)
    checkpoints_dir.mkdir()
    for target in targets.iterdir():
        (checkpoints_dir / target.name).symlink_to(target)
    save(checkpoints_dir, value=2)
    assert all(path.is_symlink() for path in checkpoints_dir.iterdir())
    assert {path.name for path in targets.iterdir()} == {
        "model_000001.pt", "meta_000001.json", "optim_000001_rank0.pt",
    }
    model, optimizer, metadata = checkpoints.load_checkpoint(str(targets), 1, "cpu", load_optimizer=True)
    assert model["weight"].item() == optimizer["moment"].item() == metadata["value"] == 2


@pytest.mark.parametrize("name", ["model_000001.pt", "meta_000001.json", "optim_000001_rank0.pt"])
def test_hard_link_preflight_preserves_aliases_before_serialization(tmp_path, monkeypatch, name):
    directory = tmp_path / "checkpoint"
    save(directory)
    alias = tmp_path / "alias"
    try:
        os.link(directory / name, alias)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    previous = {path.name: path.read_bytes() for path in directory.iterdir()}
    alias_bytes = alias.read_bytes()
    calls = []
    original_save = checkpoints.torch.save

    def observed_save(*args, **kwargs):
        calls.append(args)
        return original_save(*args, **kwargs)

    monkeypatch.setattr(checkpoints.torch, "save", observed_save)
    with pytest.raises(ValueError, match="hard-linked checkpoint"):
        save(directory, value=2)
    assert calls == []
    assert alias.read_bytes() == alias_bytes
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == previous


def test_rank_zero_without_optimizer_still_saves_model_and_metadata(tmp_path):
    checkpoints.save_checkpoint(str(tmp_path), 1, {"weight": torch.tensor([1])}, None, {"value": 1})
    model, optimizer, metadata = checkpoints.load_checkpoint(str(tmp_path), 1, "cpu")
    assert model["weight"].item() == metadata["value"] == 1
    assert optimizer is None


def test_other_rank_without_optimizer_does_not_create_a_checkpoint(tmp_path):
    directory = tmp_path / "absent"
    checkpoints.save_checkpoint(str(directory), 1, {}, None, {}, rank=1)
    assert not directory.exists()
