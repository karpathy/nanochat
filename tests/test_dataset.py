import pytest

from nanochat.dataset import MAX_SHARD, get_shard_indices_to_download, validate_num_workers


def test_get_shard_indices_to_download_includes_validation_shard():
    assert get_shard_indices_to_download(0) == [MAX_SHARD]
    assert get_shard_indices_to_download(2) == [0, 1, MAX_SHARD]


def test_get_shard_indices_to_download_caps_train_shards():
    indices = get_shard_indices_to_download(MAX_SHARD + 100)
    assert indices[0] == 0
    assert indices[-2:] == [MAX_SHARD - 1, MAX_SHARD]
    assert len(indices) == MAX_SHARD + 1


def test_get_shard_indices_to_download_rejects_negative_counts_except_all():
    with pytest.raises(ValueError, match="--num-files"):
        get_shard_indices_to_download(-2)


def test_validate_num_workers_requires_positive_count():
    assert validate_num_workers(1) == 1
    with pytest.raises(ValueError, match="--num-workers"):
        validate_num_workers(0)
