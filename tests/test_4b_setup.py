from nanochat.common import ceil_div, round_to_nearest_multiple
from nanochat.optim import _rank_first_dim_slice


def test_d36_4b_shape_math():
    depth = 36
    aspect_ratio = 64
    head_dim = 128

    model_dim = ceil_div(depth * aspect_ratio, head_dim) * head_dim
    num_heads = model_dim // head_dim

    assert model_dim == 2304
    assert num_heads == 18


def test_auto_batch_size_can_round_for_7_h200s():
    preferred_batch_size = 2**21
    world_tokens_per_fwdbwd = 7 * 8 * 2048

    compatible_batch_size = round_to_nearest_multiple(preferred_batch_size, world_tokens_per_fwdbwd)

    assert compatible_batch_size == 2_064_384
    assert compatible_batch_size % world_tokens_per_fwdbwd == 0


def test_auto_batch_size_keeps_power_of_two_when_compatible():
    preferred_batch_size = 2**21
    world_tokens_per_fwdbwd = 4 * 8 * 2048

    assert preferred_batch_size % world_tokens_per_fwdbwd == 0
    assert round_to_nearest_multiple(preferred_batch_size, world_tokens_per_fwdbwd) == preferred_batch_size


def test_adamw_dim0_padding_for_7_way_world_size():
    rank_size, padded_rows, start, end, valid_rows = _rank_first_dim_slice(
        num_rows=32768,
        world_size=7,
        rank=6,
    )

    assert rank_size == 4682
    assert padded_rows == 32774
    assert start == 28092
    assert end == 32768
    assert valid_rows == 4676


def test_adamw_dim0_padding_is_noop_when_divisible():
    rank_size, padded_rows, start, end, valid_rows = _rank_first_dim_slice(
        num_rows=32768,
        world_size=4,
        rank=3,
    )

    assert rank_size == 8192
    assert padded_rows == 32768
    assert start == 24576
    assert end == 32768
    assert valid_rows == 8192
