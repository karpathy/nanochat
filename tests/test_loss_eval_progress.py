"""Progress cadence for silent val-bpb loops (Mac/CPU can sit for minutes without logs)."""

from nanochat.loss_eval import _bpb_progress_every


def test_bpb_progress_every_covers_tiny_and_long_runs():
    assert _bpb_progress_every(0) == 1
    assert _bpb_progress_every(1) == 1
    assert _bpb_progress_every(9) == 1
    assert _bpb_progress_every(10) == 1
    assert _bpb_progress_every(320) == 32
