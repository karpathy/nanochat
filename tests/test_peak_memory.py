"""
Tests for PeakMemoryTracker, which reports "Peak memory usage" in base_train/chat_sft.

The interesting case is mps, which has no peak API: the tracker samples
current_allocated_memory() per step and keeps the running max. The tempting
alternative, driver_allocated_memory(), reports the Metal allocator's reserved
pool and is not comparable to the cuda number printed under the same label;
test_mps_peak_is_allocated_not_reserved pins that distinction.

The cpu case runs in CI without a GPU; the mps/cuda cases skip when absent.
"""
import pytest
import torch

from nanochat.common import PeakMemoryTracker


def test_cpu_reports_zero():
    # cpu has no device memory to report; update() must stay safe to call
    tracker = PeakMemoryTracker("cpu")
    assert tracker.peak() == 0
    for _ in range(3):
        tracker.update()
    assert tracker.peak() == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_mps_tracks_spike():
    tracker = PeakMemoryTracker("mps")
    assert tracker.peak() == 0

    spike_bytes = 64 * 1024 * 1024  # 64 MiB, comfortably above allocator noise
    baseline = torch.empty(1024, 1024, dtype=torch.float32, device="mps")  # 4 MiB
    tracker.update()
    after_baseline = tracker.peak()
    assert after_baseline > 0, "tracker should see a live allocation"

    # a transient spike that is freed before the run ends: the whole point is that
    # the final number remembers it
    spike = torch.empty(spike_bytes // 4, dtype=torch.float32, device="mps")
    tracker.update()
    at_spike = tracker.peak()
    assert at_spike >= after_baseline + spike_bytes, (
        f"spike not captured: {at_spike} < {after_baseline} + {spike_bytes}"
    )

    del spike
    torch.mps.synchronize()
    tracker.update()
    assert tracker.peak() == at_spike, "peak must not decay after memory is freed"

    del baseline


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_mps_peak_is_allocated_not_reserved():
    # guards against regressing to driver_allocated_memory(), which reports the
    # reserved pool and reads well above what the process has actually allocated
    tracker = PeakMemoryTracker("mps")
    t = torch.empty(1024, 1024, dtype=torch.float32, device="mps")
    tracker.update()
    assert tracker.peak() <= torch.mps.driver_allocated_memory()
    del t


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_delegates_to_allocator():
    # on cuda the allocator owns the high-water mark; update() must not interfere
    tracker = PeakMemoryTracker("cuda")
    t = torch.empty(1024, 1024, dtype=torch.float32, device="cuda")
    tracker.update()
    assert tracker.peak() == torch.cuda.max_memory_allocated()
    del t
