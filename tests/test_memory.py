"""
Test the memory estimate / auto-fit math. No real MPS needed: available bytes are injected.

Run: python -m pytest tests/test_memory.py -q
"""
import torch
import pytest
from nanochat import memory

GIB = 1024 ** 3
MARGIN = 0.6
AVAIL = 18 * GIB
DEPTH = 12
SEQ_LEN = 2048


def _fit(requested_b, avail=AVAIL, depth=DEPTH, seq_len=SEQ_LEN):
    return memory.fit_device_batch_size(requested_b, depth, seq_len, available_bytes=avail, margin=MARGIN)


def _estimate(batch, depth=DEPTH, seq_len=SEQ_LEN):
    return memory.estimate_train_peak_bytes(depth, batch, seq_len)


# ----------------------------------------------------------------------------- estimate

def test_estimate_grows_with_batch_size():
    estimates = [_estimate(b) for b in (1, 2, 4, 8)]
    assert estimates == sorted(estimates) and len(set(estimates)) == len(estimates)


def test_estimate_grows_with_sequence_length():
    assert _estimate(1, seq_len=1024) < _estimate(1, seq_len=2048)


def test_estimate_counts_the_logits_tensor_with_the_activation_fudge():
    # doubling B adds (1 + fudge) * B*T*V*4 and nothing else
    logits_bytes = 1 * SEQ_LEN * memory.VOCAB_SIZE * 4
    growth = _estimate(2) - _estimate(1)
    assert growth == pytest.approx((1 + memory.MEMORY_ACTIVATION_FUDGE) * logits_bytes)


# ----------------------------------------------------------------------------- fit

def test_fit_keeps_batch_when_estimate_fits():
    adjusted, _ = _fit(8)
    assert adjusted == 8


def test_fit_halves_until_estimate_fits_the_margin():
    budget = (_estimate(2) + _estimate(4)) / 2  # room for B=2, not for B=4
    avail = int(budget / MARGIN)
    adjusted, _ = _fit(8, avail=avail)
    assert adjusted == 2


def test_fit_floors_at_one_when_nothing_fits():
    adjusted, _ = _fit(8, avail=1 * GIB)
    assert adjusted == 1


def test_report_records_the_requested_batch_size():
    _, report = _fit(16)
    assert report["original"] == 16


def test_report_records_the_reason_when_it_shrinks():
    _, report = _fit(16, avail=1 * GIB)
    assert report["reason"] == "estimated peak exceeds margin * available memory"


def test_report_has_no_reason_when_the_batch_is_unchanged():
    _, report = _fit(8)
    assert report["reason"] is None


# ----------------------------------------------------------------------------- echo lines

def test_echo_shows_original_and_adjusted_when_it_shrinks():
    line = memory.format_auto_fit_line(_fit(16, avail=1 * GIB)[1])
    assert line.startswith("[auto-fit] device_batch_size: original/default=16 → adjusted=1 (reason: est_peak=")


def test_echo_shows_the_margin_and_available_memory_when_it_shrinks():
    line = memory.format_auto_fit_line(_fit(16, avail=1 * GIB)[1])
    assert line.endswith(" > margin*avail=0.6*1.0GiB)")


def test_echo_reports_unchanged_when_the_estimate_fits():
    line = memory.format_auto_fit_line(_fit(8)[1])
    assert line == "[auto-fit] device_batch_size=8 (unchanged; estimate fits)"


def test_echo_warns_when_the_minimum_batch_still_does_not_fit():
    line = memory.format_auto_fit_line(_fit(1, avail=1 * GIB)[1])
    assert "unchanged; already at the minimum" in line


def test_plain_language_line_explains_a_shrink():
    line = memory.format_memory_fit_hint(_fit(16, avail=1 * GIB)[1])
    assert line == "memory: aiming to use ≤60% of ~1GiB; batch 16→1 so logits fit"


def test_plain_language_line_explains_an_unchanged_batch():
    line = memory.format_memory_fit_hint(_fit(8)[1])
    assert line == "memory: aiming to use ≤60% of ~18GiB; batch 8 fits as-is"


# ----------------------------------------------------------------------------- available memory

def test_cpu_available_memory_is_a_soft_cap_so_auto_fit_never_shrinks():
    adjusted, _ = memory.fit_device_batch_size(32, DEPTH, SEQ_LEN, device_type="cpu")
    assert adjusted == 32


def test_mps_available_memory_prefers_the_torch_recommendation(monkeypatch):
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 18 * GIB, raising=False)
    assert memory.available_memory_bytes("mps") == 18 * GIB


def test_mps_available_memory_falls_back_to_a_fraction_of_system_ram(monkeypatch):
    monkeypatch.delattr(torch.mps, "recommended_max_memory", raising=False)
    monkeypatch.setattr(memory, "_system_ram_bytes", lambda: 32 * GIB)
    assert memory.available_memory_bytes("mps") == int(memory.MPS_SYSTEM_RAM_FRACTION * 32 * GIB)


def test_cuda_available_memory_uses_free_bytes_from_mem_get_info(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (7 * GIB, 24 * GIB), raising=False)
    assert memory.available_memory_bytes("cuda") == 7 * GIB


def test_device_objects_are_accepted_as_well_as_strings():
    assert memory.available_memory_bytes(torch.device("cpu")) == memory.CPU_SOFT_CAP_BYTES


# ----------------------------------------------------------------------------- OOM retry

def test_oom_rehalve_halves_the_batch_once():
    assert memory.oom_rehalve_once(4, device_type="cpu") == 2


def test_oom_rehalve_fails_with_an_actionable_message_at_the_floor():
    with pytest.raises(MemoryError, match="device_batch_size=1"):
        memory.oom_rehalve_once(1, device_type="cpu")


def test_oom_retry_echo_shows_original_and_adjusted():
    line = memory.format_oom_retry_line(4, 2)
    assert line.startswith("[auto-fit] OOM recovery: device_batch_size: original/default=4 → adjusted=2 (reason:")
