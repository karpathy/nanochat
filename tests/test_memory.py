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


def test_mps_out_of_memory_runtime_error_is_recognised():
    assert memory.is_out_of_memory_error(RuntimeError("MPS backend out of memory (MPS allocated: 17.95 GB)"))


def test_unrelated_runtime_error_is_not_treated_as_out_of_memory():
    assert not memory.is_out_of_memory_error(RuntimeError("shape mismatch"))


# ----------------------------------------------------------------------------- margin override

def test_margin_defaults_to_the_ssot_when_the_environment_is_unset(monkeypatch):
    monkeypatch.delenv(memory.MARGIN_ENV_VAR, raising=False)
    assert memory.resolve_fit_margin() == memory.MEMORY_FIT_MARGIN


def test_margin_comes_from_the_environment_when_it_is_set(monkeypatch):
    monkeypatch.setenv(memory.MARGIN_ENV_VAR, "0.25")
    assert memory.resolve_fit_margin() == 0.25


def test_a_fit_without_an_explicit_margin_honours_the_environment(monkeypatch):
    monkeypatch.setenv(memory.MARGIN_ENV_VAR, "0.25")
    _, report = memory.fit_device_batch_size(8, DEPTH, SEQ_LEN, available_bytes=AVAIL)
    assert report["margin"] == 0.25


def test_a_margin_that_is_not_a_number_is_refused_with_how_to_fix_it(monkeypatch):
    monkeypatch.setenv(memory.MARGIN_ENV_VAR, "half")
    with pytest.raises(ValueError, match="is not a number"):
        memory.resolve_fit_margin()


@pytest.mark.parametrize("out_of_range", ["0", "-0.5", "1.5"])
def test_a_margin_outside_a_usable_fraction_is_refused(monkeypatch, out_of_range):
    monkeypatch.setenv(memory.MARGIN_ENV_VAR, out_of_range)
    with pytest.raises(ValueError, match="is not a fraction of available device memory"):
        memory.resolve_fit_margin()


# ----------------------------------------------------------------------------- auto-fit switch

def test_auto_fit_is_on_by_default_on_mps():
    assert memory.auto_fit_enabled(None, "mps")


@pytest.mark.parametrize("device_type", ["cuda", "cpu"])
def test_auto_fit_is_off_by_default_away_from_mps(device_type):
    assert not memory.auto_fit_enabled(None, device_type)


def test_an_explicit_choice_beats_the_device_default():
    assert memory.auto_fit_enabled(True, "cuda") and not memory.auto_fit_enabled(False, "mps")


# ----------------------------------------------------------------------------- precedence echo

def test_precedence_echo_names_every_layer_that_set_the_batch_size():
    line = memory.format_batch_size_precedence_line([("CLI_default", 32), ("profile", 8), ("fitted", 2)])
    assert line == "[auto-fit] device_batch_size precedence: CLI_default=32 → profile=8 → fitted=2"


def test_the_echo_adds_the_precedence_chain_when_the_layers_disagree():
    lines = memory.auto_fit_echo_lines(_fit(16, avail=1 * GIB)[1], [("CLI_default", 32), ("profile", 16), ("fitted", 1)])
    assert any(line.startswith("[auto-fit] device_batch_size precedence:") for line in lines)


def test_the_echo_omits_the_precedence_chain_when_every_layer_agrees():
    lines = memory.auto_fit_echo_lines(_fit(8)[1], [("CLI_default", 8), ("profile", 8), ("fitted", 8)])
    assert lines == ["[auto-fit] device_batch_size=8 (unchanged; estimate fits)",
                     "memory: aiming to use ≤60% of ~18GiB; batch 8 fits as-is"]


def test_the_echo_keeps_the_contract_line_first_and_one_enrichment_last():
    lines = memory.auto_fit_echo_lines(_fit(16, avail=1 * GIB)[1], [("CLI_default", 32), ("profile", 16), ("fitted", 1)])
    assert lines[0].startswith("[auto-fit] device_batch_size:") and lines[-1].startswith("memory: ")
