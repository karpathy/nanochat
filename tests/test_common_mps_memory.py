import types

from nanochat.common import autodetect_device_type, get_mlx_memory_stats, get_mps_memory_stats, maybe_torch_compile, should_torch_compile


class FakeMPS:
    def __init__(self, *, allocated, driver, recommended):
        self._allocated = allocated
        self._driver = driver
        self._recommended = recommended

    def current_allocated_memory(self):
        return self._allocated

    def driver_allocated_memory(self):
        return self._driver

    def recommended_max_memory(self):
        return self._recommended


def test_get_mps_memory_stats_reports_headroom_and_budget(monkeypatch):
    gib = 1024 ** 3
    fake_torch = types.SimpleNamespace(mps=FakeMPS(allocated=12 * gib, driver=60 * gib, recommended=80 * gib))
    monkeypatch.setattr("nanochat.common.torch", fake_torch)

    stats = get_mps_memory_stats(budget_frac=0.9)

    assert stats == {
        "allocated_gb": 12.0,
        "driver_gb": 60.0,
        "recommended_gb": 80.0,
        "driver_frac": 0.75,
        "headroom_gb": 20.0,
        "headroom_frac": 0.25,
        "budget_frac": 0.9,
        "budget_limit_gb": 72.0,
        "budget_headroom_gb": 12.0,
        "exceeds_budget": False,
    }


def test_get_mps_memory_stats_flags_over_budget(monkeypatch):
    gib = 1024 ** 3
    fake_torch = types.SimpleNamespace(mps=FakeMPS(allocated=16 * gib, driver=78 * gib, recommended=80 * gib))
    monkeypatch.setattr("nanochat.common.torch", fake_torch)

    stats = get_mps_memory_stats(budget_frac=0.9)

    assert stats["headroom_gb"] == 2.0
    assert stats["budget_headroom_gb"] == -6.0
    assert stats["exceeds_budget"] is True


def test_should_torch_compile_skips_mps_by_default(monkeypatch):
    monkeypatch.delenv("NANOCHAT_COMPILE", raising=False)

    assert should_torch_compile("mps") is False
    assert should_torch_compile("cpu") is True


def test_should_torch_compile_respects_env_override(monkeypatch):
    monkeypatch.setenv("NANOCHAT_COMPILE", "1")
    assert should_torch_compile("mps") is True

    monkeypatch.setenv("NANOCHAT_COMPILE", "off")
    assert should_torch_compile("cpu") is False


def test_maybe_torch_compile_skips_compile_on_mps(monkeypatch):
    calls = []

    def fake_compile(model, *, dynamic):
        calls.append((model, dynamic))
        return "compiled"

    monkeypatch.delenv("NANOCHAT_COMPILE", raising=False)
    monkeypatch.setattr("nanochat.common.torch.compile", fake_compile)

    model = object()
    compiled = maybe_torch_compile(model, "mps", dynamic=False)

    assert compiled is model
    assert calls == []


def test_maybe_torch_compile_forced_mps_falls_back_on_error(monkeypatch):
    def fake_compile(model, *, dynamic):
        raise RuntimeError("compile exploded")

    monkeypatch.setenv("NANOCHAT_COMPILE", "1")
    monkeypatch.setattr("nanochat.common.torch.compile", fake_compile)

    model = object()
    compiled = maybe_torch_compile(model, "mps", dynamic=False)

    assert compiled is model


def test_autodetect_device_type_prefers_mps_when_cuda_unavailable(monkeypatch):
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr("nanochat.common.torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr("nanochat.common.is_mps_available", lambda: True)

    assert autodetect_device_type() == "mps"


# ---------------------------------------------------------------------------
# MLX memory telemetry
# ---------------------------------------------------------------------------

def test_get_mlx_memory_stats_returns_correct_gb_values(monkeypatch):
    gib = 1024 ** 3
    mib = 1024 ** 2

    fake_mx = types.SimpleNamespace(
        get_active_memory=lambda: 4 * gib,
        get_peak_memory=lambda: 6 * gib,
        get_cache_memory=lambda: 512 * mib,
        reset_peak_memory=lambda: None,
    )

    monkeypatch.setattr("nanochat.common._mlx", fake_mx)

    stats = get_mlx_memory_stats(reset_peak=False)

    assert stats == {
        "active_gb": 4.0,
        "peak_gb": 6.0,
        "cache_gb": 0.5,
    }


def test_get_mlx_memory_stats_calls_reset_peak_when_requested(monkeypatch):
    gib = 1024 ** 3
    reset_calls = []

    fake_mx = types.SimpleNamespace(
        get_active_memory=lambda: 2 * gib,
        get_peak_memory=lambda: 3 * gib,
        get_cache_memory=lambda: 0,
        reset_peak_memory=lambda: reset_calls.append(1),
    )

    monkeypatch.setattr("nanochat.common._mlx", fake_mx)

    get_mlx_memory_stats(reset_peak=True)

    assert len(reset_calls) == 1


def test_get_mlx_memory_stats_returns_zeros_when_mlx_unavailable(monkeypatch):
    monkeypatch.setattr("nanochat.common._mlx", None)

    stats = get_mlx_memory_stats()

    assert stats == {"active_gb": 0.0, "peak_gb": 0.0, "cache_gb": 0.0}