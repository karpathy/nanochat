import types

from nanochat.common import get_mps_memory_stats


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