from __future__ import annotations

import time

import pytest

from apdo.resources import (
    MemoryMonitor,
    available_cpu_count,
    cgroup_cpu_limit,
    cgroup_memory_limit_bytes,
    collect_environment,
    process_rss_bytes,
    system_memory_bytes,
)


def test_available_cpu_count() -> None:
    assert available_cpu_count() >= 1


def test_environment_collection() -> None:
    environment = collect_environment()

    assert environment.python_version
    assert environment.torch_version
    assert environment.cpu_count >= 1
    assert environment.effective_cpu_count >= 1
    assert environment.torch_num_threads >= 1
    assert environment.multiprocessing_start_method
    assert isinstance(environment.to_dict(), dict)
    assert environment.to_dict()["cuda_available"] == environment.cuda_available
    assert environment.to_dict()["effective_cpu_count"] >= 1


def test_cgroup_limits_are_plausible() -> None:
    memory_limit = cgroup_memory_limit_bytes()
    cpu_limit = cgroup_cpu_limit()

    if memory_limit is not None:
        assert memory_limit > 0
    if cpu_limit is not None:
        assert cpu_limit > 0


def test_memory_helpers_return_plausible_values() -> None:
    total, available = system_memory_bytes()
    rss = process_rss_bytes()

    if total is not None:
        assert total > 0
    if available is not None:
        assert available > 0
    if rss is not None:
        assert rss > 0


def test_memory_monitor_lifecycle() -> None:
    monitor = MemoryMonitor(limit_bytes=4_000_000_000)
    monitor.start()
    time.sleep(0.03)
    snapshot = monitor.stop()

    assert isinstance(snapshot.limit_exceeded, bool)
    assert snapshot.limit_bytes == 4_000_000_000
    assert isinstance(snapshot.to_dict(), dict)
    if snapshot.peak_process_rss_bytes is not None:
        assert snapshot.peak_process_rss_bytes > 0
    if snapshot.baseline_available_bytes is not None:
        assert snapshot.peak_system_used_bytes is not None
        assert snapshot.peak_system_used_bytes >= 0


def test_memory_monitor_cannot_start_twice() -> None:
    monitor = MemoryMonitor()
    monitor.start()
    try:
        with pytest.raises(RuntimeError):
            monitor.start()
    finally:
        monitor.stop()


def test_memory_monitor_reports_unavailable_system_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("apdo.resources.system_memory_bytes", lambda: (None, None))
    monitor = MemoryMonitor(limit_bytes=1)
    monitor.start()
    snapshot = monitor.stop()

    assert snapshot.baseline_available_bytes is None
    assert snapshot.peak_system_used_bytes is None
    assert snapshot.limit_enforced is False
    assert snapshot.limit_exceeded is False
