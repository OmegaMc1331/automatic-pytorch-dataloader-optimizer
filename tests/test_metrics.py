from __future__ import annotations

import math

import pytest

from apdo.metrics import (
    PassMetrics,
    bootstrap_median_ci,
    mean,
    median,
    percentile,
    stdev,
    summarize_passes,
)


def make_pass(
    throughput: float,
    *,
    startup_s: float = 0.01,
    latencies: tuple[float, ...] = (0.01, 0.02, 0.03),
) -> PassMetrics:
    return PassMetrics(
        startup_s=startup_s,
        steady_elapsed_s=1.0,
        batches=len(latencies) + 1,
        steady_samples=throughput,
        throughput=throughput,
        latencies_s=latencies,
    )


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_mean_median_and_stdev() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert mean(values) == 2.5
    assert median(values) == 2.5
    assert stdev(values) == pytest.approx(math.sqrt(5.0 / 3.0))
    assert stdev([1.0]) == 0.0


def test_percentile_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


def test_bootstrap_is_deterministic() -> None:
    values = [10.0, 12.0, 11.0, 13.0]

    first = bootstrap_median_ci(values, resamples=500, seed=7)
    second = bootstrap_median_ci(values, resamples=500, seed=7)

    assert first == second
    assert first[0] <= median(values) <= first[1]


def test_bootstrap_single_value() -> None:
    assert bootstrap_median_ci([42.0]) == (42.0, 42.0)


def test_summarize_passes() -> None:
    passes = [
        make_pass(100.0, startup_s=0.01),
        make_pass(110.0, startup_s=0.02),
        make_pass(90.0, startup_s=0.03),
    ]

    summary = summarize_passes(
        passes,
        peak_process_rss_bytes=123,
        peak_system_used_bytes=456,
        peak_gpu_memory_bytes=789,
    )

    assert summary.median_throughput == 100.0
    assert summary.mean_throughput == 100.0
    assert summary.min_throughput == 90.0
    assert summary.max_throughput == 110.0
    assert summary.median_startup_s == 0.02
    assert summary.median_latency_s == 0.02
    assert summary.p95_latency_s == pytest.approx(0.03)
    assert summary.cv_throughput > 0
    assert summary.peak_process_rss_bytes == 123
    assert summary.peak_system_used_bytes == 456
    assert summary.peak_gpu_memory_bytes == 789


def test_summarize_requires_passes() -> None:
    with pytest.raises(ValueError):
        summarize_passes([])


def test_pass_serialization() -> None:
    data = make_pass(100.0).to_dict()

    assert data["throughput"] == 100.0
    assert data["batches"] == 4
    assert data["median_latency_s"] == 0.02
    assert data["p95_latency_s"] == pytest.approx(0.029)
