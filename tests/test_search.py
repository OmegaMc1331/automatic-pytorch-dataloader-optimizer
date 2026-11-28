from __future__ import annotations

import pytest

from apdo.benchmark import StageResult
from apdo.config import LoaderConfig
from apdo.errors import ConfigurationError
from apdo.search import adaptive_worker_counts, build_search_trace
from tests.helpers import make_failed_measurement, make_measurement


def test_adaptive_worker_counts_keep_endpoints() -> None:
    assert adaptive_worker_counts((0, 1, 2, 4, 8)) == (0, 1, 2, 8)
    assert adaptive_worker_counts((0, 1, 2)) == (0, 1, 2)
    assert adaptive_worker_counts((0, 1)) == (0, 1)


def test_adaptive_worker_counts_reject_empty_input() -> None:
    with pytest.raises(ConfigurationError):
        adaptive_worker_counts(())


def test_search_trace_records_statuses_and_budget_skips() -> None:
    baseline = LoaderConfig()
    selected = LoaderConfig(num_workers=1, prefetch_factor=2)
    fastest = LoaderConfig(num_workers=2, prefetch_factor=2)
    failed = make_failed_measurement(LoaderConfig(num_workers=4))
    measurements = (
        make_measurement(baseline, (100.0, 100.0)),
        make_measurement(selected, (120.0, 121.0)),
        make_measurement(fastest, (130.0, 131.0)),
        failed,
    )
    stage = StageResult(measurements=measurements, orders=())

    trace = build_search_trace(
        strategy="adaptive",
        stages=(("screening", stage),),
        planned_rounds=2,
        selected_config=selected,
        fastest_config=fastest,
        cache_hit=True,
    )

    assert trace.strategy == "adaptive"
    assert trace.candidate_count == 4
    assert trace.measured_count == 3
    assert trace.cache_hit
    assert trace.stages[0].name == "screening"
    statuses = {candidate.config: candidate.status for candidate in trace.candidates}
    assert statuses[selected] == "finalist"
    assert statuses[fastest] == "finalist"
    assert statuses[failed.config] == "failed"
    assert "cache hint used" in trace.summary()
    assert trace.to_dict()["candidate_count"] == 4


def test_search_trace_rejects_unknown_strategy() -> None:
    with pytest.raises(ConfigurationError):
        build_search_trace(strategy="unknown", stages=(), planned_rounds=3)
