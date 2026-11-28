from __future__ import annotations

from dataclasses import replace

from apdo.config import LoaderConfig
from apdo.profitability import estimate_profitability
from tests.helpers import make_measurement


def summary(throughput: float, total_s: float, *, ci_low: float, ci_high: float):
    measurement = make_measurement(
        LoaderConfig(),
        (throughput, throughput, throughput),
        ci_low=ci_low,
        ci_high=ci_high,
    )
    assert measurement.summary is not None
    return replace(measurement.summary, median_total_elapsed_s=total_s)


def estimate(**overrides):
    values = {
        "baseline_summary": summary(100.0, 10.0, ci_low=95.0, ci_high=105.0),
        "optimized_summary": summary(200.0, 5.0, ci_low=180.0, ci_high=220.0),
        "optimization_cost_s": 2.0,
        "epochs": 1,
        "validation_status": "confirmed",
        "confirmed": True,
        "candidate_is_baseline": False,
        "planned_epochs": 1,
        "planned_sessions": 1,
        "scope": "loader",
    }
    values.update(overrides)
    return estimate_profitability(**values)


def test_confirmed_gain_can_be_profitable() -> None:
    result = estimate()

    assert result.status == "profitable"
    assert result.profitable
    assert result.saved_per_epoch_s == 5.0
    assert result.net_savings_s == 3.0
    assert result.break_even_epochs == 0.4
    assert "profitable" in result.summary()


def test_unconfirmed_gain_is_not_reported_as_profitable() -> None:
    result = estimate(confirmed=False, validation_status="inconclusive")

    assert result.status == "no_confirmed_gain"
    assert not result.profitable


def test_missing_horizon_is_not_evaluated() -> None:
    result = estimate(planned_epochs=None)

    assert result.status == "not_evaluated"
    assert result.net_savings_s is None
    assert "estimate_profitability" in result.summary()


def test_baseline_recommendation_has_no_gain() -> None:
    result = estimate(candidate_is_baseline=True)

    assert result.status == "not_profitable"
    assert result.saved_per_epoch_s == 0.0


def test_high_optimization_cost_is_not_profitable() -> None:
    result = estimate(optimization_cost_s=100.0)

    assert result.status == "not_profitable"
    assert result.net_savings_s is not None
    assert result.net_savings_s < 0.0


def test_missing_timing_is_insufficient_data() -> None:
    result = estimate(baseline_summary=None)

    assert result.status == "insufficient_data"
    assert result.baseline_epoch_s is None
