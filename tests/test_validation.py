from __future__ import annotations

import pytest

from apdo import LoaderConfig
from apdo.validation import (
    ValidationObservation,
    analyze_validation,
    final_configuration,
)
from tests.helpers import make_pass_metrics

BASELINE = LoaderConfig()
CANDIDATE = LoaderConfig(num_workers=2)


def observations(*gains: float) -> tuple[ValidationObservation, ...]:
    return tuple(
        ValidationObservation(
            round_index=index,
            baseline=make_pass_metrics(100.0),
            candidate=make_pass_metrics(100.0 * (1.0 + gain)),
        )
        for index, gain in enumerate(gains)
    )


def analyze(
    gains: tuple[float, ...],
    *,
    enabled: bool = True,
    min_improvement: float = 0.05,
):
    return analyze_validation(
        BASELINE,
        CANDIDATE,
        observations(*gains),
        enabled=enabled,
        min_improvement=min_improvement,
        seed=17,
    )


def test_confirmed_when_gain_meets_threshold_and_interval_is_positive() -> None:
    result = analyze((0.10, 0.12, 0.11))

    assert result.status == "confirmed"
    assert result.confirmed
    assert result.improvement_percent == pytest.approx(11.0)
    assert result.ci_low_percent is not None and result.ci_low_percent > 0
    assert final_configuration(BASELINE, result) == CANDIDATE


def test_confirmed_at_exact_minimum_gain() -> None:
    result = analyze((0.05, 0.05, 0.05), min_improvement=0.05)

    assert result.status == "confirmed"
    assert result.improvement_percent == pytest.approx(5.0)


def test_inconclusive_when_gain_is_below_minimum() -> None:
    result = analyze((0.03, 0.03, 0.03))

    assert result.status == "inconclusive"
    assert not result.confirmed
    assert result.improvement_percent == pytest.approx(3.0)
    assert final_configuration(BASELINE, result) == BASELINE


def test_inconclusive_when_interval_includes_zero() -> None:
    result = analyze((0.30, 0.00, 0.02))

    assert result.status == "inconclusive"
    assert result.ci_low_percent is not None and result.ci_low_percent <= 0


def test_regression_keeps_the_initial_configuration() -> None:
    result = analyze((-0.02, -0.01, -0.03))

    assert result.status == "regression"
    assert result.improvement_percent == pytest.approx(-2.0)
    assert final_configuration(BASELINE, result) == BASELINE


def test_disabled_validation_is_not_possible() -> None:
    result = analyze((0.20, 0.20, 0.20), enabled=False)

    assert result.status == "not_possible"
    assert result.enabled is False
    assert result.improvement_percent is None
    assert final_configuration(BASELINE, result) == BASELINE


def test_missing_pairs_are_not_possible() -> None:
    result = analyze(())

    assert result.status == "not_possible"
    assert "two paired" in result.reason
    assert result.baseline_summary is None
    assert result.candidate_summary is None


def test_identical_candidate_is_not_needed_even_when_disabled() -> None:
    result = analyze_validation(
        BASELINE,
        BASELINE,
        (),
        enabled=False,
        min_improvement=0.05,
        seed=17,
    )

    assert result.status == "not_needed"
    assert result.enabled is False
    assert final_configuration(BASELINE, result) == BASELINE


def test_validation_payload_contains_paired_observations() -> None:
    result = analyze((0.10, 0.10))

    payload = result.to_dict()

    assert payload["status"] == "confirmed"
    assert payload["min_improvement_percent"] == 5.0
    assert len(payload["observations"]) == 2
    assert payload["observations"][0]["gain_percent"] == pytest.approx(10.0)
    assert payload["observations"][0]["baseline"]["throughput"] == 100.0
    assert payload["observations"][0]["candidate"]["throughput"] == pytest.approx(110.0)


def test_failures_are_preserved_in_validation_result() -> None:
    result = analyze_validation(
        BASELINE,
        CANDIDATE,
        observations(0.10, 0.10),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        failures=("worker failed",),
    )

    assert result.failures == ("worker failed",)
    assert result.to_dict()["failures"] == ["worker failed"]
