from __future__ import annotations

import math
from dataclasses import replace

import pytest

from apdo import LoaderConfig
from apdo.comparability import ComparabilityResult
from apdo.validation import (
    MIN_CONFIRMATION_PAIRS,
    ValidationObservation,
    analyze_validation,
    final_configuration,
)
from tests.helpers import make_pass_metrics

BASELINE = LoaderConfig()
CANDIDATE = LoaderConfig(num_workers=2)
TEN_GAINS = (0.10, 0.12, 0.11, 0.13, 0.09, 0.10, 0.11, 0.12, 0.10, 0.11)


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
        planned_pairs=len(gains),
    )


def test_confirmed_when_gain_meets_threshold_and_interval_is_positive() -> None:
    result = analyze(TEN_GAINS)

    assert result.status == "confirmed"
    assert result.confirmed
    assert result.improvement_percent == pytest.approx(11.0)
    assert result.ci_low_percent is not None and result.ci_low_percent > 0
    assert result.valid_pairs == 10
    assert result.statistical_p_value is not None
    assert result.statistical_p_value < 0.05
    assert result.sign_test_successes == 10
    assert final_configuration(BASELINE, result) == CANDIDATE


def test_two_pairs_are_insufficient_to_confirm() -> None:
    result = analyze((0.20, 0.20))

    assert result.status == "not_possible"
    assert not result.confirmed
    assert result.valid_pairs == 2
    assert result.minimum_required_pairs == MIN_CONFIRMATION_PAIRS
    assert result.improvement_percent == pytest.approx(20.0)
    assert final_configuration(BASELINE, result) == BASELINE


def test_eight_pairs_are_the_minimum_for_confirmation() -> None:
    result = analyze((0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10))

    assert result.status == "confirmed"
    assert result.valid_pairs == 8
    assert result.statistical_p_value == pytest.approx(1.0 / 256.0)


def test_gains_exactly_at_the_threshold_are_conservative_failures() -> None:
    result = analyze((0.05,) * 10, min_improvement=0.05)

    assert result.status == "inconclusive"
    assert result.improvement_percent == pytest.approx(5.0)
    assert result.sign_test_successes == 0
    assert result.statistical_p_value == 1.0


def test_inconclusive_when_gain_is_below_minimum() -> None:
    result = analyze((0.03,) * 10)

    assert result.status == "inconclusive"
    assert not result.confirmed
    assert result.improvement_percent == pytest.approx(3.0)
    assert final_configuration(BASELINE, result) == BASELINE


def test_inconclusive_when_interval_includes_zero() -> None:
    result = analyze((0.30, 0.00, 0.02, -0.10, 0.01, 0.00, 0.02, -0.02, 0.01, 0.00))

    assert result.status == "inconclusive"
    assert result.ci_low_percent is not None and result.ci_low_percent <= 0


def test_regression_keeps_the_initial_configuration() -> None:
    result = analyze((-0.02, -0.01, -0.03, -0.02, -0.01, -0.02, -0.03, -0.01, -0.02, -0.02))

    assert result.status == "regression"
    assert result.improvement_percent == pytest.approx(-2.0)
    assert final_configuration(BASELINE, result) == BASELINE


def test_disabled_validation_is_not_possible() -> None:
    result = analyze((0.20,) * 10, enabled=False)

    assert result.status == "not_possible"
    assert result.enabled is False
    assert result.improvement_percent is None
    assert final_configuration(BASELINE, result) == BASELINE


def test_missing_pairs_are_not_possible() -> None:
    result = analyze(())

    assert result.status == "not_possible"
    assert "at least 8" in result.reason
    assert result.baseline_summary is None
    assert result.candidate_summary is None


def test_incomplete_budget_keeps_observations_but_refuses_confirmation() -> None:
    result = analyze_validation(
        BASELINE,
        CANDIDATE,
        observations(*TEN_GAINS),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        planned_pairs=20,
        validation_incomplete=True,
    )

    assert result.status == "not_possible"
    assert result.validation_incomplete is True
    assert result.completed_pairs == 10
    assert result.valid_pairs == 10
    assert result.improvement_percent is not None
    assert result.statistical_p_value is not None


def test_invalid_observations_are_excluded_and_reported() -> None:
    valid = observations(*TEN_GAINS)
    invalid = (
        ValidationObservation(
            round_index=20,
            baseline=make_pass_metrics(0.0),
            candidate=make_pass_metrics(100.0),
        ),
        ValidationObservation(
            round_index=21,
            baseline=make_pass_metrics(100.0),
            candidate=make_pass_metrics(math.nan),
        ),
        ValidationObservation(
            round_index=22,
            baseline=make_pass_metrics(100.0),
            candidate=make_pass_metrics(-1.0),
        ),
    )
    result = analyze_validation(
        BASELINE,
        CANDIDATE,
        (*valid, *invalid),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        planned_pairs=13,
    )

    payload = result.to_dict()

    assert result.status == "confirmed"
    assert result.invalid_pairs == 3
    assert result.valid_pairs == 10
    assert len(payload["rejected_observations"]) == 3
    assert len(payload["invalid_reasons"]) == 3


def test_missing_amortized_metric_makes_observation_invalid() -> None:
    valid = _amortized_observations((0.10,) * 8)
    invalid = (
        ValidationObservation(
            round_index=20,
            baseline=make_pass_metrics(100.0),
            candidate=replace(make_pass_metrics(110.0), amortized_throughput=12.0),
        ),
    )
    result = analyze_validation(
        BASELINE,
        CANDIDATE,
        (*valid, *invalid),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        selection_metric="amortized",
        planned_pairs=9,
    )

    assert result.status == "confirmed"
    assert result.valid_pairs == 8
    assert result.invalid_pairs == 1
    assert "baseline amortized throughput is unavailable" in result.invalid_reasons[0]


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


def test_validation_payload_contains_paired_observations_and_statistics() -> None:
    result = analyze(TEN_GAINS)

    payload = result.to_dict()

    assert payload["status"] == "confirmed"
    assert payload["min_improvement_percent"] == 5.0
    assert payload["planned_pairs"] == 10
    assert payload["completed_pairs"] == 10
    assert payload["valid_pairs"] == 10
    assert payload["invalid_pairs"] == 0
    assert payload["minimum_required_pairs"] == MIN_CONFIRMATION_PAIRS
    assert payload["statistical_method"] == "exact one-sided paired sign test"
    assert payload["statistical_p_value"] < 0.05
    assert payload["sign_test_successes"] == 10
    assert payload["sign_test_ties"] == "counted as failures"
    assert len(payload["observations"]) == 10
    assert payload["observations"][0]["gain_percent"] == pytest.approx(10.0)
    assert payload["observations"][0]["baseline"]["throughput"] == 100.0
    assert payload["observations"][0]["candidate"]["throughput"] == pytest.approx(110.0)


def test_failures_are_preserved_in_validation_result() -> None:
    result = analyze_validation(
        BASELINE,
        CANDIDATE,
        observations(*TEN_GAINS),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        failures=("worker failed",),
    )

    assert result.failures == ("worker failed",)
    assert result.to_dict()["failures"] == ["worker failed"]


def test_minimum_policy_requires_interval_to_reach_threshold() -> None:
    gains = (0.049, 0.049, 0.051, 0.051, 0.30) * 2
    minimum = analyze_validation(
        BASELINE,
        CANDIDATE,
        observations(*gains),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        policy="minimum",
    )
    positive = analyze_validation(
        BASELINE,
        CANDIDATE,
        observations(*gains),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        policy="positive",
    )

    assert minimum.status == "inconclusive"
    assert not minimum.threshold_guaranteed
    assert positive.status == "confirmed"
    assert positive.ci_low_percent == pytest.approx(4.9)
    assert not positive.threshold_guaranteed
    assert positive.to_dict()["policy"] == "positive"


def test_invalid_validation_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="validation policy"):
        analyze_validation(
            BASELINE,
            CANDIDATE,
            observations(0.10, 0.10),
            enabled=True,
            min_improvement=0.05,
            seed=17,
            policy="strict",
        )


def _amortized_observations(gains: tuple[float, ...]) -> tuple[ValidationObservation, ...]:
    return tuple(
        ValidationObservation(
            round_index=index,
            baseline=replace(make_pass_metrics(100.0), amortized_throughput=10.0),
            candidate=replace(
                make_pass_metrics(100.0 * (1.0 + gain)),
                amortized_throughput=10.0 * (1.0 + gain),
            ),
        )
        for index, gain in enumerate(gains)
    )


def test_selection_metric_changes_validation_gain() -> None:
    sustained = analyze_validation(
        BASELINE,
        CANDIDATE,
        observations(*((0.0,) * 10)),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        selection_metric="sustained",
    )
    amortized = analyze_validation(
        BASELINE,
        CANDIDATE,
        _amortized_observations((0.10,) * 10),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        selection_metric="amortized",
    )

    assert sustained.status == "inconclusive"
    assert amortized.status == "confirmed"
    assert amortized.selection_metric == "amortized"
    assert amortized.to_dict()["selection_metric"] == "amortized"
    assert amortized.to_dict()["observations"][0]["gain_amortized_percent"] == pytest.approx(10.0)


def test_strict_comparability_blocks_unverified_confirmation() -> None:
    unknown = ComparabilityResult(
        status="unknown",
        structural_verified=True,
        sample_identity_verified=False,
        rounds_compared=10,
        rounds_with_fingerprints=0,
        structural_mismatch_rounds=0,
        different_sample_rounds=0,
        notes=("no fingerprints",),
    )
    warn = analyze_validation(
        BASELINE,
        CANDIDATE,
        observations(*TEN_GAINS),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        comparability=unknown,
        comparability_policy="warn",
    )
    strict = analyze_validation(
        BASELINE,
        CANDIDATE,
        observations(*TEN_GAINS),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        comparability=unknown,
        comparability_policy="strict",
    )

    assert warn.status == "confirmed"
    assert strict.status == "not_possible"
    assert strict.to_dict()["comparability_policy"] == "strict"
    assert strict.to_dict()["comparability"]["sufficient_for_strict"] is False


def test_observed_comparability_problem_blocks_confirmation() -> None:
    different = ComparabilityResult(
        status="different_samples",
        structural_verified=True,
        sample_identity_verified=False,
        rounds_compared=10,
        rounds_with_fingerprints=10,
        structural_mismatch_rounds=0,
        different_sample_rounds=10,
    )
    result = analyze_validation(
        BASELINE,
        CANDIDATE,
        observations(*TEN_GAINS),
        enabled=True,
        min_improvement=0.05,
        seed=17,
        comparability=different,
        comparability_policy="warn",
    )

    assert result.status == "not_possible"
    assert "comparability" in result.reason


def test_invalid_selection_and_comparability_policies_are_rejected() -> None:
    with pytest.raises(ValueError, match="selection metric"):
        analyze_validation(
            BASELINE,
            CANDIDATE,
            observations(0.1, 0.1),
            enabled=True,
            min_improvement=0.05,
            seed=17,
            selection_metric="fastest",
        )
    with pytest.raises(ValueError, match="comparability policy"):
        analyze_validation(
            BASELINE,
            CANDIDATE,
            observations(0.1, 0.1),
            enabled=True,
            min_improvement=0.05,
            seed=17,
            comparability_policy="maybe",
        )
