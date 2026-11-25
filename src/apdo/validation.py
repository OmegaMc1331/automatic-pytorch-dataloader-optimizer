from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from apdo.comparability import ComparabilityResult
from apdo.config import (
    COMPARABILITY_POLICIES,
    SELECTION_METRICS,
    VALIDATION_POLICIES,
    LoaderConfig,
)
from apdo.metrics import (
    MetricsSummary,
    PassMetrics,
    bootstrap_median_ci,
    is_valid_throughput,
    median,
    pass_selection_throughput,
    selection_metric_label,
    summarize_passes,
)
from apdo.order import derive_seed

VALIDATION_STATUSES = (
    "confirmed",
    "inconclusive",
    "regression",
    "not_needed",
    "not_possible",
)

MIN_CONFIRMATION_PAIRS = 8
MIN_VALIDATION_PAIRS = MIN_CONFIRMATION_PAIRS
STATISTICAL_ALPHA = 0.05
STATISTICAL_METHOD = "exact one-sided paired sign test"
SIGN_TEST_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class ValidationObservation:
    round_index: int
    baseline: PassMetrics
    candidate: PassMetrics

    @property
    def gain(self) -> float | None:
        return self.gain_for("sustained")

    def gain_for(self, selection_metric: str) -> float | None:
        valid, _ = self.validity(selection_metric)
        if not valid:
            return None
        baseline = pass_selection_throughput(self.baseline, selection_metric)
        candidate = pass_selection_throughput(self.candidate, selection_metric)
        assert baseline is not None
        assert candidate is not None
        return candidate / baseline - 1.0

    def validity(self, selection_metric: str) -> tuple[bool, str | None]:
        if self.baseline.training != self.candidate.training:
            return False, "training mode differs between paired trials"
        if self.baseline.batches != self.candidate.batches:
            return False, "completed batch counts differ between paired trials"
        if (
            self.baseline.expected_batches is not None
            and self.candidate.expected_batches is not None
            and self.baseline.expected_batches != self.candidate.expected_batches
        ):
            return False, "planned batch counts differ between paired trials"
        if (
            self.baseline.total_samples > 0
            and self.candidate.total_samples > 0
            and self.baseline.total_samples != self.candidate.total_samples
        ):
            return False, "planned sample counts differ between paired trials"
        if self.baseline.full_epoch != self.candidate.full_epoch:
            return False, "epoch completeness differs between paired trials"
        baseline = pass_selection_throughput(self.baseline, selection_metric)
        candidate = pass_selection_throughput(self.candidate, selection_metric)
        label = selection_metric_label(selection_metric)
        if baseline is None:
            return False, f"baseline {label} is unavailable"
        if candidate is None:
            return False, f"candidate {label} is unavailable"
        if not is_valid_throughput(baseline):
            return False, f"baseline {label} is not a positive finite value"
        if not is_valid_throughput(candidate):
            return False, f"candidate {label} is not a positive finite value"
        gain = candidate / baseline - 1.0
        if not math.isfinite(gain):
            return False, f"{label} gain is not finite"
        return True, None

    def to_dict(self, selection_metric: str = "sustained") -> dict[str, Any]:
        valid, validity_reason = self.validity(selection_metric)
        sustained_gain = self.gain_for("sustained")
        amortized_gain = self.gain_for("amortized")
        selected_gain = self.gain_for(selection_metric)
        return {
            "round_index": self.round_index,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "gain_percent": None if sustained_gain is None else sustained_gain * 100.0,
            "gain_amortized_percent": None if amortized_gain is None else amortized_gain * 100.0,
            "gain_selection_percent": None if selected_gain is None else selected_gain * 100.0,
            "baseline_selection_throughput": pass_selection_throughput(
                self.baseline, selection_metric
            ),
            "candidate_selection_throughput": pass_selection_throughput(
                self.candidate, selection_metric
            ),
            "valid": valid,
            "validity_reason": validity_reason,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    status: str
    baseline_config: LoaderConfig
    candidate_config: LoaderConfig
    baseline_summary: MetricsSummary | None
    candidate_summary: MetricsSummary | None
    improvement_percent: float | None
    ci_low_percent: float | None
    ci_high_percent: float | None
    min_improvement_percent: float
    observations: tuple[ValidationObservation, ...]
    failures: tuple[str, ...]
    reason: str
    enabled: bool
    policy: str = "minimum"
    selection_metric: str = "sustained"
    comparability: ComparabilityResult | None = None
    comparability_policy: str = "warn"
    planned_pairs: int = 0
    statistical_p_value: float | None = None
    statistical_alpha: float = STATISTICAL_ALPHA
    statistical_method: str = STATISTICAL_METHOD
    sign_test_successes: int | None = None
    validation_incomplete: bool = False
    rejected_observations: tuple[ValidationObservation, ...] = ()

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    @property
    def completed_pairs(self) -> int:
        return len(self.observations)

    @property
    def valid_pairs(self) -> int:
        return len(self.observations) - len(self.rejected_observations)

    @property
    def invalid_pairs(self) -> int:
        return len(self.rejected_observations)

    @property
    def minimum_required_pairs(self) -> int:
        return MIN_CONFIRMATION_PAIRS

    @property
    def statistical_threshold_percent(self) -> float:
        if self.policy == "positive":
            return 0.0
        return self.min_improvement_percent

    @property
    def threshold_guaranteed(self) -> bool:
        return (
            self.ci_low_percent is not None and self.ci_low_percent >= self.min_improvement_percent
        )

    @property
    def invalid_reasons(self) -> tuple[str, ...]:
        reasons = []
        for observation in self.rejected_observations:
            _, reason = observation.validity(self.selection_metric)
            reasons.append(f"round {observation.round_index}: {reason or 'invalid observation'}")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "enabled": self.enabled,
            "baseline_config": self.baseline_config.to_dict(),
            "candidate_config": self.candidate_config.to_dict(),
            "improvement_percent": self.improvement_percent,
            "ci_low_percent": self.ci_low_percent,
            "ci_high_percent": self.ci_high_percent,
            "min_improvement_percent": self.min_improvement_percent,
            "reason": self.reason,
            "policy": self.policy,
            "selection_metric": self.selection_metric,
            "comparability_policy": self.comparability_policy,
            "comparability": (
                self.comparability.to_dict() if self.comparability is not None else None
            ),
            "threshold_guaranteed": self.threshold_guaranteed,
            "planned_pairs": self.planned_pairs,
            "completed_pairs": self.completed_pairs,
            "valid_pairs": self.valid_pairs,
            "invalid_pairs": self.invalid_pairs,
            "minimum_required_pairs": self.minimum_required_pairs,
            "validation_incomplete": self.validation_incomplete,
            "statistical_method": self.statistical_method,
            "statistical_alpha": self.statistical_alpha,
            "statistical_threshold_percent": self.statistical_threshold_percent,
            "statistical_p_value": self.statistical_p_value,
            "sign_test_successes": self.sign_test_successes,
            "sign_test_ties": "counted as failures",
            "invalid_reasons": list(self.invalid_reasons),
            "observations": [
                observation.to_dict(self.selection_metric) for observation in self.observations
            ],
            "rejected_observations": [
                observation.to_dict(self.selection_metric)
                for observation in self.rejected_observations
            ],
            "failures": list(self.failures),
        }
        if self.baseline_summary is not None:
            result["baseline_summary"] = self.baseline_summary.to_dict()
        if self.candidate_summary is not None:
            result["candidate_summary"] = self.candidate_summary.to_dict()
        return result


def analyze_validation(
    baseline_config: LoaderConfig,
    candidate_config: LoaderConfig,
    observations: Sequence[ValidationObservation],
    *,
    enabled: bool,
    min_improvement: float,
    seed: int,
    failures: Sequence[str] = (),
    policy: str = "minimum",
    selection_metric: str = "sustained",
    comparability: ComparabilityResult | None = None,
    comparability_policy: str = "warn",
    planned_pairs: int = 0,
    validation_incomplete: bool = False,
    statistical_alpha: float = STATISTICAL_ALPHA,
) -> ValidationResult:
    if policy not in VALIDATION_POLICIES:
        raise ValueError(f"unknown validation policy {policy!r}")
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"unknown selection metric {selection_metric!r}")
    if comparability_policy not in COMPARABILITY_POLICIES:
        raise ValueError(f"unknown comparability policy {comparability_policy!r}")
    if not 0.0 < statistical_alpha < 1.0:
        raise ValueError("statistical_alpha must be between 0 and 1")

    min_percent = min_improvement * 100.0
    valid_observations: list[ValidationObservation] = []
    rejected_observations: list[ValidationObservation] = []
    for observation in observations:
        valid, _ = observation.validity(selection_metric)
        if valid:
            valid_observations.append(observation)
        else:
            rejected_observations.append(observation)

    baseline_passes = [item.baseline for item in valid_observations]
    candidate_passes = [item.candidate for item in valid_observations]
    baseline_summary = summarize_passes(baseline_passes) if baseline_passes else None
    candidate_summary = summarize_passes(candidate_passes) if candidate_passes else None
    gains = [
        gain for item in valid_observations if (gain := item.gain_for(selection_metric)) is not None
    ]
    if candidate_config == baseline_config:
        return _result(
            status="not_needed",
            baseline_config=baseline_config,
            candidate_config=candidate_config,
            baseline_summary=baseline_summary,
            candidate_summary=candidate_summary,
            improvement_percent=None,
            ci_low_percent=None,
            ci_high_percent=None,
            min_percent=min_percent,
            observations=observations,
            failures=failures,
            reason=(
                "The search selected the initial configuration, so independent "
                "validation was not required."
            ),
            enabled=enabled,
            policy=policy,
            selection_metric=selection_metric,
            comparability=comparability,
            comparability_policy=comparability_policy,
            planned_pairs=planned_pairs,
            validation_incomplete=validation_incomplete,
            statistical_alpha=statistical_alpha,
            rejected_observations=rejected_observations,
        )

    if not enabled:
        return _result(
            status="not_possible",
            baseline_config=baseline_config,
            candidate_config=candidate_config,
            baseline_summary=baseline_summary,
            candidate_summary=candidate_summary,
            improvement_percent=None,
            ci_low_percent=None,
            ci_high_percent=None,
            min_percent=min_percent,
            observations=observations,
            failures=failures,
            reason=(
                "Independent validation was disabled. The initial configuration is "
                "retained because the search result is unconfirmed."
            ),
            enabled=enabled,
            policy=policy,
            selection_metric=selection_metric,
            comparability=comparability,
            comparability_policy=comparability_policy,
            planned_pairs=planned_pairs,
            validation_incomplete=validation_incomplete,
            statistical_alpha=statistical_alpha,
            rejected_observations=rejected_observations,
        )

    observed: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    successes: int | None = None
    if gains:
        observed = median(gains)
        ci_low, ci_high = bootstrap_median_ci(
            gains,
            confidence=0.95,
            resamples=4000,
            seed=derive_seed(seed, "validation", "bootstrap"),
        )
        sign_threshold = 0.0 if policy == "positive" else min_improvement
        successes, p_value = _exact_sign_test(gains, sign_threshold)
    observed_percent = None if observed is None else observed * 100.0
    ci_low_percent = None if ci_low is None else ci_low * 100.0
    ci_high_percent = None if ci_high is None else ci_high * 100.0
    metric_label = selection_metric_label(selection_metric)

    valid_count = len(valid_observations)
    minimum = MIN_CONFIRMATION_PAIRS
    if valid_count < minimum:
        status = "not_possible"
        reason = _insufficient_pairs_reason(
            completed=len(observations),
            valid=valid_count,
            rejected=len(rejected_observations),
            incomplete=validation_incomplete,
            metric_label=metric_label,
        )
    elif validation_incomplete:
        status = "not_possible"
        reason = (
            "The time budget stopped validation before the planned rounds completed. "
            f"{valid_count} valid paired rounds were retained, but confirmation requires "
            "the complete planned sample and is refused for an incomplete validation."
        )
    elif observed is not None and observed < 0.0:
        status = "regression"
        assert ci_low_percent is not None
        assert ci_high_percent is not None
        assert p_value is not None
        reason = (
            f"Validation observed a median candidate change of {observed_percent:+.1f}% "
            f"in {metric_label} with a 95% paired bootstrap interval of "
            f"[{ci_low_percent:+.1f}%, {ci_high_percent:+.1f}%]. "
            f"The exact one-sided paired sign test produced p={p_value:.4f}. "
            "The candidate was not faster, so the initial configuration is retained."
        )
    elif (
        observed is not None
        and ci_low is not None
        and p_value is not None
        and _confirmation_met(observed, ci_low, min_improvement, policy)
        and p_value < statistical_alpha
    ):
        status = "confirmed"
        assert ci_low_percent is not None
        assert ci_high_percent is not None
        assert successes is not None
        if policy == "minimum":
            bound_text = (
                f"the lower 95% bound reaches the {min_percent:.1f}% minimum improvement threshold"
            )
            test_text = f"gains above the {min_percent:.1f}% minimum threshold"
        else:
            bound_text = "the interval is strictly above zero"
            test_text = "gains above zero"
        reason = (
            f"Validation observed a median candidate improvement of "
            f"{observed_percent:+.1f}% in {metric_label} with a 95% paired bootstrap interval of "
            f"[{ci_low_percent:+.1f}%, {ci_high_percent:+.1f}%]. The observed gain "
            f"meets the {min_percent:.1f}% minimum and {bound_text}. "
            f"{successes} of {valid_count} paired rounds had {test_text}; "
            f"the exact one-sided sign test returned p={p_value:.4f}, below alpha "
            f"{statistical_alpha:.2f}."
        )
    else:
        status = "inconclusive"
        if observed is None or ci_low is None or p_value is None:
            reason = "No valid paired observation was available for a statistical decision."
        else:
            assert ci_low_percent is not None
            assert ci_high_percent is not None
            reason = (
                f"Validation observed a median candidate change of {observed_percent:+.1f}% "
                f"in {metric_label} with a 95% paired bootstrap interval of "
                f"[{ci_low_percent:+.1f}%, {ci_high_percent:+.1f}%]. "
                f"{_confirmation_failure_reason(observed, ci_low, min_improvement, policy)} "
                f"The exact one-sided sign test returned p={p_value:.4f} against alpha "
                f"{statistical_alpha:.2f}. The initial configuration is retained."
            )

    if comparability is not None and comparability_policy != "off":
        if status == "confirmed" and comparability_policy == "strict":
            if not comparability.sufficient_for_strict:
                status = "not_possible"
                reason = (
                    "Independent validation observed a candidate gain, but the strict "
                    "comparability policy requires verified identical samples and batch "
                    f"structure in every paired round. {comparability.summary}"
                )
        elif status == "confirmed" and comparability.has_observed_problem:
            status = "not_possible"
            reason = (
                "Independent validation observed a candidate gain, but batch "
                "comparability checks found an observed difference that could invalidate "
                f"the comparison. {comparability.summary}"
            )
        elif status == "confirmed":
            reason = f"{reason} {comparability.summary}"

    return _result(
        status=status,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        improvement_percent=observed_percent,
        ci_low_percent=ci_low_percent,
        ci_high_percent=ci_high_percent,
        min_percent=min_percent,
        observations=observations,
        failures=failures,
        reason=reason,
        enabled=enabled,
        policy=policy,
        selection_metric=selection_metric,
        comparability=comparability,
        comparability_policy=comparability_policy,
        planned_pairs=planned_pairs,
        statistical_p_value=p_value,
        statistical_alpha=statistical_alpha,
        sign_test_successes=successes,
        validation_incomplete=validation_incomplete,
        rejected_observations=rejected_observations,
    )


def _insufficient_pairs_reason(
    *,
    completed: int,
    valid: int,
    rejected: int,
    incomplete: bool,
    metric_label: str,
) -> str:
    minimum = MIN_CONFIRMATION_PAIRS
    if incomplete:
        detail = " The time budget stopped validation before the planned rounds completed."
    else:
        detail = ""
    if completed < minimum:
        count_text = (
            f"Only {completed} paired validation rounds completed; at least {minimum} are required."
        )
    else:
        count_text = (
            f"Only {valid} of {completed} paired validation rounds were valid for "
            f"{metric_label}; at least {minimum} valid pairs are required."
        )
    rejected_text = f" {rejected} observations were rejected as invalid." if rejected else ""
    return (
        f"{count_text}{rejected_text}{detail} The initial configuration is retained because "
        "the validation has insufficient evidence for confirmation."
    )


def _confirmation_failure_reason(
    observed: float,
    ci_low: float,
    min_improvement: float,
    policy: str,
) -> str:
    if observed < min_improvement:
        detail = "The observed median gain does not reach the minimum improvement threshold."
    elif policy == "minimum":
        if ci_low < min_improvement:
            detail = "The lower 95% bound does not reach the minimum improvement threshold."
        else:
            detail = "The sign test does not provide enough evidence at the configured alpha."
    elif ci_low <= 0.0:
        detail = "The lower 95% bound is not strictly above zero."
    else:
        detail = "The sign test does not provide enough evidence at the configured alpha."
    return detail


def _exact_sign_test(gains: Sequence[float], threshold: float) -> tuple[int, float]:
    successes = sum(1 for gain in gains if gain > threshold + SIGN_TEST_EPSILON)
    trials = len(gains)
    if trials == 0:
        return 0, 1.0
    tail = sum(math.comb(trials, index) for index in range(successes, trials + 1))
    p_value = min(1.0, tail / (2**trials))
    return successes, p_value


def _confirmation_met(
    observed: float,
    ci_low: float,
    min_improvement: float,
    policy: str,
) -> bool:
    if observed < min_improvement:
        return False
    if policy == "minimum":
        return ci_low >= min_improvement
    return ci_low > 0.0


def final_configuration(
    baseline_config: LoaderConfig,
    validation: ValidationResult,
) -> LoaderConfig:
    return validation.candidate_config if validation.confirmed else baseline_config


def _result(
    *,
    status: str,
    baseline_config: LoaderConfig,
    candidate_config: LoaderConfig,
    baseline_summary: MetricsSummary | None,
    candidate_summary: MetricsSummary | None,
    improvement_percent: float | None,
    ci_low_percent: float | None,
    ci_high_percent: float | None,
    min_percent: float,
    observations: Sequence[ValidationObservation],
    failures: Sequence[str],
    reason: str,
    enabled: bool,
    policy: str,
    selection_metric: str,
    comparability: ComparabilityResult | None = None,
    comparability_policy: str = "warn",
    planned_pairs: int = 0,
    statistical_p_value: float | None = None,
    statistical_alpha: float = STATISTICAL_ALPHA,
    sign_test_successes: int | None = None,
    validation_incomplete: bool = False,
    rejected_observations: Sequence[ValidationObservation] = (),
) -> ValidationResult:
    if status not in VALIDATION_STATUSES:
        raise ValueError(f"unknown validation status {status!r}")
    if policy not in VALIDATION_POLICIES:
        raise ValueError(f"unknown validation policy {policy!r}")
    return ValidationResult(
        status=status,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        improvement_percent=improvement_percent,
        ci_low_percent=ci_low_percent,
        ci_high_percent=ci_high_percent,
        min_improvement_percent=min_percent,
        observations=tuple(observations),
        failures=tuple(failures),
        reason=reason,
        enabled=enabled,
        policy=policy,
        selection_metric=selection_metric,
        comparability=comparability,
        comparability_policy=comparability_policy,
        planned_pairs=planned_pairs,
        statistical_p_value=statistical_p_value,
        statistical_alpha=statistical_alpha,
        statistical_method=STATISTICAL_METHOD,
        sign_test_successes=sign_test_successes,
        validation_incomplete=validation_incomplete,
        rejected_observations=tuple(rejected_observations),
    )
