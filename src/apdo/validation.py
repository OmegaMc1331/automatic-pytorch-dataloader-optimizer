from __future__ import annotations

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
from apdo.metrics import MetricsSummary, PassMetrics, bootstrap_median_ci, median, summarize_passes
from apdo.order import derive_seed

VALIDATION_STATUSES = (
    "confirmed",
    "inconclusive",
    "regression",
    "not_needed",
    "not_possible",
)

MIN_VALIDATION_PAIRS = 2


@dataclass(frozen=True, slots=True)
class ValidationObservation:
    round_index: int
    baseline: PassMetrics
    candidate: PassMetrics

    @property
    def gain(self) -> float:
        return self.gain_for("sustained")

    def gain_for(self, selection_metric: str) -> float:
        baseline = _metric_throughput(self.baseline, selection_metric)
        candidate = _metric_throughput(self.candidate, selection_metric)
        if baseline <= 0:
            return 0.0
        return candidate / baseline - 1.0

    def to_dict(self) -> dict[str, Any]:
        amortized_gain = self.gain_for("amortized")
        return {
            "round_index": self.round_index,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "gain_percent": self.gain * 100.0,
            "gain_amortized_percent": amortized_gain * 100.0,
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

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    @property
    def threshold_guaranteed(self) -> bool:
        return (
            self.ci_low_percent is not None and self.ci_low_percent >= self.min_improvement_percent
        )

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
            "observations": [observation.to_dict() for observation in self.observations],
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
) -> ValidationResult:
    baseline_passes = [item.baseline for item in observations]
    candidate_passes = [item.candidate for item in observations]
    baseline_summary = summarize_passes(baseline_passes) if baseline_passes else None
    candidate_summary = summarize_passes(candidate_passes) if candidate_passes else None
    min_percent = min_improvement * 100.0
    if policy not in VALIDATION_POLICIES:
        raise ValueError(f"unknown validation policy {policy!r}")
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"unknown selection metric {selection_metric!r}")
    if comparability_policy not in COMPARABILITY_POLICIES:
        raise ValueError(f"unknown comparability policy {comparability_policy!r}")

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
        )

    if len(observations) < MIN_VALIDATION_PAIRS:
        detail = f" Only {len(observations)} paired validation rounds completed."
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
                "At least two paired validation rounds are required to compare the "
                "candidate and the initial configuration." + detail
            ),
            enabled=enabled,
            policy=policy,
            selection_metric=selection_metric,
            comparability=comparability,
            comparability_policy=comparability_policy,
        )

    gains = [item.gain_for(selection_metric) for item in observations]
    metric_label = (
        "total-trial amortized throughput"
        if selection_metric == "amortized"
        else "sustained throughput"
    )
    observed = median(gains)
    ci_low, ci_high = bootstrap_median_ci(
        gains,
        confidence=0.95,
        resamples=4000,
        seed=derive_seed(seed, "validation", "bootstrap"),
    )
    observed_percent = observed * 100.0
    ci_low_percent = ci_low * 100.0
    ci_high_percent = ci_high * 100.0

    if observed < 0.0:
        status = "regression"
        reason = (
            f"Validation observed a median candidate change of {observed_percent:+.1f}% "
            f"in {metric_label} "
            f"with a 95% paired bootstrap interval of [{ci_low_percent:+.1f}%, "
            f"{ci_high_percent:+.1f}%]. The candidate was not faster, so the initial "
            "configuration is retained."
        )
    elif _confirmation_met(observed, ci_low, min_improvement, policy):
        status = "confirmed"
        if policy == "minimum":
            guarantee = (
                f"the lower 95% bound reaches the {min_percent:.1f}% minimum improvement threshold"
            )
        else:
            guarantee = "the interval is strictly above zero"
        reason = (
            f"Validation observed a median candidate improvement of "
            f"{observed_percent:+.1f}% in {metric_label} with a 95% paired bootstrap interval of "
            f"[{ci_low_percent:+.1f}%, {ci_high_percent:+.1f}%]. The observed gain "
            f"meets the {min_percent:.1f}% minimum and {guarantee}."
        )
    else:
        status = "inconclusive"
        if policy == "minimum":
            requirement = (
                f"the lower 95% bound does not reach the {min_percent:.1f}% minimum "
                "improvement threshold"
            )
        else:
            requirement = "the interval is not strictly above zero"
        reason = (
            f"Validation observed a median candidate change of {observed_percent:+.1f}% "
            f"in {metric_label} "
            f"with a 95% paired bootstrap interval of [{ci_low_percent:+.1f}%, "
            f"{ci_high_percent:+.1f}%]. {requirement[0].upper() + requirement[1:]}, "
            "so the initial configuration is retained."
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
    )


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
    )


def _metric_throughput(metrics: PassMetrics, selection_metric: str) -> float:
    if selection_metric == "amortized":
        return metrics.amortized_throughput or metrics.throughput
    return metrics.throughput
