from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from apdo.config import VALIDATION_POLICIES, LoaderConfig
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
        if self.baseline.throughput <= 0:
            return 0.0
        return self.candidate.throughput / self.baseline.throughput - 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "gain_percent": self.gain * 100.0,
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
) -> ValidationResult:
    baseline_passes = [item.baseline for item in observations]
    candidate_passes = [item.candidate for item in observations]
    baseline_summary = summarize_passes(baseline_passes) if baseline_passes else None
    candidate_summary = summarize_passes(candidate_passes) if candidate_passes else None
    min_percent = min_improvement * 100.0
    if policy not in VALIDATION_POLICIES:
        raise ValueError(f"unknown validation policy {policy!r}")

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
        )

    gains = [item.gain for item in observations]
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
            f"{observed_percent:+.1f}% with a 95% paired bootstrap interval of "
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
            f"with a 95% paired bootstrap interval of [{ci_low_percent:+.1f}%, "
            f"{ci_high_percent:+.1f}%]. {requirement[0].upper() + requirement[1:]}, "
            "so the initial configuration is retained."
        )

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
    )
