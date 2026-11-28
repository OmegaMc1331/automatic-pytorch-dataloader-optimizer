from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from apdo.metrics import MetricsSummary

PROFITABILITY_STATUSES = (
    "profitable",
    "not_profitable",
    "uncertain",
    "no_confirmed_gain",
    "insufficient_data",
    "not_evaluated",
)


@dataclass(frozen=True, slots=True)
class ProfitabilityEstimate:
    status: str
    scope: str
    baseline_epoch_s: float | None
    optimized_epoch_s: float | None
    saved_per_epoch_s: float | None
    optimization_cost_s: float
    planned_epochs: int | None
    planned_sessions: int
    net_savings_s: float | None
    conservative_net_savings_s: float | None
    optimistic_net_savings_s: float | None
    break_even_epochs: float | None
    validation_status: str
    reason: str

    @property
    def profitable(self) -> bool:
        return self.status == "profitable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scope": self.scope,
            "baseline_epoch_s": self.baseline_epoch_s,
            "optimized_epoch_s": self.optimized_epoch_s,
            "saved_per_epoch_s": self.saved_per_epoch_s,
            "optimization_cost_s": self.optimization_cost_s,
            "planned_epochs": self.planned_epochs,
            "planned_sessions": self.planned_sessions,
            "net_savings_s": self.net_savings_s,
            "conservative_net_savings_s": self.conservative_net_savings_s,
            "optimistic_net_savings_s": self.optimistic_net_savings_s,
            "break_even_epochs": self.break_even_epochs,
            "validation_status": self.validation_status,
            "reason": self.reason,
        }

    def summary(self) -> str:
        if self.status == "not_evaluated":
            return (
                "Profitability: not evaluated; provide a planned epoch or session horizon "
                "with result.estimate_profitability(...)."
            )
        if self.status == "insufficient_data":
            return f"Profitability: insufficient data ({self.reason})."
        parts = [
            f"Profitability: {self.status}",
            f"optimization cost {self.optimization_cost_s:.2f} s",
        ]
        if self.saved_per_epoch_s is not None:
            parts.append(f"saved {self.saved_per_epoch_s:.4f} s/epoch")
        if self.net_savings_s is not None:
            parts.append(f"net {self.net_savings_s:.2f} s over the planned horizon")
        if self.break_even_epochs is not None:
            parts.append(f"break-even at {self.break_even_epochs:.1f} epochs")
        return "; ".join(parts) + "."


def estimate_profitability(
    *,
    baseline_summary: MetricsSummary | None,
    optimized_summary: MetricsSummary | None,
    optimization_cost_s: float,
    epochs: int,
    validation_status: str,
    confirmed: bool,
    candidate_is_baseline: bool,
    planned_epochs: int | None,
    planned_sessions: int = 1,
    scope: str = "loader",
) -> ProfitabilityEstimate:
    cost = max(0.0, float(optimization_cost_s))
    if planned_sessions < 1:
        raise ValueError("planned_sessions must be at least 1")
    if planned_epochs is not None and planned_epochs < 1:
        raise ValueError("planned_epochs must be at least 1")
    if candidate_is_baseline:
        baseline_epoch = _epoch_time(baseline_summary, epochs)
        return _estimate(
            status="not_profitable",
            scope=scope,
            baseline_epoch_s=baseline_epoch,
            optimized_epoch_s=baseline_epoch,
            optimization_cost_s=cost,
            planned_epochs=planned_epochs,
            planned_sessions=planned_sessions,
            validation_status=validation_status,
            reason="the recommendation is the initial configuration, so no gain is available",
            saved_per_epoch_s=0.0,
            net_savings_s=0.0,
            conservative_net_savings_s=0.0,
            optimistic_net_savings_s=0.0,
        )
    if baseline_summary is None or optimized_summary is None:
        return _estimate(
            status="insufficient_data",
            scope=scope,
            baseline_epoch_s=None,
            optimized_epoch_s=None,
            optimization_cost_s=cost,
            planned_epochs=planned_epochs,
            planned_sessions=planned_sessions,
            validation_status=validation_status,
            reason="baseline or optimized timing is unavailable",
        )
    if not confirmed:
        return _estimate(
            status="no_confirmed_gain",
            scope=scope,
            baseline_epoch_s=_epoch_time(baseline_summary, epochs),
            optimized_epoch_s=_epoch_time(optimized_summary, epochs),
            optimization_cost_s=cost,
            planned_epochs=planned_epochs,
            planned_sessions=planned_sessions,
            validation_status=validation_status,
            reason="independent validation did not confirm the recommendation",
        )
    if planned_epochs is None:
        return _estimate(
            status="not_evaluated",
            scope=scope,
            baseline_epoch_s=_epoch_time(baseline_summary, epochs),
            optimized_epoch_s=_epoch_time(optimized_summary, epochs),
            optimization_cost_s=cost,
            planned_epochs=None,
            planned_sessions=planned_sessions,
            validation_status=validation_status,
            reason="a planned epoch horizon is required for a net-savings estimate",
        )

    baseline_epoch = _epoch_time(baseline_summary, epochs)
    optimized_epoch = _epoch_time(optimized_summary, epochs)
    if baseline_epoch is None or optimized_epoch is None or baseline_epoch <= 0.0:
        return _estimate(
            status="insufficient_data",
            scope=scope,
            baseline_epoch_s=baseline_epoch,
            optimized_epoch_s=optimized_epoch,
            optimization_cost_s=cost,
            planned_epochs=planned_epochs,
            planned_sessions=planned_sessions,
            validation_status=validation_status,
            reason="epoch timing is missing or non-positive",
        )

    saved = baseline_epoch - optimized_epoch
    horizon_epochs = planned_epochs * planned_sessions
    net = saved * horizon_epochs - cost
    conservative_saved = _conservative_saved_per_epoch(
        baseline_summary,
        optimized_summary,
        baseline_epoch,
        epochs,
    )
    optimistic_saved = _optimistic_saved_per_epoch(
        baseline_summary,
        optimized_summary,
        baseline_epoch,
        epochs,
    )
    conservative_net = (
        None if conservative_saved is None else conservative_saved * horizon_epochs - cost
    )
    optimistic_net = None if optimistic_saved is None else optimistic_saved * horizon_epochs - cost
    break_even = cost / (saved * planned_sessions) if saved > 0.0 else None

    if saved <= 0.0:
        status = "not_profitable"
        reason = "the validated recommendation did not reduce modeled epoch time"
    elif net <= 0.0:
        status = "not_profitable"
        reason = "the optimization cost exceeds modeled savings over the planned horizon"
    elif conservative_net is not None and conservative_net <= 0.0:
        status = "uncertain"
        reason = "point-estimate savings are positive but the conservative bound is not"
    else:
        status = "profitable"
        reason = "modeled savings exceed the optimization cost over the planned horizon"

    return _estimate(
        status=status,
        scope=scope,
        baseline_epoch_s=baseline_epoch,
        optimized_epoch_s=optimized_epoch,
        optimization_cost_s=cost,
        planned_epochs=planned_epochs,
        planned_sessions=planned_sessions,
        saved_per_epoch_s=saved,
        net_savings_s=net,
        conservative_net_savings_s=conservative_net,
        optimistic_net_savings_s=optimistic_net,
        break_even_epochs=break_even,
        validation_status=validation_status,
        reason=reason,
    )


def _estimate(
    *,
    status: str,
    scope: str,
    baseline_epoch_s: float | None,
    optimized_epoch_s: float | None,
    optimization_cost_s: float,
    planned_epochs: int | None,
    planned_sessions: int,
    validation_status: str,
    reason: str,
    saved_per_epoch_s: float | None = None,
    net_savings_s: float | None = None,
    conservative_net_savings_s: float | None = None,
    optimistic_net_savings_s: float | None = None,
    break_even_epochs: float | None = None,
) -> ProfitabilityEstimate:
    if saved_per_epoch_s is None and baseline_epoch_s is not None and optimized_epoch_s is not None:
        saved_per_epoch_s = baseline_epoch_s - optimized_epoch_s
    return ProfitabilityEstimate(
        status=status,
        scope=scope,
        baseline_epoch_s=baseline_epoch_s,
        optimized_epoch_s=optimized_epoch_s,
        saved_per_epoch_s=saved_per_epoch_s,
        optimization_cost_s=optimization_cost_s,
        planned_epochs=planned_epochs,
        planned_sessions=planned_sessions,
        net_savings_s=net_savings_s,
        conservative_net_savings_s=conservative_net_savings_s,
        optimistic_net_savings_s=optimistic_net_savings_s,
        break_even_epochs=break_even_epochs,
        validation_status=validation_status,
        reason=reason,
    )


def _epoch_time(summary: MetricsSummary | None, epochs: int) -> float | None:
    if summary is None or summary.median_total_elapsed_s is None:
        return None
    value = summary.median_total_elapsed_s / max(1, epochs)
    return value if math.isfinite(value) and value > 0.0 else None


def _conservative_saved_per_epoch(
    baseline: MetricsSummary,
    optimized: MetricsSummary,
    baseline_epoch_s: float,
    epochs: int,
) -> float | None:
    candidate_low = optimized.ci_low_throughput
    baseline_throughput = baseline.median_throughput
    if (
        candidate_low is None
        or not math.isfinite(candidate_low)
        or candidate_low <= 0.0
        or baseline_throughput <= 0.0
    ):
        return None
    conservative_speedup = candidate_low / baseline_throughput
    if conservative_speedup <= 1.0:
        return 0.0
    conservative_epoch = baseline_epoch_s / conservative_speedup
    return max(0.0, baseline_epoch_s - conservative_epoch)


def _optimistic_saved_per_epoch(
    baseline: MetricsSummary,
    optimized: MetricsSummary,
    baseline_epoch_s: float,
    epochs: int,
) -> float | None:
    candidate_high = optimized.ci_high_throughput
    baseline_throughput = baseline.median_throughput
    if (
        candidate_high is None
        or not math.isfinite(candidate_high)
        or candidate_high <= 0.0
        or baseline_throughput <= 0.0
    ):
        return None
    speedup = candidate_high / baseline_throughput
    if speedup <= 1.0:
        return 0.0
    optimistic_epoch = baseline_epoch_s / speedup
    return max(0.0, baseline_epoch_s - optimistic_epoch)
