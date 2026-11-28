from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from apdo.benchmark import ConfigMeasurement, StageResult
from apdo.config import LoaderConfig
from apdo.errors import ConfigurationError

SEARCH_STRATEGIES = ("grid", "adaptive")
SEARCH_STATUSES = (
    "pending",
    "screened",
    "promoted",
    "eliminated",
    "failed",
    "budget_skipped",
    "finalist",
    "validated",
)


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    config: LoaderConfig
    status: str
    phase: str
    measured_rounds: int
    selection_value: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "status": self.status,
            "phase": self.phase,
            "measured_rounds": self.measured_rounds,
            "selection_value": self.selection_value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SearchStage:
    name: str
    candidates: tuple[SearchCandidate, ...]
    planned_rounds: int
    budget_exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "planned_rounds": self.planned_rounds,
            "budget_exhausted": self.budget_exhausted,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class SearchTrace:
    strategy: str
    stages: tuple[SearchStage, ...]
    selected_config: LoaderConfig | None = None
    fastest_config: LoaderConfig | None = None
    cache_hit: bool = False

    @property
    def candidates(self) -> tuple[SearchCandidate, ...]:
        return tuple(candidate for stage in self.stages for candidate in stage.candidates)

    @property
    def candidate_count(self) -> int:
        return len({candidate.config for candidate in self.candidates})

    @property
    def measured_count(self) -> int:
        return sum(
            1
            for candidate in self.candidates
            if candidate.status in {"screened", "promoted", "finalist", "validated"}
        )

    @property
    def skipped_count(self) -> int:
        return sum(1 for candidate in self.candidates if candidate.status == "budget_skipped")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "selected_config": (
                self.selected_config.to_dict() if self.selected_config is not None else None
            ),
            "fastest_config": (
                self.fastest_config.to_dict() if self.fastest_config is not None else None
            ),
            "cache_hit": self.cache_hit,
            "candidate_count": self.candidate_count,
            "measured_count": self.measured_count,
            "skipped_count": self.skipped_count,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def summary(self) -> str:
        parts = [
            f"Search strategy: {self.strategy}",
            f"{self.candidate_count} candidates",
            f"{self.measured_count} measured",
        ]
        if self.skipped_count:
            parts.append(f"{self.skipped_count} budget-skipped")
        if self.cache_hit:
            parts.append("cache hint used")
        return "; ".join(parts) + "."


def adaptive_worker_counts(worker_counts: Sequence[int]) -> tuple[int, ...]:
    if not worker_counts:
        raise ConfigurationError("at least one worker count is required")
    maximum = max(worker_counts)
    selected = {0, 1, 2, maximum}
    return tuple(sorted(value for value in selected if value in set(worker_counts)))


def build_search_trace(
    *,
    strategy: str,
    stages: Sequence[tuple[str, StageResult]],
    planned_rounds: int,
    selected_config: LoaderConfig | None = None,
    fastest_config: LoaderConfig | None = None,
    cache_hit: bool = False,
) -> SearchTrace:
    if strategy not in SEARCH_STRATEGIES:
        raise ConfigurationError(f"search strategy must be one of {SEARCH_STRATEGIES}")
    search_stages: list[SearchStage] = []
    for name, result in stages:
        candidates = []
        for measurement in result.measurements:
            candidates.append(
                SearchCandidate(
                    config=measurement.config,
                    status=_status(measurement, selected_config, fastest_config),
                    phase=measurement.phase,
                    measured_rounds=len(measurement.rounds),
                    selection_value=_selection_value(measurement),
                    reason=measurement.error_message if not measurement.succeeded else None,
                )
            )
        search_stages.append(
            SearchStage(
                name=name,
                candidates=tuple(candidates),
                planned_rounds=planned_rounds,
                budget_exhausted=result.budget_exhausted,
            )
        )
    return SearchTrace(
        strategy=strategy,
        stages=tuple(search_stages),
        selected_config=selected_config,
        fastest_config=fastest_config,
        cache_hit=cache_hit,
    )


def _status(
    measurement: ConfigMeasurement,
    selected_config: LoaderConfig | None,
    fastest_config: LoaderConfig | None,
) -> str:
    if measurement.status == "skipped":
        return "budget_skipped"
    if not measurement.succeeded:
        return "failed"
    if selected_config is not None and measurement.config == selected_config:
        return "finalist"
    if fastest_config is not None and measurement.config == fastest_config:
        return "finalist"
    if measurement.budget_limited:
        return "screened"
    return "screened"


def _selection_value(measurement: ConfigMeasurement) -> float | None:
    if measurement.summary is None:
        return None
    return measurement.summary.median_throughput
