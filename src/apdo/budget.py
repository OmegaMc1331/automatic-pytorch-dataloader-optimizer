from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

BUDGET_CATEGORIES = ("search", "training_finalist", "validation", "cleanup", "overhead")
BUDGET_STATUSES = ("unlimited", "available", "reserved", "exhausted")


@dataclass(slots=True)
class BudgetController:
    budget_s: float | None = None
    clock: Callable[[], float] = time.monotonic
    category_elapsed_s: dict[str, float] = field(default_factory=dict)
    reserved_s: float = 0.0
    reserve_category: str = "validation"
    reserve_requested_s: float = 0.0
    exhausted: bool = False
    skipped_trials: int = 0
    measured_trials: int = 0
    trial_costs_s: list[float] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    _started_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.budget_s is not None:
            if isinstance(self.budget_s, bool) or not isinstance(self.budget_s, (int, float)):
                raise ValueError("budget_s must be a positive number or None")
            if not math.isfinite(float(self.budget_s)) or float(self.budget_s) <= 0.0:
                raise ValueError("budget_s must be a positive finite number or None")
            self.budget_s = float(self.budget_s)
        if self.reserve_category not in BUDGET_CATEGORIES:
            raise ValueError(f"reserve_category must be one of {BUDGET_CATEGORIES}")
        self._started_at = self.clock()

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self.clock() - self._started_at)

    @property
    def remaining_s(self) -> float | None:
        if self.budget_s is None:
            return None
        return max(0.0, self.budget_s - self.elapsed_s)

    @property
    def available_s(self) -> float | None:
        remaining = self.remaining_s
        if remaining is None:
            return None
        return max(0.0, remaining - self.reserved_s)

    @property
    def estimated_next_trial_s(self) -> float | None:
        if not self.trial_costs_s:
            return None
        return sum(self.trial_costs_s) / len(self.trial_costs_s)

    @property
    def status(self) -> str:
        if self.budget_s is None:
            return "unlimited"
        if self.exhausted or (self.remaining_s or 0.0) <= 0.0:
            return "exhausted"
        if self.reserved_s > 0.0:
            return "reserved"
        return "available"

    def can_start(self, category: str, estimated_s: float = 0.0) -> bool:
        if category not in BUDGET_CATEGORIES:
            raise ValueError(f"category must be one of {BUDGET_CATEGORIES}")
        if self.budget_s is None:
            return True
        if not math.isfinite(estimated_s) or estimated_s < 0.0:
            raise ValueError("estimated_s must be a non-negative finite number")
        remaining = self.remaining_s or 0.0
        if remaining <= 0.0:
            self.exhausted = True
            return False
        available = remaining
        if category != self.reserve_category:
            available = max(0.0, remaining - self.reserved_s)
        if available <= 0.0 or available < estimated_s:
            self.exhausted = True
            return False
        return True

    def expired(self) -> bool:
        if self.budget_s is None:
            return False
        if (self.remaining_s or 0.0) <= 0.0:
            self.exhausted = True
            return True
        return False

    def reserve_validation(self, amount_s: float) -> bool:
        if not math.isfinite(amount_s) or amount_s < 0.0:
            raise ValueError("amount_s must be a non-negative finite number")
        self.reserve_requested_s = amount_s
        if self.budget_s is None:
            self.reserved_s = 0.0
            self._record_event("validation_reserved", amount_s, True, "unlimited budget")
            return True
        remaining = self.remaining_s or 0.0
        sufficient = remaining >= amount_s
        self.reserved_s = amount_s if sufficient else 0.0
        self._record_event(
            "validation_reserved",
            amount_s,
            sufficient,
            "reserved for independent validation"
            if sufficient
            else "insufficient remaining budget for the planned validation",
        )
        return sufficient

    def release_reservation(self) -> None:
        self.reserved_s = 0.0
        self._record_event("validation_released", self.reserve_requested_s, True, "released")

    def validation_available(self, estimated_s: float = 0.0) -> bool:
        if self.budget_s is None:
            return True
        remaining = self.remaining_s or 0.0
        required = max(estimated_s, self.reserve_requested_s)
        return remaining >= required

    def note_trial(self, duration_s: float) -> None:
        if not math.isfinite(duration_s) or duration_s < 0.0:
            return
        self.measured_trials += 1
        self.trial_costs_s.append(duration_s)
        if len(self.trial_costs_s) > 64:
            del self.trial_costs_s[0]

    def mark_skipped(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self.skipped_trials += count

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name not in BUDGET_CATEGORIES:
            raise ValueError(f"phase must be one of {BUDGET_CATEGORIES}")
        started = self.clock()
        try:
            yield
        finally:
            elapsed = max(0.0, self.clock() - started)
            self.category_elapsed_s[name] = self.category_elapsed_s.get(name, 0.0) + elapsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_s": self.budget_s,
            "status": self.status,
            "elapsed_s": self.elapsed_s,
            "remaining_s": self.remaining_s,
            "available_s": self.available_s,
            "reserved_validation_s": self.reserved_s,
            "validation_reserve_requested_s": self.reserve_requested_s,
            "search_elapsed_s": self.category_elapsed_s.get("search", 0.0),
            "training_finalist_elapsed_s": self.category_elapsed_s.get("training_finalist", 0.0),
            "validation_elapsed_s": self.category_elapsed_s.get("validation", 0.0),
            "cleanup_elapsed_s": self.category_elapsed_s.get("cleanup", 0.0),
            "overhead_elapsed_s": self.category_elapsed_s.get("overhead", 0.0),
            "estimated_next_trial_s": self.estimated_next_trial_s,
            "measured_trials": self.measured_trials,
            "skipped_trials": self.skipped_trials,
            "exhausted": self.exhausted,
            "events": list(self.events),
        }

    def summary(self) -> str:
        if self.budget_s is None:
            budget_text = "unlimited"
        else:
            budget_text = f"{self.budget_s:.2f} s total, {self.remaining_s or 0.0:.2f} s remaining"
        parts = [
            f"Budget: {budget_text} ({self.status})",
            f"Search time: {self.category_elapsed_s.get('search', 0.0):.2f} s",
        ]
        if self.category_elapsed_s.get("training_finalist", 0.0):
            parts.append(
                f"Training finalist time: {self.category_elapsed_s['training_finalist']:.2f} s"
            )
        if self.category_elapsed_s.get("validation", 0.0):
            parts.append(f"Validation time: {self.category_elapsed_s['validation']:.2f} s")
        if self.reserved_s:
            parts.append(f"Validation reservation: {self.reserved_s:.2f} s")
        if self.skipped_trials:
            parts.append(f"Skipped trials: {self.skipped_trials}")
        return "; ".join(parts)

    def _record_event(
        self,
        name: str,
        amount_s: float,
        success: bool,
        reason: str,
    ) -> None:
        self.events.append(
            {
                "event": name,
                "amount_s": amount_s,
                "success": success,
                "reason": reason,
                "elapsed_s": self.elapsed_s,
                "remaining_s": self.remaining_s,
            }
        )


def estimate_validation_budget(
    per_trial_s: float | None,
    *,
    validation_rounds: int,
    warmup_runs: int,
    overhead_factor: float = 1.15,
) -> float:
    if per_trial_s is None or per_trial_s <= 0.0:
        return 0.0
    if validation_rounds < 1:
        raise ValueError("validation_rounds must be at least 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    trial_count = 2 * validation_rounds + 2 * warmup_runs
    return float(per_trial_s) * trial_count * overhead_factor
