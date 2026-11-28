from __future__ import annotations

import pytest

from apdo.budget import BudgetController, estimate_validation_budget


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_unlimited_budget_always_allows_trials() -> None:
    budget = BudgetController()

    assert budget.can_start("search", 10_000.0)
    assert budget.reserve_validation(10_000.0)
    assert not budget.expired()
    assert budget.status == "unlimited"
    assert budget.remaining_s is None


def test_budget_expires_and_marks_skipped_trials() -> None:
    clock = FakeClock()
    budget = BudgetController(10.0, clock=clock)

    assert budget.can_start("search")
    clock.value = 11.0
    assert not budget.can_start("search")
    assert budget.expired()
    assert budget.status == "exhausted"

    budget.mark_skipped(3)
    assert budget.skipped_trials == 3


def test_validation_reservation_protects_confirmation_capacity() -> None:
    clock = FakeClock()
    budget = BudgetController(10.0, clock=clock)
    budget.note_trial(1.0)

    assert budget.reserve_validation(6.0)
    assert budget.reserved_s == 6.0
    assert budget.status == "reserved"
    assert budget.can_start("validation", 5.0)
    assert not budget.can_start("search", 5.0)

    clock.value = 3.0
    assert budget.validation_available()
    assert budget.to_dict()["reserved_validation_s"] == 6.0


def test_insufficient_validation_reservation_is_reported() -> None:
    budget = BudgetController(1.0)

    assert not budget.reserve_validation(2.0)
    assert budget.reserved_s == 0.0
    assert budget.to_dict()["events"][0]["success"] is False


def test_trial_cost_estimate_is_bounded_and_averaged() -> None:
    budget = BudgetController(100.0)
    for _ in range(70):
        budget.note_trial(2.0)

    assert budget.measured_trials == 70
    assert len(budget.trial_costs_s) == 64
    assert budget.estimated_next_trial_s == pytest.approx(2.0)


def test_validation_budget_scales_with_pairs_and_warmup() -> None:
    assert (
        estimate_validation_budget(
            None,
            validation_rounds=10,
            warmup_runs=1,
        )
        == 0.0
    )
    assert estimate_validation_budget(
        1.0,
        validation_rounds=10,
        warmup_runs=1,
        overhead_factor=1.0,
    ) == pytest.approx(22.0)
