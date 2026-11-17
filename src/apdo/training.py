from __future__ import annotations

from collections.abc import Callable
from typing import Any

from apdo.errors import ConfigurationError

TrainingStep = Callable[[Any], None]
TrainingStepFactory = Callable[[], TrainingStep]


def create_training_step(factory: TrainingStepFactory) -> TrainingStep:
    if not callable(factory):
        raise ConfigurationError("training_step_factory must be callable")
    try:
        step = factory()
    except Exception as exc:
        raise ConfigurationError(
            f"training_step_factory failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not callable(step):
        raise ConfigurationError("training_step_factory must return a callable step")
    return step
