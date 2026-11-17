from __future__ import annotations

import contextlib
import random
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class ReproducibilityInfo:
    status: str
    managed: tuple[str, ...]
    risks: tuple[str, ...]
    reason: str

    @property
    def training_allowed(self) -> bool:
        return self.status in {"verified", "factory_managed"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "managed": list(self.managed),
            "risks": list(self.risks),
            "reason": self.reason,
            "training_allowed": self.training_allowed,
        }


def clone_generator(generator: torch.Generator) -> torch.Generator:
    cloned = torch.Generator(device=generator.device)
    cloned.set_state(generator.get_state())
    return cloned


def seed_runtime(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        with contextlib.suppress(Exception):
            numpy.random.seed(seed % (2**32))
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_rng(seed: int, device: torch.device) -> Iterator[None]:
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = None
    if numpy is not None:
        with contextlib.suppress(Exception):
            numpy_state = numpy.random.get_state()
    cuda_states = None
    if device.type == "cuda" and torch.cuda.is_available():
        with contextlib.suppress(Exception):
            cuda_states = torch.cuda.get_rng_state_all()

    seed_runtime(seed, device)
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        random.setstate(python_state)
        if numpy is not None and numpy_state is not None:
            with contextlib.suppress(Exception):
                numpy.random.set_state(numpy_state)
        if cuda_states is not None:
            with contextlib.suppress(Exception):
                torch.cuda.set_rng_state_all(cuda_states)
