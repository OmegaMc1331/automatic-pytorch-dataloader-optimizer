from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from apdo.config import LoaderConfig


@dataclass(frozen=True, slots=True)
class RoundOrder:
    phase: str
    round_index: int
    configs: tuple[LoaderConfig, ...]
    measured: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "round_index": self.round_index,
            "measured": self.measured,
            "configs": [config.to_dict() for config in self.configs],
        }


def derive_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def balanced_orders(
    configs: Sequence[LoaderConfig],
    rounds: int,
    *,
    seed: int,
    phase: str,
    measured: bool = True,
) -> tuple[RoundOrder, ...]:
    if not configs:
        return ()
    if rounds < 1:
        return ()

    base = list(configs)
    random.Random(derive_seed(seed, phase, "base")).shuffle(base)
    total = len(base)
    orders: list[RoundOrder] = []
    for round_index in range(rounds):
        offset = round_index % total
        rotated = base[offset:] + base[:offset]
        if (round_index // total) % 2 == 1:
            rotated.reverse()
        orders.append(
            RoundOrder(
                phase=phase,
                round_index=round_index,
                configs=tuple(rotated),
                measured=measured,
            )
        )
    return tuple(orders)
