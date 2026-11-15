from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apdo.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int):
            raise ConfigurationError("num_workers must be an integer")
        if self.num_workers < 0:
            raise ConfigurationError("num_workers must be greater than or equal to 0")
        if not isinstance(self.pin_memory, bool):
            raise ConfigurationError("pin_memory must be a boolean")
        if not isinstance(self.persistent_workers, bool):
            raise ConfigurationError("persistent_workers must be a boolean")

        if self.num_workers == 0:
            if self.prefetch_factor is not None:
                raise ConfigurationError("prefetch_factor requires num_workers > 0")
            if self.persistent_workers:
                raise ConfigurationError("persistent_workers requires num_workers > 0")
            return

        if self.prefetch_factor is None:
            object.__setattr__(self, "prefetch_factor", 2)
        elif isinstance(self.prefetch_factor, bool) or not isinstance(self.prefetch_factor, int):
            raise ConfigurationError("prefetch_factor must be an integer")
        elif self.prefetch_factor < 1:
            raise ConfigurationError("prefetch_factor must be at least 1")

    def to_dataloader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
        }

    def simplicity_key(self) -> tuple[int, int, int, int]:
        return (
            self.num_workers,
            int(self.pin_memory),
            int(self.persistent_workers),
            self.prefetch_factor or 0,
        )

    def describe(self) -> str:
        base = f"num_workers={self.num_workers}, pin_memory={self.pin_memory}"
        if self.num_workers == 0:
            return base
        return (
            f"{base}, prefetch_factor={self.prefetch_factor}, "
            f"persistent_workers={self.persistent_workers}"
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    batch_size: int
    requested_device: str
    device: str
    warmup_runs: int
    repetitions: int
    max_batches: int | None
    max_workers: int
    max_ram_gb: float | None
    dataset_size: int
    cpu_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "requested_device": self.requested_device,
            "device": self.device,
            "warmup_runs": self.warmup_runs,
            "repetitions": self.repetitions,
            "max_batches": self.max_batches,
            "max_workers": self.max_workers,
            "max_ram_gb": self.max_ram_gb,
            "dataset_size": self.dataset_size,
            "cpu_count": self.cpu_count,
        }
