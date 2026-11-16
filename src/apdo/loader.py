from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler

from apdo.config import LoaderConfig
from apdo.errors import ConfigurationError, DatasetError


def map_dataset_length(dataset: Any) -> int:
    if isinstance(dataset, IterableDataset):
        raise DatasetError(
            "IterableDataset is not supported in v0.2.0; use a map-style dataset "
            "with __len__ and __getitem__"
        )
    if not hasattr(dataset, "__getitem__"):
        raise DatasetError("dataset must implement __getitem__")
    try:
        length = len(dataset)
    except Exception as exc:
        raise DatasetError(f"could not determine dataset length: {exc}") from exc
    if isinstance(length, bool) or not isinstance(length, int):
        raise DatasetError("dataset length must be an integer")
    if length == 0:
        raise DatasetError("dataset is empty")
    return length


def initial_loader_config(loader: DataLoader) -> LoaderConfig:
    return LoaderConfig(
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
    )


@dataclass(frozen=True, slots=True)
class LoaderTemplate:
    dataset: Dataset
    batch_size: int | None
    sampler: Sampler | Any
    batch_sampler: Any
    collate_fn: Callable[..., Any]
    drop_last: bool
    timeout: float
    worker_init_fn: Callable[..., Any] | None
    generator: torch.Generator | None
    multiprocessing_context: Any
    in_order: bool
    pin_memory_device: str
    sample_count_fn: Callable[[Any], int] | None = None

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        batch_size: int,
        sample_count_fn: Callable[[Any], int] | None = None,
    ) -> LoaderTemplate:
        map_dataset_length(dataset)
        loader = DataLoader(dataset, batch_size=batch_size)
        return cls.from_dataloader(loader, sample_count_fn=sample_count_fn)

    @classmethod
    def from_dataloader(
        cls,
        loader: DataLoader,
        *,
        sample_count_fn: Callable[[Any], int] | None = None,
    ) -> LoaderTemplate:
        if not isinstance(loader, DataLoader):
            raise ConfigurationError("dataloader must be a torch.utils.data.DataLoader")
        if type(loader) is not DataLoader:
            raise ConfigurationError(
                "DataLoader subclasses are not supported because APDO cannot verify "
                "that their behavior is preserved; construct a standard DataLoader"
            )
        dataset = loader.dataset
        if dataset is None:
            raise DatasetError("dataloader has no dataset")
        map_dataset_length(dataset)
        if loader.batch_size is not None and loader.batch_size < 1:
            raise ConfigurationError("dataloader batch_size must be positive")
        if sample_count_fn is not None and not callable(sample_count_fn):
            raise ConfigurationError("sample_count_fn must be callable or None")
        batch_sampler = loader.batch_sampler if loader.batch_size is None else None
        return cls(
            dataset=dataset,
            batch_size=loader.batch_size,
            sampler=loader.sampler,
            batch_sampler=batch_sampler,
            collate_fn=loader.collate_fn,
            drop_last=loader.drop_last,
            timeout=loader.timeout,
            worker_init_fn=loader.worker_init_fn,
            generator=loader.generator,
            multiprocessing_context=loader.multiprocessing_context,
            in_order=getattr(loader, "in_order", True),
            pin_memory_device=getattr(loader, "pin_memory_device", ""),
            sample_count_fn=sample_count_fn,
        )

    @property
    def unbatched(self) -> bool:
        return self.batch_size is None and self.batch_sampler is None

    def expected_batches(self) -> int:
        if self.batch_sampler is not None:
            batches = _sampler_length(self.batch_sampler, "batch_sampler")
        else:
            samples = _sampler_length(self.sampler, "sampler")
            if self.batch_size is None:
                batches = samples
            elif self.drop_last:
                batches = samples // self.batch_size
            else:
                batches = (samples + self.batch_size - 1) // self.batch_size
        if batches < 1:
            raise DatasetError("sampler produces no batches")
        return batches

    def build(self, config: LoaderConfig) -> DataLoader:
        kwargs: dict[str, Any] = {
            "dataset": self.dataset,
            "collate_fn": self.collate_fn,
            "timeout": self.timeout,
            "worker_init_fn": self.worker_init_fn,
            "generator": self.generator,
            "in_order": self.in_order,
        }
        if self.batch_sampler is not None:
            kwargs["batch_sampler"] = self.batch_sampler
        else:
            kwargs["batch_size"] = self.batch_size
            kwargs["sampler"] = self.sampler
            if self.batch_size is not None:
                kwargs["drop_last"] = self.drop_last
        kwargs.update(config.to_dataloader_kwargs())
        if config.num_workers > 0 and self.multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = self.multiprocessing_context
        if config.pin_memory and self.pin_memory_device:
            kwargs["pin_memory_device"] = self.pin_memory_device
        return DataLoader(**kwargs)

    def count_samples(self, batch: Any) -> int:
        if self.unbatched:
            return 1
        if self.sample_count_fn is not None:
            return _validated_count(self.sample_count_fn(batch))
        counts = _leading_dimensions(batch)
        if not counts:
            raise DatasetError(
                "could not determine the number of samples in a batch; return tensors "
                "from collate_fn or pass sample_count_fn to optimize()"
            )
        if len(set(counts)) != 1:
            raise DatasetError(
                "batch fields have inconsistent leading dimensions "
                f"{sorted(set(counts))}; pass sample_count_fn to optimize()"
            )
        return counts[0]


def _sampler_length(sampler: Any, name: str) -> int:
    try:
        length = len(sampler)
    except (TypeError, AttributeError) as exc:
        raise ConfigurationError(
            f"the loader {name} must define __len__ so APDO can plan a finite benchmark"
        ) from exc
    except Exception as exc:
        raise ConfigurationError(f"could not determine the loader {name} length: {exc}") from exc
    if isinstance(length, bool) or not isinstance(length, int):
        raise ConfigurationError(f"the loader {name} length must be an integer")
    return length


def _validated_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetError("sample_count_fn must return an integer")
    if value < 1:
        raise DatasetError("sample_count_fn must return a positive integer")
    return value


def _leading_dimensions(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(value.shape[0]) if value.ndim > 0 else 1]
    if isinstance(value, Mapping):
        dimensions: list[int] = []
        for item in value.values():
            dimensions.extend(_leading_dimensions(item))
        return dimensions
    if isinstance(value, (tuple, list)):
        dimensions = []
        for item in value:
            dimensions.extend(_leading_dimensions(item))
        return dimensions
    return []


def compatibility_warnings(template: LoaderTemplate) -> list[str]:
    reasons: list[str] = []
    if template.batch_sampler is not None:
        reasons.append(
            "A custom batch_sampler is reused by reference. Its mutable state, sampler, "
            "and generator are shared with every benchmarked loader and are not reset."
        )
    if template.generator is not None:
        reasons.append(
            "The loader generator object is reused by reference. Iterating benchmarked "
            "loaders advances its state, including the state visible to the original loader."
        )
    if _is_random_sampler(template.sampler):
        reasons.append(
            "The loader uses a random sampler. APDO does not reset sampler, dataset, or "
            "global RNG state, so rounds can load different samples."
        )
    dataset_reason = _dataset_state_reason(template.dataset)
    if dataset_reason is not None:
        reasons.append(dataset_reason)
    if template.worker_init_fn is not None:
        reasons.append(
            "A custom worker_init_fn is reused by reference. APDO does not reset any "
            "state that it mutates."
        )
    return reasons


def _is_random_sampler(sampler: Any) -> bool:
    name = type(sampler).__name__.lower()
    return "random" in name or "shuffle" in name


def _dataset_state_reason(dataset: Any) -> str | None:
    suspects: list[Any] = [dataset]
    for name in ("transform", "transforms"):
        if hasattr(dataset, name):
            value = getattr(dataset, name)
            suspects.append(value)
            nested = getattr(value, "transforms", None)
            if isinstance(nested, (tuple, list)):
                suspects.extend(nested)
    for value in suspects:
        if isinstance(value, (torch.Generator, random.Random)):
            return (
                "The dataset or its transform graph contains a random number generator. "
                "APDO does not reset dataset or transform RNG state between rounds, so "
                "comparability may be reduced."
            )
        class_name = type(value).__name__.lower()
        if "random" in class_name or "augment" in class_name:
            return (
                "The dataset or a transform appears to use randomness. APDO does not "
                "reset dataset or transform RNG state between rounds, so comparability "
                "may be reduced."
            )
    return None
