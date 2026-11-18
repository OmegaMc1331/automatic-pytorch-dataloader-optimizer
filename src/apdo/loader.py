from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)

from apdo.config import LoaderConfig
from apdo.errors import ConfigurationError, DatasetError
from apdo.isolation import ReproducibilityInfo, clone_generator

LoaderFactory = Callable[[LoaderConfig], DataLoader]


def map_dataset_length(dataset: Any) -> int:
    if isinstance(dataset, IterableDataset):
        raise DatasetError(
            "IterableDataset is not supported in v0.4.0; use a map-style dataset "
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

    def build(self, config: LoaderConfig, *, isolate: bool = False) -> DataLoader:
        sampler = self.sampler
        batch_sampler = self.batch_sampler
        generator = self.generator
        if isolate:
            if generator is not None:
                generator = clone_generator(generator)
            if batch_sampler is not None:
                isolated_batch_sampler = _isolated_batch_sampler(batch_sampler)
                if isolated_batch_sampler is not None:
                    batch_sampler = isolated_batch_sampler
            else:
                isolated_sampler = _isolated_sampler(sampler)
                if isolated_sampler is not None:
                    sampler = isolated_sampler

        kwargs: dict[str, Any] = {
            "dataset": self.dataset,
            "collate_fn": self.collate_fn,
            "timeout": self.timeout,
            "worker_init_fn": self.worker_init_fn,
            "generator": generator,
            "in_order": self.in_order,
        }
        if batch_sampler is not None:
            kwargs["batch_sampler"] = batch_sampler
        else:
            kwargs["batch_size"] = self.batch_size
            kwargs["sampler"] = sampler
            if self.batch_size is not None:
                kwargs["drop_last"] = self.drop_last
        kwargs.update(config.to_dataloader_kwargs())
        if config.num_workers > 0 and self.multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = self.multiprocessing_context
        if config.pin_memory and self.pin_memory_device:
            kwargs["pin_memory_device"] = self.pin_memory_device
        return DataLoader(**kwargs)

    def count_samples(self, batch: Any) -> int:
        return count_batch_samples(batch, unbatched=self.unbatched, fn=self.sample_count_fn)

    def isolation_info(self) -> ReproducibilityInfo:
        risks: list[str] = []
        managed = ["global_rng"]
        severe = False

        if self.generator is not None:
            managed.append("loader_generator")
        if self.batch_sampler is not None:
            if _isolated_batch_sampler(self.batch_sampler, clone=False) is None:
                severe = True
                risks.append(
                    "The custom batch sampler cannot be reconstructed safely and is reused "
                    "by reference, so its mutable state is shared with the original loader."
                )
            else:
                managed.extend(("sampler", "batch_sampler"))
        elif _isolated_sampler(self.sampler, clone=False) is None:
            severe = True
            risks.append(
                "The custom sampler cannot be reconstructed safely and is reused by "
                "reference, so its mutable state is shared with the original loader."
            )
        else:
            managed.append("sampler")

        dataset_risk, dataset_severe = _dataset_state_risk(self.dataset)
        if dataset_risk is not None:
            risks.append(dataset_risk)
            severe = severe or dataset_severe
        if self.worker_init_fn is not None:
            risks.append(
                "A custom worker_init_fn is reused by reference. APDO cannot inspect or "
                "reset state that it mutates inside worker processes."
            )

        if severe:
            status = "not_guaranteed"
        elif risks:
            status = "partial"
        else:
            status = "verified"
        reason = _isolation_reason(status, managed, risks)
        return ReproducibilityInfo(
            status=status,
            managed=tuple(managed),
            risks=tuple(risks),
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class LoaderSource:
    factory: LoaderFactory | None
    template: LoaderTemplate | None
    expected_batches: int
    unbatched: bool
    batch_size: int | None
    dataset_size: int
    sample_count_fn: Callable[[Any], int] | None
    isolation: ReproducibilityInfo
    input_kind: str

    @classmethod
    def from_template(cls, template: LoaderTemplate, input_kind: str) -> LoaderSource:
        return cls(
            factory=None,
            template=template,
            expected_batches=template.expected_batches(),
            unbatched=template.unbatched,
            batch_size=template.batch_size,
            dataset_size=map_dataset_length(template.dataset),
            sample_count_fn=template.sample_count_fn,
            isolation=template.isolation_info(),
            input_kind=input_kind,
        )

    @classmethod
    def from_factory(
        cls,
        factory: Callable[[LoaderConfig], DataLoader],
        *,
        sample_count_fn: Callable[[Any], int] | None,
    ) -> LoaderSource:
        if not callable(factory):
            raise ConfigurationError("dataloader_factory must be callable")
        try:
            loader = factory(LoaderConfig())
        except Exception as exc:
            raise ConfigurationError(
                f"dataloader_factory failed for the initial configuration: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(loader, DataLoader):
            raise ConfigurationError("dataloader_factory must return a torch DataLoader")
        if type(loader) is not DataLoader:
            raise ConfigurationError(
                "dataloader_factory must return a standard torch.utils.data.DataLoader"
            )
        actual = initial_loader_config(loader)
        if actual != LoaderConfig():
            raise ConfigurationError(
                "dataloader_factory must honor the requested LoaderConfig; the initial "
                f"request was {LoaderConfig().describe()} but the returned loader uses "
                f"{actual.describe()}"
            )
        template = LoaderTemplate.from_dataloader(loader, sample_count_fn=sample_count_fn)
        info = ReproducibilityInfo(
            status="factory_managed",
            managed=("loader", "sampler", "generator", "training_environment", "global_rng"),
            risks=(),
            reason=(
                "The explicit factory is responsible for returning independent loader and "
                "training objects. APDO manages global RNG isolation around each trial but "
                "does not inspect objects created inside the factory."
            ),
        )
        return cls(
            factory=factory,
            template=template,
            expected_batches=template.expected_batches(),
            unbatched=template.unbatched,
            batch_size=template.batch_size,
            dataset_size=map_dataset_length(template.dataset),
            sample_count_fn=sample_count_fn,
            isolation=info,
            input_kind="dataloader_factory",
        )

    def build(self, config: LoaderConfig, *, isolate: bool = True) -> DataLoader:
        if self.factory is None:
            if self.template is None:
                raise ConfigurationError("this loader source has no factory")
            return self.template.build(config, isolate=isolate)
        loader = self.factory(config)
        if type(loader) is not DataLoader:
            raise ConfigurationError(
                "dataloader_factory must return a standard torch.utils.data.DataLoader"
            )
        actual = initial_loader_config(loader)
        if actual != config:
            raise ConfigurationError(
                f"dataloader_factory returned {actual.describe()} for requested {config.describe()}"
            )
        return loader

    def count_samples(self, batch: Any) -> int:
        return count_batch_samples(batch, unbatched=self.unbatched, fn=self.sample_count_fn)

    def compatibility_warnings(self) -> list[str]:
        if self.factory is not None:
            return [
                "The loader factory contract requires independent mutable loader state for "
                "every trial; APDO cannot verify objects created inside the factory."
            ]
        if self.template is None:
            return []
        return compatibility_warnings(self.template)


def count_batch_samples(
    batch: Any,
    *,
    unbatched: bool,
    fn: Callable[[Any], int] | None,
) -> int:
    if unbatched:
        return 1
    if fn is not None:
        return _validated_count(fn(batch))
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


def compatibility_warnings(template: LoaderTemplate) -> list[str]:
    info = template.isolation_info()
    return list(info.risks)


def _isolation_reason(
    status: str,
    managed: list[str],
    risks: list[str],
) -> str:
    if status == "verified":
        return (
            "Supported loader generator, sampler, batch sampler, and global RNG state are "
            "isolated for each trial. Undetected mutation inside arbitrary dataset or "
            "transform code cannot be excluded."
        )
    if status == "partial":
        return (
            "Supported loader state is isolated, but detected dataset, transform, or "
            "worker initialization behavior cannot be fully controlled: " + " ".join(risks)
        )
    return (
        "One or more loader-owned objects cannot be reconstructed independently. APDO "
        "does not claim full reproducibility for this pipeline: " + " ".join(risks)
    )


def _dataset_state_risk(dataset: Any) -> tuple[str | None, bool]:
    suspects: list[Any] = [dataset]
    for name in ("transform", "transforms", "generator"):
        if hasattr(dataset, name):
            value = getattr(dataset, name)
            suspects.append(value)
            nested = getattr(value, "transforms", None)
            if isinstance(nested, (tuple, list)):
                suspects.extend(nested)
    for value in suspects:
        if isinstance(value, (torch.Generator, random.Random)):
            return (
                "The dataset or its transform graph contains an independent random number "
                "generator that APDO cannot clone. Use dataloader_factory for training mode.",
                True,
            )
        class_name = type(value).__name__.lower()
        module = type(value).__module__.lower()
        if module.startswith("numpy.random") and "generator" in class_name:
            return (
                "The dataset or its transform graph contains a NumPy random generator "
                "that APDO cannot clone. Use dataloader_factory for training mode.",
                True,
            )
        if "random" in class_name or "augment" in class_name:
            return (
                "The dataset or a transform appears to use randomness. Its internal state "
                "is not isolated between trials, so comparability may be reduced.",
                False,
            )
    return None, False


def _isolated_sampler(sampler: Any, *, clone: bool = True) -> Any:
    generator = getattr(sampler, "generator", None)
    cloned_generator = clone_generator(generator) if clone and generator is not None else generator
    if type(sampler) is SequentialSampler:
        return SequentialSampler(sampler.data_source)
    if type(sampler) is RandomSampler:
        if not clone:
            return sampler
        return RandomSampler(
            sampler.data_source,
            replacement=sampler.replacement,
            num_samples=sampler.num_samples,
            generator=cloned_generator,
        )
    if type(sampler) is SubsetRandomSampler:
        if not clone:
            return sampler
        return SubsetRandomSampler(sampler.indices, generator=cloned_generator)
    if type(sampler) is WeightedRandomSampler:
        if not clone:
            return sampler
        return WeightedRandomSampler(
            sampler.weights,
            sampler.num_samples,
            sampler.replacement,
            generator=cloned_generator,
        )
    return None


def _isolated_batch_sampler(batch_sampler: Any, *, clone: bool = True) -> Any:
    if type(batch_sampler) is not BatchSampler:
        return None
    sampler = _isolated_sampler(batch_sampler.sampler, clone=clone)
    if sampler is None:
        return None
    if not clone:
        return batch_sampler
    return BatchSampler(
        sampler,
        batch_size=batch_sampler.batch_size,
        drop_last=batch_sampler.drop_last,
    )


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
