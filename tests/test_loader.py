from __future__ import annotations

import pytest
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    IterableDataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    TensorDataset,
)

from apdo import LoaderConfig
from apdo.errors import ConfigurationError, DatasetError
from apdo.loader import (
    LoaderSource,
    LoaderTemplate,
    compatibility_warnings,
    initial_loader_config,
    map_dataset_length,
)


class CustomLoader(DataLoader):
    pass


class NoLengthSampler(SequentialSampler):
    def __len__(self) -> int:
        raise TypeError("length is unavailable")


class NoLengthBatchSampler(BatchSampler):
    def __len__(self) -> int:
        raise TypeError("length is unavailable")


class RandomTransform:
    pass


class TransformDataset(TensorDataset):
    def __init__(self, size: int) -> None:
        super().__init__(torch.arange(size).reshape(-1, 1), torch.arange(size))
        self.transform = RandomTransform()


class StreamingDataset(IterableDataset):
    def __iter__(self):
        yield torch.zeros(1)


class UnindexableDataset:
    def __len__(self) -> int:
        return 4


def make_dataset(size: int = 10) -> TensorDataset:
    return TensorDataset(torch.arange(size).reshape(-1, 1), torch.arange(size))


def test_initial_config_reads_worker_settings() -> None:
    loader = DataLoader(make_dataset(), batch_size=2, num_workers=1)

    config = initial_loader_config(loader)

    assert config == LoaderConfig(num_workers=1, pin_memory=False, prefetch_factor=2)


def test_loader_template_preserves_standard_loader_options() -> None:
    dataset = make_dataset()
    sampler = SequentialSampler(dataset)
    generator = torch.Generator().manual_seed(5)
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        drop_last=True,
        timeout=2,
        worker_init_fn=None,
        generator=generator,
        in_order=False,
        num_workers=0,
    )
    template = LoaderTemplate.from_dataloader(loader)

    built = template.build(LoaderConfig(num_workers=0, pin_memory=False))

    assert built.dataset is dataset
    assert built.batch_size == 4
    assert built.sampler is sampler
    assert built.drop_last is True
    assert built.timeout == 2
    assert built.generator is generator
    assert built.in_order is False
    assert built.collate_fn is loader.collate_fn
    assert template.expected_batches() == 2
    assert loader.batch_size == 4
    assert loader.num_workers == 0


def test_loader_template_preserves_custom_batch_sampler() -> None:
    dataset = make_dataset()
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=3, drop_last=False)
    loader = DataLoader(dataset, batch_sampler=batch_sampler)
    template = LoaderTemplate.from_dataloader(loader)

    built = template.build(LoaderConfig(num_workers=0))

    assert template.batch_size is None
    assert template.expected_batches() == 4
    assert built.batch_sampler is batch_sampler
    assert built.batch_size is None


def test_unbatched_loader_is_supported() -> None:
    loader = DataLoader(make_dataset(4), batch_size=None)
    template = LoaderTemplate.from_dataloader(loader)

    assert template.unbatched
    assert template.expected_batches() == 4
    assert template.count_samples(torch.zeros(3)) == 1
    assert template.build(LoaderConfig()).batch_size is None


def test_loader_subclasses_are_rejected_explicitly() -> None:
    loader = CustomLoader(make_dataset(), batch_size=2)

    with pytest.raises(ConfigurationError, match="subclasses"):
        LoaderTemplate.from_dataloader(loader)


def test_unsupported_datasets_are_rejected() -> None:
    with pytest.raises(DatasetError, match="IterableDataset"):
        map_dataset_length(StreamingDataset())
    with pytest.raises(DatasetError, match="__getitem__"):
        map_dataset_length(UnindexableDataset())
    with pytest.raises(DatasetError, match="empty"):
        map_dataset_length(TensorDataset(torch.empty(0, 1)))
    with pytest.raises(DatasetError, match="IterableDataset"):
        LoaderTemplate.from_dataloader(DataLoader(StreamingDataset()))


def test_sampler_length_is_required() -> None:
    dataset = make_dataset()
    loader = DataLoader(dataset, batch_size=2, sampler=NoLengthSampler(dataset))
    template = LoaderTemplate.from_dataloader(loader)

    with pytest.raises(ConfigurationError, match="sampler"):
        template.expected_batches()

    batch_sampler = NoLengthBatchSampler(SequentialSampler(dataset), batch_size=2, drop_last=False)
    template = LoaderTemplate.from_dataloader(DataLoader(dataset, batch_sampler=batch_sampler))

    with pytest.raises(ConfigurationError, match="batch_sampler"):
        template.expected_batches()


def test_sample_counts_are_inferred_from_batch_structure() -> None:
    template = LoaderTemplate.from_dataloader(DataLoader(make_dataset(), batch_size=4))

    assert template.count_samples((torch.zeros(4, 2), torch.zeros(4))) == 4
    assert template.count_samples({"features": torch.zeros(3, 5)}) == 3
    assert template.count_samples([torch.zeros(2), {"label": torch.zeros(2)}]) == 2
    assert template.count_samples(torch.tensor(1.0)) == 1


def test_sample_count_ambiguities_fail_cleanly() -> None:
    template = LoaderTemplate.from_dataloader(DataLoader(make_dataset(), batch_size=4))

    with pytest.raises(DatasetError, match="inconsistent"):
        template.count_samples((torch.zeros(4), torch.zeros(3)))
    with pytest.raises(DatasetError, match="could not determine"):
        template.count_samples({"text": "not a tensor"})


def test_custom_sample_count_function_is_validated() -> None:
    calls: list[object] = []

    def count(batch: object) -> int:
        calls.append(batch)
        return 7

    template = LoaderTemplate.from_dataloader(
        DataLoader(make_dataset(), batch_size=4),
        sample_count_fn=count,
    )

    assert template.count_samples("custom batch") == 7
    assert calls == ["custom batch"]

    for value in (True, 0, -1, 1.5):
        bad = LoaderTemplate.from_dataloader(
            DataLoader(make_dataset(), batch_size=4),
            sample_count_fn=lambda batch, value=value: value,
        )
        with pytest.raises(DatasetError):
            bad.count_samples("custom batch")


def test_compatibility_warnings_describe_shared_state() -> None:
    dataset = TransformDataset(8)
    generator = torch.Generator().manual_seed(1)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=RandomSampler(dataset, generator=generator),
        generator=generator,
        worker_init_fn=lambda worker_id: None,
    )

    template = LoaderTemplate.from_dataloader(loader)
    warnings = compatibility_warnings(template)
    info = template.isolation_info()

    assert info.status == "partial"
    assert any("randomness" in warning for warning in warnings)
    assert any("worker_init_fn" in warning for warning in warnings)


def test_custom_batch_sampler_warns_about_shared_state() -> None:
    dataset = make_dataset()
    batch_sampler = BatchSampler(SequentialSampler(dataset), batch_size=2, drop_last=False)
    template = LoaderTemplate.from_dataloader(DataLoader(dataset, batch_sampler=batch_sampler))

    warnings = compatibility_warnings(template)

    assert warnings == []
    assert template.isolation_info().status == "verified"
    assert "sampler" in template.isolation_info().managed
    assert "batch_sampler" in template.isolation_info().managed


class StatefulSampler(Sampler[int]):
    def __init__(self, size: int) -> None:
        self.size = size
        self.offset = 0

    def __iter__(self):
        indices = list(range(self.size))
        self.offset += 1
        return iter(indices[self.offset % 2 :] + indices[: self.offset % 2])

    def __len__(self) -> int:
        return self.size


def test_isolated_build_clones_generator_and_preserves_original() -> None:
    dataset = make_dataset()
    generator = torch.Generator().manual_seed(9)
    loader = DataLoader(dataset, batch_size=2, generator=generator)
    template = LoaderTemplate.from_dataloader(loader)
    state_before = generator.get_state().clone()

    built = template.build(LoaderConfig(), isolate=True)
    assert built.generator is not None
    assert built.generator is not generator
    assert torch.equal(built.generator.get_state(), state_before)

    next(iter(built))

    assert torch.equal(generator.get_state(), state_before)


def test_isolated_build_falls_back_to_custom_sampler() -> None:
    dataset = make_dataset()
    sampler = StatefulSampler(len(dataset))
    loader = DataLoader(dataset, batch_size=2, sampler=sampler)
    template = LoaderTemplate.from_dataloader(loader)

    built = template.build(LoaderConfig(), isolate=True)
    info = template.isolation_info()

    assert built.sampler is sampler
    assert info.status == "not_guaranteed"
    assert not info.training_allowed


def test_loader_source_from_factory_enforces_contract() -> None:
    def factory(config: LoaderConfig) -> DataLoader:
        return DataLoader(make_dataset(), batch_size=2, **config.to_dataloader_kwargs())

    source = LoaderSource.from_factory(factory, sample_count_fn=None)

    assert source.input_kind == "dataloader_factory"
    assert source.isolation.status == "factory_managed"
    assert source.isolation.training_allowed
    assert source.build(LoaderConfig()).batch_size == 2

    def invalid(config: LoaderConfig) -> DataLoader:
        return DataLoader(make_dataset(), batch_size=2, num_workers=1)

    with pytest.raises(ConfigurationError, match="honor"):
        LoaderSource.from_factory(invalid, sample_count_fn=None)
