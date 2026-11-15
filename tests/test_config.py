from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from apdo import LoaderConfig
from apdo.errors import ConfigurationError


def test_default_config_is_valid() -> None:
    config = LoaderConfig()

    assert config.num_workers == 0
    assert config.pin_memory is False
    assert config.prefetch_factor is None
    assert config.persistent_workers is False


def test_worker_config_uses_pytorch_default_prefetch() -> None:
    config = LoaderConfig(num_workers=4)

    assert config.prefetch_factor == 2
    assert config.to_dataloader_kwargs() == {
        "num_workers": 4,
        "pin_memory": False,
        "prefetch_factor": 2,
        "persistent_workers": False,
    }


def test_single_process_kwargs_omit_worker_only_options() -> None:
    kwargs = LoaderConfig(pin_memory=True).to_dataloader_kwargs()

    assert kwargs == {"num_workers": 0, "pin_memory": True}


def test_describe_includes_relevant_fields() -> None:
    config = LoaderConfig(
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    description = config.describe()

    assert "num_workers=2" in description
    assert "pin_memory=True" in description
    assert "prefetch_factor=4" in description
    assert "persistent_workers=True" in description


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_workers": -1},
        {"num_workers": True},
        {"num_workers": 1.5},
        {"pin_memory": 1},
        {"persistent_workers": "yes"},
        {"prefetch_factor": 2},
        {"persistent_workers": True},
        {"num_workers": 2, "prefetch_factor": 0},
        {"num_workers": 2, "prefetch_factor": True},
        {"num_workers": 2, "prefetch_factor": 2.5},
    ],
)
def test_invalid_configs_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        LoaderConfig(**kwargs)


def test_config_is_immutable() -> None:
    config = LoaderConfig(num_workers=2)

    with pytest.raises(FrozenInstanceError):
        config.num_workers = 4


def test_simplicity_key_prefers_fewer_workers() -> None:
    baseline = LoaderConfig()
    simple = LoaderConfig(num_workers=1)
    complex_config = LoaderConfig(
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    assert baseline.simplicity_key() < simple.simplicity_key()
    assert simple.simplicity_key() < complex_config.simplicity_key()
