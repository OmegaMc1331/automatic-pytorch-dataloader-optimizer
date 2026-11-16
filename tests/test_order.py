from __future__ import annotations

from apdo import LoaderConfig
from apdo.order import balanced_orders, derive_seed


def test_balanced_orders_cover_each_position_once() -> None:
    configs = (LoaderConfig(), LoaderConfig(1), LoaderConfig(2))

    orders = balanced_orders(configs, 3, seed=123, phase="search")

    assert len(orders) == 3
    assert all(order.phase == "search" for order in orders)
    assert all(order.measured for order in orders)
    assert all(set(order.configs) == set(configs) for order in orders)
    for position in range(3):
        assert {order.configs[position] for order in orders} == set(configs)


def test_balanced_orders_are_seeded_and_record_round_numbers() -> None:
    configs = (LoaderConfig(), LoaderConfig(1), LoaderConfig(2), LoaderConfig(4))

    first = balanced_orders(configs, 5, seed=7, phase="search")
    second = balanced_orders(configs, 5, seed=7, phase="search")
    changed = balanced_orders(configs, 5, seed=8, phase="search")

    assert first == second
    assert first != changed
    assert [order.round_index for order in first] == [0, 1, 2, 3, 4]
    assert all(set(order.configs) == set(configs) for order in first)


def test_balanced_orders_support_empty_and_warmup_requests() -> None:
    assert balanced_orders((), 3, seed=1, phase="search") == ()
    assert balanced_orders((LoaderConfig(),), 0, seed=1, phase="search") == ()

    orders = balanced_orders(
        (LoaderConfig(), LoaderConfig(1)),
        2,
        seed=1,
        phase="search",
        measured=False,
    )

    assert len(orders) == 2
    assert all(not order.measured for order in orders)
    assert orders[0].to_dict()["phase"] == "search"


def test_derive_seed_is_stable_and_part_sensitive() -> None:
    assert derive_seed(42, "a", "b") == derive_seed(42, "a", "b")
    assert derive_seed(42, "a", "b") != derive_seed(42, "b", "a")
    assert derive_seed(42, "a") != derive_seed(43, "a")
