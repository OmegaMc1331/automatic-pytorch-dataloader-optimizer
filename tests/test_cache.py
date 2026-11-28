from __future__ import annotations

import json
from pathlib import Path

from apdo.cache import ConfigCache, build_cache_entry, cache_identity
from apdo.config import BENCHMARK_PROTOCOL_VERSION, LoaderConfig


def make_entry():
    return build_cache_entry(
        key="workload-key",
        identity=cache_identity({"dataset": "synthetic", "version": 1}),
        config=LoaderConfig(),
        recommended_config=LoaderConfig(num_workers=1, prefetch_factor=2),
        validation_status="confirmed",
        improvement_percent=20.0,
        search_improvement_percent=25.0,
        selection_metric="sustained",
        benchmark_mode="loader",
        environment={"torch_version": "2.14.0"},
        source={"dataset_size": 64},
        evidence={"valid_pairs": 10},
    )


def test_cache_round_trip_and_identity_check(tmp_path: Path) -> None:
    cache = ConfigCache(tmp_path)
    entry = make_entry()

    path = cache.store(entry)
    assert path.is_file()
    assert entry.protocol_version == BENCHMARK_PROTOCOL_VERSION

    lookup = cache.lookup(entry.key, entry.identity)
    assert lookup.hit
    assert lookup.entry is not None
    assert lookup.entry.recommended_config == entry.recommended_config

    miss = cache.lookup(entry.key, "different-identity")
    assert not miss.hit
    assert "identity" in miss.reason


def test_cache_rejects_stale_protocol(tmp_path: Path) -> None:
    cache = ConfigCache(tmp_path)
    entry = make_entry()
    path = cache.store(entry)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol_version"] = "4.1"
    path.write_text(json.dumps(payload), encoding="utf-8")

    lookup = cache.lookup(entry.key, entry.identity)
    assert not lookup.hit
    assert "protocol" in lookup.reason


def test_cache_tolerates_corrupt_entries(tmp_path: Path) -> None:
    cache = ConfigCache(tmp_path)
    entry = make_entry()
    path = cache.store(entry)
    path.write_text("{not valid json", encoding="utf-8")

    lookup = cache.lookup(entry.key, entry.identity)
    assert not lookup.hit
    assert cache.list_entries() == ()


def test_cache_clear_only_removes_apdo_entries(tmp_path: Path) -> None:
    cache = ConfigCache(tmp_path)
    entry = make_entry()
    cache.store(entry)
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    assert cache.clear() == 1
    assert unrelated.is_file()
    assert cache.list_entries() == ()
