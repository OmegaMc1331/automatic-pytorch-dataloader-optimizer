from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apdo._version import __version__
from apdo.config import BENCHMARK_PROTOCOL_VERSION, CACHE_POLICIES, LoaderConfig
from apdo.errors import ConfigurationError

CACHE_SCHEMA_VERSION = 1
MAX_CACHE_ENTRIES = 128


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: str
    identity: str
    config: LoaderConfig
    recommended_config: LoaderConfig
    validation_status: str
    improvement_percent: float | None
    search_improvement_percent: float | None
    selection_metric: str
    benchmark_mode: str
    protocol_version: str
    created_at: str
    environment: dict[str, Any]
    source: dict[str, Any]
    evidence: dict[str, Any]
    apdo_version: str = __version__
    schema_version: int = CACHE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "apdo_version": self.apdo_version,
            "protocol_version": self.protocol_version,
            "key": self.key,
            "identity": self.identity,
            "config": self.config.to_dict(),
            "recommended_config": self.recommended_config.to_dict(),
            "validation_status": self.validation_status,
            "improvement_percent": self.improvement_percent,
            "search_improvement_percent": self.search_improvement_percent,
            "selection_metric": self.selection_metric,
            "benchmark_mode": self.benchmark_mode,
            "created_at": self.created_at,
            "environment": self.environment,
            "source": self.source,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CacheEntry:
        if not isinstance(payload, dict):
            raise ValueError("cache entry must be a JSON object")
        schema_version = payload.get("schema_version")
        if schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported cache schema version: {schema_version!r}")
        required = (
            "key",
            "identity",
            "config",
            "recommended_config",
            "validation_status",
            "selection_metric",
            "benchmark_mode",
            "protocol_version",
            "created_at",
            "environment",
            "source",
            "evidence",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"cache entry is missing fields: {', '.join(missing)}")
        return cls(
            key=_string(payload["key"], "key"),
            identity=_string(payload["identity"], "identity"),
            config=_config(payload["config"]),
            recommended_config=_config(payload["recommended_config"]),
            validation_status=_string(payload["validation_status"], "validation_status"),
            improvement_percent=_optional_float(
                payload.get("improvement_percent"), "improvement_percent"
            ),
            search_improvement_percent=_optional_float(
                payload.get("search_improvement_percent"), "search_improvement_percent"
            ),
            selection_metric=_string(payload["selection_metric"], "selection_metric"),
            benchmark_mode=_string(payload["benchmark_mode"], "benchmark_mode"),
            protocol_version=_string(payload["protocol_version"], "protocol_version"),
            created_at=_string(payload["created_at"], "created_at"),
            environment=_mapping(payload["environment"], "environment"),
            source=_mapping(payload["source"], "source"),
            evidence=_mapping(payload["evidence"], "evidence"),
            apdo_version=_string(payload.get("apdo_version", __version__), "apdo_version"),
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True)
class CacheLookup:
    hit: bool
    entry: CacheEntry | None
    reason: str


class ConfigCache:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = resolve_cache_directory(directory)

    def lookup(self, key: str, identity: str) -> CacheLookup:
        if not key or not identity:
            return CacheLookup(False, None, "cache key and identity are required")
        path = self._path_for(key)
        if not path.is_file():
            return CacheLookup(False, None, "no matching cache entry")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entry = CacheEntry.from_dict(payload)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return CacheLookup(False, None, f"cache entry could not be read: {exc}")
        if entry.key != key:
            return CacheLookup(False, None, "cache entry key does not match")
        if entry.identity != identity:
            return CacheLookup(False, None, "cache identity does not match")
        if entry.protocol_version != BENCHMARK_PROTOCOL_VERSION:
            return CacheLookup(False, None, "cache protocol version does not match")
        if entry.apdo_version != __version__:
            return CacheLookup(False, None, "cache package version does not match")
        return CacheLookup(True, entry, "cache entry matched")

    def store(self, entry: CacheEntry) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path_for(entry.key)
        temporary = target.with_suffix(target.suffix + ".tmp")
        payload = json.dumps(
            entry.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, target)
        self._prune()
        return target

    def list_entries(self) -> tuple[CacheEntry, ...]:
        if not self.directory.is_dir():
            return ()
        entries: list[CacheEntry] = []
        for path in sorted(self.directory.glob("apdo-*.json")):
            try:
                entry = CacheEntry.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            entries.append(entry)
        entries.sort(key=lambda item: item.created_at, reverse=True)
        return tuple(entries)

    def clear(self) -> int:
        if not self.directory.is_dir():
            return 0
        removed = 0
        for path in self.directory.glob("apdo-*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"apdo-{digest}.json"

    def _prune(self) -> None:
        entries = self.list_entries()
        for entry in entries[MAX_CACHE_ENTRIES:]:
            try:
                self._path_for(entry.key).unlink()
            except OSError:
                continue


def resolve_cache_directory(directory: str | Path | None = None) -> Path:
    if directory is not None:
        return Path(directory).expanduser()
    return Path.home() / ".apdo" / "cache"


def cache_identity(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_cache_entry(
    *,
    key: str,
    identity: str,
    config: LoaderConfig,
    recommended_config: LoaderConfig,
    validation_status: str,
    improvement_percent: float | None,
    search_improvement_percent: float | None,
    selection_metric: str,
    benchmark_mode: str,
    environment: dict[str, Any],
    source: dict[str, Any],
    evidence: dict[str, Any],
) -> CacheEntry:
    return CacheEntry(
        key=key,
        identity=identity,
        config=config,
        recommended_config=recommended_config,
        validation_status=validation_status,
        improvement_percent=improvement_percent,
        search_improvement_percent=search_improvement_percent,
        selection_metric=selection_metric,
        benchmark_mode=benchmark_mode,
        protocol_version=BENCHMARK_PROTOCOL_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        environment=environment,
        source=source,
        evidence=evidence,
    )


def _config(payload: Any) -> LoaderConfig:
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON object")
    return LoaderConfig(
        num_workers=_integer(payload.get("num_workers", 0), "num_workers"),
        pin_memory=_boolean(payload.get("pin_memory", False), "pin_memory"),
        prefetch_factor=(
            None
            if payload.get("prefetch_factor") is None
            else _integer(payload["prefetch_factor"], "prefetch_factor")
        ),
        persistent_workers=_boolean(payload.get("persistent_workers", False), "persistent_workers"),
    )


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null")
    return float(value)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def validate_cache_policy(policy: str) -> None:
    if policy not in CACHE_POLICIES:
        raise ConfigurationError(f"cache_policy must be one of {CACHE_POLICIES}")
