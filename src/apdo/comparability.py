from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from apdo.config import COMPARABILITY_POLICIES
from apdo.errors import ConfigurationError, FingerprintError

if TYPE_CHECKING:
    from apdo.benchmark import ConfigMeasurement
    from apdo.metrics import CycleMetrics

COMPARABILITY_STATUSES = (
    "same_order",
    "same_sample_set",
    "different_samples",
    "unknown",
    "not_compared",
)
MAX_FINGERPRINT_DEPTH = 64
SampleFingerprintFn = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class BatchObservations:
    structures: tuple[str, ...]
    sample_counts: tuple[int, ...]
    fingerprints: tuple[str, ...] | None = None

    @property
    def batch_count(self) -> int:
        return len(self.structures)

    @property
    def fingerprints_available(self) -> bool:
        return self.fingerprints is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_count": self.batch_count,
            "sample_counts": list(self.sample_counts),
            "structure_sequence_digest": _sequence_digest(self.structures),
            "fingerprint_sequence_digest": (
                _sequence_digest(self.fingerprints) if self.fingerprints is not None else None
            ),
            "fingerprints_available": self.fingerprints_available,
        }


@dataclass(frozen=True, slots=True)
class ComparabilityResult:
    status: str
    structural_verified: bool
    sample_identity_verified: bool
    rounds_compared: int
    rounds_with_fingerprints: int
    structural_mismatch_rounds: int
    different_sample_rounds: int
    notes: tuple[str, ...] = ()

    @property
    def sufficient_for_strict(self) -> bool:
        return (
            self.rounds_compared > 0 and self.structural_verified and self.sample_identity_verified
        )

    @property
    def safe_for_confirmation(self) -> bool:
        return (
            self.structural_verified
            and self.different_sample_rounds == 0
            and self.status != "different_samples"
        )

    @property
    def has_observed_problem(self) -> bool:
        return self.rounds_compared > 0 and (
            self.structural_mismatch_rounds > 0
            or self.different_sample_rounds > 0
            or self.status == "different_samples"
        )

    @property
    def confidence(self) -> str:
        if self.status in ("same_order", "same_sample_set") and self.structural_verified:
            return "verified"
        if self.status == "unknown" and self.structural_verified:
            return "partial"
        if self.status in ("different_samples", "not_compared") or not self.structural_verified:
            return "failed"
        return "partial"

    @property
    def summary(self) -> str:
        if self.status == "not_compared":
            return "No paired batch observations were available for comparison."
        if self.status == "different_samples":
            return (
                f"Sample fingerprints differed in {self.different_sample_rounds} of "
                f"{self.rounds_compared} paired rounds; the compared configurations did "
                "not observe the same samples."
            )
        if not self.structural_verified:
            return (
                "Batch structural comparability was not verified for every paired round: "
                f"{self.structural_mismatch_rounds} structural mismatches across "
                f"{self.rounds_compared} compared rounds."
            )
        if self.status == "same_order":
            return (
                f"Batch structures and sample fingerprints matched in the same order in all "
                f"{self.rounds_compared} paired rounds."
            )
        if self.status == "same_sample_set":
            return (
                f"Batch structures and sample sets matched, with order differences, in all "
                f"{self.rounds_compared} paired rounds."
            )
        if self.rounds_with_fingerprints == 0:
            return (
                f"Batch structures matched in all {self.rounds_compared} paired rounds, but "
                "sample identity was not verified because no sample fingerprint function "
                "was supplied."
            )
        return (
            f"Batch structures matched in all {self.rounds_compared} paired rounds, but "
            "sample identity was only partially verified."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "structural_verified": self.structural_verified,
            "sample_identity_verified": self.sample_identity_verified,
            "rounds_compared": self.rounds_compared,
            "rounds_with_fingerprints": self.rounds_with_fingerprints,
            "structural_mismatch_rounds": self.structural_mismatch_rounds,
            "different_sample_rounds": self.different_sample_rounds,
            "sufficient_for_strict": self.sufficient_for_strict,
            "notes": list(self.notes),
            "summary": self.summary,
        }


def batch_structure(batch: Any) -> str:
    digest = hashlib.sha256()
    _update_structure(digest, batch, 0)
    return digest.hexdigest()


def batch_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    try:
        _update_fingerprint_value(digest, value, 0)
    except FingerprintError:
        raise
    except (TypeError, ValueError) as exc:
        raise FingerprintError(
            "sample_fingerprint_fn must return stable scalar values or supported "
            f"collections, got {type(value).__name__}: {exc}"
        ) from exc
    return digest.hexdigest()


def combine_cycles(
    cycles: Sequence[CycleMetrics],
    *,
    fingerprint_fn_present: bool,
) -> BatchObservations | None:
    observations = [
        cycle.batch_observations for cycle in cycles if cycle.batch_observations is not None
    ]
    if not observations:
        return None
    structures = tuple(structure for item in observations for structure in item.structures)
    sample_counts = tuple(count for item in observations for count in item.sample_counts)
    if fingerprint_fn_present and all(item.fingerprints is not None for item in observations):
        fingerprints = tuple(
            fingerprint for item in observations for fingerprint in (item.fingerprints or ())
        )
    else:
        fingerprints = None
    return BatchObservations(
        structures=structures,
        sample_counts=sample_counts,
        fingerprints=fingerprints,
    )


def compare_measurements(
    baseline: ConfigMeasurement,
    candidate: ConfigMeasurement,
) -> ComparabilityResult:
    baseline_rounds = {item.round_index: item for item in baseline.rounds}
    candidate_rounds = {item.round_index: item for item in candidate.rounds}
    shared = sorted(set(baseline_rounds) & set(candidate_rounds))
    if not shared:
        return ComparabilityResult(
            status="not_compared",
            structural_verified=False,
            sample_identity_verified=False,
            rounds_compared=0,
            rounds_with_fingerprints=0,
            structural_mismatch_rounds=0,
            different_sample_rounds=0,
            notes=("No matching measured round indexes were available.",),
        )

    structural_mismatches = 0
    observation_pairs = 0
    fingerprint_pairs = 0
    same_order = 0
    same_set = 0
    different = 0
    for round_index in shared:
        left = baseline_rounds[round_index].pass_metrics
        right = candidate_rounds[round_index].pass_metrics
        left_obs = combine_cycles(left.cycles, fingerprint_fn_present=True)
        right_obs = combine_cycles(right.cycles, fingerprint_fn_present=True)
        if left_obs is None or right_obs is None:
            continue
        observation_pairs += 1
        if (
            left_obs.structures != right_obs.structures
            or left_obs.sample_counts != right_obs.sample_counts
        ):
            structural_mismatches += 1
        if left_obs.fingerprints is None or right_obs.fingerprints is None:
            continue
        fingerprint_pairs += 1
        if left_obs.fingerprints == right_obs.fingerprints:
            same_order += 1
        elif Counter(left_obs.fingerprints) == Counter(right_obs.fingerprints):
            same_set += 1
        else:
            different += 1

    structural_verified = structural_mismatches == 0 and observation_pairs == len(shared)
    if different > 0:
        status = "different_samples"
    elif fingerprint_pairs == 0:
        status = "unknown"
    elif same_set > 0:
        status = "same_sample_set"
    elif same_order == fingerprint_pairs:
        status = "same_order"
    else:
        status = "unknown"

    sample_identity_verified = (
        structural_verified
        and fingerprint_pairs == len(shared)
        and status in ("same_order", "same_sample_set")
    )
    notes: list[str] = []
    if structural_mismatches:
        notes.append("Batch structure or sample counts differed in one or more paired rounds.")
    if observation_pairs < len(shared):
        notes.append(
            "One or more paired rounds had no batch observations, so structural "
            "equality was not verified for every round."
        )
    if fingerprint_pairs == 0:
        notes.append(
            "No sample fingerprints were available; structural equality alone cannot "
            "verify sample identity."
        )
    elif fingerprint_pairs < len(shared):
        notes.append(
            "Some paired rounds had no sample fingerprints, so identity was not verified "
            "for every round."
        )
    if different:
        notes.append("Fingerprint sequences differed in content in one or more rounds.")
    return ComparabilityResult(
        status=status,
        structural_verified=structural_verified,
        sample_identity_verified=sample_identity_verified,
        rounds_compared=len(shared),
        rounds_with_fingerprints=fingerprint_pairs,
        structural_mismatch_rounds=structural_mismatches,
        different_sample_rounds=different,
        notes=tuple(notes),
    )


def validate_comparability_policy(policy: str) -> None:
    if policy not in COMPARABILITY_POLICIES:
        raise ConfigurationError(f"comparability_policy must be one of {COMPARABILITY_POLICIES}")


def _sequence_digest(values: Sequence[str] | None) -> str | None:
    if values is None:
        return None
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _update_structure(digest: Any, value: Any, depth: int) -> None:
    if depth > MAX_FINGERPRINT_DEPTH:
        raise FingerprintError("batch structure exceeds the maximum nesting depth")
    if isinstance(value, torch.Tensor):
        digest.update(b"T")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        return
    if isinstance(value, Mapping):
        digest.update(b"M")
        keys = sorted(_structure_key(key) for key in value)
        for key in keys:
            digest.update(len(key).to_bytes(8, "big"))
            digest.update(key)
        for _, item in sorted(value.items(), key=lambda pair: _structure_key(pair[0])):
            _update_structure(digest, item, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"L")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_structure(digest, item, depth + 1)
        return
    digest.update(b"S")
    digest.update(type(value).__module__.encode("utf-8"))
    digest.update(b".")
    digest.update(type(value).__qualname__.encode("utf-8"))


def _structure_key(value: Any) -> bytes:
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}".encode(
        "utf-8",
        errors="backslashreplace",
    )


def _update_fingerprint_value(digest: Any, value: Any, depth: int) -> None:
    if depth > MAX_FINGERPRINT_DEPTH:
        raise FingerprintError("sample fingerprint exceeds the maximum nesting depth")
    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
        return
    if isinstance(value, int):
        encoded = str(value).encode("ascii")
        digest.update(b"I")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        return
    if isinstance(value, float):
        encoded = repr(value).encode("ascii")
        digest.update(b"F")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"U")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
        digest.update(b"Y")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        return
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise FingerprintError(
                "sample fingerprint tensors must be scalar; return a stable scalar "
                "value or a collection of scalar values"
            )
        digest.update(b"T")
        digest.update(str(value.dtype).encode("ascii"))
        scalar = value.detach().cpu().item()
        _update_fingerprint_value(digest, scalar, depth + 1)
        return
    if isinstance(value, Mapping):
        digest.update(b"M")
        entries = [(_structure_key(key), key, item) for key, item in value.items()]
        entries.sort(key=lambda entry: entry[0])
        for _, key, item in entries:
            _update_fingerprint_value(digest, key, depth + 1)
            _update_fingerprint_value(digest, item, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"L")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_fingerprint_value(digest, item, depth + 1)
        return
    raise FingerprintError(
        f"sample_fingerprint_fn returned an unsupported value of type {type(value).__name__}"
    )
