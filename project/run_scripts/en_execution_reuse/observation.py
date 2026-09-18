"""Exact-identity, complete-only endpoint observations, detached CPU FP32.

Stores only final-normalized hidden and caller-computed score/statistic records;
it never runs a suffix/head, changes a head shape, or retains logits/KV/graphs.
Producers must preserve the legacy guard target-union and invariant 16-position
head schedules. Physical validation must bypass this cache entirely.

Callers build coverage from all protected rows, all current caches and their
full valid-position vectors, then put hidden/current/past rows and ``seal``
only AFTER the endpoint readonly scope exits successfully. Reads inside a
readonly phase avoid per-sequence endpoint byte hashing. Reads outside a phase
perform a full endpoint validation, and are intended for phase-level rows.
This enforces declared coverage, not the truth of a caller's input manifest:
the integration must derive manifests from actual pack/mask/position bytes.
"""
from dataclasses import asdict, dataclass
import json
import math
from typing import Mapping

import torch

from .endpoint_session import (EndpointHandle, EndpointIdentity, EndpointSession,
                               EndpointSessionError, identity_sha256,
                               tensor_sha256)


class ObservationError(EndpointSessionError):
    pass


def _json(value):
    # JSON conversion both rejects tensors/graphs and breaks all source aliases.
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _cache_id(value):
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError("OBSERVATION_STRING_OR_INT_CACHE_ID_REQUIRED")
    # Type prefix avoids collisions between integer 1 and string '1'.
    return f"{type(value).__name__}:{value}"


def _rows(rows):
    records = []
    seen = set()
    for row in rows:
        row = json.loads(_json(row))
        if not isinstance(row, dict):
            raise ValueError("OBSERVATION_ROW_OBJECT_REQUIRED")
        required = ("sequence_id", "cache", "positions", "labels")
        if any(key not in row for key in required):
            raise ValueError("OBSERVATION_ROW_ID_CACHE_POSITION_LABEL_REQUIRED")
        sid = row["sequence_id"]
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("OBSERVATION_UNIQUE_NONEMPTY_SEQUENCE_ID_REQUIRED")
        positions, labels = row["positions"], row["labels"]
        if (not isinstance(positions, list) or not isinstance(labels, list)
                or not positions or len(positions) != len(labels)
                or any(type(v) is not int or v < 0 for v in positions + labels)):
            raise ValueError("OBSERVATION_ROW_POSITION_LABEL_CONTRACT")
        _cache_id(row["cache"])
        seen.add(sid)
        records.append(_json(row))
    return tuple(records)


@dataclass(frozen=True)
class CoverageSpec:
    """Full declared hidden shapes/valid positions and exact score-row metadata."""
    hidden_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    valid_positions: tuple[tuple[str, tuple[int, ...]], ...]
    current_rows_json: tuple[str, ...]
    past_rows_json: tuple[str, ...]

    def __post_init__(self):
        # frozen=True alone would still allow mutable lists supplied directly
        # to this constructor to alter coverage after its key was issued.
        fields = (self.hidden_shapes, self.valid_positions,
                  self.current_rows_json, self.past_rows_json)
        if any(type(value) is not tuple for value in fields):
            raise TypeError("OBSERVATION_IMMUTABLE_COVERAGE_TUPLES_REQUIRED")
        for entries in (self.hidden_shapes, self.valid_positions):
            if any(type(pair) is not tuple or len(pair) != 2
                   or not isinstance(pair[0], str) or type(pair[1]) is not tuple
                   for pair in entries):
                raise TypeError("OBSERVATION_IMMUTABLE_COVERAGE_ENTRIES_REQUIRED")
            if len(dict(entries)) != len(entries):
                raise ValueError("OBSERVATION_DUPLICATE_COVERAGE_CACHE")
        shapes, positions = dict(self.hidden_shapes), dict(self.valid_positions)
        if set(shapes) != set(positions):
            raise ValueError("OBSERVATION_HIDDEN_VALID_POSITION_INVENTORY")
        for cache, shape in shapes.items():
            if (len(shape) != 3 or shape[0] != 1
                    or any(type(v) is not int or v <= 0 for v in shape)):
                raise ValueError("OBSERVATION_HIDDEN_ONE_SEQUENCE_POSITIVE_SHAPE")
            valid = positions[cache]
            if (not valid or any(type(p) is not int or not 0 <= p < shape[1] for p in valid)
                    or tuple(sorted(set(valid))) != valid):
                raise ValueError("OBSERVATION_ORDERED_ALL_VALID_POSITIONS_REQUIRED")
        for records in (self.current_rows_json, self.past_rows_json):
            if any(not isinstance(row, str) for row in records):
                raise TypeError("OBSERVATION_IMMUTABLE_CANONICAL_ROW_STRINGS_REQUIRED")
            if _rows([json.loads(row) for row in records]) != records:
                raise ValueError("OBSERVATION_CANONICAL_ROW_IDENTITY_REQUIRED")
        for encoded in self.current_rows_json:
            row = json.loads(encoded)
            cache = _cache_id(row["cache"])
            if cache not in shapes or not set(row["positions"]).issubset(positions[cache]):
                raise ValueError("OBSERVATION_CURRENT_ROW_OUTSIDE_VALID_COVERAGE")

    @classmethod
    def from_rows(cls, current_rows, *, hidden_shapes: Mapping,
                  valid_positions: Mapping, past_rows=()):
        shapes = {_cache_id(key): tuple(value) for key, value in hidden_shapes.items()}
        positions = {_cache_id(key): tuple(value) for key, value in valid_positions.items()}
        current, past = _rows(current_rows), _rows(past_rows)
        return cls(tuple(sorted(shapes.items())), tuple(sorted(positions.items())), current, past)

    @property
    def sha256(self):
        return identity_sha256(asdict(self))

    def rows(self, kind):
        if kind not in ("current", "past"):
            raise ValueError("OBSERVATION_CURRENT_OR_PAST_REQUIRED")
        encoded = self.current_rows_json if kind == "current" else self.past_rows_json
        return tuple(json.loads(row) for row in encoded)


@dataclass(frozen=True)
class ObservationKey:
    endpoint_identity: EndpointIdentity
    input_manifest_sha256: str
    rows_identity_sha256: str
    teacher_identity_sha256: str
    coverage_sha256: str
    model_epoch_sha256: str
    trial_identity_sha256: str | None

    @classmethod
    def create(cls, handle: EndpointHandle, *, input_manifest,
               teacher_identity, coverage: CoverageSpec):
        if input_manifest is None or teacher_identity is None:
            raise ValueError("OBSERVATION_EXPLICIT_INPUT_AND_TEACHER_IDENTITY_REQUIRED")
        endpoint = handle.identity
        return cls(endpoint, identity_sha256(input_manifest),
                   identity_sha256((coverage.current_rows_json, coverage.past_rows_json)),
                   identity_sha256(teacher_identity), coverage.sha256,
                   endpoint.model_epoch_sha256, endpoint.trial_identity_sha256)


class ObservationBundle:
    """Native or candidate cache with invalidation subscription and no graphs.

    ``lookup_*`` returns None on incomplete coverage or a nonmatching key, but
    raises on stale endpoint/closed/corrupt state: corruption is NOT a cache
    miss or native-success fallback. Successful reads return owned CPU clones
    or deserialized scalar records, never writable cache aliases.
    """

    def __init__(self, session: EndpointSession, handle: EndpointHandle, *,
                 input_manifest, teacher_identity, coverage: CoverageSpec):
        if not isinstance(coverage, CoverageSpec):
            raise TypeError("OBSERVATION_COVERAGE_SPEC_REQUIRED")
        self.session, self.handle, self.coverage = session, handle, coverage
        self.key = ObservationKey.create(handle, input_manifest=input_manifest,
                                        teacher_identity=teacher_identity, coverage=coverage)
        self._hidden = {}
        self._rows = {}
        self._stats = {}
        self._complete = False
        self._closed = False
        self._unsubscribe = None
        self.work = {key: 0 for key in ("hidden_hits", "row_hits", "misses",
                                      "hidden_bytes", "hidden_sha_checks", "seals")}
        self._unsubscribe = session.subscribe_invalidation(handle, self.close)

    @property
    def closed(self):
        return self._closed

    @property
    def complete_coverage(self):
        return self._complete and not self.closed

    @property
    def complete_marker(self):
        if not self.complete_coverage:
            return None
        self._guard()
        return self.key

    def _guard(self):
        if self.closed:
            raise ObservationError("OBSERVATION_CLOSED_OR_INVALIDATED")
        # Full bytes outside a borrow are a new validation boundary. Inside,
        # the enclosing successful exit is required before consuming outputs.
        self.session.validate_handle(
            self.handle, full=not self.session.has_active_borrow(self.handle))

    def _writable(self):
        self._guard()
        if self._complete:
            raise ObservationError("OBSERVATION_COMPLETE_IS_IMMUTABLE")

    def put_hidden(self, cache_id, hidden):
        self._writable()
        cache = _cache_id(cache_id)
        shape = dict(self.coverage.hidden_shapes).get(cache)
        if shape is None or cache in self._hidden:
            raise ObservationError("OBSERVATION_HIDDEN_UNKNOWN_OR_DUPLICATE_CACHE")
        if (not isinstance(hidden, torch.Tensor) or hidden.dtype != torch.float32
                or tuple(hidden.shape) != shape or hidden.layout != torch.strided):
            raise ValueError("OBSERVATION_FP32_HIDDEN_COVERAGE_SHAPE")
        value = hidden.detach().cpu().contiguous().clone()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("OBSERVATION_NONFINITE_HIDDEN")
        self._hidden[cache] = (value, tensor_sha256(value), value._version)
        self.work["hidden_bytes"] += value.numel() * value.element_size()

    def put_rows(self, kind, rows):
        self._writable()
        expected = {row["sequence_id"]: row for row in self.coverage.rows(kind)}
        if kind in self._rows or not isinstance(rows, Mapping) or set(rows) != set(expected):
            raise ObservationError("OBSERVATION_EXACT_ROW_INVENTORY_OR_DUPLICATE")
        payload = json.loads(_json(rows))
        for sid, metadata in expected.items():
            row = payload[sid]
            if (not isinstance(row, dict)
                    or any(field not in row or _json(row[field]) != _json(value)
                           for field, value in metadata.items())):
                raise ObservationError("OBSERVATION_SCORE_LABEL_POSITION_METADATA_MISMATCH")
            nll, strict, predictions = row.get("nll"), row.get("strict"), row.get("predictions")
            if (type(nll) not in (float, int) or not math.isfinite(nll)
                    or type(strict) is not bool or not isinstance(predictions, list)
                    or len(predictions) != len(metadata["labels"])
                    or any(type(v) is not int or v < 0 for v in predictions)):
                raise ValueError("OBSERVATION_FINITE_NLL_STRICT_PREDICTIONS_REQUIRED")
        self._rows[kind] = _json(payload)

    def put_stats(self, name, payload):
        """Optional scalar-only quality/invariant/geometry receipts, not graphs."""
        self._writable()
        if name not in ("quality_decisions", "invariant_stats", "geometry_anchor_stats"):
            raise ValueError("OBSERVATION_UNKNOWN_STATS_KIND")
        if name in self._stats:
            raise ObservationError("OBSERVATION_DUPLICATE_STATS")
        self._stats[name] = _json(payload)

    def seal(self):
        """Issue completion only after successful checked endpoint scope exit."""
        self._writable()
        if self.session.has_active_borrow(self.handle):
            raise ObservationError("OBSERVATION_SEAL_AFTER_READONLY_SCOPE_EXIT_REQUIRED")
        if (set(self._hidden) != set(dict(self.coverage.hidden_shapes))
                or set(self._rows) != {"current", "past"}):
            raise ObservationError("OBSERVATION_PARTIAL_COVERAGE_CANNOT_SEAL")
        for cache in self._hidden:
            self._validate_hidden(cache)
        self._complete = True
        self.work["seals"] += 1
        return self.key

    def _hit(self, key):
        self._guard()
        if not self.complete_coverage or key != self.key:
            self.work["misses"] += 1
            return False
        return True

    def _validate_hidden(self, cache):
        value, digest, version = self._hidden[cache]
        self.work["hidden_sha_checks"] += 1
        if (value.device.type != "cpu" or value.dtype != torch.float32
                or value.requires_grad or value.grad_fn is not None or value.grad is not None
                or tuple(value.shape) != dict(self.coverage.hidden_shapes)[cache]
                or value._version != version or tensor_sha256(value) != digest):
            self.close()
            raise ObservationError("OBSERVATION_HIDDEN_BYTES_MUTATED_OR_GRAPH_ATTACHED")
        return value

    def lookup_hidden(self, cache_id, *, key: ObservationKey):
        if not self._hit(key):
            return None
        cache = _cache_id(cache_id)
        if cache not in self._hidden:
            self.work["misses"] += 1
            return None
        value = self._validate_hidden(cache)
        self.work["hidden_hits"] += 1
        return value.clone()

    def lookup_rows(self, kind, *, key: ObservationKey):
        if kind not in ("current", "past"):
            raise ValueError("OBSERVATION_CURRENT_OR_PAST_REQUIRED")
        if not self._hit(key):
            return None
        self.work["row_hits"] += 1
        return json.loads(self._rows[kind])

    def lookup_stats(self, name, *, key: ObservationKey):
        if not self._hit(key):
            return None
        return json.loads(self._stats[name]) if name in self._stats else None

    def close(self):
        self._closed = True
        self._complete = False
        self._hidden.clear()
        self._rows.clear()
        self._stats.clear()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def __enter__(self):
        self._guard()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


# Same lifecycle/coverage protocol; endpoint.kind distinguishes provenance.
NativeObservation = ObservationBundle
CandidateObservation = ObservationBundle
