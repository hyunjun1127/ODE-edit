"""Bounded, native-only v5 replay memory. No persistence/checkpoint API.

The caller snapshots *before* sampling, keeps entry teachers/pending events in
its RAM transaction, and calls ``admit`` only after the physical commit succeeds.
Sampling is fixed until that commit; quality/loss values never drive selection.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata(value: Any) -> Any:
    """Copy JSON-like native metadata without admitting opaque payload objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
        return {k: _metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata(v) for v in value]
    raise ValueError("native metadata must contain finite JSON-compatible values")


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _identity(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True, repr=False)
class _Snapshot:
    config: tuple[int, int, int]
    state: dict[str, Any] = field(repr=False)
    rng_state: tuple = field(repr=False)

    def __repr__(self) -> str:
        return "<NativeMemory snapshot: RAM only>"

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("NativeMemory snapshots are RAM-only, not checkpoints")


class NativeMemory:
    """Algorithm R over first-observed unique fact IDs, with frozen KL anchors.

    Events contain ``fact_id``, ``version``, ``record`` (only native
    ``requested_rewrite`` and optional ``case_id``), ``context_nll``,
    ``kl_identity``, ``kl_input`` (JSON-like token/mask/position/readout metadata),
    and a CPU full-vocabulary log-probability ``teacher`` tensor. The native
    adapter owns identity construction and vocabulary/context shape validation.

    ``sample`` returns defensive copies with those same fields. Within a returned
    sample, owners of one KL identity share one cloned teacher/input tuple.
    ``admit`` preserves stream order, including contradictory fact occurrences.
    """

    def __init__(self, capacity: int = 128, reference_cap: int = 16,
                 seed: int = 20261002) -> None:
        self.capacity = _nonnegative_int("capacity", capacity)
        self.reference_cap = _nonnegative_int("reference_cap", reference_cap)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self._rng = random.Random(seed)
        self._state: dict[str, Any] = {
            "slots": [], "by_fact": {}, "anchors": {}, "seen": set(),
            "pending_sample": None,
            "counts": {"commits": 0, "events": 0, "admissions": 0,
                       "evictions": 0, "resident_repeats": 0,
                       "nonresident_repeats": 0},
        }

    def __len__(self) -> int:
        return len(self._state["slots"])

    @property
    def fact_ids(self) -> tuple[str, ...]:
        """Resident IDs in stable reservoir-slot order (not score order)."""
        return tuple(r["fact_id"] for r in self._state["slots"])

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("NativeMemory is RAM-only, not a checkpoint")

    def snapshot(self) -> _Snapshot:
        """Independent RAM copy, including RNG and in-flight sample metadata."""
        return _Snapshot((self.capacity, self.reference_cap, self.seed),
                         copy.deepcopy(self._state), self._rng.getstate())

    def restore(self, snapshot: _Snapshot) -> None:
        """Restore without consuming the snapshot; repeated rollback is exact."""
        if not isinstance(snapshot, _Snapshot):
            raise TypeError("restore requires a NativeMemory RAM snapshot")
        if snapshot.config != (self.capacity, self.reference_cap, self.seed):
            raise ValueError("snapshot belongs to a different memory configuration")
        restored = copy.deepcopy(snapshot.state)
        self._rng.setstate(snapshot.rng_state)
        self._state = restored

    def sample(self, current_fact_ids: Iterable[str], actual_B: int) -> list[dict]:
        """Uniform without replacement, once per commit boundary.

        Repeated calls for the same batch return the original selection, without
        any RNG draw. A changed batch before ``admit`` is a transaction error.
        Exclusion is by fact ID, not by a shared KL identity.
        """
        actual_B = _nonnegative_int("actual_B", actual_B)
        if isinstance(current_fact_ids, (str, bytes)):
            raise ValueError("current_fact_ids must be an iterable of fact IDs")
        current = tuple(sorted({_identity("fact_id", f) for f in current_fact_ids}))
        pending = self._state["pending_sample"]
        if pending is not None:
            if pending["current"] != current or pending["actual_B"] != actual_B:
                raise RuntimeError("batch changed before commit/rollback of memory sample")
            return self._records(pending["ids"])
        excluded = set(current)
        eligible = [f for f in self.fact_ids if f not in excluded]
        count = min(self.reference_cap, actual_B, len(eligible))
        selected = tuple(self._rng.sample(eligible, count)) if count else ()
        self._state["pending_sample"] = {
            "current": current, "actual_B": actual_B, "ids": selected,
        }
        return self._records(selected)

    def _records(self, fact_ids: Iterable[str]) -> list[dict]:
        result = []
        anchors: dict[str, dict] = {}
        for fact_id in fact_ids:
            resident = copy.deepcopy(
                self._state["slots"][self._state["by_fact"][fact_id]])
            identity = resident["kl_identity"]
            if identity not in anchors:
                anchor = self._state["anchors"][identity]
                anchors[identity] = {"kl_input": copy.deepcopy(anchor["kl_input"]),
                                     "teacher": anchor["teacher"].clone()}
            resident.update(anchors[identity])
            result.append(resident)
        return result

    @staticmethod
    def _event(event: Mapping[str, Any]) -> dict:
        required = {"fact_id", "version", "record", "context_nll",
                    "kl_identity", "kl_input", "teacher"}
        if not isinstance(event, Mapping) or set(event) != required:
            raise ValueError("memory event must contain exactly the native event fields")
        result = {name: _identity(name, event[name])
                  for name in ("fact_id", "version", "kl_identity")}
        record = event["record"]
        if (not isinstance(record, Mapping)
                or "requested_rewrite" not in record
                or set(record) - {"case_id", "requested_rewrite"}
                or not isinstance(record["requested_rewrite"], Mapping)):
            raise ValueError("record allows only requested_rewrite and optional case_id")
        result["record"] = _metadata(record)
        if not isinstance(event["kl_input"], Mapping):
            raise ValueError("kl_input must be native token metadata")
        result["kl_input"] = _metadata(event["kl_input"])
        nll = event["context_nll"]
        if not isinstance(nll, (list, tuple)) or not nll:
            raise ValueError("context_nll must contain every native context scalar")
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) or x < 0 for x in nll):
            raise ValueError("context_nll must contain finite nonnegative scalars")
        result["context_nll"] = [float(x) for x in nll]
        teacher = event["teacher"]
        if (not isinstance(teacher, torch.Tensor) or teacher.device.type != "cpu"
                or not teacher.is_floating_point() or teacher.ndim < 1
                or teacher.numel() == 0):
            raise ValueError("teacher must be a CPU full-vocabulary log-probability tensor")
        teacher = teacher.detach().to(dtype=torch.float32).contiguous().clone()
        if not bool(torch.isfinite(teacher).all()):
            raise ValueError("teacher must be finite in FP32")
        result["teacher"] = teacher
        return result

    def admit(self, events: Iterable[Mapping[str, Any]]) -> dict[str, int]:
        """Publish committed events atomically in stream order, then end batch.

        Entry teachers in events are ignored on resident repeats. Last-owner
        eviction drops the entire KL tuple before any new owner is admitted,
        including a replacement carrying the same identity in this same batch.
        The outer transaction must also restore W/H and pending caller events.
        """
        prepared = [self._event(e) for e in events]
        # Internal operations never mutate stored tensors/metadata in place, so
        # a shallow anchor copy suffices for this short-lived rollback backup.
        previous = self._state
        self._state = {
            "slots": list(previous["slots"]), "by_fact": dict(previous["by_fact"]),
            "anchors": {k: dict(v) for k, v in previous["anchors"].items()},
            "seen": set(previous["seen"]),
            "pending_sample": previous["pending_sample"],
            "counts": dict(previous["counts"]),
        }
        rng_before = self._rng.getstate()
        counts_before = dict(self._state["counts"])
        try:
            for event in prepared:
                self._apply(event)
            self._state["counts"]["commits"] += 1
            self._state["pending_sample"] = None
        except Exception:
            self._state = previous
            self._rng.setstate(rng_before)
            raise
        return {k: v - counts_before[k] for k, v in self._state["counts"].items()}

    def _apply(self, event: dict) -> None:
        state = self._state
        fact_id = event["fact_id"]
        state["counts"]["events"] += 1
        if fact_id in state["seen"]:
            slot = state["by_fact"].get(fact_id)
            if slot is None:
                state["counts"]["nonresident_repeats"] += 1
                return
            # Freeze BOTH old token tuple/identity and its admission teacher.
            resident = dict(state["slots"][slot])
            for key in ("version", "record", "context_nll"):
                resident[key] = event[key]
            state["slots"][slot] = resident
            state["counts"]["resident_repeats"] += 1
            return
        state["seen"].add(fact_id)
        n = len(state["seen"])
        if n <= self.capacity:
            slot = len(state["slots"])
        else:
            draw = self._rng.randint(1, n)
            if draw > self.capacity:
                return
            slot = draw - 1
        if slot < len(state["slots"]):
            evicted = state["slots"][slot]
            del state["by_fact"][evicted["fact_id"]]
            anchor = state["anchors"][evicted["kl_identity"]]
            anchor["refcount"] -= 1
            if anchor["refcount"] == 0:
                del state["anchors"][evicted["kl_identity"]]
            state["counts"]["evictions"] += 1
        identity = event["kl_identity"]
        anchor = state["anchors"].get(identity)
        if anchor is None:
            teacher = event["teacher"]
            anchor = {
                "kl_input": event["kl_input"], "teacher": teacher, "refcount": 0,
                "input_hash": _hash(event["kl_input"]),
                "teacher_hash": hashlib.sha256(teacher.numpy().tobytes()).hexdigest(),
            }
            state["anchors"][identity] = anchor
        elif anchor["input_hash"] != _hash(event["kl_input"]):
            raise ValueError("one KL identity cannot refer to different native input tuples")
        elif anchor["teacher"].shape != event["teacher"].shape:
            raise ValueError("shared KL identity changed full-vocabulary teacher shape")
        anchor["refcount"] += 1
        resident = {k: event[k] for k in
                    ("fact_id", "version", "record", "context_nll", "kl_identity")}
        if slot == len(state["slots"]):
            state["slots"].append(resident)
        else:
            state["slots"][slot] = resident
        state["by_fact"][fact_id] = slot
        state["counts"]["admissions"] += 1

    def summary(self) -> dict[str, Any]:
        """Export hashes/scalars only; never native text, token arrays or tensors."""
        anchors = self._state["anchors"]
        pending = self._state["pending_sample"]
        result = {
            "capacity": self.capacity, "reference_cap": self.reference_cap,
            "seed": self.seed, "resident_count": len(self),
            "seen_count": len(self._state["seen"]), "kl_identity_count": len(anchors),
            "teacher_bytes": sum(a["teacher"].numel() * a["teacher"].element_size()
                                 for a in anchors.values()),
            "owner_count": sum(a["refcount"] for a in anchors.values()),
            "batch_sampled": pending is not None,
            "sample_size": len(pending["ids"]) if pending is not None else 0,
            "seen_hash": _hash(sorted(self._state["seen"])),
            "resident_hash": _hash(self._state["slots"]),
            "anchor_hash": _hash({k: {"input_hash": a["input_hash"],
                                      "teacher_hash": a["teacher_hash"],
                                      "shape": list(a["teacher"].shape),
                                      "refcount": a["refcount"]}
                                  for k, a in anchors.items()}),
            "rng_hash": _hash(self._rng.getstate()),
            "pending_sample_hash": _hash(pending),
            **self._state["counts"],
        }
        result["state_hash"] = _hash(result)
        return result
