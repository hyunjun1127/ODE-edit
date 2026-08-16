"""Immutable, batch-transactional ODE-side history and key registry."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .contracts import (
    BATCH_SIZE,
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
    finite,
    positive_integer,
)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    logical = tensor.detach().to(device="cpu").contiguous().view(-1)
    byte_view = logical.view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    request_sha256: str
    case_id: int
    collision_key_sha256: str
    target_sha256: str
    version: int
    terminal_event_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("request_sha256", self.request_sha256),
            ("collision_key_sha256", self.collision_key_sha256),
            ("target_sha256", self.target_sha256),
            ("terminal_event_sha256", self.terminal_event_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ODEBFContractError(f"{name} is not a lowercase SHA-256")
        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int) or self.case_id < 0:
            raise ODEBFContractError("history case_id is invalid")
        positive_integer("history version", self.version)


@dataclass(frozen=True, slots=True)
class ProjectedBatchKeys:
    layer: int
    request_order: tuple[str, ...]
    keys: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.request_order) != BATCH_SIZE or len(set(self.request_order)) != BATCH_SIZE:
            raise ODEBFContractError("projected key order is not a distinct B10 batch")
        if self.keys.ndim != 2 or self.keys.shape[1] != BATCH_SIZE:
            raise ODEBFContractError("projected keys must have shape [key_dim,10]")
        if self.keys.dtype not in (torch.float32, torch.float64):
            raise ODEBFContractError("projected keys must use FP32 or FP64")
        if not torch.isfinite(self.keys).all():
            raise ODEBFContractError("projected keys contain non-finite values")
        object.__setattr__(self, "keys", self.keys.detach().to(device="cpu").clone())

    @property
    def structural_payload(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "request_order": list(self.request_order),
            "shape": list(self.keys.shape),
            "dtype": str(self.keys.dtype),
            "sha256": _tensor_sha256(self.keys),
        }


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    version: int
    active_records: tuple[HistoryRecord, ...]
    audit_obsolete_records: tuple[HistoryRecord, ...]
    projected_keys_by_layer: tuple[tuple[int, torch.Tensor], ...]
    finalized_transactions: tuple[tuple[str, str], ...]
    cumulative_load: tuple[tuple[int, float], ...]
    digest: str


@dataclass(frozen=True, slots=True)
class ProspectiveHistoryBatch:
    transaction_id: str
    expected_version: int
    records: tuple[HistoryRecord, ...]
    projected: tuple[ProjectedBatchKeys, ...]
    obsolete_request_sha256: tuple[str, ...]
    payload_digest: str


@dataclass(frozen=True, slots=True)
class HistoryFinalizeReceipt:
    transaction_id: str
    before_version: int
    after_version: int
    appended_count: int
    obsolete_count: int
    idempotent_replay: bool
    state_digest: str


class HistoryLedger:
    """The ODE-side source of truth; EasyEdit ``cache_c`` is never mutated."""

    def __init__(self, *, max_active_records: int, layer_order: Sequence[int]) -> None:
        if max_active_records < BATCH_SIZE:
            raise ODEBFContractError("history bound cannot hold one joint batch")
        self.max_active_records = positive_integer("max_active_records", max_active_records)
        self.layer_order = tuple(layer_order)
        if len(self.layer_order) < 2 or len(set(self.layer_order)) != len(self.layer_order):
            raise ODEBFContractError("history registry requires distinct candidate layers")
        self._lock = threading.RLock()
        self._version = 0
        self._active: tuple[HistoryRecord, ...] = ()
        self._obsolete: tuple[HistoryRecord, ...] = ()
        self._keys: dict[int, tuple[torch.Tensor, ...]] = {layer: () for layer in self.layer_order}
        self._key_requests: dict[int, tuple[str, ...]] = {layer: () for layer in self.layer_order}
        self._finalized: dict[str, str] = {}
        self._load: dict[int, float] = {layer: 0.0 for layer in self.layer_order}

    @property
    def version(self) -> int:
        return self._version

    def _state_payload(self) -> dict[str, object]:
        return {
            "version": self._version,
            "active": [record.request_sha256 for record in self._active],
            "obsolete": [record.request_sha256 for record in self._obsolete],
            "keys": {
                str(layer): [
                    {
                        "request": request,
                        "sha256": _tensor_sha256(column),
                        "shape": list(column.shape),
                        "dtype": str(column.dtype),
                    }
                    for request, column in zip(
                        self._key_requests[layer], self._keys[layer]
                    )
                ]
                for layer in self.layer_order
            },
            "finalized": sorted(self._finalized.items()),
            "cumulative_load": sorted(self._load.items()),
        }

    def snapshot(self) -> HistorySnapshot:
        with self._lock:
            payload = self._state_payload()
            return HistorySnapshot(
                self._version,
                self._active,
                self._obsolete,
                tuple(
                    (
                        layer,
                        torch.cat(self._keys[layer], dim=1).clone()
                        if self._keys[layer]
                        else torch.empty((0, 0), dtype=torch.float64),
                    )
                    for layer in self.layer_order
                ),
                tuple(sorted(self._finalized.items())),
                tuple(sorted(self._load.items())),
                canonical_hash(payload),
            )

    def prospective(
        self,
        *,
        transaction_id: str,
        expected_version: int,
        records: Sequence[HistoryRecord],
        projected_keys: Mapping[int, torch.Tensor],
    ) -> ProspectiveHistoryBatch:
        if not transaction_id or transaction_id.strip() != transaction_id:
            raise ODEBFContractError("transaction identity is empty or noncanonical")
        batch = tuple(records)
        if len(batch) != BATCH_SIZE:
            raise ODEBFContractError("history finalization requires exactly ten records")
        request_order = tuple(record.request_sha256 for record in batch)
        if len(set(request_order)) != BATCH_SIZE:
            raise ODEBFContractError("history batch contains duplicate requests")
        if len({record.case_id for record in batch}) != BATCH_SIZE:
            raise ODEBFContractError("history batch contains duplicate cases")
        collision_targets = [(record.collision_key_sha256, record.target_sha256) for record in batch]
        if len(set(collision_targets)) != BATCH_SIZE:
            raise ODEBFContractError("joint batch repeats the same normalized target")
        if len({record.collision_key_sha256 for record in batch}) != BATCH_SIZE:
            raise ODEBFContractError(
                "joint batch contains an ambiguous subject-relation-template collision"
            )
        if any(record.version != expected_version + 1 for record in batch):
            raise ODEBFContractError("joint history records do not share the next version")
        if set(projected_keys) != set(self.layer_order):
            raise ODEBFContractError("projected-key layer registry is incomplete")
        projected = tuple(
            ProjectedBatchKeys(layer, request_order, projected_keys[layer])
            for layer in self.layer_order
        )
        with self._lock:
            if expected_version != self._version:
                raise ODEBFStateError("history compare-and-swap version differs")
            incoming_by_collision = {
                record.collision_key_sha256: record.target_sha256 for record in batch
            }
            obsolete = tuple(
                record.request_sha256
                for record in self._active
                if record.collision_key_sha256 in incoming_by_collision
                and incoming_by_collision[record.collision_key_sha256] != record.target_sha256
            )
            remaining = len(self._active) - len(obsolete)
            if remaining + BATCH_SIZE > self.max_active_records:
                raise ODEBFStateError("bounded history registry capacity would be exceeded")
        payload = {
            "transaction_id": transaction_id,
            "expected_version": expected_version,
            "records": [
                {
                    "request_sha256": record.request_sha256,
                    "case_id": record.case_id,
                    "collision_key_sha256": record.collision_key_sha256,
                    "target_sha256": record.target_sha256,
                    "version": record.version,
                    "terminal_event_sha256": record.terminal_event_sha256,
                }
                for record in batch
            ],
            "projected": [item.structural_payload for item in projected],
            "obsolete": list(obsolete),
        }
        return ProspectiveHistoryBatch(
            transaction_id,
            expected_version,
            batch,
            projected,
            obsolete,
            canonical_hash(payload),
        )

    def finalize(
        self,
        prospective: ProspectiveHistoryBatch,
        *,
        post_commit_verified: bool,
        load_increment_by_layer: Mapping[int, float],
        fault_phase: str | None = None,
    ) -> HistoryFinalizeReceipt:
        if not post_commit_verified:
            raise ODEBFStateError("history cannot finalize before post-commit verification")
        if set(load_increment_by_layer) != set(self.layer_order):
            raise ODEBFContractError("cumulative load update is incomplete")
        increments = {
            layer: finite("load increment", load_increment_by_layer[layer])
            for layer in self.layer_order
        }
        if any(value < 0.0 for value in increments.values()):
            raise ODEBFContractError("cumulative load cannot decrease")
        if fault_phase not in (None, "early", "middle", "late"):
            raise ODEBFContractError("history fault phase is invalid")

        with self._lock:
            finalized_digest = self._finalized.get(prospective.transaction_id)
            if finalized_digest is not None:
                if finalized_digest != prospective.payload_digest:
                    raise ODEBFStateError("transaction ID was reused for another batch")
                return HistoryFinalizeReceipt(
                    prospective.transaction_id,
                    self._version,
                    self._version,
                    0,
                    0,
                    True,
                    self.snapshot().digest,
                )
            if self._version != prospective.expected_version:
                raise ODEBFStateError("history compare-and-swap failed at finalization")
            before = self.snapshot()
            if fault_phase == "early":
                raise RuntimeError("injected early history finalization fault")

            obsolete_set = set(prospective.obsolete_request_sha256)
            kept_active = tuple(
                record for record in self._active if record.request_sha256 not in obsolete_set
            )
            moved_obsolete = tuple(
                record for record in self._active if record.request_sha256 in obsolete_set
            )
            new_active = kept_active + prospective.records
            new_obsolete = self._obsolete + moved_obsolete

            new_keys: dict[int, tuple[torch.Tensor, ...]] = {}
            new_requests: dict[int, tuple[str, ...]] = {}
            for projected in prospective.projected:
                retained_pairs = [
                    (request, column)
                    for request, column in zip(
                        self._key_requests[projected.layer], self._keys[projected.layer]
                    )
                    if request not in obsolete_set
                ]
                appended = [
                    projected.keys[:, index : index + 1].clone()
                    for index in range(BATCH_SIZE)
                ]
                new_requests[projected.layer] = tuple(
                    [request for request, _ in retained_pairs]
                    + list(projected.request_order)
                )
                new_keys[projected.layer] = tuple(
                    [column for _, column in retained_pairs] + appended
                )
            if fault_phase == "middle":
                if self.snapshot().digest != before.digest:
                    raise ODEBFStateError("prospective history construction mutated state")
                raise RuntimeError("injected middle history finalization fault")

            new_load = {
                layer: self._load[layer] + increments[layer] for layer in self.layer_order
            }
            new_finalized = dict(self._finalized)
            new_finalized[prospective.transaction_id] = prospective.payload_digest
            if fault_phase == "late":
                if self.snapshot().digest != before.digest:
                    raise ODEBFStateError("late history construction mutated state")
                raise RuntimeError("injected late history finalization fault")

            # One atomic Python-state swap after every component was validated.
            self._active = new_active
            self._obsolete = new_obsolete
            self._keys = new_keys
            self._key_requests = new_requests
            self._load = new_load
            self._finalized = new_finalized
            self._version += 1
            after = self.snapshot()
            return HistoryFinalizeReceipt(
                prospective.transaction_id,
                before.version,
                after.version,
                BATCH_SIZE,
                len(obsolete_set),
                False,
                after.digest,
            )

    def solve_key_view(self, layer: int) -> torch.Tensor:
        with self._lock:
            if layer not in self._keys:
                raise ODEBFContractError("history layer is absent")
            if not self._keys[layer]:
                return torch.empty((0, 0), dtype=torch.float64)
            return torch.cat(self._keys[layer], dim=1).clone()

    def deterministic_weighted_view(
        self,
        layer: int,
        weights_by_request: Mapping[str, float],
    ) -> torch.Tensor:
        """Apply weights at use time; the persistent registry stays unweighted."""

        with self._lock:
            requests = self._key_requests.get(layer)
            columns = self._keys.get(layer)
            if requests is None or columns is None:
                raise ODEBFContractError("history layer is absent")
            if set(weights_by_request) != set(requests):
                raise ODEBFContractError("history weight view is incomplete")
            weighted = [
                column * finite("history weight", weights_by_request[request])
                for request, column in zip(requests, columns)
            ]
            return (
                torch.cat(weighted, dim=1)
                if weighted
                else torch.empty((0, 0), dtype=torch.float64)
            )
