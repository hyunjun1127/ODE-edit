"""Sequential four-arm B10 state, endpoint, history, and resource contracts."""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import torch

from .benchmark import BatchSuccessReceipt
from .contracts import BATCH_SIZE, FIXED_K, ODEBFContractError, ODEBFStateError, canonical_hash, finite
from .first_hit import FeasibilityVerdict
from .functional import WaypointFactor, tensor_sha256


class P1Arm(str, Enum):
    N32_NATIVE = "N32_NATIVE"
    F_G = "F_G"
    F_BF = "F_BF"
    R_BF = "R_BF"


P1_ARM_ORDER = tuple(P1Arm)


@dataclass(frozen=True, slots=True)
class P1Waypoint:
    stage: int
    accepted: bool
    success: BatchSuccessReceipt
    feasibility: FeasibilityVerdict
    snapshot_sha256: str
    field_sha256: str
    raw_velocity_sha256: str
    accepted_beta: float
    rejection_consumed_slot: bool

    def __post_init__(self) -> None:
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or not 0 <= self.stage <= FIXED_K:
            raise ODEBFContractError("P1 waypoint stage is outside s=0..8")
        for value in (self.snapshot_sha256, self.field_sha256, self.raw_velocity_sha256):
            if len(value) != 64:
                raise ODEBFContractError("P1 waypoint digest differs")
        beta = finite("accepted beta", self.accepted_beta)
        if self.stage == 0:
            if self.accepted or beta != 0.0 or self.rejection_consumed_slot:
                raise ODEBFContractError("P1 entry waypoint transition fields differ")
        elif self.accepted:
            if beta <= 0.0 or beta > 1.0 or self.rejection_consumed_slot:
                raise ODEBFContractError("P1 accepted waypoint beta differs")
        elif beta != 0.0 or not self.rejection_consumed_slot:
            raise ODEBFContractError("P1 rejection did not consume exactly one stage slot")

    @property
    def exact_feasible_hit(self) -> bool:
        return self.success.joint_exact_success and self.feasibility.all_pass


@dataclass(frozen=True, slots=True)
class P1EndpointSelection:
    status: str
    selected_stage: int
    selected_snapshot_sha256: str
    joint_first_exact_hit_step: int | None
    all_exact_hit_steps: tuple[int, ...]
    per_request_first_hit_step: tuple[int | None, ...]
    persistence_after_first_hit: tuple[bool, ...]
    reversal_after_first_hit: bool
    fixed_budget_terminal_success_count: int
    fixed_budget_terminal_feasible: bool
    max_official_success_count: int
    earliest_max_success_step: int
    executed_slots: int
    accepted_slots: int
    rejected_slots: int
    scientific_endpoint_commit_permitted: bool

    def __post_init__(self) -> None:
        if self.status not in ("EXACT_BATCH_HIT", "NO_EXACT_BATCH_HIT"):
            raise ODEBFContractError("P1 endpoint status differs")
        if not 0 <= self.selected_stage <= FIXED_K or len(self.selected_snapshot_sha256) != 64:
            raise ODEBFContractError("P1 selected endpoint identity differs")
        if self.executed_slots != FIXED_K or self.accepted_slots + self.rejected_slots != FIXED_K:
            raise ODEBFContractError("P1 endpoint did not execute fixed K8")
        if len(self.per_request_first_hit_step) != BATCH_SIZE:
            raise ODEBFContractError("P1 endpoint lacks per-request first-hit vector")
        if not self.scientific_endpoint_commit_permitted:
            raise ODEBFContractError("P1 selected endpoint is not feasible for a transaction")


class FixedK8P1Selector:
    """Select first exact feasible hit, otherwise the feasible K8 state."""

    def __init__(self) -> None:
        self._records: list[P1Waypoint] = []
        self._selection: P1EndpointSelection | None = None

    def append(self, record: P1Waypoint) -> None:
        if self._selection is not None:
            raise ODEBFStateError("P1 endpoint selector is already finalized")
        if record.stage != len(self._records):
            raise ODEBFStateError("P1 waypoint stages are not contiguous")
        if record.stage > FIXED_K:
            raise ODEBFStateError("P1 waypoint exceeds fixed K8")
        if self._records and not record.accepted:
            previous = self._records[-1]
            if record.snapshot_sha256 != previous.snapshot_sha256:
                raise ODEBFStateError("rejected P1 slot changed the virtual state")
        self._records.append(record)

    @property
    def records(self) -> tuple[P1Waypoint, ...]:
        return tuple(self._records)

    def finalize(self) -> P1EndpointSelection:
        if self._selection is not None:
            return self._selection
        if len(self._records) != FIXED_K + 1:
            raise ODEBFStateError("P1 selector lacks entry plus eight stage receipts")
        exact = tuple(record.stage for record in self._records if record.exact_feasible_hit)
        first = exact[0] if exact else None
        selected = self._records[first] if first is not None else self._records[-1]
        if not selected.feasibility.all_pass:
            raise ODEBFStateError("P1 fixed-K terminal endpoint is not feasible")
        per_request: list[int | None] = [None] * BATCH_SIZE
        for record in self._records:
            if not record.feasibility.all_pass:
                continue
            for index, bit in enumerate(record.success.request_success_vector):
                if bit and per_request[index] is None:
                    per_request[index] = record.stage
        persistence = (
            tuple(
                record.exact_feasible_hit
                for record in self._records
                if first is not None and record.stage > first
            )
            if first is not None
            else ()
        )
        counts = tuple(record.success.numerator for record in self._records)
        maximum = max(counts)
        earliest_max = counts.index(maximum)
        self._selection = P1EndpointSelection(
            "EXACT_BATCH_HIT" if first is not None else "NO_EXACT_BATCH_HIT",
            selected.stage,
            selected.snapshot_sha256,
            first,
            exact,
            tuple(per_request),
            persistence,
            any(not value for value in persistence),
            self._records[-1].success.numerator,
            self._records[-1].feasibility.all_pass,
            maximum,
            earliest_max,
            FIXED_K,
            sum(int(record.accepted) for record in self._records[1:]),
            sum(int(not record.accepted) for record in self._records[1:]),
            True,
        )
        return self._selection

    def verify_post_commit(
        self,
        *,
        committed_snapshot_sha256: str,
        success: BatchSuccessReceipt,
        feasibility: FeasibilityVerdict,
    ) -> None:
        selection = self.finalize()
        selected = self._records[selection.selected_stage]
        if committed_snapshot_sha256 != selection.selected_snapshot_sha256:
            raise ODEBFContractError("P1 committed endpoint is not the frozen selection")
        if success.raw_free_payload() != selected.success.raw_free_payload():
            raise ODEBFContractError("P1 post-commit official efficacy vector differs")
        if not feasibility.all_pass:
            raise ODEBFContractError("P1 post-commit endpoint is not feasible")
        if selection.status == "EXACT_BATCH_HIT" and not success.joint_exact_success:
            raise ODEBFContractError("P1 exact first-hit post-commit verification differs")


@dataclass(frozen=True, slots=True)
class P1HistoryRecord:
    request_sha256: str
    case_id: int
    collision_sha256: str
    target_sha256: str
    terminal_event_sha256: str
    version: int

    def __post_init__(self) -> None:
        for value in (
            self.request_sha256,
            self.collision_sha256,
            self.target_sha256,
            self.terminal_event_sha256,
        ):
            if len(value) != 64:
                raise ODEBFContractError("P1 history record digest differs")
        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int) or self.case_id < 0:
            raise ODEBFContractError("P1 history case identity differs")
        if self.version <= 0:
            raise ODEBFContractError("P1 history version differs")


@dataclass(frozen=True, slots=True)
class P1HistorySnapshot:
    version: int
    active_records: tuple[P1HistoryRecord, ...]
    audit_obsolete_records: tuple[P1HistoryRecord, ...]
    solve_key_sha256: tuple[tuple[int, str], ...]
    risk_key_sha256: tuple[tuple[int, str], ...]
    cumulative_load: tuple[tuple[int, float], ...]
    finalized_transactions: tuple[str, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class ProspectiveP1HistoryBatch:
    transaction_id: str
    expected_version: int
    records: tuple[P1HistoryRecord, ...]
    solve_keys: tuple[tuple[int, torch.Tensor], ...]
    risk_keys: tuple[tuple[int, torch.Tensor], ...]
    obsolete_request_sha256: tuple[str, ...]
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class P1HistoryFinalizeReceipt:
    transaction_id: str
    before_version: int
    after_version: int
    appended_count: int
    obsolete_count: int
    idempotent_replay: bool
    state_sha256: str


@dataclass(frozen=True, slots=True)
class P1HistoryRollbackState:
    """Opaque in-process rollback token for a larger all-or-nothing transaction."""

    version: int
    active_records: tuple[P1HistoryRecord, ...]
    obsolete_records: tuple[P1HistoryRecord, ...]
    solve_keys: tuple[tuple[int, torch.Tensor], ...]
    risk_keys: tuple[tuple[int, torch.Tensor], ...]
    cumulative_load: tuple[tuple[int, float], ...]
    finalized_transactions: tuple[tuple[str, str], ...]
    state_sha256: str


class P1HistoryLedger:
    """Atomic raw-solve/projected-risk registry plus immutable request metadata."""

    def __init__(
        self,
        *,
        layer_order: Sequence[int],
        maximum_records: int = 40,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.layer_order = tuple(int(layer) for layer in layer_order)
        if len(self.layer_order) < 2 or len(set(self.layer_order)) != len(self.layer_order):
            raise ODEBFContractError("P1 history layer order differs")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
            or maximum_records < 4 * batch_size
        ):
            raise ODEBFContractError("P1 history bound cannot hold four B10 transactions")
        self.maximum_records = maximum_records
        self.batch_size = batch_size
        self._lock = threading.RLock()
        self._version = 0
        self._active: tuple[P1HistoryRecord, ...] = ()
        self._obsolete: tuple[P1HistoryRecord, ...] = ()
        self._solve: dict[int, torch.Tensor] = {
            layer: torch.empty((0, 0), dtype=torch.float32) for layer in self.layer_order
        }
        self._risk: dict[int, torch.Tensor] = {
            layer: torch.empty((0, 0), dtype=torch.float32) for layer in self.layer_order
        }
        self._load: dict[int, float] = {layer: 0.0 for layer in self.layer_order}
        self._finalized: dict[str, str] = {}

    @property
    def version(self) -> int:
        return self._version

    def _key_hashes(self, values: Mapping[int, torch.Tensor]) -> tuple[tuple[int, str], ...]:
        return tuple(
            (layer, canonical_hash({"shape": list(values[layer].shape), "sha256": tensor_sha256(values[layer])}))
            for layer in self.layer_order
        )

    def snapshot(self) -> P1HistorySnapshot:
        with self._lock:
            solve_hashes = self._key_hashes(self._solve)
            risk_hashes = self._key_hashes(self._risk)
            payload = {
                "version": self._version,
                "active": [record.request_sha256 for record in self._active],
                "obsolete": [record.request_sha256 for record in self._obsolete],
                "solve": solve_hashes,
                "risk": risk_hashes,
                "load": sorted(self._load.items()),
                "finalized": sorted(self._finalized),
            }
            return P1HistorySnapshot(
                self._version,
                self._active,
                self._obsolete,
                solve_hashes,
                risk_hashes,
                tuple(sorted(self._load.items())),
                tuple(sorted(self._finalized)),
                canonical_hash(payload),
            )

    def transaction_checkpoint(self) -> P1HistoryRollbackState:
        """Capture exact mutable ledger state without exposing it to serialization."""

        with self._lock:
            snapshot = self.snapshot()
            return P1HistoryRollbackState(
                self._version,
                self._active,
                self._obsolete,
                tuple((layer, self._solve[layer].clone()) for layer in self.layer_order),
                tuple((layer, self._risk[layer].clone()) for layer in self.layer_order),
                tuple(sorted(self._load.items())),
                tuple(sorted(self._finalized.items())),
                snapshot.digest,
            )

    def restore_transaction_checkpoint(self, checkpoint: P1HistoryRollbackState) -> None:
        """Restore a token captured immediately before a failed enclosing commit."""

        if not isinstance(checkpoint, P1HistoryRollbackState):
            raise ODEBFContractError("P1 history rollback token differs")
        solve = dict(checkpoint.solve_keys)
        risk = dict(checkpoint.risk_keys)
        if set(solve) != set(self.layer_order) or set(risk) != set(self.layer_order):
            raise ODEBFContractError("P1 history rollback layer set differs")
        with self._lock:
            self._version = checkpoint.version
            self._active = checkpoint.active_records
            self._obsolete = checkpoint.obsolete_records
            self._solve = {layer: solve[layer].clone() for layer in self.layer_order}
            self._risk = {layer: risk[layer].clone() for layer in self.layer_order}
            self._load = dict(checkpoint.cumulative_load)
            self._finalized = dict(checkpoint.finalized_transactions)
            if self.snapshot().digest != checkpoint.state_sha256:
                raise ODEBFStateError("P1 history rollback did not restore exact state")

    def _validate_keys(
        self,
        values: Mapping[int, torch.Tensor],
        layers: tuple[int, ...],
    ) -> tuple[tuple[int, torch.Tensor], ...]:
        if set(values) != set(layers):
            raise ODEBFContractError("P1 history key layer set is incomplete")
        result: list[tuple[int, torch.Tensor]] = []
        dimensions: dict[int, int] = {}
        for layer in layers:
            tensor = values[layer]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 2
                or tensor.shape[1] != self.batch_size
                or tensor.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(tensor).all()
            ):
                raise ODEBFContractError("P1 history keys are not finite joint-rank10 tensors")
            dimensions[layer] = tensor.shape[0]
            result.append((layer, tensor.detach().to(device="cpu", dtype=torch.float32).clone()))
        return tuple(result)

    def prospective(
        self,
        *,
        transaction_id: str,
        expected_version: int,
        records: Sequence[P1HistoryRecord],
        solve_keys_by_layer: Mapping[int, torch.Tensor],
        risk_keys_by_layer: Mapping[int, torch.Tensor],
    ) -> ProspectiveP1HistoryBatch:
        batch = tuple(records)
        if not transaction_id or len(batch) != self.batch_size:
            raise ODEBFContractError("P1 history prospective batch differs")
        if len({item.request_sha256 for item in batch}) != self.batch_size or len(
            {item.case_id for item in batch}
        ) != self.batch_size:
            raise ODEBFContractError("P1 history batch is not a distinct B10")
        if len({item.collision_sha256 for item in batch}) != self.batch_size:
            raise ODEBFContractError("P1 history batch has an ambiguous collision")
        if any(item.version != expected_version + 1 for item in batch):
            raise ODEBFContractError("P1 history records do not share the next version")
        solve = self._validate_keys(solve_keys_by_layer, self.layer_order)
        risk = self._validate_keys(risk_keys_by_layer, self.layer_order)
        if any(solve_item[1].shape != risk_item[1].shape for solve_item, risk_item in zip(solve, risk)):
            raise ODEBFContractError("P1 solve/risk key geometries differ")
        incoming = {item.collision_sha256: item.target_sha256 for item in batch}
        with self._lock:
            if expected_version != self._version:
                raise ODEBFStateError("P1 history compare-and-swap version differs")
            obsolete = tuple(
                item.request_sha256
                for item in self._active
                if item.collision_sha256 in incoming
                and incoming[item.collision_sha256] != item.target_sha256
            )
            if len(self._active) - len(obsolete) + self.batch_size > self.maximum_records:
                raise ODEBFStateError("P1 history capacity would be exceeded")
        payload = {
            "transaction_id": transaction_id,
            "expected_version": expected_version,
            "records": [item.request_sha256 for item in batch],
            "solve": [(layer, tensor_sha256(value)) for layer, value in solve],
            "risk": [(layer, tensor_sha256(value)) for layer, value in risk],
            "obsolete": list(obsolete),
        }
        return ProspectiveP1HistoryBatch(
            transaction_id,
            expected_version,
            batch,
            solve,
            risk,
            obsolete,
            canonical_hash(payload),
        )

    def finalize(
        self,
        prospective: ProspectiveP1HistoryBatch,
        *,
        post_commit_verified: bool,
        load_increment_by_layer: Mapping[int, float],
        fault_phase: str | None = None,
    ) -> P1HistoryFinalizeReceipt:
        if not post_commit_verified:
            raise ODEBFStateError("P1 history cannot append before post-commit verification")
        if set(load_increment_by_layer) != set(self.layer_order):
            raise ODEBFContractError("P1 history load update is incomplete")
        increments = {
            layer: finite("P1 load increment", load_increment_by_layer[layer])
            for layer in self.layer_order
        }
        if any(value < 0.0 for value in increments.values()):
            raise ODEBFContractError("P1 history load cannot decrease")
        if fault_phase not in (None, "early", "middle", "late"):
            raise ODEBFContractError("P1 history fault phase differs")
        with self._lock:
            prior = self._finalized.get(prospective.transaction_id)
            if prior is not None:
                if prior != prospective.payload_sha256:
                    raise ODEBFStateError("P1 transaction identity was reused")
                return P1HistoryFinalizeReceipt(
                    prospective.transaction_id,
                    self._version,
                    self._version,
                    0,
                    0,
                    True,
                    self.snapshot().digest,
                )
            if prospective.expected_version != self._version:
                raise ODEBFStateError("P1 history finalization version differs")
            before = self.snapshot()
            if fault_phase == "early":
                raise RuntimeError("injected early P1 history fault")
            obsolete = set(prospective.obsolete_request_sha256)
            kept_indices = [
                index
                for index, item in enumerate(self._active)
                if item.request_sha256 not in obsolete
            ]
            kept = tuple(self._active[index] for index in kept_indices)
            moved = tuple(item for item in self._active if item.request_sha256 in obsolete)
            solve_new: dict[int, torch.Tensor] = {}
            risk_new: dict[int, torch.Tensor] = {}
            for layer, incoming in prospective.solve_keys:
                existing = self._solve[layer]
                retained = (
                    existing[:, kept_indices]
                    if existing.numel() and kept_indices
                    else torch.empty((incoming.shape[0], 0), dtype=torch.float32)
                )
                solve_new[layer] = torch.cat((retained, incoming), dim=1)
            for layer, incoming in prospective.risk_keys:
                existing = self._risk[layer]
                retained = (
                    existing[:, kept_indices]
                    if existing.numel() and kept_indices
                    else torch.empty((incoming.shape[0], 0), dtype=torch.float32)
                )
                risk_new[layer] = torch.cat((retained, incoming), dim=1)
            if fault_phase == "middle":
                if self.snapshot().digest != before.digest:
                    raise ODEBFStateError("P1 prospective history mutated state")
                raise RuntimeError("injected middle P1 history fault")
            load_new = {
                layer: self._load[layer] + increments[layer] for layer in self.layer_order
            }
            finalized_new = dict(self._finalized)
            finalized_new[prospective.transaction_id] = prospective.payload_sha256
            if fault_phase == "late":
                if self.snapshot().digest != before.digest:
                    raise ODEBFStateError("P1 late history construction mutated state")
                raise RuntimeError("injected late P1 history fault")
            self._active = kept + prospective.records
            self._obsolete = self._obsolete + moved
            self._solve = solve_new
            self._risk = risk_new
            self._load = load_new
            self._finalized = finalized_new
            self._version += 1
            after = self.snapshot()
            return P1HistoryFinalizeReceipt(
                prospective.transaction_id,
                before.version,
                after.version,
                self.batch_size,
                len(obsolete),
                False,
                after.digest,
            )

    def solve_keys(self, layer: int) -> torch.Tensor:
        with self._lock:
            if layer not in self._solve:
                raise ODEBFContractError("P1 history solve layer is absent")
            return self._solve[layer].clone()

    def risk_keys(self, layer: int) -> torch.Tensor:
        with self._lock:
            if layer not in self._risk:
                raise ODEBFContractError("P1 history risk layer is absent")
            return self._risk[layer].clone()

    def cumulative_load(self) -> dict[int, float]:
        with self._lock:
            return dict(self._load)

    def rotating_records(self, *, sequential_batch: int, waypoint: int) -> tuple[P1HistoryRecord, ...]:
        with self._lock:
            if not self._active:
                return ()
            ranked = sorted(
                self._active,
                key=lambda item: hashlib.sha256(
                    f"odebf-p1-h|{sequential_batch}|{waypoint}|{item.request_sha256}".encode("utf-8")
                ).hexdigest(),
            )
            return tuple(ranked[: min(self.batch_size, len(ranked))])


@dataclass(frozen=True, slots=True)
class ArmWeightSnapshot:
    arm: P1Arm
    sequential_batch_completed: int
    parameter_sha256: tuple[tuple[str, str], ...]
    state_sha256: str


def snapshot_touched_weights(
    arm: P1Arm,
    sequential_batch_completed: int,
    parameters: Mapping[str, torch.nn.Parameter],
) -> tuple[ArmWeightSnapshot, dict[str, torch.Tensor]]:
    if not parameters:
        raise ODEBFContractError("P1 arm snapshot has no touched weights")
    values = {
        name: parameter.detach().to(device="cpu").clone()
        for name, parameter in sorted(parameters.items())
    }
    hashes = tuple((name, tensor_sha256(value)) for name, value in values.items())
    return (
        ArmWeightSnapshot(
            arm,
            sequential_batch_completed,
            hashes,
            canonical_hash({"arm": arm.value, "batch": sequential_batch_completed, "weights": hashes}),
        ),
        values,
    )


def restore_arm_snapshot(
    parameters: Mapping[str, torch.nn.Parameter],
    values: Mapping[str, torch.Tensor],
    receipt: ArmWeightSnapshot,
) -> None:
    if set(parameters) != set(values) or set(parameters) != {name for name, _ in receipt.parameter_sha256}:
        raise ODEBFContractError("P1 arm restore parameter set differs")
    pointers = {name: parameter.data_ptr() for name, parameter in parameters.items()}
    with torch.no_grad():
        for name in sorted(parameters):
            parameters[name].copy_(values[name].to(device=parameters[name].device))
    observed = tuple((name, tensor_sha256(parameters[name])) for name in sorted(parameters))
    if observed != receipt.parameter_sha256 or any(
        parameters[name].data_ptr() != pointers[name] for name in parameters
    ):
        raise ODEBFStateError("P1 arm snapshot restore was not byte/pointer exact")


@dataclass(frozen=True, slots=True)
class P1AResourceGate:
    peak_reserved_mib: float
    process_maxrss_mib: float
    elapsed_first_batch_all_arms_seconds: float
    extrapolated_total_with_reserve_seconds: float
    reserve_fraction: float
    passed: bool
    status: str


def evaluate_p1a_resource_gate(
    *,
    peak_reserved_bytes: int,
    process_maxrss_kib: int,
    elapsed_first_batch_all_arms_seconds: float,
) -> P1AResourceGate:
    if peak_reserved_bytes < 0 or process_maxrss_kib < 0:
        raise ODEBFContractError("P1A resource observation is negative")
    elapsed = finite("P1A first-batch elapsed", elapsed_first_batch_all_arms_seconds)
    if elapsed <= 0.0:
        raise ODEBFContractError("P1A first-batch elapsed is not positive")
    reserved_mib = peak_reserved_bytes / (1024.0**2)
    rss_mib = process_maxrss_kib / 1024.0
    forecast = elapsed * 4.0 * 1.10
    passed = reserved_mib < 52000.0 and rss_mib < 48000.0 and forecast < 24.0 * 3600.0
    return P1AResourceGate(
        reserved_mib,
        rss_mib,
        elapsed,
        forecast,
        0.10,
        passed,
        "P1A_CONTINUE" if passed else "P1A_RESOURCE_HOLD",
    )
