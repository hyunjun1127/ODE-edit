"""Batch-size-independent primitives for the P1R23 streaming ODE-BF runtime.

This module deliberately contains no model-specific routing policy.  It owns
the discrete scientific invariants that are easy to get wrong when a global
request batch is split into physical GPU microbatches: deterministic request
partitioning, inverse row mapping, uniform request/context weighting, Native
``1+5`` key aggregation, exact ``R0=0`` construction, accepted-state refresh
chaining, and phase-separated compute accounting.

The implementation is additive.  P1R19 and P1R22 remain immutable references.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .request_digest import ordered_request_digest_scalable_v1
from .functional import tensor_sha256


P1R23_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-SCALABLE-BATCHED-RUNTIME-P1R23-V1"
)
P1R23_METHOD_ID = "SCALABLE_STREAMING_DYNAMIC_BG_ODE_BF_V1"
P1R23_LAYER_ORDER = (4, 5, 6, 7, 8)


def scalable_ordered_request_digest(
    request_sha256: Sequence[str],
) -> str:
    """Bind P1R24 B1 smoke, legacy B10, and atomic B100 distinctly.

    The B10 branch is deliberately byte-identical to the P1R19/R13 order
    identity.  The B100 branch is distinct and cannot be confused with ten
    sequential B10 rounds.
    """

    values = tuple(request_sha256)
    if len(values) not in (1, 10, 100):
        raise ODEBFContractError("P1R23 scalable request count differs")
    try:
        return ordered_request_digest_scalable_v1(values)
    except ODEBFContractError as exc:
        raise ODEBFContractError("P1R23 scalable request identity differs") from exc
P1R23_CONTEXT_GROUP_SIZES = (1, 5)
P1R23_CONTEXTS_PER_REQUEST = sum(P1R23_CONTEXT_GROUP_SIZES)
P1R23_GRID_COUNT = 8
P1R23_H = 1.0 / P1R23_GRID_COUNT


def _finite_float(value: Any, *, label: str) -> float:
    observed = float(value)
    if not math.isfinite(observed):
        raise ODEBFContractError(f"P1R23 {label} is non-finite")
    return observed


@dataclass(frozen=True, slots=True)
class StreamingMicrobatch:
    """One deterministic physical request microbatch."""

    batch_index: int
    request_ordinals: tuple[int, ...]
    request_identity_sha256: tuple[str, ...]
    maximum_unpadded_length: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "batch_index": self.batch_index,
            "request_ordinals": list(self.request_ordinals),
            "request_identity_sha256": canonical_hash(
                list(self.request_identity_sha256)
            ),
            "request_count": len(self.request_ordinals),
            "row_count": len(self.request_ordinals)
            * P1R23_CONTEXTS_PER_REQUEST,
            "maximum_unpadded_length": self.maximum_unpadded_length,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class StreamingBatchPlan:
    """Stable length-bucket partition plus its exact inverse permutation."""

    request_count: int
    microbatch_size: int
    request_identity_sha256: tuple[str, ...]
    request_lengths: tuple[int, ...]
    bucket_order: tuple[int, ...]
    inverse_order: tuple[int, ...]
    batches: tuple[StreamingMicrobatch, ...]
    identity_sha256: str

    @property
    def physical_microbatch_count(self) -> int:
        return len(self.batches)

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r23-streaming-batch-plan/v1",
            "request_count": self.request_count,
            "contexts_per_request": P1R23_CONTEXTS_PER_REQUEST,
            "microbatch_size": self.microbatch_size,
            "physical_microbatch_count": self.physical_microbatch_count,
            "request_identity_vector_sha256": canonical_hash(
                list(self.request_identity_sha256)
            ),
            "request_length_vector_sha256": canonical_hash(
                list(self.request_lengths)
            ),
            "bucket_order": list(self.bucket_order),
            "inverse_order": list(self.inverse_order),
            "batches": [item.raw_free_payload() for item in self.batches],
            "identity_sha256": self.identity_sha256,
        }


def build_streaming_batch_plan(
    request_identity_sha256: Sequence[str],
    request_lengths: Sequence[int],
    *,
    microbatch_size: int,
) -> StreamingBatchPlan:
    """Create one model-independent deterministic request partition.

    Length is a technical scheduling attribute only.  Stable request ordinal
    breaks ties and ``inverse_order`` restores the original scientific order.
    """

    identities = tuple(str(item) for item in request_identity_sha256)
    lengths = tuple(int(item) for item in request_lengths)
    request_count = len(identities)
    if (
        request_count <= 0
        or len(lengths) != request_count
        or isinstance(microbatch_size, bool)
        or not isinstance(microbatch_size, int)
        or microbatch_size <= 0
        or microbatch_size > request_count
    ):
        raise ODEBFContractError("P1R23 streaming partition geometry differs")
    if any(len(item) != 64 for item in identities) or len(set(identities)) != request_count:
        raise ODEBFContractError("P1R23 request identity inventory differs")
    if any(item <= 0 for item in lengths):
        raise ODEBFContractError("P1R23 request length inventory differs")
    order = tuple(sorted(range(request_count), key=lambda item: (lengths[item], item)))
    inverse = [0] * request_count
    for bucket_position, request_ordinal in enumerate(order):
        inverse[request_ordinal] = bucket_position
    batches: list[StreamingMicrobatch] = []
    for start in range(0, request_count, microbatch_size):
        selected = order[start : start + microbatch_size]
        batches.append(
            StreamingMicrobatch(
                len(batches),
                selected,
                tuple(identities[item] for item in selected),
                max(lengths[item] for item in selected),
            )
        )
    flattened = tuple(
        ordinal for batch in batches for ordinal in batch.request_ordinals
    )
    if flattened != order or set(flattened) != set(range(request_count)):
        raise ODEBFStateError("P1R23 streaming partition lost or duplicated a request")
    payload = {
        "request_count": request_count,
        "microbatch_size": microbatch_size,
        "request_identity_sha256": list(identities),
        "request_lengths": list(lengths),
        "bucket_order": list(order),
        "inverse_order": inverse,
        "batch_ordinals": [list(item.request_ordinals) for item in batches],
    }
    return StreamingBatchPlan(
        request_count,
        microbatch_size,
        identities,
        lengths,
        order,
        tuple(inverse),
        tuple(batches),
        canonical_hash(payload),
    )


@dataclass(slots=True)
class UniformRequestAccumulator:
    """Accumulate request means without ever averaging microbatch means."""

    request_count: int
    _values: list[float | None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.request_count <= 0:
            raise ODEBFContractError("P1R23 accumulator request count differs")
        self._values = [None] * self.request_count

    def add(
        self,
        request_ordinals: Sequence[int],
        request_context_means: Sequence[float],
    ) -> None:
        ordinals = tuple(int(item) for item in request_ordinals)
        values = tuple(
            _finite_float(item, label="request-context mean")
            for item in request_context_means
        )
        if len(ordinals) != len(values) or len(set(ordinals)) != len(ordinals):
            raise ODEBFContractError("P1R23 accumulator microbatch geometry differs")
        for ordinal, value in zip(ordinals, values, strict=True):
            if ordinal < 0 or ordinal >= self.request_count:
                raise ODEBFContractError("P1R23 accumulator ordinal is out of range")
            if self._values[ordinal] is not None:
                raise ODEBFContractError("P1R23 accumulator duplicated a request")
            self._values[ordinal] = value

    def finalize(self) -> tuple[float, tuple[float, ...]]:
        if any(item is None for item in self._values):
            raise ODEBFContractError("P1R23 accumulator omitted a request")
        values = tuple(float(item) for item in self._values if item is not None)
        return math.fsum(values) / self.request_count, values


def uniform_context_mean(
    values: torch.Tensor,
    *,
    context_count: int = P1R23_CONTEXTS_PER_REQUEST,
) -> torch.Tensor:
    """Return per-request uniform context means from ``[request,context]``."""

    if (
        values.ndim < 2
        or values.shape[1] != context_count
        or context_count != P1R23_CONTEXTS_PER_REQUEST
        or not torch.isfinite(values).all()
    ):
        raise ODEBFContractError("P1R23 context objective geometry differs")
    return values.to(dtype=torch.float64).mean(dim=1)


def aggregate_native_one_plus_five(context_rows: torch.Tensor) -> torch.Tensor:
    """Aggregate ``[B,6,D]`` rows as 1/2 canonical + 1/2 mean(paraphrase5)."""

    if (
        context_rows.ndim != 3
        or context_rows.shape[1] != P1R23_CONTEXTS_PER_REQUEST
        or context_rows.shape[0] <= 0
        or not torch.isfinite(context_rows).all()
    ):
        raise ODEBFContractError("P1R23 key context geometry differs")
    rows64 = context_rows.to(dtype=torch.float64)
    aggregated = 0.5 * (rows64[:, 0, :] + rows64[:, 1:, :].mean(dim=1))
    return aggregated.to(dtype=torch.float32).T.contiguous()


@dataclass(frozen=True, slots=True)
class StreamingCaptureMicrobatch:
    """Already-computed raw rows returned by one physical microbatch pass."""

    request_ordinals: tuple[int, ...]
    keys_by_layer: Mapping[int, torch.Tensor]
    canonical_terminal: torch.Tensor
    model_state_before_sha256: str
    model_state_after_sha256: str
    processed_token_count: int
    padded_token_count: int


@dataclass(frozen=True, slots=True)
class StreamingPhysicalCapture:
    """Global state capture reconstructed from physical request microbatches."""

    keys_by_layer: Mapping[int, torch.Tensor]
    terminal_z: torch.Tensor
    batch_plan_sha256: str
    model_state_sha256: str
    physical_forward_count: int
    processed_token_count: int
    padded_token_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r23-streaming-physical-capture/v1",
            "layer_order": list(P1R23_LAYER_ORDER),
            "key_sha256": {
                str(layer): tensor_sha256(self.keys_by_layer[layer])
                for layer in P1R23_LAYER_ORDER
            },
            "terminal_z_sha256": tensor_sha256(self.terminal_z),
            "batch_plan_sha256": self.batch_plan_sha256,
            "model_state_sha256": self.model_state_sha256,
            "logical_capture_group_count": 1,
            "physical_forward_count": self.physical_forward_count,
            "processed_token_count": self.processed_token_count,
            "padded_token_count": self.padded_token_count,
            "identity_sha256": self.identity_sha256,
        }


def accumulate_streaming_capture(
    plan: StreamingBatchPlan,
    forward_microbatch: Callable[[StreamingMicrobatch], StreamingCaptureMicrobatch],
) -> StreamingPhysicalCapture:
    """Run one logical capture group and restore global request order.

    ``forward_microbatch`` is the only model-dependent boundary.  Behavioral
    tests therefore exercise the same accumulation/state checks as production
    without loading a model.
    """

    key_rows: dict[int, list[torch.Tensor | None]] = {
        layer: [None] * plan.request_count for layer in P1R23_LAYER_ORDER
    }
    terminal_rows: list[torch.Tensor | None] = [None] * plan.request_count
    state_identity: str | None = None
    processed = 0
    padded = 0
    for batch in plan.batches:
        observed = forward_microbatch(batch)
        if observed.request_ordinals != batch.request_ordinals:
            raise ODEBFContractError("P1R23 capture row mapping differs")
        if observed.model_state_before_sha256 != observed.model_state_after_sha256:
            raise ODEBFStateError("P1R23 model changed within a capture microbatch")
        if state_identity is None:
            state_identity = observed.model_state_before_sha256
        elif state_identity != observed.model_state_before_sha256:
            raise ODEBFStateError("P1R23 model changed between capture microbatches")
        if set(observed.keys_by_layer) != set(P1R23_LAYER_ORDER):
            raise ODEBFContractError("P1R23 capture layer inventory differs")
        local_count = len(batch.request_ordinals)
        if (
            observed.canonical_terminal.ndim != 2
            or observed.canonical_terminal.shape[0] != local_count
            or not torch.isfinite(observed.canonical_terminal).all()
        ):
            raise ODEBFContractError("P1R23 capture terminal geometry differs")
        for layer in P1R23_LAYER_ORDER:
            rows = observed.keys_by_layer[layer]
            if (
                rows.ndim != 3
                or rows.shape[:2] != (local_count, P1R23_CONTEXTS_PER_REQUEST)
                or not torch.isfinite(rows).all()
            ):
                raise ODEBFContractError("P1R23 capture key geometry differs")
            reduced = aggregate_native_one_plus_five(rows).T.contiguous()
            for local, ordinal in enumerate(batch.request_ordinals):
                if key_rows[layer][ordinal] is not None:
                    raise ODEBFContractError("P1R23 capture duplicated a key row")
                key_rows[layer][ordinal] = reduced[local].detach().cpu()
        for local, ordinal in enumerate(batch.request_ordinals):
            if terminal_rows[ordinal] is not None:
                raise ODEBFContractError("P1R23 capture duplicated a terminal row")
            terminal_rows[ordinal] = observed.canonical_terminal[local].detach().cpu()
        processed += int(observed.processed_token_count)
        padded += int(observed.padded_token_count)
    if state_identity is None:
        raise ODEBFContractError("P1R23 capture did not run")
    if any(item is None for item in terminal_rows) or any(
        any(item is None for item in key_rows[layer]) for layer in P1R23_LAYER_ORDER
    ):
        raise ODEBFContractError("P1R23 capture omitted a request")
    keys = {
        layer: torch.stack(
            [item for item in key_rows[layer] if item is not None]
        ).T.to(dtype=torch.float32).contiguous()
        for layer in P1R23_LAYER_ORDER
    }
    terminal = torch.stack(
        [item for item in terminal_rows if item is not None]
    ).T.to(dtype=torch.float32).contiguous()
    payload = {
        "key_sha256": [
            (layer, tensor_sha256(keys[layer])) for layer in P1R23_LAYER_ORDER
        ],
        "terminal_z_sha256": tensor_sha256(terminal),
        "batch_plan_sha256": plan.identity_sha256,
        "model_state_sha256": state_identity,
        "physical_forward_count": len(plan.batches),
        "processed_token_count": processed,
        "padded_token_count": padded,
    }
    return StreamingPhysicalCapture(
        keys,
        terminal,
        plan.identity_sha256,
        state_identity,
        len(plan.batches),
        processed,
        padded,
        canonical_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class InitialTargetState:
    target_z: torch.Tensor
    current_terminal_z: torch.Tensor
    residual: torch.Tensor
    capture_identity_sha256: str
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r23-initial-target/v1",
            "target_z_sha256": tensor_sha256(self.target_z),
            "current_terminal_z_sha256": tensor_sha256(self.current_terminal_z),
            "residual_sha256": tensor_sha256(self.residual),
            "capture_identity_sha256": self.capture_identity_sha256,
            "residual_exact_zero": bool(torch.count_nonzero(self.residual) == 0),
            "identity_sha256": self.identity_sha256,
        }


def initial_target_from_capture(capture: StreamingPhysicalCapture) -> InitialTargetState:
    """Clone z0 and current terminal from the exact same operator instance."""

    current = capture.terminal_z.detach().cpu().to(dtype=torch.float32).contiguous()
    if current.ndim != 2 or current.shape[1] <= 0 or not torch.isfinite(current).all():
        raise ODEBFContractError("P1R23 initial terminal geometry differs")
    target = current.clone()
    residual = (target - current).contiguous()
    if torch.count_nonzero(residual) != 0:
        raise ODEBFStateError("P1R23 initial residual is not exact zero")
    payload = {
        "target_z_sha256": tensor_sha256(target),
        "current_terminal_z_sha256": tensor_sha256(current),
        "residual_sha256": tensor_sha256(residual),
        "capture_identity_sha256": capture.identity_sha256,
        "operator_instance_shared": True,
    }
    return InitialTargetState(
        target,
        current,
        residual,
        capture.identity_sha256,
        canonical_hash(payload),
    )


class AcceptedPhysicalAdvancePolicy(str, Enum):
    EVERY_OUTER = "EVERY_OUTER"
    FINAL_Z_ONESHOT = "FINAL_Z_ONESHOT"


@dataclass(slots=True)
class DynamicRefreshLedger:
    """Prove that K8 fields are rebuilt on the accepted-state chain."""

    expected_steps: int = P1R23_GRID_COUNT
    physical_advance_policy: AcceptedPhysicalAdvancePolicy = (
        AcceptedPhysicalAdvancePolicy.EVERY_OUTER
    )
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            self.expected_steps != P1R23_GRID_COUNT
            or not isinstance(
                self.physical_advance_policy, AcceptedPhysicalAdvancePolicy
            )
        ):
            raise ODEBFContractError("P1R23 refresh policy differs")

    def record(
        self,
        *,
        step_index: int,
        accepted_state_sha256: str,
        target_sha256: str,
        key_inventory_sha256: str,
        slope_sha256: str,
        field_sha256: str,
        field_invocation_index: int,
        physical_state_advanced: bool | None = None,
    ) -> None:
        if step_index != len(self.records) or field_invocation_index != step_index + 1:
            raise ODEBFStateError("P1R23 field refresh sequence differs")
        values = (
            accepted_state_sha256,
            target_sha256,
            key_inventory_sha256,
            slope_sha256,
            field_sha256,
        )
        if any(not isinstance(item, str) or len(item) != 64 for item in values):
            raise ODEBFContractError("P1R23 field refresh identity differs")
        if self.physical_advance_policy is AcceptedPhysicalAdvancePolicy.EVERY_OUTER:
            if physical_state_advanced is not None:
                raise ODEBFContractError(
                    "legacy refresh received cadence-specific state"
                )
            if self.records and accepted_state_sha256 == self.records[-1]["accepted_state_sha256"]:
                raise ODEBFStateError("P1R23 accepted physical state did not advance")
        else:
            if not isinstance(physical_state_advanced, bool):
                raise ODEBFContractError(
                    "final-z refresh physical transition status is absent"
                )
            expected_advance = step_index == self.expected_steps - 1
            if physical_state_advanced != expected_advance:
                raise ODEBFStateError(
                    "final-z physical advance cadence differs"
                )
            if self.records:
                if accepted_state_sha256 != self.records[-1]["accepted_state_sha256"]:
                    raise ODEBFStateError(
                        "final-z stationary physical prefix differs"
                    )
                if target_sha256 == self.records[-1]["target_sha256"]:
                    raise ODEBFStateError(
                        "final-z target command did not advance"
                    )
        record = {
            "step_index": step_index,
            "field_invocation_index": field_invocation_index,
            "accepted_state_sha256": accepted_state_sha256,
            "target_sha256": target_sha256,
            "key_inventory_sha256": key_inventory_sha256,
            "slope_sha256": slope_sha256,
            "field_sha256": field_sha256,
        }
        if self.physical_advance_policy is AcceptedPhysicalAdvancePolicy.FINAL_Z_ONESHOT:
            record["physical_state_advanced"] = physical_state_advanced
        record["identity_sha256"] = canonical_hash(record)
        self.records.append(record)

    def finalize(self) -> dict[str, Any]:
        if len(self.records) != self.expected_steps:
            raise ODEBFStateError("P1R23 dynamic field count differs")
        payload = {
            "schema": "ode-edit-s05-p1r23-dynamic-refresh-ledger/v1",
            "expected_steps": self.expected_steps,
            "field_build_count": len(self.records),
            "frozen_initial_field_reuse_count": 0,
            "static_split_count": 0,
            "records": self.records,
        }
        if self.physical_advance_policy is AcceptedPhysicalAdvancePolicy.FINAL_Z_ONESHOT:
            stationary_count = sum(
                int(not bool(item["physical_state_advanced"]))
                for item in self.records
            )
            advance_count = sum(
                int(bool(item["physical_state_advanced"]))
                for item in self.records
            )
            if stationary_count != self.expected_steps - 1 or advance_count != 1:
                raise ODEBFStateError("final-z refresh terminal cadence differs")
            payload.update(
                {
                    "physical_advance_policy": self.physical_advance_policy.value,
                    "stationary_physical_state_count": stationary_count,
                    "physical_advance_count": advance_count,
                    "target_command_advance_count": self.expected_steps - 1,
                }
            )
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


_COMPUTE_COUNTERS = (
    "logical_forward_groups",
    "model_forward_calls",
    "physical_microbatch_graphs",
    "autograd_invocations",
    "backward_calls",
    "processed_tokens",
    "padded_tokens",
    "target_backward_calls",
    "slope_backward_calls",
    "capture_forward_calls",
    "materialization_count",
    "dense_assembly_count",
    "full_weight_hash_count",
    "hot_hook_dense_assembly_count",
    "hot_hook_full_weight_hash_count",
)


@dataclass(slots=True)
class ScalableComputeLedger:
    """Phase-aware logical/physical accounting for B-independent execution."""

    phases: dict[str, dict[str, int]] = field(default_factory=dict)
    wall_seconds: dict[str, float] = field(default_factory=dict)

    def increment(self, phase: str, **values: int) -> None:
        if not phase:
            raise ODEBFContractError("P1R23 compute phase is absent")
        target = self.phases.setdefault(
            phase, {name: 0 for name in _COMPUTE_COUNTERS}
        )
        for name, value in values.items():
            if name not in target or isinstance(value, bool) or int(value) < 0:
                raise ODEBFContractError("P1R23 compute counter differs")
            target[name] += int(value)

    def add_wall(self, phase: str, seconds: float) -> None:
        observed = _finite_float(seconds, label="phase wall time")
        if observed < 0.0:
            raise ODEBFContractError("P1R23 phase wall time is negative")
        self.wall_seconds[phase] = self.wall_seconds.get(phase, 0.0) + observed

    def totals(self) -> dict[str, int]:
        return {
            name: sum(values[name] for values in self.phases.values())
            for name in _COMPUTE_COUNTERS
        }

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "ode-edit-s05-p1r23-compute-ledger/v1",
            "phases": {key: self.phases[key] for key in sorted(self.phases)},
            "wall_seconds": {
                key: self.wall_seconds[key] for key in sorted(self.wall_seconds)
            },
            "totals": self.totals(),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


def expected_streaming_gradient_counts(
    *,
    request_count: int,
    microbatch_size: int,
    grid_count: int = P1R23_GRID_COUNT,
) -> dict[str, int]:
    """Return honest logical versus physical K8 target/slope accounting."""

    if request_count <= 0 or microbatch_size <= 0 or grid_count <= 0:
        raise ODEBFContractError("P1R23 compute geometry differs")
    q = math.ceil(request_count / microbatch_size)
    return {
        "request_count": request_count,
        "microbatch_size": microbatch_size,
        "physical_microbatch_count_per_group": q,
        "logical_target_gradient_groups": grid_count,
        "logical_slope_gradient_groups": grid_count,
        "physical_target_forward_graphs": grid_count * q,
        "physical_target_backward_calls": grid_count * q,
        "physical_slope_forward_graphs": grid_count * q,
        "physical_slope_backward_calls": grid_count * q,
        "initial_capture_forward_calls": q,
        "terminal_objective_forward_calls": q,
        "accepted_materialization_count": grid_count,
    }


__all__ = [
    "DynamicRefreshLedger",
    "InitialTargetState",
    "P1R23_CONTEXTS_PER_REQUEST",
    "P1R23_CONTEXT_GROUP_SIZES",
    "P1R23_GRID_COUNT",
    "P1R23_H",
    "P1R23_INSTRUCTION_ID",
    "P1R23_LAYER_ORDER",
    "P1R23_METHOD_ID",
    "ScalableComputeLedger",
    "StreamingBatchPlan",
    "StreamingCaptureMicrobatch",
    "StreamingMicrobatch",
    "StreamingPhysicalCapture",
    "UniformRequestAccumulator",
    "accumulate_streaming_capture",
    "aggregate_native_one_plus_five",
    "build_streaming_batch_plan",
    "expected_streaming_gradient_counts",
    "initial_target_from_capture",
    "uniform_context_mean",
]
