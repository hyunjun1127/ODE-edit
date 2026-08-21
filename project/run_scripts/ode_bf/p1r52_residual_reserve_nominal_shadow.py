"""Uniform nominal residual-reserve shadow sweep with exact native restore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r52_residual_reserve_fp32_transaction import (
    FP32TransactionMode,
    FP32TransactionReceipt,
    LayerParameterBinding,
    OfficialStyleFP32SequentialTransaction,
    RESIDUAL_RESERVE_LAYER_ORDER,
)
from .p1r52_residual_reserve_geometry import (
    ResidualReserveGeometry,
    build_residual_reserve_geometry,
)
from .p1r52_residual_reserve_layer_step import (
    SHADOW_APPLICATION_PROOF_STATUS,
    NominalReferenceFactorResult,
    ResidualReserveLayerStepPlan,
    derive_nominal_reference_factor,
    plan_residual_reserve_layer_step,
)
from .p1r52_residual_reserve_pc_inventory import LowRankFP32Factor
from .p1r52_residual_reserve_update_binding import (
    AppliedPreparedLowRankUpdate,
    apply_prepared_low_rank_update,
)


_UNIFORM_NOMINAL_PI_FP32 = float(torch.tensor(0.2, dtype=torch.float32))
UNIFORM_NOMINAL_PI = (_UNIFORM_NOMINAL_PI_FP32,) * 5
SHADOW_PROOF_STATUS = "CLOSED_BY_M3D_B2B1_SHADOW_APPLY_AND_RESTORE"
PROCESS_LOCAL_IDENTITY_EXCLUSIONS = (
    "overall.transaction_id",
    "target.pointer",
    "entry_weight_state[*].pointer",
    "restored_weight_state[*].pointer",
    "layers[*].observation.pre_observation_weight_state[*].pointer",
    "layers[*].observation.terminal_identity.pointer",
    "layers[*].observation.joint_keys_identity.pointer",
    "layers[*].observation.identity_sha256",
    "layers[*].q_solve.input_identities[*].pointer",
    "layers[*].q_solve.identity_sha256",
    "layers[*].plan.input_identities[*].pointer",
    "layers[*].plan.q_solve_receipt_identity",
    "layers[*].plan.construction_receipt_identity",
    "layers[*].plan.identity_sha256",
    "layers[*].construction.raw_update_pointer",
    "layers[*].construction.matched_update_pointer",
    "layers[*].construction.identity_sha256",
    "layers[*].application.construction_receipt_identity",
    "layers[*].application.matched_update_pointer",
    "layers[*].application.m3a_layer_receipt_identity",
    "layers[*].application.identity_sha256",
    "layers[*].m3a_layer.entry_parameter_pointer",
    "layers[*].m3a_layer.prepared_fp32_update_pointer",
    "layers[*].m3a_layer.post_storage_parameter_pointer",
    "layers[*].nominal.layer_step_receipt_identity",
    "layers[*].nominal.construction_receipt_identity",
    "layers[*].nominal.identity_sha256",
    "layers[*].closure.observation_receipt_identity",
    "layers[*].closure.layer_step_receipt_identity",
    "layers[*].closure.nominal_factor_receipt_identity",
    "layers[*].closure.construction_receipt_identity",
    "layers[*].closure.application_receipt_identity",
    "layers[*].closure.transaction_layer_receipt_identity",
    "layers[*].closure.identity_sha256",
    "transaction.transaction_id",
    "transaction.layer_receipts[*].entry_parameter_pointer",
    "transaction.layer_receipts[*].prepared_fp32_update_pointer",
    "transaction.layer_receipts[*].post_storage_parameter_pointer",
    "transaction.final_parameter_pointers",
    "transaction.identity_sha256",
)


@dataclass(frozen=True, slots=True)
class ShadowWeightStateIdentity:
    layer: int
    weight_name: str
    shape: tuple[int, int]
    dtype: str
    device: str
    sha256: str
    pointer: int

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "weight_name": self.weight_name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "sha256": self.sha256,
            "pointer": self.pointer,
        }

    def scientific_payload(self) -> dict[str, Any]:
        payload = self.raw_free_payload()
        payload.pop("pointer")
        return payload


@dataclass(frozen=True, slots=True)
class ShadowTensorIdentity:
    shape: tuple[int, ...]
    dtype: str
    device: str
    sha256: str
    pointer: int
    version: int
    requires_grad: bool

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "sha256": self.sha256,
            "pointer": self.pointer,
            "version": self.version,
            "requires_grad": self.requires_grad,
        }

    def scientific_payload(self) -> dict[str, Any]:
        payload = self.raw_free_payload()
        payload.pop("pointer")
        return payload


@dataclass(frozen=True, slots=True)
class PrefixObservationReceipt:
    layer: int
    pre_observation_weight_state: tuple[ShadowWeightStateIdentity, ...]
    terminal_identity: ShadowTensorIdentity
    joint_keys_identity: ShadowTensorIdentity
    terminal_forward_count: int
    key_forward_count: int
    model_backward_count: int
    semantic_backward_count: int
    slope_backward_count: int
    action_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": self.layer,
            "pre_observation_weight_state": [
                item.raw_free_payload()
                for item in self.pre_observation_weight_state
            ],
            "terminal_identity": self.terminal_identity.raw_free_payload(),
            "joint_keys_identity": self.joint_keys_identity.raw_free_payload(),
            "terminal_forward_count": self.terminal_forward_count,
            "key_forward_count": self.key_forward_count,
            "model_backward_count": self.model_backward_count,
            "semantic_backward_count": self.semantic_backward_count,
            "slope_backward_count": self.slope_backward_count,
            "action_influence_count": self.action_influence_count,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "pre_observation_weight_state": [
                item.scientific_payload()
                for item in self.pre_observation_weight_state
            ],
            "terminal_identity": self.terminal_identity.scientific_payload(),
            "joint_keys_identity": self.joint_keys_identity.scientific_payload(),
            "terminal_forward_count": self.terminal_forward_count,
            "key_forward_count": self.key_forward_count,
            "model_backward_count": self.model_backward_count,
            "semantic_backward_count": self.semantic_backward_count,
            "slope_backward_count": self.slope_backward_count,
            "action_influence_count": self.action_influence_count,
        }


@dataclass(frozen=True, slots=True)
class PrefixObservation:
    current_terminal32: torch.Tensor
    joint_keys32: torch.Tensor
    receipt: PrefixObservationReceipt


@dataclass(frozen=True, slots=True)
class ShadowAlphaLayerContext:
    layer: int
    weight_name: str
    parameter_shape: tuple[int, int]
    projector32: torch.Tensor
    committed_covariance32: torch.Tensor
    regularization32: torch.Tensor
    q_residual_tolerance: float
    q_condition_max_dimension: int
    construction_id: str


@dataclass(frozen=True, slots=True)
class NominalShadowLayerReceipt:
    layer: int
    observation_receipt_identity: str
    layer_step_receipt_identity: str
    nominal_factor_receipt_identity: str
    nominal_factor_identity: str
    construction_receipt_identity: str
    application_receipt_identity: str
    transaction_layer_receipt_identity: str
    shadow_proof_status: str
    b2a_input_shadow_proof_status: str
    exact_construction_apply_count: int
    authoritative_transaction_commit_claim_count: int
    post_storage_decision_influence_count: int
    pre_observation_weight_hashes: tuple[tuple[int, str], ...]
    terminal_sha256: str
    joint_keys_sha256: str
    residual_sha256: str
    q_sha256: str
    beta_applied_left_sha256: str
    matched_update_sha256: str
    post_storage_parameter_sha256: str
    router_call_count: int
    heldout_evaluator_count: int
    external_materializer_count: int
    ledger_append_count: int
    history_append_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": self.layer,
            "observation_receipt_identity": self.observation_receipt_identity,
            "layer_step_receipt_identity": self.layer_step_receipt_identity,
            "nominal_factor_receipt_identity": (
                self.nominal_factor_receipt_identity
            ),
            "nominal_factor_identity": self.nominal_factor_identity,
            "construction_receipt_identity": self.construction_receipt_identity,
            "application_receipt_identity": self.application_receipt_identity,
            "transaction_layer_receipt_identity": (
                self.transaction_layer_receipt_identity
            ),
            "shadow_proof_status": self.shadow_proof_status,
            "b2a_input_shadow_proof_status": (
                self.b2a_input_shadow_proof_status
            ),
            "exact_construction_apply_count": (
                self.exact_construction_apply_count
            ),
            "authoritative_transaction_commit_claim_count": (
                self.authoritative_transaction_commit_claim_count
            ),
            "post_storage_decision_influence_count": (
                self.post_storage_decision_influence_count
            ),
            "pre_observation_weight_hashes": [
                [layer, sha256]
                for layer, sha256 in self.pre_observation_weight_hashes
            ],
            "terminal_sha256": self.terminal_sha256,
            "joint_keys_sha256": self.joint_keys_sha256,
            "residual_sha256": self.residual_sha256,
            "q_sha256": self.q_sha256,
            "beta_applied_left_sha256": self.beta_applied_left_sha256,
            "matched_update_sha256": self.matched_update_sha256,
            "post_storage_parameter_sha256": (
                self.post_storage_parameter_sha256
            ),
            "router_call_count": self.router_call_count,
            "heldout_evaluator_count": self.heldout_evaluator_count,
            "external_materializer_count": self.external_materializer_count,
            "ledger_append_count": self.ledger_append_count,
            "history_append_count": self.history_append_count,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class NominalShadowLayerResult:
    observation: PrefixObservation
    plan: ResidualReserveLayerStepPlan
    application: AppliedPreparedLowRankUpdate
    nominal: NominalReferenceFactorResult
    receipt: NominalShadowLayerReceipt


@dataclass(frozen=True, slots=True)
class NominalShadowDerivedLedger:
    terminal_forward_count: int
    key_forward_count: int
    q_solve_count: int
    dense_update_construction_count: int
    temporary_native_apply_count: int
    native_storage_assignment_count: int
    storage_cast_boundary_count: int
    restore_count: int
    model_backward_count: int
    semantic_backward_count: int
    slope_backward_count: int
    post_storage_decision_influence_count: int
    router_call_count: int
    heldout_evaluator_count: int
    external_materializer_count: int
    candidate_materialization_count: int
    ledger_append_count: int
    history_append_count: int
    logical_outer_commit_count: int
    persistent_commit_count: int

    def raw_free_payload(self) -> dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class NominalShadowReceipt:
    transaction_id: str
    transaction_mode: str
    uniform_pi: tuple[float, ...]
    uniform_pi_sha256: str
    geometry_receipt_identity: str
    target_identity: ShadowTensorIdentity
    entry_weight_state: tuple[ShadowWeightStateIdentity, ...]
    restored_weight_state: tuple[ShadowWeightStateIdentity, ...]
    layer_receipts: tuple[NominalShadowLayerReceipt, ...]
    transaction_receipt_identity: str
    shadow_proof_status: str
    native_storage_assignment_count: int
    storage_cast_boundary_count: int
    restore_count: int
    logical_outer_commit_count: int
    persistent_commit_count: int
    external_materializer_count: int
    candidate_materialization_count: int
    ledger_append_count: int
    history_append_count: int
    terminal_forward_count: int
    key_forward_count: int
    q_solve_count: int
    dense_update_construction_count: int
    temporary_native_apply_count: int
    model_backward_count: int
    semantic_backward_count: int
    slope_backward_count: int
    router_call_count: int
    heldout_evaluator_count: int
    post_storage_decision_influence_count: int
    process_local_identity_exclusions: tuple[str, ...]
    scientific_identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "transaction_id": self.transaction_id,
            "transaction_mode": self.transaction_mode,
            "uniform_pi": list(self.uniform_pi),
            "uniform_pi_sha256": self.uniform_pi_sha256,
            "geometry_receipt_identity": self.geometry_receipt_identity,
            "target_identity": self.target_identity.raw_free_payload(),
            "entry_weight_state": [
                item.raw_free_payload() for item in self.entry_weight_state
            ],
            "restored_weight_state": [
                item.raw_free_payload() for item in self.restored_weight_state
            ],
            "layer_receipts": [
                item.raw_free_payload() for item in self.layer_receipts
            ],
            "transaction_receipt_identity": self.transaction_receipt_identity,
            "shadow_proof_status": self.shadow_proof_status,
            "native_storage_assignment_count": (
                self.native_storage_assignment_count
            ),
            "storage_cast_boundary_count": self.storage_cast_boundary_count,
            "restore_count": self.restore_count,
            "logical_outer_commit_count": self.logical_outer_commit_count,
            "persistent_commit_count": self.persistent_commit_count,
            "external_materializer_count": self.external_materializer_count,
            "candidate_materialization_count": (
                self.candidate_materialization_count
            ),
            "ledger_append_count": self.ledger_append_count,
            "history_append_count": self.history_append_count,
            "terminal_forward_count": self.terminal_forward_count,
            "key_forward_count": self.key_forward_count,
            "q_solve_count": self.q_solve_count,
            "dense_update_construction_count": (
                self.dense_update_construction_count
            ),
            "temporary_native_apply_count": self.temporary_native_apply_count,
            "model_backward_count": self.model_backward_count,
            "semantic_backward_count": self.semantic_backward_count,
            "slope_backward_count": self.slope_backward_count,
            "router_call_count": self.router_call_count,
            "heldout_evaluator_count": self.heldout_evaluator_count,
            "post_storage_decision_influence_count": (
                self.post_storage_decision_influence_count
            ),
            "process_local_identity_exclusions": list(
                self.process_local_identity_exclusions
            ),
            "scientific_identity_sha256": self.scientific_identity_sha256,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class NominalShadowResult:
    geometry: ResidualReserveGeometry
    factors: tuple[LowRankFP32Factor, ...]
    layer_results: tuple[NominalShadowLayerResult, ...]
    transaction_receipt: FP32TransactionReceipt
    receipt: NominalShadowReceipt


PrefixObservationProvider = Callable[
    [int, tuple[ShadowWeightStateIdentity, ...]], PrefixObservation
]


def _tensor_identity(value: torch.Tensor) -> ShadowTensorIdentity:
    return ShadowTensorIdentity(
        shape=tuple(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        sha256=tensor_sha256(value),
        pointer=int(value.data_ptr()),
        version=int(value._version),
        requires_grad=bool(value.requires_grad),
    )


def _resolve_bindings(
    bindings: Mapping[int, LayerParameterBinding],
) -> dict[int, LayerParameterBinding]:
    if set(bindings) != set(RESIDUAL_RESERVE_LAYER_ORDER):
        raise ODEBFContractError("nominal shadow binding inventory differs")
    resolved: dict[int, LayerParameterBinding] = {}
    for layer in RESIDUAL_RESERVE_LAYER_ORDER:
        binding = bindings[layer]
        if (
            not isinstance(binding, LayerParameterBinding)
            or binding.layer != layer
            or not isinstance(binding.parameter, torch.nn.Parameter)
        ):
            raise ODEBFContractError("nominal shadow layer binding differs")
        resolved[layer] = binding
    return resolved


def _weight_state(
    bindings: Mapping[int, LayerParameterBinding],
) -> tuple[ShadowWeightStateIdentity, ...]:
    result = []
    for layer in RESIDUAL_RESERVE_LAYER_ORDER:
        binding = bindings[layer]
        value = binding.parameter
        result.append(
            ShadowWeightStateIdentity(
                layer=layer,
                weight_name=binding.weight_name,
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                sha256=tensor_sha256(value),
                pointer=int(value.data_ptr()),
            )
        )
    return tuple(result)


def build_prefix_observation(
    *,
    layer: int,
    pre_observation_weight_state: tuple[ShadowWeightStateIdentity, ...],
    current_terminal32: torch.Tensor,
    joint_keys32: torch.Tensor,
    terminal_forward_count: int = 1,
    key_forward_count: int = 1,
    model_backward_count: int = 0,
) -> PrefixObservation:
    """Build a typed callback result without embedding model logic."""

    if layer not in RESIDUAL_RESERVE_LAYER_ORDER:
        raise ODEBFContractError("prefix observation layer differs")
    if (
        len(pre_observation_weight_state) != len(RESIDUAL_RESERVE_LAYER_ORDER)
        or tuple(item.layer for item in pre_observation_weight_state)
        != RESIDUAL_RESERVE_LAYER_ORDER
        or any(
            not isinstance(item, ShadowWeightStateIdentity)
            for item in pre_observation_weight_state
        )
    ):
        raise ODEBFContractError("prefix observation weight state differs")
    for name, value in (
        ("terminal", current_terminal32),
        ("joint keys", joint_keys32),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype is not torch.float32
            or value.ndim != 2
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
        ):
            raise ODEBFContractError(f"prefix observation {name} differs")
    if current_terminal32.device != joint_keys32.device:
        raise ODEBFContractError("prefix observation devices differ")
    if (
        terminal_forward_count != 1
        or key_forward_count != 1
        or model_backward_count != 0
    ):
        raise ODEBFContractError("prefix observation compute ledger differs")
    terminal = current_terminal32.detach().clone().contiguous()
    keys = joint_keys32.detach().clone().contiguous()
    receipt = PrefixObservationReceipt(
        layer=layer,
        pre_observation_weight_state=pre_observation_weight_state,
        terminal_identity=_tensor_identity(terminal),
        joint_keys_identity=_tensor_identity(keys),
        terminal_forward_count=1,
        key_forward_count=1,
        model_backward_count=0,
        semantic_backward_count=0,
        slope_backward_count=0,
        action_influence_count=0,
    )
    receipt.identity_sha256
    return PrefixObservation(terminal, keys, receipt)


def _validate_observation(
    observation: PrefixObservation,
    *,
    layer: int,
    expected_weight_state: tuple[ShadowWeightStateIdentity, ...],
    device: torch.device,
) -> None:
    if (
        not isinstance(observation, PrefixObservation)
        or not isinstance(observation.receipt, PrefixObservationReceipt)
    ):
        raise ODEBFContractError("prefix observation result type differs")
    receipt = observation.receipt
    terminal = observation.current_terminal32
    keys = observation.joint_keys32
    if (
        receipt.layer != layer
        or receipt.pre_observation_weight_state != expected_weight_state
        or receipt.terminal_forward_count != 1
        or receipt.key_forward_count != 1
        or receipt.model_backward_count != 0
        or receipt.semantic_backward_count != 0
        or receipt.slope_backward_count != 0
        or receipt.action_influence_count != 0
        or terminal.dtype is not torch.float32
        or keys.dtype is not torch.float32
        or terminal.device != device
        or keys.device != device
        or terminal.ndim != 2
        or keys.ndim != 2
        or terminal.requires_grad
        or keys.requires_grad
        or not bool(torch.isfinite(terminal).all())
        or not bool(torch.isfinite(keys).all())
        or receipt.terminal_identity != _tensor_identity(terminal)
        or receipt.joint_keys_identity != _tensor_identity(keys)
    ):
        raise ODEBFStateError("prefix observation receipt binding differs")


def _context_guard(context: ShadowAlphaLayerContext) -> tuple[Any, ...]:
    return (
        context.layer,
        context.weight_name,
        context.parameter_shape,
        tensor_sha256(context.projector32),
        int(context.projector32.data_ptr()),
        int(context.projector32._version),
        tensor_sha256(context.committed_covariance32),
        int(context.committed_covariance32.data_ptr()),
        int(context.committed_covariance32._version),
        tensor_sha256(context.regularization32),
        int(context.regularization32.data_ptr()),
        int(context.regularization32._version),
        context.q_residual_tolerance,
        context.q_condition_max_dimension,
        context.construction_id,
    )


def _validate_contexts(
    contexts: Mapping[int, ShadowAlphaLayerContext],
    bindings: Mapping[int, LayerParameterBinding],
    *,
    device: torch.device,
) -> tuple[tuple[Any, ...], ...]:
    if set(contexts) != set(RESIDUAL_RESERVE_LAYER_ORDER):
        raise ODEBFContractError("nominal shadow alpha context inventory differs")
    guards = []
    ids: set[str] = set()
    for layer in RESIDUAL_RESERVE_LAYER_ORDER:
        context = contexts[layer]
        binding = bindings[layer]
        if (
            not isinstance(context, ShadowAlphaLayerContext)
            or context.layer != layer
            or context.weight_name != binding.weight_name
            or context.parameter_shape != tuple(binding.parameter.shape)
            or not isinstance(context.construction_id, str)
            or not context.construction_id
            or context.construction_id in ids
        ):
            raise ODEBFContractError("nominal shadow alpha context differs")
        ids.add(context.construction_id)
        for value in (
            context.projector32,
            context.committed_covariance32,
            context.regularization32,
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype is not torch.float32
                or value.device != device
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
            ):
                raise ODEBFContractError("nominal shadow alpha tensor differs")
        guards.append(_context_guard(context))
    return tuple(guards)


def _drop(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    projected = dict(payload)
    for key in keys:
        projected.pop(key, None)
    return projected


def _project_observation(receipt: PrefixObservationReceipt) -> dict[str, Any]:
    payload = _drop(receipt.raw_free_payload(), "identity_sha256")
    payload["pre_observation_weight_state"] = [
        item.scientific_payload()
        for item in receipt.pre_observation_weight_state
    ]
    payload["terminal_identity"] = receipt.terminal_identity.scientific_payload()
    payload["joint_keys_identity"] = (
        receipt.joint_keys_identity.scientific_payload()
    )
    return payload


def _project_q_solve(result: NominalShadowLayerResult) -> dict[str, Any]:
    payload = _drop(
        result.plan.q_solve.receipt.raw_free_payload(),
        "identity_sha256",
    )
    payload["input_identities"] = [
        _drop(item, "pointer") for item in payload["input_identities"]
    ]
    return payload


def _project_layer_step(result: NominalShadowLayerResult) -> dict[str, Any]:
    payload = _drop(
        result.plan.receipt.raw_free_payload(),
        "q_solve_receipt_identity",
        "construction_receipt_identity",
        "identity_sha256",
    )
    payload["input_identities"] = [
        _drop(item, "pointer") for item in payload["input_identities"]
    ]
    return payload


def _project_construction(result: NominalShadowLayerResult) -> dict[str, Any]:
    return _drop(
        result.plan.prepared_update.receipt.raw_free_payload(),
        "raw_update_pointer",
        "matched_update_pointer",
        "identity_sha256",
    )


def _project_m3a_layer(result: NominalShadowLayerResult) -> dict[str, Any]:
    return _drop(
        result.application.m3a_layer_receipt.raw_free_payload(),
        "entry_parameter_pointer",
        "prepared_fp32_update_pointer",
        "post_storage_parameter_pointer",
    )


def _project_application(result: NominalShadowLayerResult) -> dict[str, Any]:
    return _drop(
        result.application.raw_free_payload(),
        "construction_receipt_identity",
        "matched_update_pointer",
        "m3a_layer_receipt_identity",
        "identity_sha256",
    )


def _project_nominal(result: NominalShadowLayerResult) -> dict[str, Any]:
    return _drop(
        result.nominal.receipt.raw_free_payload(),
        "layer_step_receipt_identity",
        "construction_receipt_identity",
        "identity_sha256",
    )


def _project_closure(result: NominalShadowLayerResult) -> dict[str, Any]:
    return _drop(
        result.receipt.raw_free_payload(),
        "observation_receipt_identity",
        "layer_step_receipt_identity",
        "nominal_factor_receipt_identity",
        "construction_receipt_identity",
        "application_receipt_identity",
        "transaction_layer_receipt_identity",
        "identity_sha256",
    )


def _project_transaction(receipt: FP32TransactionReceipt) -> dict[str, Any]:
    payload = _drop(
        receipt.raw_free_payload(),
        "transaction_id",
        "final_parameter_pointers",
        "identity_sha256",
    )
    payload["layer_receipts"] = [
        _drop(
            item.raw_free_payload(),
            "entry_parameter_pointer",
            "prepared_fp32_update_pointer",
            "post_storage_parameter_pointer",
        )
        for item in receipt.layer_receipts
    ]
    return payload


def _stable_scientific_projection(
    *,
    geometry: ResidualReserveGeometry,
    target: ShadowTensorIdentity,
    entry: tuple[ShadowWeightStateIdentity, ...],
    restored: tuple[ShadowWeightStateIdentity, ...],
    layer_results: tuple[NominalShadowLayerResult, ...],
    transaction: FP32TransactionReceipt,
) -> dict[str, Any]:
    return {
        "uniform_pi": list(UNIFORM_NOMINAL_PI),
        "geometry": geometry.receipt.raw_free_payload(),
        "target": target.scientific_payload(),
        "entry": [item.scientific_payload() for item in entry],
        "restored": [item.scientific_payload() for item in restored],
        "layers": [
            {
                "observation": _project_observation(item.observation.receipt),
                "q_solve": _project_q_solve(item),
                "plan": _project_layer_step(item),
                "construction": _project_construction(item),
                "prepared_factor": (
                    item.plan.prepared_update.factor.raw_free_payload()
                ),
                "application": _project_application(item),
                "m3a_layer": _project_m3a_layer(item),
                "nominal": _project_nominal(item),
                "nominal_factor": item.nominal.factor.raw_free_payload(),
                "closure": _project_closure(item),
            }
            for item in layer_results
        ],
        "transaction": _project_transaction(transaction),
        "process_local_identity_exclusions": list(
            PROCESS_LOCAL_IDENTITY_EXCLUSIONS
        ),
    }


def _stable_scientific_identity(
    *,
    geometry: ResidualReserveGeometry,
    target: ShadowTensorIdentity,
    entry: tuple[ShadowWeightStateIdentity, ...],
    restored: tuple[ShadowWeightStateIdentity, ...],
    layer_results: tuple[NominalShadowLayerResult, ...],
    transaction: FP32TransactionReceipt,
) -> str:
    return canonical_hash(
        _stable_scientific_projection(
            geometry=geometry,
            target=target,
            entry=entry,
            restored=restored,
            layer_results=layer_results,
            transaction=transaction,
        )
    )


def _validate_nested_shadow_contract(
    layer_results: tuple[NominalShadowLayerResult, ...],
    transaction: FP32TransactionReceipt,
) -> None:
    """Validate every nested compute/influence receipt before aggregation."""

    if (
        tuple(item.receipt.layer for item in layer_results)
        != RESIDUAL_RESERVE_LAYER_ORDER
        or len(transaction.layer_receipts) != len(layer_results)
    ):
        raise ODEBFStateError("nominal shadow nested layer inventory differs")
    for item, transaction_layer in zip(
        layer_results,
        transaction.layer_receipts,
        strict=True,
    ):
        observation = item.observation.receipt
        q_receipt = item.plan.q_solve.receipt
        plan_receipt = item.plan.receipt
        construction = item.plan.prepared_update.receipt
        application = item.application
        m3a_layer = application.m3a_layer_receipt
        nominal = item.nominal.receipt
        closure = item.receipt
        application.validate()
        expected_fp64_scalar_reductions = (
            3 if q_receipt.condition_estimate is not None else 2
        )
        if (
            observation.terminal_forward_count != 1
            or observation.key_forward_count != 1
            or observation.model_backward_count != 0
            or observation.semantic_backward_count != 0
            or observation.slope_backward_count != 0
            or observation.action_influence_count != 0
            or q_receipt.logical_q_solve_count != 1
            or q_receipt.dense_writer_update_construction_count != 0
            or q_receipt.fp64_algorithm_tensor_count != 0
            or q_receipt.fp64_scalar_reduction_count
            != expected_fp64_scalar_reductions
            or q_receipt.model_forward_count != 0
            or q_receipt.model_backward_count != 0
            or q_receipt.semantic_backward_count != 0
            or q_receipt.slope_backward_count != 0
            or q_receipt.materialization_count != 0
            or plan_receipt.beta_application_count != 1
            or plan_receipt.internal_beta_reapplication_count != 0
            or plan_receipt.q_solve_count != q_receipt.logical_q_solve_count
            or plan_receipt.dense_construction_count != 1
            or plan_receipt.second_dense_construction_count != 0
            or plan_receipt.algorithm_tensor_requires_grad_count != 0
            or plan_receipt.model_forward_count != 0
            or plan_receipt.model_backward_count != 0
            or plan_receipt.semantic_backward_count != 0
            or plan_receipt.slope_backward_count != 0
            or plan_receipt.apply_count != 0
            or plan_receipt.commit_count != 0
            or plan_receipt.materialization_count != 0
            or construction.official_matcher_call_count != 1
            or construction.dense_construction_count != 1
            or construction.second_dense_construction_count != 0
            or construction.fp64_algorithm_tensor_count != 0
            or construction.internal_beta_reapplication_count != 0
            or not construction.factor_input_immutability_verified
            or not construction.factor_input_nonalias_verified
            or construction.postcast_decision_influence_count != 0
            or application.exact_object_apply_count != 1
            or not application.input_pointer_unchanged
            or not application.input_version_unchanged
            or m3a_layer.storage_assignment_count != 1
            or m3a_layer.storage_cast_boundary_count != 1
            or m3a_layer.numeric_storage_cast_count
            != int(m3a_layer.storage_cast_required)
            or not m3a_layer.official_reference_endpoint_byte_exact
            or m3a_layer.official_reference_endpoint_sha256
            != m3a_layer.post_storage_parameter_sha256
            or m3a_layer.rounding_telemetry_decision_influence_count != 0
            or m3a_layer.bf16_path_call_count != 0
            or m3a_layer.bf16_path_decision_influence_count != 0
            or m3a_layer.autocast_count != 0
            or m3a_layer.downcast_count != 0
            or m3a_layer.quantization_count != 0
            or m3a_layer.postcast_decision_influence_count != 0
            or nominal.shadow_application_proof_status
            != SHADOW_APPLICATION_PROOF_STATUS
            or nominal.authoritative_transaction_committed_claim
            or nominal.division_count != 1
            or nominal.dense_factor_materialization_count != 0
            or nominal.dense_update_construction_count != 0
            or nominal.second_update_construction_count != 0
            or not nominal.quota_byte_exact_to_geometry_omega
            or not nominal.source_tensor_immutability_verified
            or not nominal.input_pointer_version_hash_immutability_verified
            or not nominal.nominal_factor_nonalias_verified
            or closure.shadow_proof_status != SHADOW_PROOF_STATUS
            or closure.b2a_input_shadow_proof_status
            != SHADOW_APPLICATION_PROOF_STATUS
            or closure.exact_construction_apply_count
            != application.exact_object_apply_count
            or closure.authoritative_transaction_commit_claim_count != 0
            or closure.post_storage_decision_influence_count
            != m3a_layer.postcast_decision_influence_count
            or closure.router_call_count != 0
            or closure.heldout_evaluator_count != 0
            or closure.external_materializer_count != 0
            or closure.ledger_append_count != 0
            or closure.history_append_count != 0
            or m3a_layer != transaction_layer
            or closure.transaction_layer_receipt_identity
            != transaction_layer.identity_sha256
        ):
            raise ODEBFStateError("nominal shadow nested contract differs")
    if (
        transaction.mode != FP32TransactionMode.SHADOW.value
        or transaction.layer_order != RESIDUAL_RESERVE_LAYER_ORDER
        or transaction.native_storage_assignment_count != 5
        or transaction.storage_cast_boundary_count != 5
        or transaction.numeric_storage_cast_count
        != sum(
            item.numeric_storage_cast_count
            for item in transaction.layer_receipts
        )
        or transaction.rounding_observation_count != 5
        or transaction.rounding_mismatch_count
        != sum(
            not item.prepared_vs_actual_byte_exact
            for item in transaction.layer_receipts
        )
        or transaction.bf16_path_call_count != 0
        or transaction.bf16_path_decision_influence_count != 0
        or transaction.logical_outer_commit_count != 0
        or transaction.persistent_commit_count != 0
        or transaction.rollback_count != 1
        or not transaction.restored_entry_bytes
        or not transaction.restored_entry_pointers
        or transaction.postcast_decision_influence_count != 0
        or transaction.model_forward_count != 0
        or transaction.model_backward_count != 0
        or transaction.candidate_materialization_count != 0
        or transaction.external_materializer_call_count != 0
    ):
        raise ODEBFStateError("nominal shadow transaction contract differs")


def _derive_and_validate_ledger(
    layer_results: tuple[NominalShadowLayerResult, ...],
    transaction: FP32TransactionReceipt,
) -> NominalShadowDerivedLedger:
    _validate_nested_shadow_contract(layer_results, transaction)
    ledger = NominalShadowDerivedLedger(
        terminal_forward_count=sum(
            item.observation.receipt.terminal_forward_count
            for item in layer_results
        ),
        key_forward_count=sum(
            item.observation.receipt.key_forward_count
            for item in layer_results
        ),
        q_solve_count=sum(
            item.plan.q_solve.receipt.logical_q_solve_count
            for item in layer_results
        ),
        dense_update_construction_count=sum(
            item.plan.prepared_update.receipt.dense_construction_count
            for item in layer_results
        ),
        temporary_native_apply_count=sum(
            item.application.exact_object_apply_count
            for item in layer_results
        ),
        native_storage_assignment_count=sum(
            item.application.m3a_layer_receipt.storage_assignment_count
            for item in layer_results
        ),
        storage_cast_boundary_count=sum(
            item.application.m3a_layer_receipt.storage_cast_boundary_count
            for item in layer_results
        ),
        restore_count=transaction.rollback_count,
        model_backward_count=transaction.model_backward_count,
        semantic_backward_count=sum(
            item.plan.receipt.semantic_backward_count
            for item in layer_results
        ),
        slope_backward_count=sum(
            item.plan.receipt.slope_backward_count
            for item in layer_results
        ),
        post_storage_decision_influence_count=(
            transaction.postcast_decision_influence_count
        ),
        router_call_count=sum(
            item.receipt.router_call_count for item in layer_results
        ),
        heldout_evaluator_count=sum(
            item.receipt.heldout_evaluator_count for item in layer_results
        ),
        external_materializer_count=transaction.external_materializer_call_count,
        candidate_materialization_count=(
            transaction.candidate_materialization_count
        ),
        ledger_append_count=sum(
            item.receipt.ledger_append_count for item in layer_results
        ),
        history_append_count=sum(
            item.receipt.history_append_count for item in layer_results
        ),
        logical_outer_commit_count=transaction.logical_outer_commit_count,
        persistent_commit_count=transaction.persistent_commit_count,
    )
    locked = {
        "terminal_forward_count": 5,
        "key_forward_count": 5,
        "q_solve_count": 5,
        "dense_update_construction_count": 5,
        "temporary_native_apply_count": 5,
        "native_storage_assignment_count": 5,
        "storage_cast_boundary_count": 5,
        "restore_count": 1,
        "model_backward_count": 0,
        "semantic_backward_count": 0,
        "slope_backward_count": 0,
        "post_storage_decision_influence_count": 0,
        "router_call_count": 0,
        "heldout_evaluator_count": 0,
        "external_materializer_count": 0,
        "candidate_materialization_count": 0,
        "ledger_append_count": 0,
        "history_append_count": 0,
        "logical_outer_commit_count": 0,
        "persistent_commit_count": 0,
    }
    if ledger.raw_free_payload() != locked:
        raise ODEBFStateError("nominal shadow derived compute ledger differs")
    if (
        transaction.native_storage_assignment_count
        != ledger.native_storage_assignment_count
        or transaction.storage_cast_boundary_count
        != ledger.storage_cast_boundary_count
    ):
        raise ODEBFStateError("nominal shadow transaction ledger differs")
    return ledger


def run_uniform_nominal_shadow_probe(
    target_proposal32: torch.Tensor,
    bindings: Mapping[int, LayerParameterBinding],
    alpha_contexts: Mapping[int, ShadowAlphaLayerContext],
    observation_provider: PrefixObservationProvider,
    *,
    transaction_id: str,
) -> NominalShadowResult:
    """Run exactly one uniform five-layer native shadow and restore entry W."""

    if (
        not isinstance(target_proposal32, torch.Tensor)
        or target_proposal32.dtype is not torch.float32
        or target_proposal32.ndim != 2
        or target_proposal32.requires_grad
        or not bool(torch.isfinite(target_proposal32).all())
    ):
        raise ODEBFContractError("nominal shadow target differs")
    if not callable(observation_provider):
        raise ODEBFContractError("nominal shadow observation provider differs")
    resolved = _resolve_bindings(bindings)
    device = target_proposal32.device
    if any(binding.parameter.device != device for binding in resolved.values()):
        raise ODEBFContractError("nominal shadow parameter devices differ")
    context_guards = _validate_contexts(
        alpha_contexts,
        resolved,
        device=device,
    )
    target_guard = _tensor_identity(target_proposal32)
    entry_state = _weight_state(resolved)
    uniform_pi32 = torch.full(
        (len(RESIDUAL_RESERVE_LAYER_ORDER),),
        0.2,
        dtype=torch.float32,
        device=device,
    )
    if tuple(float(item) for item in uniform_pi32.to(device="cpu")) != UNIFORM_NOMINAL_PI:
        raise ODEBFStateError("uniform nominal pi bytes differ")
    geometry = build_residual_reserve_geometry(uniform_pi32)
    transaction: OfficialStyleFP32SequentialTransaction | None = None
    layer_results: list[NominalShadowLayerResult] = []
    try:
        transaction = OfficialStyleFP32SequentialTransaction(
            resolved,
            mode=FP32TransactionMode.SHADOW,
            transaction_id=transaction_id,
        )
        for layer in RESIDUAL_RESERVE_LAYER_ORDER:
            pre_observation_state = _weight_state(resolved)
            observation = observation_provider(layer, pre_observation_state)
            if _weight_state(resolved) != pre_observation_state:
                raise ODEBFStateError("prefix observation mutated live weights")
            _validate_observation(
                observation,
                layer=layer,
                expected_weight_state=pre_observation_state,
                device=device,
            )
            if _validate_contexts(
                alpha_contexts,
                resolved,
                device=device,
            ) != context_guards or _tensor_identity(target_proposal32) != target_guard:
                raise ODEBFStateError("nominal shadow fixed input mutated")
            context = alpha_contexts[layer]
            plan = plan_residual_reserve_layer_step(
                target_proposal32,
                observation.current_terminal32,
                observation.joint_keys32,
                context.projector32,
                context.committed_covariance32,
                context.regularization32,
                geometry,
                layer=layer,
                weight_name=context.weight_name,
                parameter_shape=context.parameter_shape,
                construction_id=context.construction_id,
                q_residual_tolerance=context.q_residual_tolerance,
                q_condition_max_dimension=context.q_condition_max_dimension,
            )
            application = apply_prepared_low_rank_update(
                transaction,
                plan.prepared_update,
            )
            nominal = derive_nominal_reference_factor(
                plan,
                geometry,
                layer=layer,
            )
            application.validate()
            transaction_layer_receipt = application.m3a_layer_receipt
            layer_receipt = NominalShadowLayerReceipt(
                layer=layer,
                observation_receipt_identity=observation.receipt.identity_sha256,
                layer_step_receipt_identity=plan.receipt.identity_sha256,
                nominal_factor_receipt_identity=nominal.receipt.identity_sha256,
                nominal_factor_identity=nominal.factor.identity_sha256,
                construction_receipt_identity=(
                    plan.prepared_update.receipt.identity_sha256
                ),
                application_receipt_identity=application.identity_sha256,
                transaction_layer_receipt_identity=(
                    transaction_layer_receipt.identity_sha256
                ),
                shadow_proof_status=SHADOW_PROOF_STATUS,
                b2a_input_shadow_proof_status=(
                    nominal.receipt.shadow_application_proof_status
                ),
                exact_construction_apply_count=application.exact_object_apply_count,
                authoritative_transaction_commit_claim_count=0,
                post_storage_decision_influence_count=(
                    transaction_layer_receipt.postcast_decision_influence_count
                ),
                pre_observation_weight_hashes=tuple(
                    (item.layer, item.sha256)
                    for item in pre_observation_state
                ),
                terminal_sha256=observation.receipt.terminal_identity.sha256,
                joint_keys_sha256=observation.receipt.joint_keys_identity.sha256,
                residual_sha256=plan.receipt.residual_sha256,
                q_sha256=plan.receipt.q_sha256,
                beta_applied_left_sha256=(
                    plan.receipt.beta_applied_left_sha256
                ),
                matched_update_sha256=(
                    plan.prepared_update.receipt.matched_update_sha256
                ),
                post_storage_parameter_sha256=(
                    transaction_layer_receipt.post_storage_parameter_sha256
                ),
                router_call_count=0,
                heldout_evaluator_count=0,
                external_materializer_count=0,
                ledger_append_count=0,
                history_append_count=0,
            )
            if (
                nominal.receipt.shadow_application_proof_status
                != SHADOW_APPLICATION_PROOF_STATUS
                or nominal.receipt.authoritative_transaction_committed_claim
                or layer_receipt.exact_construction_apply_count != 1
            ):
                raise ODEBFStateError("nominal shadow proof closure differs")
            layer_results.append(
                NominalShadowLayerResult(
                    observation,
                    plan,
                    application,
                    nominal,
                    layer_receipt,
                )
            )
        transaction_receipt = transaction.finish_shadow()
    except BaseException:
        if transaction is not None and not transaction.finalized:
            transaction.abort_and_rollback()
        raise

    restored_state = _weight_state(resolved)
    if (
        restored_state != entry_state
        or transaction_receipt.mode != FP32TransactionMode.SHADOW.value
        or transaction_receipt.layer_order != RESIDUAL_RESERVE_LAYER_ORDER
        or not transaction_receipt.restored_entry_bytes
        or not transaction_receipt.restored_entry_pointers
        or len(layer_results) != 5
    ):
        raise ODEBFStateError("nominal shadow final restore receipt differs")
    frozen_layer_results = tuple(layer_results)
    layer_receipts = tuple(item.receipt for item in frozen_layer_results)
    derived_ledger = _derive_and_validate_ledger(
        frozen_layer_results,
        transaction_receipt,
    )
    scientific_identity = _stable_scientific_identity(
        geometry=geometry,
        target=target_guard,
        entry=entry_state,
        restored=restored_state,
        layer_results=frozen_layer_results,
        transaction=transaction_receipt,
    )
    receipt = NominalShadowReceipt(
        transaction_id=transaction_id,
        transaction_mode=FP32TransactionMode.SHADOW.value,
        uniform_pi=UNIFORM_NOMINAL_PI,
        uniform_pi_sha256=tensor_sha256(uniform_pi32),
        geometry_receipt_identity=geometry.receipt.raw_free_payload()[
            "identity_sha256"
        ],
        target_identity=target_guard,
        entry_weight_state=entry_state,
        restored_weight_state=restored_state,
        layer_receipts=layer_receipts,
        transaction_receipt_identity=transaction_receipt.identity_sha256,
        shadow_proof_status=SHADOW_PROOF_STATUS,
        native_storage_assignment_count=(
            derived_ledger.native_storage_assignment_count
        ),
        storage_cast_boundary_count=derived_ledger.storage_cast_boundary_count,
        restore_count=derived_ledger.restore_count,
        logical_outer_commit_count=derived_ledger.logical_outer_commit_count,
        persistent_commit_count=derived_ledger.persistent_commit_count,
        external_materializer_count=derived_ledger.external_materializer_count,
        candidate_materialization_count=(
            derived_ledger.candidate_materialization_count
        ),
        ledger_append_count=derived_ledger.ledger_append_count,
        history_append_count=derived_ledger.history_append_count,
        terminal_forward_count=derived_ledger.terminal_forward_count,
        key_forward_count=derived_ledger.key_forward_count,
        q_solve_count=derived_ledger.q_solve_count,
        dense_update_construction_count=(
            derived_ledger.dense_update_construction_count
        ),
        temporary_native_apply_count=derived_ledger.temporary_native_apply_count,
        model_backward_count=derived_ledger.model_backward_count,
        semantic_backward_count=derived_ledger.semantic_backward_count,
        slope_backward_count=derived_ledger.slope_backward_count,
        router_call_count=derived_ledger.router_call_count,
        heldout_evaluator_count=derived_ledger.heldout_evaluator_count,
        post_storage_decision_influence_count=(
            derived_ledger.post_storage_decision_influence_count
        ),
        process_local_identity_exclusions=PROCESS_LOCAL_IDENTITY_EXCLUSIONS,
        scientific_identity_sha256=scientific_identity,
    )
    receipt.identity_sha256
    factors = tuple(item.nominal.factor for item in layer_results)
    factor_tensor_pointers = tuple(
        pointer
        for item in factors
        for pointer in (int(item.left.data_ptr()), int(item.right.data_ptr()))
    )
    if (
        tuple(item.layer for item in factors) != RESIDUAL_RESERVE_LAYER_ORDER
        or len(set(factor_tensor_pointers)) != 10
    ):
        raise ODEBFStateError("nominal shadow factor inventory aliases")
    return NominalShadowResult(
        geometry,
        factors,
        frozen_layer_results,
        transaction_receipt,
        receipt,
    )


__all__ = [
    "PROCESS_LOCAL_IDENTITY_EXCLUSIONS",
    "SHADOW_PROOF_STATUS",
    "UNIFORM_NOMINAL_PI",
    "NominalShadowDerivedLedger",
    "NominalShadowLayerReceipt",
    "NominalShadowLayerResult",
    "NominalShadowReceipt",
    "NominalShadowResult",
    "PrefixObservation",
    "PrefixObservationProvider",
    "PrefixObservationReceipt",
    "ShadowAlphaLayerContext",
    "ShadowTensorIdentity",
    "ShadowWeightStateIdentity",
    "build_prefix_observation",
    "run_uniform_nominal_shadow_probe",
]
