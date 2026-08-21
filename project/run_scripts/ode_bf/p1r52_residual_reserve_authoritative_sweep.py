"""Authoritative residual-reserve prefix sweep and coordinated ledger commit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r52_residual_reserve_committed_state import (
    CommittedGrossLoadLedger,
    LedgerCommitResult,
    LedgerPublicationPreviewReceipt,
    LedgerStageReceipt,
)
from .p1r52_residual_reserve_fp32_transaction import (
    FP32PreparedCommitReceipt,
    FP32TransactionMode,
    LayerParameterBinding,
    OfficialStyleFP32SequentialTransaction,
    RESIDUAL_RESERVE_LAYER_ORDER,
)
from .p1r52_residual_reserve_layer_step import (
    ResidualReserveLayerStepPlan,
    plan_residual_reserve_layer_step,
)
from .p1r52_residual_reserve_nominal_shadow import (
    NominalShadowResult,
    PrefixObservation,
    PrefixObservationProvider,
    ShadowAlphaLayerContext,
    ShadowTensorIdentity,
    ShadowWeightStateIdentity,
)
from .p1r52_residual_reserve_pc_inventory import LowRankFactorSource
from .p1r52_residual_reserve_route_assembly import (
    ROUTE_ASSEMBLY_STATUS,
    ResidualReserveRouteAssemblyResult,
)
from .p1r52_residual_reserve_update_binding import (
    EXACT_SINGLE_CONSTRUCTION_STATUS,
    AppliedPreparedLowRankUpdate,
    apply_prepared_low_rank_update,
)


AUTHORITATIVE_SWEEP_STATUS = "M3D_B2B2_B_AUTHORITATIVE_COMMITTED"


@dataclass(frozen=True, slots=True)
class AuthoritativeSweepLayerReceipt:
    layer: int
    pre_observation_weight_hashes: tuple[tuple[int, str], ...]
    observation_receipt_identity: str
    terminal_sha256: str
    joint_keys_sha256: str
    residual_sha256: str
    q_sha256: str
    plan_receipt_identity: str
    construction_receipt_identity: str
    construction_factor_identity: str
    construction_update_sha256: str
    construction_orientation: str
    application_receipt_identity: str
    transaction_layer_receipt_identity: str
    post_storage_parameter_sha256: str
    actual_post_storage_delta_sha256: str
    pre_cast_update_norm: float
    pre_cast_update_energy: float
    actual_post_storage_delta_norm: float
    actual_post_storage_delta_energy: float
    factor_update_equivalence_status: str
    terminal_forward_count: int
    key_forward_count: int
    q_solve_count: int
    dense_update_construction_count: int
    exact_apply_count: int
    storage_assignment_count: int
    storage_cast_boundary_count: int
    model_backward_count: int
    semantic_backward_count: int
    slope_backward_count: int
    post_storage_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "pre_observation_weight_hashes": [
                [layer, sha256]
                for layer, sha256 in self.pre_observation_weight_hashes
            ],
            "observation_receipt_identity": self.observation_receipt_identity,
            "terminal_sha256": self.terminal_sha256,
            "joint_keys_sha256": self.joint_keys_sha256,
            "residual_sha256": self.residual_sha256,
            "q_sha256": self.q_sha256,
            "plan_receipt_identity": self.plan_receipt_identity,
            "construction_receipt_identity": self.construction_receipt_identity,
            "construction_factor_identity": self.construction_factor_identity,
            "construction_update_sha256": self.construction_update_sha256,
            "construction_orientation": self.construction_orientation,
            "application_receipt_identity": self.application_receipt_identity,
            "transaction_layer_receipt_identity": (
                self.transaction_layer_receipt_identity
            ),
            "post_storage_parameter_sha256": self.post_storage_parameter_sha256,
            "actual_post_storage_delta_sha256": (
                self.actual_post_storage_delta_sha256
            ),
            "pre_cast_update_norm": self.pre_cast_update_norm,
            "pre_cast_update_energy": self.pre_cast_update_energy,
            "actual_post_storage_delta_norm": (
                self.actual_post_storage_delta_norm
            ),
            "actual_post_storage_delta_energy": (
                self.actual_post_storage_delta_energy
            ),
            "factor_update_equivalence_status": (
                self.factor_update_equivalence_status
            ),
            "terminal_forward_count": self.terminal_forward_count,
            "key_forward_count": self.key_forward_count,
            "q_solve_count": self.q_solve_count,
            "dense_update_construction_count": (
                self.dense_update_construction_count
            ),
            "exact_apply_count": self.exact_apply_count,
            "storage_assignment_count": self.storage_assignment_count,
            "storage_cast_boundary_count": self.storage_cast_boundary_count,
            "model_backward_count": self.model_backward_count,
            "semantic_backward_count": self.semantic_backward_count,
            "slope_backward_count": self.slope_backward_count,
            "post_storage_decision_influence_count": (
                self.post_storage_decision_influence_count
            ),
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_hash(self.raw_free_payload())


@dataclass(frozen=True, slots=True)
class AuthoritativeSweepLedger:
    terminal_forward_count: int
    key_forward_count: int
    q_solve_count: int
    dense_update_construction_count: int
    exact_apply_count: int
    native_storage_assignment_count: int
    storage_cast_boundary_count: int
    logical_outer_commit_count: int
    persistent_commit_count: int
    ledger_commit_count: int
    successful_factor_append_count: int
    model_backward_count: int
    semantic_backward_count: int
    slope_backward_count: int
    router_call_count: int
    inventory_build_count: int
    heldout_evaluator_count: int
    external_materializer_count: int
    candidate_materialization_count: int
    post_storage_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class AuthoritativeSweepReceipt:
    status: str
    route_assembly_identity: str
    nominal_shadow_scientific_identity: str
    nominal_shadow_receipt_identity: str
    selected_pi_sha256: str
    selected_geometry_identity: str
    selected_geometry_pi_sha256: str
    selected_execution_device: str
    entry_weight_state_identity: str
    final_weight_state_identity: str
    layer_receipts: tuple[AuthoritativeSweepLayerReceipt, ...]
    prepared_commit_identity: str
    staged_ledger_identity: str
    final_transaction_receipt_identity: str
    ledger_commit_receipt_identity: str
    committed_state_id: str
    committed_state_before_version: int
    committed_state_after_version: int
    committed_state_before_identity: str
    committed_state_after_identity: str
    committed_state_before_decision_identity: str
    committed_state_after_decision_identity: str
    factor_update_equivalence_status: str
    input_immutability_verified: bool
    ledger: AuthoritativeSweepLedger

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "route_assembly_identity": self.route_assembly_identity,
            "nominal_shadow_scientific_identity": (
                self.nominal_shadow_scientific_identity
            ),
            "nominal_shadow_receipt_identity": self.nominal_shadow_receipt_identity,
            "selected_pi_sha256": self.selected_pi_sha256,
            "selected_geometry_identity": self.selected_geometry_identity,
            "selected_geometry_pi_sha256": self.selected_geometry_pi_sha256,
            "selected_execution_device": self.selected_execution_device,
            "entry_weight_state_identity": self.entry_weight_state_identity,
            "final_weight_state_identity": self.final_weight_state_identity,
            "layer_receipts": [
                item.raw_free_payload() for item in self.layer_receipts
            ],
            "prepared_commit_identity": self.prepared_commit_identity,
            "staged_ledger_identity": self.staged_ledger_identity,
            "final_transaction_receipt_identity": (
                self.final_transaction_receipt_identity
            ),
            "ledger_commit_receipt_identity": (
                self.ledger_commit_receipt_identity
            ),
            "committed_state_id": self.committed_state_id,
            "committed_state_before_version": self.committed_state_before_version,
            "committed_state_after_version": self.committed_state_after_version,
            "committed_state_before_identity": self.committed_state_before_identity,
            "committed_state_after_identity": self.committed_state_after_identity,
            "committed_state_before_decision_identity": (
                self.committed_state_before_decision_identity
            ),
            "committed_state_after_decision_identity": (
                self.committed_state_after_decision_identity
            ),
            "factor_update_equivalence_status": (
                self.factor_update_equivalence_status
            ),
            "input_immutability_verified": self.input_immutability_verified,
            "ledger": self.ledger.raw_free_payload(),
            "actual_post_storage_delta_role": "OBSERVATION_ONLY",
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class AuthoritativeSweepLayerResult:
    observation: PrefixObservation
    plan: ResidualReserveLayerStepPlan
    application: AppliedPreparedLowRankUpdate
    receipt: AuthoritativeSweepLayerReceipt


@dataclass(frozen=True, slots=True)
class AuthoritativeSweepResult:
    layer_results: tuple[AuthoritativeSweepLayerResult, ...]
    prepared_receipt: FP32PreparedCommitReceipt
    stage_receipt: LedgerStageReceipt
    ledger_commit: LedgerCommitResult
    receipt: AuthoritativeSweepReceipt


def _weight_state(
    bindings: Mapping[int, LayerParameterBinding],
) -> tuple[ShadowWeightStateIdentity, ...]:
    return tuple(
        ShadowWeightStateIdentity(
            layer=layer,
            weight_name=bindings[layer].weight_name,
            shape=tuple(bindings[layer].parameter.shape),
            dtype=str(bindings[layer].parameter.dtype),
            device=str(bindings[layer].parameter.device),
            sha256=tensor_sha256(bindings[layer].parameter),
            pointer=int(bindings[layer].parameter.data_ptr()),
        )
        for layer in RESIDUAL_RESERVE_LAYER_ORDER
    )


def _scientific_weight_state_identity(
    state: tuple[ShadowWeightStateIdentity, ...],
) -> str:
    return canonical_hash([item.scientific_payload() for item in state])


def _tensor_identity_matches(
    value: torch.Tensor,
    identity: ShadowTensorIdentity,
) -> bool:
    return (
        tuple(value.shape) == identity.shape
        and str(value.dtype) == identity.dtype
        and str(value.device) == identity.device
        and tensor_sha256(value) == identity.sha256
        and int(value.data_ptr()) == identity.pointer
        and int(value._version) == identity.version
        and bool(value.requires_grad) == identity.requires_grad
    )


def _tensor_guard(value: torch.Tensor) -> tuple[str, int, int]:
    return (
        tensor_sha256(value),
        int(value.data_ptr()),
        int(value._version),
    )


def _route_tensor_guard(
    route: ResidualReserveRouteAssemblyResult,
) -> tuple[tuple[str, int, int], ...]:
    geometry = route.selected_geometry
    return tuple(
        _tensor_guard(value)
        for value in (
            route.selected_pi,
            geometry.pi,
            geometry.omega,
            geometry.suffix_retention,
            geometry.beta,
            geometry.rho,
            geometry.mass,
        )
    )


def _validate_selected_geometry(
    route: ResidualReserveRouteAssemblyResult,
) -> None:
    geometry = route.selected_geometry
    receipt = geometry.receipt
    tensors = (
        geometry.pi,
        geometry.omega,
        geometry.suffix_retention,
        geometry.beta,
        geometry.rho,
        geometry.mass,
    )
    if (
        any(value.dtype is not torch.float32 for value in tensors)
        or any(value.device != route.selected_pi.device for value in tensors)
        or any(value.requires_grad for value in tensors)
        or any(not bool(torch.isfinite(value).all()) for value in tensors)
        or tensor_sha256(route.selected_pi) != tensor_sha256(geometry.pi)
        or tuple(float(item) for item in geometry.pi.detach().cpu()) != receipt.pi
        or tuple(float(item) for item in geometry.omega.detach().cpu())
        != receipt.omega
        or tuple(float(item) for item in geometry.suffix_retention.detach().cpu())
        != receipt.suffix_retention
        or tuple(float(item) for item in geometry.beta.detach().cpu())
        != receipt.beta
        or float(geometry.rho.detach().cpu()) != receipt.rho
        or float(geometry.mass.detach().cpu()) != receipt.mass
    ):
        raise ODEBFStateError("authoritative selected geometry tensor differs")


def _resolve_bindings(
    bindings: Mapping[int, LayerParameterBinding],
) -> dict[int, LayerParameterBinding]:
    if set(bindings) != set(RESIDUAL_RESERVE_LAYER_ORDER):
        raise ODEBFContractError("authoritative sweep binding inventory differs")
    result: dict[int, LayerParameterBinding] = {}
    for layer in RESIDUAL_RESERVE_LAYER_ORDER:
        binding = bindings[layer]
        if (
            not isinstance(binding, LayerParameterBinding)
            or binding.layer != layer
            or not isinstance(binding.parameter, torch.nn.Parameter)
        ):
            raise ODEBFContractError("authoritative sweep layer binding differs")
        result[layer] = binding
    return result


def _validate_route_and_entry(
    route: ResidualReserveRouteAssemblyResult,
    nominal_shadow: NominalShadowResult,
    ledger: CommittedGrossLoadLedger,
    target_proposal32: torch.Tensor,
    entry_state: tuple[ShadowWeightStateIdentity, ...],
) -> None:
    if (
        not isinstance(route, ResidualReserveRouteAssemblyResult)
        or not isinstance(nominal_shadow, NominalShadowResult)
        or not isinstance(ledger, CommittedGrossLoadLedger)
    ):
        raise ODEBFContractError("authoritative sweep provenance type differs")
    route_receipt = route.receipt
    state = ledger.state
    geometry_identity = route.selected_geometry.receipt.raw_free_payload()[
        "identity_sha256"
    ]
    _validate_selected_geometry(route)
    if (
        route_receipt.status != ROUTE_ASSEMBLY_STATUS
        or route_receipt.nominal_scientific_identity
        != nominal_shadow.receipt.scientific_identity_sha256
        or route_receipt.nominal_receipt_identity
        != nominal_shadow.receipt.identity_sha256
        or route_receipt.nominal_restored_entry_identity
        != _scientific_weight_state_identity(entry_state)
        or nominal_shadow.receipt.restored_weight_state != entry_state
        or nominal_shadow.receipt.entry_weight_state != entry_state
        or route_receipt.committed_state_id != state.state_id
        or route_receipt.committed_state_version != state.version
        or route_receipt.committed_state_identity != state.identity_sha256
        or route_receipt.committed_state_decision_identity
        != state.decision_identity_sha256
        or route_receipt.inventory_receipt_identity
        != route.inventory.receipt.identity_sha256
        or route_receipt.p_proxy_identity
        != route.inventory.p_proxy.raw_free_payload()["identity_sha256"]
        or route_receipt.c_proxy_identity
        != route.inventory.c_proxy.raw_free_payload()["identity_sha256"]
        or route_receipt.router_receipt_identity
        != route.routing.receipt.raw_free_payload()["identity_sha256"]
        or route.routing.receipt.p_proxy
        != route.inventory.p_proxy.raw_free_payload()
        or route.routing.receipt.c_proxy
        != route.inventory.c_proxy.raw_free_payload()
        or route_receipt.router_selected_pi_sha256
        != tensor_sha256(route.routing.pi_balanced)
        or route_receipt.selected_pi_sha256 != tensor_sha256(route.selected_pi)
        or route_receipt.selected_pi_sha256
        != tensor_sha256(route.selected_geometry.pi)
        or route_receipt.selected_pi
        != tuple(float(item) for item in route.selected_pi.detach().cpu())
        or route_receipt.selected_geometry_identity != geometry_identity
        or route_receipt.selected_geometry_pi_sha256
        != tensor_sha256(route.selected_geometry.pi)
        or route_receipt.selected_execution_device != str(route.selected_pi.device)
        or route.selected_pi.device != route.selected_geometry.pi.device
        or route.selected_pi.dtype is not torch.float32
        or route.selected_geometry.pi.dtype is not torch.float32
        or route_receipt.ledger.router_call_count != 1
        or route_receipt.ledger.inventory_build_count != 1
        or route_receipt.ledger.post_storage_decision_influence_count != 0
        or not _tensor_identity_matches(
            target_proposal32,
            nominal_shadow.receipt.target_identity,
        )
    ):
        raise ODEBFStateError("authoritative sweep frozen route/entry differs")


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
    nominal_shadow: NominalShadowResult,
    *,
    device: torch.device,
) -> tuple[tuple[Any, ...], ...]:
    if set(contexts) != set(RESIDUAL_RESERVE_LAYER_ORDER):
        raise ODEBFContractError("authoritative sweep alpha contexts differ")
    guards = []
    construction_ids: set[str] = set()
    for layer, nominal_layer in zip(
        RESIDUAL_RESERVE_LAYER_ORDER,
        nominal_shadow.layer_results,
        strict=True,
    ):
        context = contexts[layer]
        binding = bindings[layer]
        nominal_inputs = {
            item.name: item for item in nominal_layer.plan.receipt.input_identities
        }
        if (
            not isinstance(context, ShadowAlphaLayerContext)
            or context.layer != layer
            or context.weight_name != binding.weight_name
            or context.parameter_shape != tuple(binding.parameter.shape)
            or not isinstance(context.construction_id, str)
            or not context.construction_id
            or context.construction_id in construction_ids
            or tuple(nominal_inputs)[:6]
            != (
                "target_proposal",
                "current_terminal",
                "joint_keys",
                "projector",
                "committed_covariance",
                "regularization",
            )
            or nominal_inputs["projector"].sha256
            != tensor_sha256(context.projector32)
            or nominal_inputs["committed_covariance"].sha256
            != tensor_sha256(context.committed_covariance32)
            or nominal_inputs["regularization"].sha256
            != tensor_sha256(context.regularization32)
        ):
            raise ODEBFStateError("authoritative sweep alpha context provenance differs")
        construction_ids.add(context.construction_id)
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
                raise ODEBFContractError(
                    "authoritative sweep alpha context tensor differs"
                )
        guards.append(_context_guard(context))
    return tuple(guards)


def _validate_observation(
    observation: PrefixObservation,
    *,
    layer: int,
    expected_weight_state: tuple[ShadowWeightStateIdentity, ...],
    device: torch.device,
) -> None:
    if not isinstance(observation, PrefixObservation):
        raise ODEBFContractError("authoritative prefix observation type differs")
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
        or terminal.requires_grad
        or keys.requires_grad
        or not bool(torch.isfinite(terminal).all())
        or not bool(torch.isfinite(keys).all())
        or not _tensor_identity_matches(terminal, receipt.terminal_identity)
        or not _tensor_identity_matches(keys, receipt.joint_keys_identity)
    ):
        raise ODEBFStateError("authoritative prefix observation binding differs")


def _derive_ledger(
    layers: tuple[AuthoritativeSweepLayerResult, ...],
    preview: LedgerPublicationPreviewReceipt,
) -> AuthoritativeSweepLedger:
    ledger = AuthoritativeSweepLedger(
        terminal_forward_count=sum(
            item.observation.receipt.terminal_forward_count for item in layers
        ),
        key_forward_count=sum(
            item.observation.receipt.key_forward_count for item in layers
        ),
        q_solve_count=sum(item.plan.receipt.q_solve_count for item in layers),
        dense_update_construction_count=sum(
            item.plan.receipt.dense_construction_count for item in layers
        ),
        exact_apply_count=sum(
            item.application.exact_object_apply_count for item in layers
        ),
        native_storage_assignment_count=(
            preview.native_storage_assignment_count
        ),
        storage_cast_boundary_count=preview.storage_cast_boundary_count,
        logical_outer_commit_count=preview.logical_outer_commit_count,
        persistent_commit_count=preview.persistent_commit_count,
        ledger_commit_count=preview.logical_commit_count,
        successful_factor_append_count=preview.factor_append_count,
        model_backward_count=preview.transaction_model_backward_count,
        semantic_backward_count=sum(
            item.plan.receipt.semantic_backward_count for item in layers
        ),
        slope_backward_count=sum(
            item.plan.receipt.slope_backward_count for item in layers
        ),
        router_call_count=0,
        inventory_build_count=0,
        heldout_evaluator_count=0,
        external_materializer_count=preview.external_materializer_count,
        candidate_materialization_count=preview.candidate_materialization_count,
        post_storage_decision_influence_count=(
            preview.postcast_decision_influence_count
        ),
    )
    locked = {
        "terminal_forward_count": 5,
        "key_forward_count": 5,
        "q_solve_count": 5,
        "dense_update_construction_count": 5,
        "exact_apply_count": 5,
        "native_storage_assignment_count": 5,
        "storage_cast_boundary_count": 5,
        "logical_outer_commit_count": 1,
        "persistent_commit_count": 1,
        "ledger_commit_count": 1,
        "successful_factor_append_count": 5,
        "model_backward_count": 0,
        "semantic_backward_count": 0,
        "slope_backward_count": 0,
        "router_call_count": 0,
        "inventory_build_count": 0,
        "heldout_evaluator_count": 0,
        "external_materializer_count": 0,
        "candidate_materialization_count": 0,
        "post_storage_decision_influence_count": 0,
    }
    if ledger.raw_free_payload() != locked:
        raise ODEBFStateError("authoritative sweep derived ledger differs")
    return ledger


def _validate_precommit_layers(
    layers: tuple[AuthoritativeSweepLayerResult, ...],
    route: ResidualReserveRouteAssemblyResult,
) -> None:
    if tuple(item.receipt.layer for item in layers) != RESIDUAL_RESERVE_LAYER_ORDER:
        raise ODEBFStateError("authoritative precommit layer order differs")
    for item in layers:
        observation = item.observation.receipt
        plan = item.plan.receipt
        q_solve = item.plan.q_solve.receipt
        construction = item.plan.prepared_update.receipt
        application = item.application
        m3a = application.m3a_layer_receipt
        receipt = item.receipt
        application.validate()
        expected_pre_hashes = tuple(
            (state.layer, state.sha256)
            for state in observation.pre_observation_weight_state
        )
        if (
            observation.terminal_forward_count != 1
            or observation.key_forward_count != 1
            or observation.model_backward_count != 0
            or observation.semantic_backward_count != 0
            or observation.slope_backward_count != 0
            or observation.action_influence_count != 0
            or q_solve.logical_q_solve_count != 1
            or q_solve.dense_writer_update_construction_count != 0
            or q_solve.model_forward_count != 0
            or q_solve.model_backward_count != 0
            or q_solve.semantic_backward_count != 0
            or q_solve.slope_backward_count != 0
            or q_solve.materialization_count != 0
            or plan.q_solve_count != 1
            or plan.geometry_receipt_identity
            != route.receipt.selected_geometry_identity
            or plan.geometry_beta_tensor_device
            != route.receipt.selected_execution_device
            or plan.q_solve_receipt_identity != q_solve.identity_sha256
            or plan.q_sha256 != q_solve.q_sha256
            or plan.construction_receipt_identity
            != construction.identity_sha256
            or plan.construction_factor_identity
            != item.plan.prepared_update.factor.identity_sha256
            or plan.construction_update_sha256
            != construction.matched_update_sha256
            or plan.construction_orientation
            != construction.matched_orientation
            or plan.dense_construction_count != 1
            or plan.second_dense_construction_count != 0
            or plan.model_forward_count != 0
            or plan.model_backward_count != 0
            or plan.semantic_backward_count != 0
            or plan.slope_backward_count != 0
            or plan.apply_count != 0
            or plan.commit_count != 0
            or plan.materialization_count != 0
            or construction.dense_construction_count != 1
            or construction.second_dense_construction_count != 0
            or construction.postcast_decision_influence_count != 0
            or application.exact_object_apply_count != 1
            or m3a.storage_assignment_count != 1
            or m3a.storage_cast_boundary_count != 1
            or m3a.postcast_decision_influence_count != 0
            or receipt.observation_receipt_identity
            != observation.identity_sha256
            or receipt.pre_observation_weight_hashes != expected_pre_hashes
            or receipt.terminal_sha256 != observation.terminal_identity.sha256
            or receipt.terminal_sha256
            != tensor_sha256(item.observation.current_terminal32)
            or receipt.joint_keys_sha256
            != observation.joint_keys_identity.sha256
            or receipt.joint_keys_sha256
            != tensor_sha256(item.observation.joint_keys32)
            or receipt.plan_receipt_identity != plan.identity_sha256
            or receipt.construction_receipt_identity
            != construction.identity_sha256
            or receipt.construction_factor_identity
            != item.plan.prepared_update.factor.identity_sha256
            or receipt.construction_update_sha256
            != construction.matched_update_sha256
            or receipt.construction_orientation
            != construction.matched_orientation
            or receipt.residual_sha256 != plan.residual_sha256
            or receipt.residual_sha256 != tensor_sha256(item.plan.residual32)
            or receipt.q_sha256 != plan.q_sha256
            or receipt.q_sha256 != q_solve.q_sha256
            or receipt.q_sha256 != tensor_sha256(item.plan.q32)
            or receipt.application_receipt_identity != application.identity_sha256
            or receipt.transaction_layer_receipt_identity != m3a.identity_sha256
            or receipt.post_storage_parameter_sha256
            != m3a.post_storage_parameter_sha256
            or receipt.actual_post_storage_delta_sha256
            != m3a.actual_post_storage_delta32_sha256
            or receipt.pre_cast_update_norm != m3a.pre_cast_fp32_update_norm
            or receipt.pre_cast_update_energy != m3a.pre_cast_fp32_update_energy
            or receipt.actual_post_storage_delta_norm
            != m3a.actual_post_storage_delta32_norm
            or receipt.actual_post_storage_delta_energy
            != m3a.actual_post_storage_delta32_energy
            or receipt.terminal_forward_count != observation.terminal_forward_count
            or receipt.key_forward_count != observation.key_forward_count
            or receipt.q_solve_count != plan.q_solve_count
            or receipt.dense_update_construction_count
            != plan.dense_construction_count
            or receipt.exact_apply_count != application.exact_object_apply_count
            or receipt.storage_assignment_count != m3a.storage_assignment_count
            or receipt.storage_cast_boundary_count
            != m3a.storage_cast_boundary_count
            or receipt.model_backward_count != 0
            or receipt.model_backward_count != observation.model_backward_count
            or receipt.semantic_backward_count != plan.semantic_backward_count
            or receipt.slope_backward_count != plan.slope_backward_count
            or receipt.post_storage_decision_influence_count
            != m3a.postcast_decision_influence_count
            or receipt.factor_update_equivalence_status
            != EXACT_SINGLE_CONSTRUCTION_STATUS
            or item.plan.prepared_update.factor.source
            is not LowRankFactorSource.AUTHORITATIVE_PREPARED_PRECAST_FP32
        ):
            raise ODEBFStateError(
                "authoritative nested precommit receipt differs"
            )


def _validate_publication_preview(
    preview: LedgerPublicationPreviewReceipt,
    stage: LedgerStageReceipt,
    prepared: FP32PreparedCommitReceipt,
    layers: tuple[AuthoritativeSweepLayerResult, ...],
    *,
    before_state_identity: str,
    before_decision_identity: str,
    before_version: int,
) -> None:
    if (
        preview.stage_identity != stage.stage_identity
        or preview.transaction_id != prepared.transaction_id
        or preview.prepared_commit_identity != prepared.identity_sha256
        or preview.future_final_transaction_receipt_identity
        != prepared.future_final_receipt.identity_sha256
        or preview.before_version != before_version
        or preview.after_version != before_version + 1
        or preview.before_state_identity != before_state_identity
        or preview.before_decision_identity != before_decision_identity
        or preview.layers != RESIDUAL_RESERVE_LAYER_ORDER
        or preview.prepared_factor_identities
        != tuple(item.plan.prepared_update.factor.identity_sha256 for item in layers)
        or preview.prepared_factor_sources
        != (LowRankFactorSource.AUTHORITATIVE_PREPARED_PRECAST_FP32.value,) * 5
        or preview.committed_factor_sources
        != (LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32.value,) * 5
        or preview.provenance_transition_counts != (1,) * 5
        or preview.factor_update_equivalence_statuses
        != (EXACT_SINGLE_CONSTRUCTION_STATUS,) * 5
        or preview.factor_append_counts != (1,) * 5
        or preview.factor_append_count != 5
        or preview.logical_commit_count != 1
        or preview.shadow_commit_count != 0
        or preview.failed_commit_count != 0
        or preview.duplicate_commit_count != 0
        or preview.post_storage_decision_influence_count != 0
        or preview.model_forward_count != 0
        or preview.model_backward_count != 0
        or preview.dense_materialization_count != 0
        or preview.native_storage_assignment_count != 5
        or preview.storage_cast_boundary_count != 5
        or preview.logical_outer_commit_count != 1
        or preview.persistent_commit_count != 1
        or preview.transaction_model_forward_count != 0
        or preview.transaction_model_backward_count != 0
        or preview.external_materializer_count != 0
        or preview.candidate_materialization_count != 0
        or preview.postcast_decision_influence_count != 0
    ):
        raise ODEBFStateError("authoritative publication preview differs")


def run_authoritative_residual_reserve_sweep(
    target_proposal32: torch.Tensor,
    nominal_shadow: NominalShadowResult,
    route: ResidualReserveRouteAssemblyResult,
    bindings: Mapping[int, LayerParameterBinding],
    alpha_contexts: Mapping[int, ShadowAlphaLayerContext],
    observation_provider: PrefixObservationProvider,
    committed_ledger: CommittedGrossLoadLedger,
    *,
    transaction_id: str,
) -> AuthoritativeSweepResult:
    """Run one frozen-route native prefix sweep and atomically publish its ledger."""

    if not callable(observation_provider):
        raise ODEBFContractError("authoritative observation provider differs")
    resolved = _resolve_bindings(bindings)
    device = target_proposal32.device
    if (
        target_proposal32.dtype is not torch.float32
        or target_proposal32.requires_grad
        or any(item.parameter.device != device for item in resolved.values())
        or route.selected_pi.device != device
        or route.selected_geometry.pi.device != device
    ):
        raise ODEBFContractError("authoritative sweep execution device differs")
    entry_state = _weight_state(resolved)
    _validate_route_and_entry(
        route,
        nominal_shadow,
        committed_ledger,
        target_proposal32,
        entry_state,
    )
    context_guards = _validate_contexts(
        alpha_contexts,
        resolved,
        nominal_shadow,
        device=device,
    )
    route_guards = _route_tensor_guard(route)
    route_receipt_guard = route.receipt.identity_sha256
    target_guard = _tensor_guard(target_proposal32)
    state_before = committed_ledger.state
    before_state_identity = state_before.identity_sha256
    before_decision_identity = state_before.decision_identity_sha256
    transaction: OfficialStyleFP32SequentialTransaction | None = None
    stage: LedgerStageReceipt | None = None
    layer_results: list[AuthoritativeSweepLayerResult] = []
    try:
        transaction = OfficialStyleFP32SequentialTransaction(
            resolved,
            mode=FP32TransactionMode.AUTHORITATIVE,
            transaction_id=transaction_id,
        )
        for layer in RESIDUAL_RESERVE_LAYER_ORDER:
            pre_state = _weight_state(resolved)
            observation = observation_provider(layer, pre_state)
            if _weight_state(resolved) != pre_state:
                raise ODEBFStateError(
                    "authoritative observation mutated live weights"
                )
            _validate_observation(
                observation,
                layer=layer,
                expected_weight_state=pre_state,
                device=device,
            )
            if (
                _validate_contexts(
                    alpha_contexts,
                    resolved,
                    nominal_shadow,
                    device=device,
                )
                != context_guards
                or _route_tensor_guard(route) != route_guards
                or route.receipt.identity_sha256 != route_receipt_guard
                or _tensor_guard(target_proposal32) != target_guard
            ):
                raise ODEBFStateError("authoritative fixed input mutated")
            context = alpha_contexts[layer]
            plan = plan_residual_reserve_layer_step(
                target_proposal32,
                observation.current_terminal32,
                observation.joint_keys32,
                context.projector32,
                context.committed_covariance32,
                context.regularization32,
                route.selected_geometry,
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
            application.validate()
            m3a = application.m3a_layer_receipt
            layer_receipt = AuthoritativeSweepLayerReceipt(
                layer=layer,
                pre_observation_weight_hashes=tuple(
                    (item.layer, item.sha256) for item in pre_state
                ),
                observation_receipt_identity=observation.receipt.identity_sha256,
                terminal_sha256=observation.receipt.terminal_identity.sha256,
                joint_keys_sha256=observation.receipt.joint_keys_identity.sha256,
                residual_sha256=plan.receipt.residual_sha256,
                q_sha256=plan.receipt.q_sha256,
                plan_receipt_identity=plan.receipt.identity_sha256,
                construction_receipt_identity=(
                    plan.prepared_update.receipt.identity_sha256
                ),
                construction_factor_identity=(
                    plan.prepared_update.factor.identity_sha256
                ),
                construction_update_sha256=(
                    plan.prepared_update.receipt.matched_update_sha256
                ),
                construction_orientation=(
                    plan.prepared_update.receipt.matched_orientation
                ),
                application_receipt_identity=application.identity_sha256,
                transaction_layer_receipt_identity=m3a.identity_sha256,
                post_storage_parameter_sha256=m3a.post_storage_parameter_sha256,
                actual_post_storage_delta_sha256=(
                    m3a.actual_post_storage_delta32_sha256
                ),
                pre_cast_update_norm=m3a.pre_cast_fp32_update_norm,
                pre_cast_update_energy=m3a.pre_cast_fp32_update_energy,
                actual_post_storage_delta_norm=(
                    m3a.actual_post_storage_delta32_norm
                ),
                actual_post_storage_delta_energy=(
                    m3a.actual_post_storage_delta32_energy
                ),
                factor_update_equivalence_status=(
                    EXACT_SINGLE_CONSTRUCTION_STATUS
                ),
                terminal_forward_count=observation.receipt.terminal_forward_count,
                key_forward_count=observation.receipt.key_forward_count,
                q_solve_count=plan.receipt.q_solve_count,
                dense_update_construction_count=(
                    plan.receipt.dense_construction_count
                ),
                exact_apply_count=application.exact_object_apply_count,
                storage_assignment_count=m3a.storage_assignment_count,
                storage_cast_boundary_count=m3a.storage_cast_boundary_count,
                model_backward_count=observation.receipt.model_backward_count,
                semantic_backward_count=plan.receipt.semantic_backward_count,
                slope_backward_count=plan.receipt.slope_backward_count,
                post_storage_decision_influence_count=(
                    m3a.postcast_decision_influence_count
                ),
            )
            layer_results.append(
                AuthoritativeSweepLayerResult(
                    observation,
                    plan,
                    application,
                    layer_receipt,
                )
            )
        frozen_precommit_layers = tuple(layer_results)
        _validate_precommit_layers(frozen_precommit_layers, route)
        if (
            _validate_contexts(
                alpha_contexts,
                resolved,
                nominal_shadow,
                device=device,
            )
            != context_guards
            or _route_tensor_guard(route) != route_guards
            or route.receipt.identity_sha256 != route_receipt_guard
            or _tensor_guard(target_proposal32) != target_guard
        ):
            raise ODEBFStateError("authoritative precommit fixed input mutated")
        prepared = transaction.prepare_authoritative_commit()
        stage = committed_ledger.stage_authoritative_commit(
            transaction,
            prepared,
            tuple(item.application for item in layer_results),
        )
        preview = committed_ledger.preview_staged_publication(
            stage.stage_identity,
            prepared.identity_sha256,
        )
        _validate_publication_preview(
            preview,
            stage,
            prepared,
            frozen_precommit_layers,
            before_state_identity=before_state_identity,
            before_decision_identity=before_decision_identity,
            before_version=state_before.version,
        )
        final_state = _weight_state(resolved)
        derived_ledger = _derive_ledger(frozen_precommit_layers, preview)
        receipt = AuthoritativeSweepReceipt(
            status=AUTHORITATIVE_SWEEP_STATUS,
            route_assembly_identity=route.receipt.identity_sha256,
            nominal_shadow_scientific_identity=(
                nominal_shadow.receipt.scientific_identity_sha256
            ),
            nominal_shadow_receipt_identity=(
                nominal_shadow.receipt.identity_sha256
            ),
            selected_pi_sha256=tensor_sha256(route.selected_pi),
            selected_geometry_identity=(
                route.selected_geometry.receipt.raw_free_payload()[
                    "identity_sha256"
                ]
            ),
            selected_geometry_pi_sha256=tensor_sha256(
                route.selected_geometry.pi
            ),
            selected_execution_device=str(route.selected_pi.device),
            entry_weight_state_identity=(
                _scientific_weight_state_identity(entry_state)
            ),
            final_weight_state_identity=(
                _scientific_weight_state_identity(final_state)
            ),
            layer_receipts=tuple(
                item.receipt for item in frozen_precommit_layers
            ),
            prepared_commit_identity=prepared.identity_sha256,
            staged_ledger_identity=stage.stage_identity,
            final_transaction_receipt_identity=(
                preview.future_final_transaction_receipt_identity
            ),
            ledger_commit_receipt_identity=(
                preview.ledger_commit_receipt_identity
            ),
            committed_state_id=preview.committed_state_id,
            committed_state_before_version=preview.before_version,
            committed_state_after_version=preview.after_version,
            committed_state_before_identity=preview.before_state_identity,
            committed_state_after_identity=preview.after_state_identity,
            committed_state_before_decision_identity=(
                preview.before_decision_identity
            ),
            committed_state_after_decision_identity=(
                preview.after_decision_identity
            ),
            factor_update_equivalence_status=(
                EXACT_SINGLE_CONSTRUCTION_STATUS
            ),
            input_immutability_verified=True,
            ledger=derived_ledger,
        )
        receipt.identity_sha256
        commit = committed_ledger.commit_staged_with_transaction(
            transaction,
            stage.stage_identity,
            prepared.identity_sha256,
            preview.identity_sha256,
        )
        return AuthoritativeSweepResult(
            frozen_precommit_layers,
            prepared,
            stage,
            commit,
            receipt,
        )
    except BaseException:
        if (
            stage is not None
            and transaction is not None
            and not transaction.finalized
        ):
            committed_ledger.abort_staged(stage.stage_identity)
        elif transaction is not None and not transaction.finalized:
            transaction.abort_and_rollback()
        raise


__all__ = [
    "AUTHORITATIVE_SWEEP_STATUS",
    "AuthoritativeSweepLayerReceipt",
    "AuthoritativeSweepLayerResult",
    "AuthoritativeSweepLedger",
    "AuthoritativeSweepReceipt",
    "AuthoritativeSweepResult",
    "run_authoritative_residual_reserve_sweep",
]
