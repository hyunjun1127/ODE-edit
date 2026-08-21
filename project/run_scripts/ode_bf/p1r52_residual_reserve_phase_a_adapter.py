"""Thin one-outer Phase-A adapter for residual-reserve writer isolation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import P1R24TargetStep
from .p1r51_requestwise_semantic_allocation import P1R51ControllerState
from .p1r52_r42_safe_kdc import (
    P1R52_INSTRUCTION_ID,
    P1R52_METHOD_ID,
    P1R52_REPAIR_REASON,
    P1R52_REPAIR_REVISION,
    P1R52SelectedTarget,
)
from .p1r52_residual_reserve_authoritative_sweep import (
    AuthoritativeSweepPrecommitView,
    AuthoritativeSweepPreparedPublication,
    AuthoritativeSweepPublicationBuilder,
    AuthoritativeSweepReceipt,
    ResidualReserveAuthoritativeRoute,
    run_authoritative_residual_reserve_sweep_with_publication,
)
from .p1r52_residual_reserve_committed_state import (
    CommittedGrossLoadLedger,
    CommittedGrossLoadState,
)
from .p1r52_residual_reserve_fp32_transaction import (
    LayerParameterBinding,
    RESIDUAL_RESERVE_LAYER_ORDER,
)
from .p1r52_residual_reserve_nominal_shadow import (
    NominalShadowResult,
    PrefixObservationProvider,
    ShadowAlphaLayerContext,
    ShadowWeightStateIdentity,
    run_uniform_nominal_shadow_probe,
)
from .p1r52_residual_reserve_pc_inventory import (
    SealedPrevalidatedCovariance,
)
from .p1r52_residual_reserve_route_assembly import (
    ROUTE_ASSEMBLY_STATUS,
    UNIFORM_ROUTE_MODE,
    UNIFORM_ROUTE_STATUS,
    ResidualReserveRouteAssemblyReceipt,
    ResidualReserveRouteAssemblyResult,
    ResidualReserveUniformRouteReceipt,
    ResidualReserveUniformRouteResult,
    assemble_residual_reserve_pc_route,
    assemble_residual_reserve_uniform_route,
)
from .scalable_batched_model import ScalableObjectiveResult


PHASE_A_ADAPTER_STATUS = "M4A_PHASE_A_ONE_OUTER_COMMITTED"


class ResidualReservePhaseAArm(str, Enum):
    RR_UNIFORM = "RR_UNIFORM"
    RR_PCSOFT = "RR_PCSOFT"


@dataclass(frozen=True, slots=True)
class SelectedTargetTensorGuard:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    sha256: str
    pointer: int
    version: int
    requires_grad: bool

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "sha256": self.sha256,
            "pointer": self.pointer,
            "version": self.version,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True, slots=True)
class P1R52SelectedTargetBindingReceipt:
    instruction_id: str
    method_id: str
    repair_revision: str
    repair_reason: str
    step_index: int
    selected_target_receipt_identity: str
    target_step_receipt_identity: str
    selected_and_step_receipt_equal: bool
    target_next_sha256: str
    target_next_shape: tuple[int, int]
    target_next_dtype: str
    target_next_device: str
    target_tensor_guards: tuple[SelectedTargetTensorGuard, ...]
    selected_endpoint_identity: str
    selected_endpoint_guard_identity: str
    controller_state_identity: str
    action_freeze_identity: str
    target_recompute_count: int
    immutable_verified: bool

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instruction_id": self.instruction_id,
            "method_id": self.method_id,
            "repair_revision": self.repair_revision,
            "repair_reason": self.repair_reason,
            "step_index": self.step_index,
            "selected_target_receipt_identity": (
                self.selected_target_receipt_identity
            ),
            "target_step_receipt_identity": self.target_step_receipt_identity,
            "selected_and_step_receipt_equal": (
                self.selected_and_step_receipt_equal
            ),
            "target_next_sha256": self.target_next_sha256,
            "target_next_shape": list(self.target_next_shape),
            "target_next_dtype": self.target_next_dtype,
            "target_next_device": self.target_next_device,
            "target_tensor_guards": [
                item.raw_free_payload() for item in self.target_tensor_guards
            ],
            "selected_endpoint_identity": self.selected_endpoint_identity,
            "selected_endpoint_guard_identity": (
                self.selected_endpoint_guard_identity
            ),
            "controller_state_identity": self.controller_state_identity,
            "action_freeze_identity": self.action_freeze_identity,
            "target_recompute_count": self.target_recompute_count,
            "immutable_verified": self.immutable_verified,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class ResidualReservePhaseAActionFreezeReceipt:
    selected_target_binding_identity: str
    selected_target_action_identity: str
    target_next_sha256: str
    entry_weight_scientific_identity: str
    action_freeze_identity: str
    target_recompute_count: int
    weight_mutation_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "selected_target_binding_identity": (
                self.selected_target_binding_identity
            ),
            "selected_target_action_identity": (
                self.selected_target_action_identity
            ),
            "target_next_sha256": self.target_next_sha256,
            "entry_weight_scientific_identity": (
                self.entry_weight_scientific_identity
            ),
            "action_freeze_identity": self.action_freeze_identity,
            "target_recompute_count": self.target_recompute_count,
            "weight_mutation_count": self.weight_mutation_count,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class PhaseAAdapterComputeLedger:
    shadow_terminal_forward_count: int
    shadow_key_forward_count: int
    shadow_q_solve_count: int
    shadow_dense_update_construction_count: int
    shadow_temporary_native_apply_count: int
    shadow_native_storage_assignment_count: int
    shadow_storage_cast_boundary_count: int
    shadow_restore_count: int
    route_inventory_build_count: int
    route_covariance_right_matmul_count: int
    route_solver_call_count: int
    authoritative_terminal_forward_count: int
    authoritative_key_forward_count: int
    authoritative_q_solve_count: int
    authoritative_dense_update_construction_count: int
    authoritative_exact_apply_count: int
    authoritative_native_storage_assignment_count: int
    authoritative_storage_cast_boundary_count: int
    authoritative_logical_outer_commit_count: int
    authoritative_ledger_commit_count: int
    authoritative_factor_append_count: int
    total_terminal_forward_count: int
    total_key_forward_count: int
    total_q_solve_count: int
    total_dense_update_construction_count: int
    total_native_apply_count: int
    total_native_storage_assignment_count: int
    total_storage_cast_boundary_count: int
    model_backward_count: int
    semantic_backward_count: int
    slope_backward_count: int
    heldout_evaluator_count: int
    alpha_history_consume_count: int
    alpha_history_append_count: int
    alpha_history_finalize_count: int
    target_recompute_count: int
    external_materializer_count: int
    post_storage_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class ResidualReservePhaseAAdapterReceipt:
    status: str
    arm: str
    selected_target_binding: P1R52SelectedTargetBindingReceipt
    action_freeze_identity: str
    entry_weight_scientific_identity: str
    nominal_shadow_scientific_identity: str
    nominal_shadow_receipt_identity: str
    nominal_restored_entry_identity: str
    route_mode: str
    route_receipt_identity: str
    route_selected_pi_sha256: str
    route_selected_geometry_identity: str
    route_solver_call_count: int
    authoritative_receipt_identity: str
    authoritative_entry_weight_identity: str
    authoritative_final_weight_identity: str
    committed_state_id: str
    committed_state_before_version: int
    committed_state_after_version: int
    committed_state_before_identity: str
    committed_state_after_identity: str
    successful_factor_append_count: int
    alpha_history_consume_count: int
    alpha_history_append_count: int
    alpha_history_finalize_count: int
    heldout_evaluator_count: int
    target_controller_decision_influence_count: int
    cross_arm_state_access_count: int
    input_immutability_verified: bool
    compute: PhaseAAdapterComputeLedger

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "arm": self.arm,
            "selected_target_binding": (
                self.selected_target_binding.raw_free_payload()
            ),
            "action_freeze_identity": self.action_freeze_identity,
            "entry_weight_scientific_identity": (
                self.entry_weight_scientific_identity
            ),
            "nominal_shadow_scientific_identity": (
                self.nominal_shadow_scientific_identity
            ),
            "nominal_shadow_receipt_identity": self.nominal_shadow_receipt_identity,
            "nominal_restored_entry_identity": (
                self.nominal_restored_entry_identity
            ),
            "route_mode": self.route_mode,
            "route_receipt_identity": self.route_receipt_identity,
            "route_selected_pi_sha256": self.route_selected_pi_sha256,
            "route_selected_geometry_identity": (
                self.route_selected_geometry_identity
            ),
            "route_solver_call_count": self.route_solver_call_count,
            "authoritative_receipt_identity": self.authoritative_receipt_identity,
            "authoritative_entry_weight_identity": (
                self.authoritative_entry_weight_identity
            ),
            "authoritative_final_weight_identity": (
                self.authoritative_final_weight_identity
            ),
            "committed_state_id": self.committed_state_id,
            "committed_state_before_version": self.committed_state_before_version,
            "committed_state_after_version": self.committed_state_after_version,
            "committed_state_before_identity": self.committed_state_before_identity,
            "committed_state_after_identity": self.committed_state_after_identity,
            "successful_factor_append_count": self.successful_factor_append_count,
            "alpha_history_consume_count": self.alpha_history_consume_count,
            "alpha_history_append_count": self.alpha_history_append_count,
            "alpha_history_finalize_count": self.alpha_history_finalize_count,
            "heldout_evaluator_count": self.heldout_evaluator_count,
            "target_controller_decision_influence_count": (
                self.target_controller_decision_influence_count
            ),
            "cross_arm_state_access_count": self.cross_arm_state_access_count,
            "input_immutability_verified": self.input_immutability_verified,
            "compute": self.compute.raw_free_payload(),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class ResidualReservePhaseAAdapterResult(
    AuthoritativeSweepPreparedPublication,
):
    nominal_shadow: NominalShadowResult
    route: ResidualReserveAuthoritativeRoute
    authoritative: AuthoritativeSweepPrecommitView
    receipt: ResidualReservePhaseAAdapterReceipt

    @property
    def authoritative_sweep_receipt_identity(self) -> str:
        return self.authoritative.receipt.identity_sha256

    @property
    def publication_identity_sha256(self) -> str:
        return self.receipt.identity_sha256


@dataclass(frozen=True, slots=True)
class _EntryParameterSnapshot:
    binding: LayerParameterBinding
    parameter: torch.nn.Parameter
    storage_alias: torch.Tensor
    value: torch.Tensor
    pointer: int
    sha256: str


def _mapping_identity(value: Mapping[str, Any], *, name: str) -> str:
    if not isinstance(value, Mapping):
        raise ODEBFContractError(f"Phase-A {name} receipt differs")
    payload = dict(value)
    claimed = payload.pop("identity_sha256", None)
    if not isinstance(claimed, str) or canonical_hash(payload) != claimed:
        raise ODEBFStateError(f"Phase-A {name} receipt identity differs")
    return claimed


def _tensor_guard(name: str, value: torch.Tensor) -> SelectedTargetTensorGuard:
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ODEBFContractError(f"Phase-A target tensor {name} differs")
    return SelectedTargetTensorGuard(
        name=name,
        shape=tuple(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        sha256=tensor_sha256(value),
        pointer=int(value.data_ptr()),
        version=int(value._version),
        requires_grad=bool(value.requires_grad),
    )


def _selected_target_binding(
    selected: P1R52SelectedTarget,
) -> P1R52SelectedTargetBindingReceipt:
    if (
        not isinstance(selected, P1R52SelectedTarget)
        or not isinstance(selected.target_step, P1R24TargetStep)
        or not isinstance(selected.selected_endpoint, ScalableObjectiveResult)
        or not isinstance(selected.next_state, P1R51ControllerState)
    ):
        raise ODEBFContractError("Phase-A selected target type differs")
    step = selected.target_step
    target = step.target_next
    selected_identity = _mapping_identity(
        selected.receipt,
        name="selected-target",
    )
    step_identity = _mapping_identity(step.receipt, name="target-step")
    receipt = dict(selected.receipt)
    step_index = receipt.get("k")
    if (
        target.dtype is not torch.float32
        or target.ndim != 2
        or target.requires_grad
        or not bool(torch.isfinite(target).all())
        or selected.receipt != step.receipt
        or selected_identity != step_identity
        or receipt.get("instruction_id") != P1R52_INSTRUCTION_ID
        or receipt.get("method_id") != P1R52_METHOD_ID
        or receipt.get("repair_revision") != P1R52_REPAIR_REVISION
        or receipt.get("repair_reason") != P1R52_REPAIR_REASON
        or isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or step_index < 0
        or step_index >= 8
        or receipt.get("target_next_sha256") != tensor_sha256(target)
    ):
        raise ODEBFStateError("Phase-A selected target contract differs")
    tensor_guards = tuple(
        _tensor_guard(name, value)
        for name, value in (
            ("target_next", step.target_next),
            ("target_displacement", step.target_displacement),
            ("required_displacement", step.required_displacement),
            ("write_velocity", step.write_velocity),
            ("nll_gradient", step.nll_gradient),
            ("combined_gradient", step.combined_gradient),
        )
    )
    if any(item.requires_grad for item in tensor_guards):
        raise ODEBFContractError("Phase-A target tensor requires grad")
    endpoint_payload = selected.selected_endpoint.raw_free_payload()
    endpoint_guard = canonical_hash(endpoint_payload)
    controller_identity = canonical_hash(
        {
            "entry_semantic_gradient_norm": (
                None
                if selected.next_state.entry_semantic_gradient_norm is None
                else list(selected.next_state.entry_semantic_gradient_norm)
            ),
            "cumulative_accepted_activation_path": list(
                selected.next_state.cumulative_accepted_activation_path
            ),
        }
    )
    action_freeze = canonical_hash(
        {
            "instruction_id": P1R52_INSTRUCTION_ID,
            "method_id": P1R52_METHOD_ID,
            "repair_revision": P1R52_REPAIR_REVISION,
            "step_index": step_index,
            "selected_target_receipt_identity": selected_identity,
            "target_step_receipt_identity": step_identity,
            "target_next_sha256": tensor_sha256(target),
            "selected_endpoint_identity": selected.selected_endpoint.identity_sha256,
            "controller_state_identity": controller_identity,
        }
    )
    return P1R52SelectedTargetBindingReceipt(
        instruction_id=P1R52_INSTRUCTION_ID,
        method_id=P1R52_METHOD_ID,
        repair_revision=P1R52_REPAIR_REVISION,
        repair_reason=P1R52_REPAIR_REASON,
        step_index=step_index,
        selected_target_receipt_identity=selected_identity,
        target_step_receipt_identity=step_identity,
        selected_and_step_receipt_equal=True,
        target_next_sha256=tensor_sha256(target),
        target_next_shape=tuple(target.shape),
        target_next_dtype=str(target.dtype),
        target_next_device=str(target.device),
        target_tensor_guards=tensor_guards,
        selected_endpoint_identity=selected.selected_endpoint.identity_sha256,
        selected_endpoint_guard_identity=endpoint_guard,
        controller_state_identity=controller_identity,
        action_freeze_identity=action_freeze,
        target_recompute_count=0,
        immutable_verified=True,
    )


def _capture_entry(
    bindings: Mapping[int, LayerParameterBinding],
) -> tuple[
    tuple[_EntryParameterSnapshot, ...],
    tuple[ShadowWeightStateIdentity, ...],
    str,
]:
    if set(bindings) != set(RESIDUAL_RESERVE_LAYER_ORDER):
        raise ODEBFContractError("Phase-A weight binding inventory differs")
    snapshots: list[_EntryParameterSnapshot] = []
    states: list[ShadowWeightStateIdentity] = []
    for layer in RESIDUAL_RESERVE_LAYER_ORDER:
        binding = bindings[layer]
        if (
            not isinstance(binding, LayerParameterBinding)
            or binding.layer != layer
            or not isinstance(binding.parameter, torch.nn.Parameter)
        ):
            raise ODEBFContractError("Phase-A weight binding differs")
        parameter = binding.parameter
        sha256 = tensor_sha256(parameter)
        snapshots.append(
            _EntryParameterSnapshot(
                binding=binding,
                parameter=parameter,
                storage_alias=parameter.data,
                value=parameter.detach().clone(),
                pointer=int(parameter.data_ptr()),
                sha256=sha256,
            )
        )
        states.append(
            ShadowWeightStateIdentity(
                layer=layer,
                weight_name=binding.weight_name,
                shape=tuple(parameter.shape),
                dtype=str(parameter.dtype),
                device=str(parameter.device),
                sha256=sha256,
                pointer=int(parameter.data_ptr()),
            )
        )
    frozen_states = tuple(states)
    scientific_identity = canonical_hash(
        [item.scientific_payload() for item in frozen_states]
    )
    return tuple(snapshots), frozen_states, scientific_identity


def _entry_scientific_identity(
    bindings: Mapping[int, LayerParameterBinding],
) -> str:
    if set(bindings) != set(RESIDUAL_RESERVE_LAYER_ORDER):
        raise ODEBFContractError("Phase-A action-freeze binding inventory differs")
    payload = []
    for layer in RESIDUAL_RESERVE_LAYER_ORDER:
        binding = bindings[layer]
        if (
            not isinstance(binding, LayerParameterBinding)
            or binding.layer != layer
            or not isinstance(binding.parameter, torch.nn.Parameter)
        ):
            raise ODEBFContractError("Phase-A action-freeze binding differs")
        parameter = binding.parameter
        payload.append(
            {
                "layer": layer,
                "weight_name": binding.weight_name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
                "sha256": tensor_sha256(parameter),
            }
        )
    return canonical_hash(payload)


def freeze_residual_reserve_phase_a_action(
    selected_target: P1R52SelectedTarget,
    bindings: Mapping[int, LayerParameterBinding],
) -> ResidualReservePhaseAActionFreezeReceipt:
    """Freeze one accepted target against one scientific entry-W state."""

    target_binding = _selected_target_binding(selected_target)
    entry_identity = _entry_scientific_identity(bindings)
    action_identity = canonical_hash(
        {
            "selected_target_action_identity": (
                target_binding.action_freeze_identity
            ),
            "target_next_sha256": target_binding.target_next_sha256,
            "entry_weight_scientific_identity": entry_identity,
        }
    )
    receipt = ResidualReservePhaseAActionFreezeReceipt(
        selected_target_binding_identity=target_binding.identity_sha256,
        selected_target_action_identity=target_binding.action_freeze_identity,
        target_next_sha256=target_binding.target_next_sha256,
        entry_weight_scientific_identity=entry_identity,
        action_freeze_identity=action_identity,
        target_recompute_count=0,
        weight_mutation_count=0,
    )
    receipt.identity_sha256
    return receipt


def _entry_is_exact(snapshots: tuple[_EntryParameterSnapshot, ...]) -> bool:
    return all(
        snapshot.binding.parameter is snapshot.parameter
        and int(snapshot.parameter.data_ptr()) == snapshot.pointer
        and tensor_sha256(snapshot.parameter) == snapshot.sha256
        for snapshot in snapshots
    )


def _restore_entry(snapshots: tuple[_EntryParameterSnapshot, ...]) -> None:
    with torch.no_grad():
        for snapshot in snapshots:
            parameter = snapshot.binding.parameter
            if parameter is not snapshot.parameter:
                raise ODEBFStateError("Phase-A Parameter object changed")
            if int(parameter.data_ptr()) != snapshot.pointer:
                parameter.data = snapshot.storage_alias
            parameter.copy_(snapshot.value)
    if not _entry_is_exact(snapshots):
        raise ODEBFStateError("Phase-A entry rollback was not pointer/byte exact")


def _validate_nominal_binding(
    nominal: NominalShadowResult,
    target: P1R52SelectedTargetBindingReceipt,
    entry_state: tuple[ShadowWeightStateIdentity, ...],
    entry_identity: str,
) -> None:
    if (
        nominal.receipt.target_identity.sha256 != target.target_next_sha256
        or nominal.receipt.target_identity.pointer
        != target.target_tensor_guards[0].pointer
        or nominal.receipt.target_identity.version
        != target.target_tensor_guards[0].version
        or nominal.receipt.entry_weight_state != entry_state
        or nominal.receipt.restored_weight_state != entry_state
        or canonical_hash(
            [
                item.scientific_payload()
                for item in nominal.receipt.restored_weight_state
            ]
        )
        != entry_identity
    ):
        raise ODEBFStateError("Phase-A nominal target/entry binding differs")


def _derive_compute_ledger(
    nominal: NominalShadowResult,
    route: ResidualReserveAuthoritativeRoute,
    authoritative: AuthoritativeSweepReceipt,
) -> PhaseAAdapterComputeLedger:
    shadow = nominal.receipt
    route_ledger = route.receipt.ledger
    sweep = authoritative.ledger
    ledger = PhaseAAdapterComputeLedger(
        shadow_terminal_forward_count=shadow.terminal_forward_count,
        shadow_key_forward_count=shadow.key_forward_count,
        shadow_q_solve_count=shadow.q_solve_count,
        shadow_dense_update_construction_count=(
            shadow.dense_update_construction_count
        ),
        shadow_temporary_native_apply_count=shadow.temporary_native_apply_count,
        shadow_native_storage_assignment_count=(
            shadow.native_storage_assignment_count
        ),
        shadow_storage_cast_boundary_count=shadow.storage_cast_boundary_count,
        shadow_restore_count=shadow.restore_count,
        route_inventory_build_count=route_ledger.inventory_build_count,
        route_covariance_right_matmul_count=(
            route_ledger.covariance_right_matmul_count
        ),
        route_solver_call_count=route_ledger.router_call_count,
        authoritative_terminal_forward_count=sweep.terminal_forward_count,
        authoritative_key_forward_count=sweep.key_forward_count,
        authoritative_q_solve_count=sweep.q_solve_count,
        authoritative_dense_update_construction_count=(
            sweep.dense_update_construction_count
        ),
        authoritative_exact_apply_count=sweep.exact_apply_count,
        authoritative_native_storage_assignment_count=(
            sweep.native_storage_assignment_count
        ),
        authoritative_storage_cast_boundary_count=(
            sweep.storage_cast_boundary_count
        ),
        authoritative_logical_outer_commit_count=(
            sweep.logical_outer_commit_count
        ),
        authoritative_ledger_commit_count=sweep.ledger_commit_count,
        authoritative_factor_append_count=sweep.successful_factor_append_count,
        total_terminal_forward_count=(
            shadow.terminal_forward_count + sweep.terminal_forward_count
        ),
        total_key_forward_count=shadow.key_forward_count + sweep.key_forward_count,
        total_q_solve_count=shadow.q_solve_count + sweep.q_solve_count,
        total_dense_update_construction_count=(
            shadow.dense_update_construction_count
            + sweep.dense_update_construction_count
        ),
        total_native_apply_count=(
            shadow.temporary_native_apply_count + sweep.exact_apply_count
        ),
        total_native_storage_assignment_count=(
            shadow.native_storage_assignment_count
            + sweep.native_storage_assignment_count
        ),
        total_storage_cast_boundary_count=(
            shadow.storage_cast_boundary_count
            + sweep.storage_cast_boundary_count
        ),
        model_backward_count=shadow.model_backward_count + sweep.model_backward_count,
        semantic_backward_count=(
            shadow.semantic_backward_count + sweep.semantic_backward_count
        ),
        slope_backward_count=shadow.slope_backward_count + sweep.slope_backward_count,
        heldout_evaluator_count=(
            shadow.heldout_evaluator_count + sweep.heldout_evaluator_count
        ),
        alpha_history_consume_count=0,
        alpha_history_append_count=0,
        alpha_history_finalize_count=0,
        target_recompute_count=0,
        external_materializer_count=(
            shadow.external_materializer_count + sweep.external_materializer_count
        ),
        post_storage_decision_influence_count=(
            shadow.post_storage_decision_influence_count
            + sweep.post_storage_decision_influence_count
        ),
    )
    expected = {
        "shadow_terminal_forward_count": 5,
        "shadow_key_forward_count": 5,
        "shadow_q_solve_count": 5,
        "shadow_dense_update_construction_count": 5,
        "shadow_temporary_native_apply_count": 5,
        "shadow_native_storage_assignment_count": 5,
        "shadow_storage_cast_boundary_count": 5,
        "shadow_restore_count": 1,
        "route_inventory_build_count": 1,
        "route_covariance_right_matmul_count": 5,
        "route_solver_call_count": route.receipt.ledger.router_call_count,
        "authoritative_terminal_forward_count": 5,
        "authoritative_key_forward_count": 5,
        "authoritative_q_solve_count": 5,
        "authoritative_dense_update_construction_count": 5,
        "authoritative_exact_apply_count": 5,
        "authoritative_native_storage_assignment_count": 5,
        "authoritative_storage_cast_boundary_count": 5,
        "authoritative_logical_outer_commit_count": 1,
        "authoritative_ledger_commit_count": 1,
        "authoritative_factor_append_count": 5,
        "total_terminal_forward_count": 10,
        "total_key_forward_count": 10,
        "total_q_solve_count": 10,
        "total_dense_update_construction_count": 10,
        "total_native_apply_count": 10,
        "total_native_storage_assignment_count": 10,
        "total_storage_cast_boundary_count": 10,
        "model_backward_count": 0,
        "semantic_backward_count": 0,
        "slope_backward_count": 0,
        "heldout_evaluator_count": 0,
        "alpha_history_consume_count": 0,
        "alpha_history_append_count": 0,
        "alpha_history_finalize_count": 0,
        "target_recompute_count": 0,
        "external_materializer_count": 0,
        "post_storage_decision_influence_count": 0,
    }
    if (
        ledger.raw_free_payload() != expected
        or ledger.route_solver_call_count not in (0, 1)
    ):
        raise ODEBFStateError("Phase-A adapter compute ledger differs")
    return ledger


@dataclass(slots=True)
class _PhaseAAdapterPublicationBuilder(
    AuthoritativeSweepPublicationBuilder,
):
    selected_target: P1R52SelectedTarget
    target_binding: P1R52SelectedTargetBindingReceipt
    action_freeze: ResidualReservePhaseAActionFreezeReceipt
    arm: ResidualReservePhaseAArm
    entry_state: tuple[ShadowWeightStateIdentity, ...]
    entry_identity: str
    nominal: NominalShadowResult
    route: ResidualReserveAuthoritativeRoute
    route_mode: str
    committed_ledger: CommittedGrossLoadLedger
    before_state: CommittedGrossLoadState
    before_identity: str
    before_decision_identity: str

    def prepare(
        self,
        view: AuthoritativeSweepPrecommitView,
    ) -> ResidualReservePhaseAAdapterResult:
        """Build the complete adapter result while W+ledger remain reversible."""

        if not isinstance(view, AuthoritativeSweepPrecommitView):
            raise ODEBFContractError("Phase-A authoritative preview type differs")
        _validate_nominal_binding(
            self.nominal,
            self.target_binding,
            self.entry_state,
            self.entry_identity,
        )
        route_identity = self.route.receipt.identity_sha256
        route_geometry_identity = (
            self.route.selected_geometry.receipt.raw_free_payload()[
                "identity_sha256"
            ]
        )
        authoritative = view.receipt
        preview = view.publication_preview
        if (
            _selected_target_binding(self.selected_target)
            != self.target_binding
            or self.action_freeze.selected_target_binding_identity
            != self.target_binding.identity_sha256
            or self.action_freeze.selected_target_action_identity
            != self.target_binding.action_freeze_identity
            or self.action_freeze.target_next_sha256
            != self.target_binding.target_next_sha256
            or self.action_freeze.entry_weight_scientific_identity
            != self.entry_identity
            or self.committed_ledger.state is not self.before_state
            or self.before_state.identity_sha256 != self.before_identity
            or self.before_state.decision_identity_sha256
            != self.before_decision_identity
            or authoritative.route_assembly_identity != route_identity
            or authoritative.nominal_shadow_scientific_identity
            != self.nominal.receipt.scientific_identity_sha256
            or authoritative.nominal_shadow_receipt_identity
            != self.nominal.receipt.identity_sha256
            or authoritative.entry_weight_state_identity != self.entry_identity
            or authoritative.selected_pi_sha256
            != tensor_sha256(self.route.selected_pi)
            or authoritative.selected_geometry_identity
            != route_geometry_identity
            or authoritative.committed_state_before_version
            != self.before_state.version
            or authoritative.committed_state_after_version
            != self.before_state.version + 1
            or authoritative.committed_state_before_identity
            != self.before_identity
            or authoritative.committed_state_after_identity
            != preview.after_state_identity
            or authoritative.committed_state_before_decision_identity
            != self.before_decision_identity
            or authoritative.committed_state_after_decision_identity
            != preview.after_decision_identity
            or preview.committed_factor_counts_after
            != tuple(
                len(layer.committed_precast_factors) + 1
                for layer in self.before_state.layers
            )
            or preview.factor_append_count != 5
            or authoritative.ledger.successful_factor_append_count != 5
        ):
            raise ODEBFStateError(
                "Phase-A precommit target/route/state binding differs"
            )
        compute = _derive_compute_ledger(
            self.nominal,
            self.route,
            authoritative,
        )
        receipt = ResidualReservePhaseAAdapterReceipt(
            status=PHASE_A_ADAPTER_STATUS,
            arm=self.arm.value,
            selected_target_binding=self.target_binding,
            action_freeze_identity=self.action_freeze.action_freeze_identity,
            entry_weight_scientific_identity=self.entry_identity,
            nominal_shadow_scientific_identity=(
                self.nominal.receipt.scientific_identity_sha256
            ),
            nominal_shadow_receipt_identity=self.nominal.receipt.identity_sha256,
            nominal_restored_entry_identity=canonical_hash(
                [
                    item.scientific_payload()
                    for item in self.nominal.receipt.restored_weight_state
                ]
            ),
            route_mode=self.route_mode,
            route_receipt_identity=route_identity,
            route_selected_pi_sha256=tensor_sha256(self.route.selected_pi),
            route_selected_geometry_identity=route_geometry_identity,
            route_solver_call_count=self.route.receipt.ledger.router_call_count,
            authoritative_receipt_identity=authoritative.identity_sha256,
            authoritative_entry_weight_identity=(
                authoritative.entry_weight_state_identity
            ),
            authoritative_final_weight_identity=(
                authoritative.final_weight_state_identity
            ),
            committed_state_id=authoritative.committed_state_id,
            committed_state_before_version=self.before_state.version,
            committed_state_after_version=preview.after_version,
            committed_state_before_identity=self.before_identity,
            committed_state_after_identity=preview.after_state_identity,
            successful_factor_append_count=5,
            alpha_history_consume_count=0,
            alpha_history_append_count=0,
            alpha_history_finalize_count=0,
            heldout_evaluator_count=0,
            target_controller_decision_influence_count=0,
            cross_arm_state_access_count=0,
            input_immutability_verified=True,
            compute=compute,
        )
        receipt.identity_sha256
        result = ResidualReservePhaseAAdapterResult(
            self.nominal,
            self.route,
            view,
            receipt,
        )
        result.publication_identity_sha256
        return result


def run_residual_reserve_phase_a_outer(
    selected_target: P1R52SelectedTarget,
    action_freeze: ResidualReservePhaseAActionFreezeReceipt,
    arm: ResidualReservePhaseAArm,
    bindings: Mapping[int, LayerParameterBinding],
    alpha_contexts: Mapping[int, ShadowAlphaLayerContext],
    observation_provider: PrefixObservationProvider,
    committed_ledger: CommittedGrossLoadLedger,
    covariances: tuple[SealedPrevalidatedCovariance, ...],
    *,
    execution_id: str,
) -> ResidualReservePhaseAAdapterResult:
    """Apply one accepted P1R52 target through one isolated Phase-A arm."""

    if not isinstance(arm, ResidualReservePhaseAArm):
        raise ODEBFContractError("Phase-A arm type differs")
    if not isinstance(committed_ledger, CommittedGrossLoadLedger):
        raise ODEBFContractError("Phase-A gross ledger type differs")
    if not isinstance(execution_id, str) or not execution_id:
        raise ODEBFContractError("Phase-A execution identity differs")
    if not callable(observation_provider):
        raise ODEBFContractError("Phase-A observation provider differs")
    target_binding = _selected_target_binding(selected_target)
    entry_identity = _entry_scientific_identity(bindings)
    if (
        not isinstance(action_freeze, ResidualReservePhaseAActionFreezeReceipt)
        or action_freeze.selected_target_binding_identity
        != target_binding.identity_sha256
        or action_freeze.selected_target_action_identity
        != target_binding.action_freeze_identity
        or action_freeze.target_next_sha256 != target_binding.target_next_sha256
        or action_freeze.entry_weight_scientific_identity != entry_identity
        or action_freeze.action_freeze_identity
        != canonical_hash(
            {
                "selected_target_action_identity": (
                    target_binding.action_freeze_identity
                ),
                "target_next_sha256": target_binding.target_next_sha256,
                "entry_weight_scientific_identity": entry_identity,
            }
        )
        or action_freeze.target_recompute_count != 0
        or action_freeze.weight_mutation_count != 0
    ):
        raise ODEBFStateError("Phase-A action freeze is stale")
    entry_snapshots, entry_state, captured_entry_identity = _capture_entry(bindings)
    if captured_entry_identity != entry_identity:
        raise ODEBFStateError("Phase-A entry changed during snapshot")
    target = selected_target.target_step.target_next
    if target.device != entry_snapshots[0].parameter.device:
        raise ODEBFContractError("Phase-A target/weight device differs")
    before_state = committed_ledger.state
    before_identity = before_state.identity_sha256
    before_decision_identity = before_state.decision_identity_sha256

    def validate_fixed_inputs(*, require_entry: bool) -> None:
        if _selected_target_binding(selected_target) != target_binding:
            raise ODEBFStateError("Phase-A selected target/controller mutated")
        if require_entry and not _entry_is_exact(entry_snapshots):
            raise ODEBFStateError("Phase-A entry W mutated before execution")
        if (
            committed_ledger.state is not before_state
            or committed_ledger.state.identity_sha256 != before_identity
            or committed_ledger.state.decision_identity_sha256
            != before_decision_identity
        ):
            raise ODEBFStateError("Phase-A gross ledger mutated before commit")

    validate_fixed_inputs(require_entry=True)
    try:
        nominal = run_uniform_nominal_shadow_probe(
            target,
            bindings,
            alpha_contexts,
            observation_provider,
            transaction_id=f"{execution_id}:nominal-shadow",
        )
        validate_fixed_inputs(require_entry=True)
        _validate_nominal_binding(
            nominal,
            target_binding,
            entry_state,
            entry_identity,
        )

        if arm is ResidualReservePhaseAArm.RR_UNIFORM:
            route: ResidualReserveAuthoritativeRoute = (
                assemble_residual_reserve_uniform_route(
                    nominal,
                    committed_ledger.state,
                    covariances,
                )
            )
            if (
                not isinstance(route, ResidualReserveUniformRouteResult)
                or not isinstance(
                    route.receipt,
                    ResidualReserveUniformRouteReceipt,
                )
                or route.receipt.status != UNIFORM_ROUTE_STATUS
                or route.receipt.route_mode != UNIFORM_ROUTE_MODE
                or route.receipt.pc_solver_call_count != 0
                or route.receipt.routing_decision_influence_count != 0
            ):
                raise ODEBFStateError("Phase-A uniform route differs")
            route_mode = ResidualReservePhaseAArm.RR_UNIFORM.value
        else:
            route = assemble_residual_reserve_pc_route(
                nominal,
                committed_ledger.state,
                covariances,
            )
            if (
                not isinstance(route, ResidualReserveRouteAssemblyResult)
                or not isinstance(
                    route.receipt,
                    ResidualReserveRouteAssemblyReceipt,
                )
                or route.receipt.status != ROUTE_ASSEMBLY_STATUS
                or route.receipt.ledger.router_call_count != 1
                or route.receipt.router_selected_status
                != route.routing.receipt.selected_status
                or route.receipt.selected_pi
                != route.receipt.router_selected_pi
                or route.receipt.selected_pi_sha256
                != route.receipt.router_selected_pi_sha256
                or route.receipt.selected_pi_sha256
                != tensor_sha256(route.selected_pi)
                or route.receipt.router_selected_pi_sha256
                != tensor_sha256(route.routing.pi_balanced)
            ):
                raise ODEBFStateError("Phase-A PCSOFT route differs")
            route_mode = ResidualReservePhaseAArm.RR_PCSOFT.value
        validate_fixed_inputs(require_entry=True)
        _validate_nominal_binding(
            nominal,
            target_binding,
            entry_state,
            entry_identity,
        )

        def guarded_provider(
            layer: int,
            pre_state: tuple[ShadowWeightStateIdentity, ...],
        ):
            if _selected_target_binding(selected_target) != target_binding:
                raise ODEBFStateError("Phase-A target mutated before prefix capture")
            observation = observation_provider(layer, pre_state)
            if _selected_target_binding(selected_target) != target_binding:
                raise ODEBFStateError("Phase-A target mutated during prefix capture")
            return observation

        publication_builder = _PhaseAAdapterPublicationBuilder(
            selected_target=selected_target,
            target_binding=target_binding,
            action_freeze=action_freeze,
            arm=arm,
            entry_state=entry_state,
            entry_identity=entry_identity,
            nominal=nominal,
            route=route,
            route_mode=route_mode,
            committed_ledger=committed_ledger,
            before_state=before_state,
            before_identity=before_identity,
            before_decision_identity=before_decision_identity,
        )
        result = run_authoritative_residual_reserve_sweep_with_publication(
            target,
            nominal,
            route,
            bindings,
            alpha_contexts,
            guarded_provider,
            committed_ledger,
            publication_builder,
            transaction_id=f"{execution_id}:authoritative",
        )
    except BaseException:
        if committed_ledger.state is before_state:
            _restore_entry(entry_snapshots)
        raise
    return result


__all__ = [
    "PHASE_A_ADAPTER_STATUS",
    "P1R52SelectedTargetBindingReceipt",
    "PhaseAAdapterComputeLedger",
    "ResidualReservePhaseAActionFreezeReceipt",
    "ResidualReservePhaseAAdapterReceipt",
    "ResidualReservePhaseAAdapterResult",
    "ResidualReservePhaseAArm",
    "SelectedTargetTensorGuard",
    "freeze_residual_reserve_phase_a_action",
    "run_residual_reserve_phase_a_outer",
]
