"""Pure P/C route assembly for the residual-reserve sequential writer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r52_residual_reserve_committed_state import CommittedGrossLoadState
from .p1r52_residual_reserve_geometry import (
    FP32_SIMPLEX_SUM_TOLERANCE,
    ResidualReserveGeometry,
    build_residual_reserve_geometry,
)
from .p1r52_residual_reserve_nominal_shadow import (
    SHADOW_PROOF_STATUS,
    UNIFORM_NOMINAL_PI,
    NominalShadowResult,
)
from .p1r52_residual_reserve_pc_inventory import (
    LayerPCInventoryInput,
    LowRankFactorSource,
    PCInventoryResult,
    RESIDUAL_RESERVE_INVENTORY_LAYERS,
    SealedPrevalidatedCovariance,
    build_residual_reserve_pc_inventory,
)
from .p1r52_residual_reserve_pc_router import (
    ResidualReservePCRoutingResult,
    evaluate_quadratic_proxy,
    solve_residual_reserve_pc_router,
)


ROUTE_ASSEMBLY_STATUS = "M3D_B2B2_A_ROUTE_ASSEMBLY_ONLY"
UNIFORM_DECISION_OFF_STATUS = "UNIFORM_OBSERVATION_WITHOUT_SECOND_ROUTE"
_LAYER_ORDER = RESIDUAL_RESERVE_INVENTORY_LAYERS
_SHADOW_MODE = "SHADOW"


@dataclass(frozen=True, slots=True)
class RouteAssemblyLayerBindingReceipt:
    layer: int
    nominal_factor_identity: str
    nominal_shadow_layer_receipt_identity: str
    committed_layer_decision_identity: str
    committed_factor_identities: tuple[str, ...]
    committed_factor_count: int
    covariance_identity: str
    covariance_artifact_identity: str
    covariance_anchor_identity: str
    committed_p_constant: float
    cumulative_precast_lambda: float
    pretrained_weight_norm_squared: float
    current_candidate_factor_count: int
    current_prefix_factor_count: int
    input_alias_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "nominal_factor_identity": self.nominal_factor_identity,
            "nominal_shadow_layer_receipt_identity": (
                self.nominal_shadow_layer_receipt_identity
            ),
            "committed_layer_decision_identity": (
                self.committed_layer_decision_identity
            ),
            "committed_factor_identities": list(self.committed_factor_identities),
            "committed_factor_count": self.committed_factor_count,
            "covariance_identity": self.covariance_identity,
            "covariance_artifact_identity": self.covariance_artifact_identity,
            "covariance_anchor_identity": self.covariance_anchor_identity,
            "committed_p_constant": self.committed_p_constant,
            "cumulative_precast_lambda": self.cumulative_precast_lambda,
            "pretrained_weight_norm_squared": self.pretrained_weight_norm_squared,
            "current_candidate_factor_count": self.current_candidate_factor_count,
            "current_prefix_factor_count": self.current_prefix_factor_count,
            "input_alias_count": self.input_alias_count,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_hash(self.raw_free_payload())


@dataclass(frozen=True, slots=True)
class RouteAssemblyLedger:
    inventory_build_count: int
    covariance_right_matmul_count: int
    router_call_count: int
    router_decision_to_execution_pi_copy_count: int
    cross_device_pi_copy_count: int
    selected_geometry_build_count: int
    model_forward_count: int
    model_backward_count: int
    prefix_observation_count: int
    q_solve_count: int
    dense_writer_update_construction_count: int
    native_apply_count: int
    storage_cast_count: int
    transaction_count: int
    materialization_count: int
    ledger_append_count: int
    history_append_count: int
    heldout_evaluator_count: int
    weight_mutation_count: int
    post_storage_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class ResidualReserveRouteAssemblyReceipt:
    status: str
    nominal_scientific_identity: str
    nominal_receipt_identity: str
    nominal_restored_entry_identity: str
    committed_state_id: str
    committed_state_version: int
    committed_state_identity: str
    committed_state_decision_identity: str
    layer_bindings: tuple[RouteAssemblyLayerBindingReceipt, ...]
    covariance_artifact_identities: tuple[str, ...]
    inventory_receipt_identity: str
    p_proxy_identity: str
    c_proxy_identity: str
    router_receipt_identity: str
    router_selected_status: str
    router_selected_pi: tuple[float, ...]
    router_selected_pi_sha256: str
    router_decision_device: str
    selected_pi: tuple[float, ...]
    selected_pi_sha256: str
    selected_execution_device: str
    router_decision_to_execution_pi_copy_count: int
    cross_device_pi_copy_count: int
    device_independent_pi_bytes_exact: bool
    selected_geometry_identity: str
    selected_geometry_pi_sha256: str
    selected_geometry_build_count: int
    selected_pi_frozen_nonalias: bool
    input_immutability_verified: bool
    ledger: RouteAssemblyLedger

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "nominal_scientific_identity": self.nominal_scientific_identity,
            "nominal_receipt_identity": self.nominal_receipt_identity,
            "nominal_restored_entry_identity": (
                self.nominal_restored_entry_identity
            ),
            "committed_state_id": self.committed_state_id,
            "committed_state_version": self.committed_state_version,
            "committed_state_identity": self.committed_state_identity,
            "committed_state_decision_identity": (
                self.committed_state_decision_identity
            ),
            "layer_bindings": [
                item.raw_free_payload() for item in self.layer_bindings
            ],
            "covariance_artifact_identities": list(
                self.covariance_artifact_identities
            ),
            "inventory_receipt_identity": self.inventory_receipt_identity,
            "p_proxy_identity": self.p_proxy_identity,
            "c_proxy_identity": self.c_proxy_identity,
            "router_receipt_identity": self.router_receipt_identity,
            "router_selected_status": self.router_selected_status,
            "router_selected_pi": list(self.router_selected_pi),
            "router_selected_pi_sha256": self.router_selected_pi_sha256,
            "router_decision_device": self.router_decision_device,
            "selected_pi": list(self.selected_pi),
            "selected_pi_sha256": self.selected_pi_sha256,
            "selected_execution_device": self.selected_execution_device,
            "router_decision_to_execution_pi_copy_count": (
                self.router_decision_to_execution_pi_copy_count
            ),
            "cross_device_pi_copy_count": self.cross_device_pi_copy_count,
            "device_independent_pi_bytes_exact": (
                self.device_independent_pi_bytes_exact
            ),
            "selected_geometry_identity": self.selected_geometry_identity,
            "selected_geometry_pi_sha256": self.selected_geometry_pi_sha256,
            "selected_geometry_build_count": self.selected_geometry_build_count,
            "selected_pi_frozen_nonalias": self.selected_pi_frozen_nonalias,
            "input_immutability_verified": self.input_immutability_verified,
            "ledger": self.ledger.raw_free_payload(),
            "decision_inputs": (
                "NOMINAL_SURROGATE_PLUS_SUCCESSFUL_COMMITTED_PRECAST_ONLY"
            ),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class ResidualReserveRouteAssemblyResult:
    inventory: PCInventoryResult
    routing: ResidualReservePCRoutingResult
    selected_pi: torch.Tensor
    selected_geometry: ResidualReserveGeometry
    receipt: ResidualReserveRouteAssemblyReceipt


@dataclass(frozen=True, slots=True)
class UniformDecisionOffObservation:
    pi: torch.Tensor
    geometry: ResidualReserveGeometry
    p_value: float
    c_value: float
    source_route_assembly_identity: str
    router_pi_sha256: str
    router_decision_device: str
    execution_device: str
    router_decision_to_execution_pi_copy_count: int
    cross_device_pi_copy_count: int
    device_independent_pi_bytes_exact: bool
    router_call_count: int
    decision_influence_count: int
    status: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pi": [float(item) for item in self.pi],
            "pi_sha256": tensor_sha256(self.pi),
            "geometry_identity": self.geometry.receipt.raw_free_payload()[
                "identity_sha256"
            ],
            "p_value": self.p_value,
            "c_value": self.c_value,
            "source_route_assembly_identity": self.source_route_assembly_identity,
            "router_pi_sha256": self.router_pi_sha256,
            "router_decision_device": self.router_decision_device,
            "execution_device": self.execution_device,
            "router_decision_to_execution_pi_copy_count": (
                self.router_decision_to_execution_pi_copy_count
            ),
            "cross_device_pi_copy_count": self.cross_device_pi_copy_count,
            "device_independent_pi_bytes_exact": (
                self.device_independent_pi_bytes_exact
            ),
            "router_call_count": self.router_call_count,
            "decision_influence_count": self.decision_influence_count,
            "status": self.status,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class _TensorGuard:
    tensor: torch.Tensor
    pointer: int
    version: int
    sha256: str


def _guard_tensors(values: Iterable[torch.Tensor]) -> tuple[_TensorGuard, ...]:
    result = []
    for value in values:
        result.append(
            _TensorGuard(
                tensor=value,
                pointer=int(value.data_ptr()),
                version=int(value._version),
                sha256=tensor_sha256(value),
            )
        )
    return tuple(result)


def _verify_guards(guards: tuple[_TensorGuard, ...]) -> None:
    if any(
        int(item.tensor.data_ptr()) != item.pointer
        or int(item.tensor._version) != item.version
        or tensor_sha256(item.tensor) != item.sha256
        for item in guards
    ):
        raise ODEBFStateError("route assembly input tensor mutated")


def _nominal_input_tensors(result: NominalShadowResult) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = [
        result.geometry.pi,
        result.geometry.omega,
        result.geometry.suffix_retention,
        result.geometry.beta,
        result.geometry.rho,
        result.geometry.mass,
    ]
    for factor in result.factors:
        tensors.extend((factor.left, factor.right))
    return tuple(tensors)


def _committed_input_tensors(
    state: CommittedGrossLoadState,
) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []
    for layer in state.layers:
        tensors.append(layer.covariance.value)
        for factor in layer.committed_precast_factors:
            tensors.extend((factor.left, factor.right))
    return tuple(tensors)


def _validate_nominal_shadow(result: NominalShadowResult) -> None:
    if not isinstance(result, NominalShadowResult):
        raise ODEBFContractError("route assembly nominal shadow type differs")
    receipt = result.receipt
    transaction = result.transaction_receipt
    if (
        tuple(item.layer for item in result.factors)
        != _LAYER_ORDER
        or tuple(item.receipt.layer for item in result.layer_results)
        != _LAYER_ORDER
        or tuple(item.receipt for item in result.layer_results)
        != receipt.layer_receipts
        or tuple(item.nominal.factor.identity_sha256 for item in result.layer_results)
        != tuple(item.identity_sha256 for item in result.factors)
        or any(
            factor.source
            is not LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32
            for factor in result.factors
        )
        or receipt.shadow_proof_status != SHADOW_PROOF_STATUS
        or receipt.transaction_mode != _SHADOW_MODE
        or transaction.mode != _SHADOW_MODE
        or receipt.transaction_receipt_identity != transaction.identity_sha256
        or receipt.geometry_receipt_identity
        != result.geometry.receipt.raw_free_payload()["identity_sha256"]
        or receipt.entry_weight_state != receipt.restored_weight_state
        or not transaction.restored_entry_bytes
        or not transaction.restored_entry_pointers
        or receipt.restore_count != 1
        or receipt.logical_outer_commit_count != 0
        or receipt.persistent_commit_count != 0
        or receipt.router_call_count != 0
        or receipt.ledger_append_count != 0
        or receipt.history_append_count != 0
        or receipt.heldout_evaluator_count != 0
        or receipt.external_materializer_count != 0
        or receipt.candidate_materialization_count != 0
        or receipt.model_backward_count != 0
        or receipt.semantic_backward_count != 0
        or receipt.slope_backward_count != 0
        or receipt.post_storage_decision_influence_count != 0
        or tuple(float(item) for item in result.geometry.pi.detach().cpu())
        != UNIFORM_NOMINAL_PI
    ):
        raise ODEBFStateError("route assembly nominal shadow binding differs")
    pointers = [
        int(tensor.data_ptr())
        for factor in result.factors
        for tensor in (factor.left, factor.right)
    ]
    if len(set(pointers)) != len(pointers):
        raise ODEBFStateError("route assembly nominal factors alias")


def _validate_committed_and_covariance_inputs(
    state: CommittedGrossLoadState,
    covariances: tuple[SealedPrevalidatedCovariance, ...],
) -> None:
    if not isinstance(state, CommittedGrossLoadState):
        raise ODEBFContractError("route assembly committed state type differs")
    if (
        not isinstance(covariances, tuple)
        or tuple(item.layer for item in covariances)
        != _LAYER_ORDER
        or not all(isinstance(item, SealedPrevalidatedCovariance) for item in covariances)
    ):
        raise ODEBFContractError("route assembly covariance inventory differs")
    for layer_state, covariance in zip(state.layers, covariances, strict=True):
        if (
            layer_state.covariance.identity_sha256 != covariance.identity_sha256
            or any(
                factor.source
                is not LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32
                for factor in layer_state.committed_precast_factors
            )
        ):
            raise ODEBFStateError("route assembly committed anchor is stale")


def _validate_inventory_result(
    inventory: PCInventoryResult,
    layer_inputs: tuple[LayerPCInventoryInput, ...],
    geometry: ResidualReserveGeometry,
) -> None:
    if not isinstance(inventory, PCInventoryResult):
        raise ODEBFContractError("route assembly inventory result type differs")
    receipt = inventory.receipt
    expected_p_identity = inventory.p_proxy.raw_free_payload()["identity_sha256"]
    expected_c_identity = inventory.c_proxy.raw_free_payload()["identity_sha256"]
    if (
        receipt.mass != float(geometry.mass.detach().to(device="cpu"))
        or receipt.mass_sha256 != tensor_sha256(geometry.mass)
        or receipt.p_proxy_identity != expected_p_identity
        or receipt.c_proxy_identity != expected_c_identity
        or len(receipt.layer_receipts) != len(layer_inputs)
        or receipt.model_forward_count != 0
        or receipt.model_backward_count != 0
        or receipt.dense_materialization_count != 0
        or receipt.post_storage_decision_influence_count != 0
    ):
        raise ODEBFStateError("route assembly inventory provenance differs")
    p_linear = torch.tensor(
        [item.p_linear for item in receipt.layer_receipts],
        dtype=torch.float32,
    )
    p_quadratic = torch.diag(
        torch.tensor(
            [item.p_quadratic_diagonal for item in receipt.layer_receipts],
            dtype=torch.float32,
        )
    )
    c_quadratic = torch.diag(
        torch.tensor(
            [item.c_quadratic_diagonal for item in receipt.layer_receipts],
            dtype=torch.float32,
        )
    )
    if (
        inventory.p_proxy.constant
        != math.fsum(item.committed_p_constant for item in layer_inputs)
        or not torch.equal(inventory.p_proxy.linear, p_linear)
        or not torch.equal(inventory.p_proxy.quadratic, p_quadratic)
        or inventory.c_proxy.constant != 0.0
        or not torch.equal(inventory.c_proxy.linear, torch.zeros(5))
        or not torch.equal(inventory.c_proxy.quadratic, c_quadratic)
    ):
        raise ODEBFStateError("route assembly proxy coefficient provenance differs")
    for item, layer_receipt in zip(
        layer_inputs,
        receipt.layer_receipts,
        strict=True,
    ):
        expected_committed_identities = tuple(
            factor.identity_sha256 for factor in item.committed_drift_factors
        )
        if (
            layer_receipt.layer != item.layer
            or layer_receipt.parameter_shape != item.nominal_factor.parameter_shape
            or layer_receipt.nominal_factor_identity
            != item.nominal_factor.identity_sha256
            or layer_receipt.committed_factor_count
            != len(item.committed_drift_factors)
            or layer_receipt.committed_factor_identities
            != expected_committed_identities
            or layer_receipt.covariance_identity != item.covariance.identity_sha256
            or layer_receipt.covariance_artifact_identity
            != item.covariance.artifact_identity
            or layer_receipt.committed_p_constant != item.committed_p_constant
            or layer_receipt.cumulative_precast_lambda
            != item.cumulative_precast_lambda
            or layer_receipt.pretrained_weight_norm_squared
            != item.pretrained_weight_norm_squared
            or layer_receipt.covariance_right_matmul_count != 1
            or layer_receipt.dense_drift_materialization_count != 0
            or layer_receipt.dense_nominal_materialization_count != 0
            or layer_receipt.dense_candidate_materialization_count != 0
            or layer_receipt.post_storage_decision_influence_count != 0
        ):
            raise ODEBFStateError(
                "route assembly layer inventory provenance differs"
            )


def _pi_tuple(value: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().to(device="cpu"))


def _validate_router_pi(
    value: torch.Tensor,
    *,
    expected: tuple[float, ...],
    name: str,
) -> None:
    expected_tensor = torch.tensor(expected, dtype=torch.float32, device="cpu")
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype is not torch.float32
        or value.device.type != "cpu"
        or value.shape != (len(_LAYER_ORDER),)
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
        or bool((value < 0.0).any())
        or abs(float(value.sum(dtype=torch.float32)) - 1.0)
        > FP32_SIMPLEX_SUM_TOLERANCE
        or _pi_tuple(value) != expected
        or tensor_sha256(value) != tensor_sha256(expected_tensor)
    ):
        raise ODEBFStateError(f"route assembly {name} router pi differs")


def _validate_routing_result(
    routing: ResidualReservePCRoutingResult,
    inventory: PCInventoryResult,
) -> torch.Tensor:
    if not isinstance(routing, ResidualReservePCRoutingResult):
        raise ODEBFContractError("route assembly router result type differs")
    receipt = routing.receipt
    if (
        receipt.p_proxy != inventory.p_proxy.raw_free_payload()
        or receipt.c_proxy != inventory.c_proxy.raw_free_payload()
        or tuple(item.role for item in receipt.candidates)
        != ("P_ONLY", "C_ONLY", "BALANCED", "UNIFORM")
    ):
        raise ODEBFStateError("route assembly router proxy provenance differs")
    outputs = (
        ("P_ONLY", routing.pi_p),
        ("C_ONLY", routing.pi_c),
        ("BALANCED", routing.pi_balanced),
        ("UNIFORM", routing.pi_uniform),
    )
    pointers: list[int] = []
    for (role, value), candidate in zip(outputs, receipt.candidates, strict=True):
        if candidate.role != role:
            raise ODEBFStateError("route assembly router role differs")
        _validate_router_pi(value, expected=candidate.pi, name=role)
        pointers.append(int(value.data_ptr()))
    balanced = receipt.candidates[2]
    if (
        len(set(pointers)) != len(pointers)
        or balanced.solver_status != receipt.selected_status
        or balanced.pi != _pi_tuple(routing.pi_balanced)
    ):
        raise ODEBFStateError("route assembly selected router pi provenance differs")
    return routing.pi_balanced


def assemble_residual_reserve_pc_route(
    nominal_shadow: NominalShadowResult,
    committed_state: CommittedGrossLoadState,
    covariances: tuple[SealedPrevalidatedCovariance, ...],
) -> ResidualReserveRouteAssemblyResult:
    """Build P/C proxies, solve once, and freeze one selected M1 geometry."""

    _validate_nominal_shadow(nominal_shadow)
    _validate_committed_and_covariance_inputs(committed_state, covariances)
    guards = _guard_tensors(
        _nominal_input_tensors(nominal_shadow)
        + _committed_input_tensors(committed_state)
        + tuple(item.value for item in covariances)
    )
    all_factor_pointers = [
        int(tensor.data_ptr())
        for factor in nominal_shadow.factors
        for tensor in (factor.left, factor.right)
    ] + [
        int(tensor.data_ptr())
        for layer in committed_state.layers
        for factor in layer.committed_precast_factors
        for tensor in (factor.left, factor.right)
    ]
    if len(all_factor_pointers) != len(set(all_factor_pointers)):
        raise ODEBFStateError("route assembly factor inventory aliases")

    layer_inputs: list[LayerPCInventoryInput] = []
    layer_receipts: list[RouteAssemblyLayerBindingReceipt] = []
    for nominal, shadow_layer, state_layer, covariance in zip(
        nominal_shadow.factors,
        nominal_shadow.layer_results,
        committed_state.layers,
        covariances,
        strict=True,
    ):
        layer = state_layer.layer
        if (
            nominal.layer != layer
            or shadow_layer.receipt.layer != layer
            or shadow_layer.receipt.nominal_factor_identity
            != nominal.identity_sha256
            or nominal.parameter_shape
            != (
                state_layer.committed_precast_factors[0].parameter_shape
                if state_layer.committed_precast_factors
                else nominal.parameter_shape
            )
        ):
            raise ODEBFStateError("route assembly layer component is stale")
        layer_inputs.append(
            LayerPCInventoryInput(
                layer=layer,
                nominal_factor=nominal,
                committed_drift_factors=state_layer.committed_precast_factors,
                covariance=covariance,
                committed_p_constant=state_layer.structural_p_constant,
                cumulative_precast_lambda=state_layer.cumulative_precast_lambda,
                pretrained_weight_norm_squared=(
                    state_layer.pretrained_weight_norm_squared
                ),
            )
        )
        layer_receipts.append(
            RouteAssemblyLayerBindingReceipt(
                layer=layer,
                nominal_factor_identity=nominal.identity_sha256,
                nominal_shadow_layer_receipt_identity=(
                    shadow_layer.receipt.identity_sha256
                ),
                committed_layer_decision_identity=(
                    state_layer.decision_identity_sha256
                ),
                committed_factor_identities=tuple(
                    factor.identity_sha256
                    for factor in state_layer.committed_precast_factors
                ),
                committed_factor_count=len(
                    state_layer.committed_precast_factors
                ),
                covariance_identity=covariance.identity_sha256,
                covariance_artifact_identity=covariance.artifact_identity,
                covariance_anchor_identity=state_layer.covariance.identity_sha256,
                committed_p_constant=state_layer.structural_p_constant,
                cumulative_precast_lambda=state_layer.cumulative_precast_lambda,
                pretrained_weight_norm_squared=(
                    state_layer.pretrained_weight_norm_squared
                ),
                current_candidate_factor_count=0,
                current_prefix_factor_count=0,
                input_alias_count=0,
            )
        )

    frozen_layer_inputs = tuple(layer_inputs)
    inventory = build_residual_reserve_pc_inventory(
        frozen_layer_inputs,
        mass=nominal_shadow.geometry.mass,
    )
    _validate_inventory_result(
        inventory,
        frozen_layer_inputs,
        nominal_shadow.geometry,
    )
    routing = solve_residual_reserve_pc_router(
        inventory.p_proxy,
        inventory.c_proxy,
    )
    router_selected_pi = _validate_routing_result(routing, inventory)
    execution_device = nominal_shadow.geometry.pi.device
    selected_pi = (
        router_selected_pi.detach()
        .to(device=execution_device, dtype=torch.float32, copy=True)
        .contiguous()
    )
    selected_geometry = build_residual_reserve_geometry(selected_pi)
    _verify_guards(guards)
    router_pi_sha256 = tensor_sha256(router_selected_pi)
    execution_pi_sha256 = tensor_sha256(selected_pi)
    geometry_pi_sha256 = tensor_sha256(selected_geometry.pi)
    cross_device_copy_count = int(router_selected_pi.device != execution_device)
    if (
        selected_pi.dtype is not torch.float32
        or selected_pi.device != execution_device
        or selected_pi.requires_grad
        or int(selected_pi.data_ptr()) == int(routing.pi_balanced.data_ptr())
        or selected_geometry.pi.device != execution_device
        or int(selected_pi.data_ptr()) == int(selected_geometry.pi.data_ptr())
        or router_pi_sha256 != execution_pi_sha256
        or execution_pi_sha256 != geometry_pi_sha256
        or inventory.receipt.p_proxy_identity
        != inventory.p_proxy.raw_free_payload()["identity_sha256"]
        or inventory.receipt.c_proxy_identity
        != inventory.c_proxy.raw_free_payload()["identity_sha256"]
    ):
        raise ODEBFStateError("route assembly frozen selected route differs")

    ledger = RouteAssemblyLedger(
        inventory_build_count=1,
        covariance_right_matmul_count=sum(
            item.covariance_right_matmul_count
            for item in inventory.receipt.layer_receipts
        ),
        router_call_count=1,
        router_decision_to_execution_pi_copy_count=1,
        cross_device_pi_copy_count=cross_device_copy_count,
        selected_geometry_build_count=1,
        model_forward_count=inventory.receipt.model_forward_count,
        model_backward_count=inventory.receipt.model_backward_count,
        prefix_observation_count=0,
        q_solve_count=0,
        dense_writer_update_construction_count=0,
        native_apply_count=0,
        storage_cast_count=0,
        transaction_count=0,
        materialization_count=inventory.receipt.dense_materialization_count,
        ledger_append_count=0,
        history_append_count=0,
        heldout_evaluator_count=0,
        weight_mutation_count=0,
        post_storage_decision_influence_count=(
            inventory.receipt.post_storage_decision_influence_count
        ),
    )
    expected_ledger = {
        "inventory_build_count": 1,
        "covariance_right_matmul_count": 5,
        "router_call_count": 1,
        "router_decision_to_execution_pi_copy_count": 1,
        "cross_device_pi_copy_count": cross_device_copy_count,
        "selected_geometry_build_count": 1,
        "model_forward_count": 0,
        "model_backward_count": 0,
        "prefix_observation_count": 0,
        "q_solve_count": 0,
        "dense_writer_update_construction_count": 0,
        "native_apply_count": 0,
        "storage_cast_count": 0,
        "transaction_count": 0,
        "materialization_count": 0,
        "ledger_append_count": 0,
        "history_append_count": 0,
        "heldout_evaluator_count": 0,
        "weight_mutation_count": 0,
        "post_storage_decision_influence_count": 0,
    }
    if ledger.raw_free_payload() != expected_ledger:
        raise ODEBFStateError("route assembly compute ledger differs")

    receipt = ResidualReserveRouteAssemblyReceipt(
        status=ROUTE_ASSEMBLY_STATUS,
        nominal_scientific_identity=nominal_shadow.receipt.scientific_identity_sha256,
        nominal_receipt_identity=nominal_shadow.receipt.identity_sha256,
        nominal_restored_entry_identity=canonical_hash(
            [item.scientific_payload() for item in nominal_shadow.receipt.restored_weight_state]
        ),
        committed_state_id=committed_state.state_id,
        committed_state_version=committed_state.version,
        committed_state_identity=committed_state.identity_sha256,
        committed_state_decision_identity=committed_state.decision_identity_sha256,
        layer_bindings=tuple(layer_receipts),
        covariance_artifact_identities=tuple(
            item.artifact_identity for item in covariances
        ),
        inventory_receipt_identity=inventory.receipt.identity_sha256,
        p_proxy_identity=inventory.receipt.p_proxy_identity,
        c_proxy_identity=inventory.receipt.c_proxy_identity,
        router_receipt_identity=routing.receipt.raw_free_payload()["identity_sha256"],
        router_selected_status=routing.receipt.selected_status,
        router_selected_pi=_pi_tuple(router_selected_pi),
        router_selected_pi_sha256=router_pi_sha256,
        router_decision_device=str(router_selected_pi.device),
        selected_pi=_pi_tuple(selected_pi),
        selected_pi_sha256=execution_pi_sha256,
        selected_execution_device=str(execution_device),
        router_decision_to_execution_pi_copy_count=1,
        cross_device_pi_copy_count=cross_device_copy_count,
        device_independent_pi_bytes_exact=True,
        selected_geometry_identity=selected_geometry.receipt.raw_free_payload()[
            "identity_sha256"
        ],
        selected_geometry_pi_sha256=geometry_pi_sha256,
        selected_geometry_build_count=1,
        selected_pi_frozen_nonalias=True,
        input_immutability_verified=True,
        ledger=ledger,
    )
    receipt.identity_sha256
    return ResidualReserveRouteAssemblyResult(
        inventory,
        routing,
        selected_pi,
        selected_geometry,
        receipt,
    )


def observe_uniform_route_without_resolve(
    result: ResidualReserveRouteAssemblyResult,
) -> UniformDecisionOffObservation:
    """Evaluate the frozen uniform candidate without a second router call."""

    if (
        not isinstance(result, ResidualReserveRouteAssemblyResult)
        or result.receipt.status != ROUTE_ASSEMBLY_STATUS
        or result.receipt.inventory_receipt_identity
        != result.inventory.receipt.identity_sha256
        or result.receipt.router_receipt_identity
        != result.routing.receipt.raw_free_payload()["identity_sha256"]
        or result.receipt.selected_pi_sha256 != tensor_sha256(result.selected_pi)
        or result.receipt.router_selected_pi_sha256
        != tensor_sha256(result.routing.pi_balanced)
        or result.receipt.router_selected_pi_sha256
        != result.receipt.selected_pi_sha256
        or result.receipt.router_selected_pi
        != _pi_tuple(result.routing.pi_balanced)
        or result.receipt.selected_pi != _pi_tuple(result.selected_pi)
        or result.receipt.selected_geometry_pi_sha256
        != tensor_sha256(result.selected_geometry.pi)
        or result.receipt.selected_execution_device
        != str(result.selected_pi.device)
        or result.selected_geometry.pi.device != result.selected_pi.device
    ):
        raise ODEBFContractError("uniform decision-off source differs")
    _validate_routing_result(result.routing, result.inventory)
    router_uniform_pi = result.routing.pi_uniform
    uniform_pi = (
        router_uniform_pi.detach()
        .to(device=result.selected_pi.device, dtype=torch.float32, copy=True)
        .contiguous()
    )
    geometry = build_residual_reserve_geometry(uniform_pi)
    router_pi_sha256 = tensor_sha256(router_uniform_pi)
    execution_pi_sha256 = tensor_sha256(uniform_pi)
    cross_device_copy_count = int(
        router_uniform_pi.device != result.selected_pi.device
    )
    if (
        router_pi_sha256 != execution_pi_sha256
        or execution_pi_sha256 != tensor_sha256(geometry.pi)
        or uniform_pi.device != result.selected_pi.device
        or geometry.pi.device != result.selected_pi.device
        or int(uniform_pi.data_ptr()) == int(router_uniform_pi.data_ptr())
        or int(uniform_pi.data_ptr()) == int(geometry.pi.data_ptr())
    ):
        raise ODEBFStateError("uniform decision-off execution pi differs")
    observation = UniformDecisionOffObservation(
        pi=uniform_pi,
        geometry=geometry,
        p_value=evaluate_quadratic_proxy(
            result.inventory.p_proxy,
            router_uniform_pi,
        ),
        c_value=evaluate_quadratic_proxy(
            result.inventory.c_proxy,
            router_uniform_pi,
        ),
        source_route_assembly_identity=result.receipt.identity_sha256,
        router_pi_sha256=router_pi_sha256,
        router_decision_device=str(router_uniform_pi.device),
        execution_device=str(uniform_pi.device),
        router_decision_to_execution_pi_copy_count=1,
        cross_device_pi_copy_count=cross_device_copy_count,
        device_independent_pi_bytes_exact=True,
        router_call_count=0,
        decision_influence_count=0,
        status=UNIFORM_DECISION_OFF_STATUS,
    )
    observation.raw_free_payload()
    return observation


__all__ = [
    "ROUTE_ASSEMBLY_STATUS",
    "UNIFORM_DECISION_OFF_STATUS",
    "ResidualReserveRouteAssemblyReceipt",
    "ResidualReserveRouteAssemblyResult",
    "RouteAssemblyLayerBindingReceipt",
    "RouteAssemblyLedger",
    "UniformDecisionOffObservation",
    "assemble_residual_reserve_pc_route",
    "observe_uniform_route_without_resolve",
]
