"""Pure one-layer residual-reserve planner and nominal-factor provenance."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r52_residual_reserve_alpha_q_solve import (
    ALPHA_Q_SOLVE_CONDITION_MAX_DIMENSION,
    AlphaQOnlySolveResult,
    solve_residual_independent_alpha_q_fp32,
)
from .p1r52_residual_reserve_geometry import (
    RESIDUAL_RESERVE_LAYER_COUNT,
    ResidualReserveGeometry,
    current_prefix_residual,
)
from .p1r52_residual_reserve_pc_inventory import (
    LowRankFP32Factor,
    LowRankFactorSource,
)
from .p1r52_residual_reserve_update_binding import (
    INPUT_BETA_APPLICATION_PROOF_STATUS,
    PreparedLowRankUpdate,
    build_prepared_low_rank_update,
)


RESIDUAL_RESERVE_LAYER_ORDER = (4, 5, 6, 7, 8)
LAYER_STEP_BETA_PROOF_STATUS = "CLOSED_BY_M3D_B2A_LAYER_STEP"
NOMINAL_REFERENCE_RECEIPT_NAME = (
    "NOMINAL_PATH_LAGGED_CROSS_AWARE_SURROGATE"
)
NOMINAL_FACTOR_FP32_RTOL = 4.0 * torch.finfo(torch.float32).eps


@dataclass(frozen=True, slots=True)
class LayerStepInputIdentity:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    sha256: str
    pointer: int
    version: int

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "sha256": self.sha256,
            "pointer": self.pointer,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ResidualReserveLayerStepReceipt:
    layer: int
    layer_index: int
    geometry_receipt_identity: str
    geometry_beta_scalar: float
    geometry_beta_tensor_shape: tuple[int, ...]
    geometry_beta_tensor_dtype: str
    geometry_beta_tensor_device: str
    geometry_beta_tensor_sha256: str
    input_identities: tuple[LayerStepInputIdentity, ...]
    residual_shape: tuple[int, int]
    residual_sha256: str
    beta_applied_left_shape: tuple[int, int]
    beta_applied_left_sha256: str
    q_solve_receipt_identity: str
    q_sha256: str
    construction_receipt_identity: str
    construction_factor_identity: str
    construction_update_sha256: str
    construction_orientation: str
    construction_input_beta_proof_status: str
    beta_proof_status: str
    beta_application_count: int
    internal_beta_reapplication_count: int
    q_solve_count: int
    dense_construction_count: int
    second_dense_construction_count: int
    input_pointer_version_hash_immutability_verified: bool
    intermediate_nonalias_verified: bool
    model_forward_count: int
    model_backward_count: int
    semantic_backward_count: int
    slope_backward_count: int
    apply_count: int
    commit_count: int
    materialization_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": self.layer,
            "layer_index": self.layer_index,
            "geometry_receipt_identity": self.geometry_receipt_identity,
            "geometry_beta_scalar": self.geometry_beta_scalar,
            "geometry_beta_tensor_shape": list(
                self.geometry_beta_tensor_shape
            ),
            "geometry_beta_tensor_dtype": self.geometry_beta_tensor_dtype,
            "geometry_beta_tensor_device": self.geometry_beta_tensor_device,
            "geometry_beta_tensor_sha256": self.geometry_beta_tensor_sha256,
            "input_identities": [
                item.raw_free_payload() for item in self.input_identities
            ],
            "residual_shape": list(self.residual_shape),
            "residual_sha256": self.residual_sha256,
            "beta_applied_left_shape": list(self.beta_applied_left_shape),
            "beta_applied_left_sha256": self.beta_applied_left_sha256,
            "q_solve_receipt_identity": self.q_solve_receipt_identity,
            "q_sha256": self.q_sha256,
            "construction_receipt_identity": self.construction_receipt_identity,
            "construction_factor_identity": self.construction_factor_identity,
            "construction_update_sha256": self.construction_update_sha256,
            "construction_orientation": self.construction_orientation,
            "construction_input_beta_proof_status": (
                self.construction_input_beta_proof_status
            ),
            "beta_proof_status": self.beta_proof_status,
            "beta_application_count": self.beta_application_count,
            "internal_beta_reapplication_count": (
                self.internal_beta_reapplication_count
            ),
            "q_solve_count": self.q_solve_count,
            "dense_construction_count": self.dense_construction_count,
            "second_dense_construction_count": (
                self.second_dense_construction_count
            ),
            "input_pointer_version_hash_immutability_verified": (
                self.input_pointer_version_hash_immutability_verified
            ),
            "intermediate_nonalias_verified": self.intermediate_nonalias_verified,
            "model_forward_count": self.model_forward_count,
            "model_backward_count": self.model_backward_count,
            "semantic_backward_count": self.semantic_backward_count,
            "slope_backward_count": self.slope_backward_count,
            "apply_count": self.apply_count,
            "commit_count": self.commit_count,
            "materialization_count": self.materialization_count,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class ResidualReserveLayerStepPlan:
    residual32: torch.Tensor
    q32: torch.Tensor
    beta_applied_left32: torch.Tensor
    prepared_update: PreparedLowRankUpdate
    q_solve: AlphaQOnlySolveResult
    receipt: ResidualReserveLayerStepReceipt


@dataclass(frozen=True, slots=True)
class NominalReferenceFactorReceipt:
    name: str
    layer: int
    construction_receipt_identity: str
    source_factor_identity: str
    source_factor_source: str
    nominal_factor_identity: str
    nominal_factor_source: str
    shadow_prepared_not_applied: bool
    authoritative_transaction_committed_claim: bool
    mass_scalar: float
    pi_reference_scalar: float
    quota_scalar: float
    quota_tensor_sha256: str
    divided_oriented_side: str
    division_count: int
    dense_factor_materialization_count: int
    dense_update_construction_count: int
    second_update_construction_count: int
    source_tensor_byte_identity_before_division: bool
    input_pointer_version_hash_immutability_verified: bool
    nominal_factor_nonalias_verified: bool
    fp32_equivalence_rtol: float

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "layer": self.layer,
            "construction_receipt_identity": (
                self.construction_receipt_identity
            ),
            "source_factor_identity": self.source_factor_identity,
            "source_factor_source": self.source_factor_source,
            "nominal_factor_identity": self.nominal_factor_identity,
            "nominal_factor_source": self.nominal_factor_source,
            "shadow_prepared_not_applied": self.shadow_prepared_not_applied,
            "authoritative_transaction_committed_claim": (
                self.authoritative_transaction_committed_claim
            ),
            "mass_scalar": self.mass_scalar,
            "pi_reference_scalar": self.pi_reference_scalar,
            "quota_scalar": self.quota_scalar,
            "quota_tensor_sha256": self.quota_tensor_sha256,
            "divided_oriented_side": self.divided_oriented_side,
            "division_count": self.division_count,
            "dense_factor_materialization_count": (
                self.dense_factor_materialization_count
            ),
            "dense_update_construction_count": (
                self.dense_update_construction_count
            ),
            "second_update_construction_count": (
                self.second_update_construction_count
            ),
            "source_tensor_byte_identity_before_division": (
                self.source_tensor_byte_identity_before_division
            ),
            "input_pointer_version_hash_immutability_verified": (
                self.input_pointer_version_hash_immutability_verified
            ),
            "nominal_factor_nonalias_verified": (
                self.nominal_factor_nonalias_verified
            ),
            "fp32_equivalence_rtol": self.fp32_equivalence_rtol,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class NominalReferenceFactorResult:
    factor: LowRankFP32Factor
    receipt: NominalReferenceFactorReceipt


def _layer_index(layer: int) -> int:
    if isinstance(layer, bool) or layer not in RESIDUAL_RESERVE_LAYER_ORDER:
        raise ODEBFContractError("residual-reserve layer-step layer is invalid")
    return layer - RESIDUAL_RESERVE_LAYER_ORDER[0]


def _tensor_identity(name: str, value: torch.Tensor) -> LayerStepInputIdentity:
    return LayerStepInputIdentity(
        name=name,
        shape=tuple(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        sha256=tensor_sha256(value),
        pointer=int(value.data_ptr()),
        version=int(value._version),
    )


def _validate_geometry(
    geometry: ResidualReserveGeometry,
    *,
    device: torch.device,
) -> str:
    if not isinstance(geometry, ResidualReserveGeometry):
        raise ODEBFContractError("residual-reserve layer-step geometry type differs")
    tensors = (
        geometry.pi,
        geometry.omega,
        geometry.suffix_retention,
        geometry.beta,
    )
    if (
        any(
            not isinstance(value, torch.Tensor)
            or value.dtype is not torch.float32
            or value.device != device
            or value.shape != (RESIDUAL_RESERVE_LAYER_COUNT,)
            or not bool(torch.isfinite(value).all())
            for value in tensors
        )
        or geometry.rho.dtype is not torch.float32
        or geometry.mass.dtype is not torch.float32
        or geometry.rho.device != device
        or geometry.mass.device != device
        or geometry.rho.ndim != 0
        or geometry.mass.ndim != 0
    ):
        raise ODEBFContractError("residual-reserve layer-step geometry differs")
    receipt = geometry.receipt.raw_free_payload()
    if (
        tuple(float(item) for item in geometry.pi.detach().to(device="cpu"))
        != geometry.receipt.pi
        or tuple(float(item) for item in geometry.beta.detach().to(device="cpu"))
        != geometry.receipt.beta
        or float(geometry.mass) != geometry.receipt.mass
        or float(geometry.rho) != geometry.receipt.rho
    ):
        raise ODEBFStateError("residual-reserve geometry tensor/receipt differs")
    return str(receipt["identity_sha256"])


def _guard_inputs(
    named: tuple[tuple[str, torch.Tensor], ...],
) -> tuple[LayerStepInputIdentity, ...]:
    return tuple(_tensor_identity(name, value) for name, value in named)


def plan_residual_reserve_layer_step(
    target_proposal32: torch.Tensor,
    current_terminal32: torch.Tensor,
    joint_keys32: torch.Tensor,
    projector32: torch.Tensor,
    committed_covariance32: torch.Tensor,
    regularization32: torch.Tensor,
    geometry: ResidualReserveGeometry,
    *,
    layer: int,
    weight_name: str,
    parameter_shape: tuple[int, int],
    construction_id: str,
    q_residual_tolerance: float,
    q_condition_max_dimension: int = ALPHA_Q_SOLVE_CONDITION_MAX_DIMENSION,
) -> ResidualReserveLayerStepPlan:
    """Plan one layer without apply, transaction, model, or route mutation."""

    layer_index = _layer_index(layer)
    named_inputs = (
        ("target_proposal", target_proposal32),
        ("current_terminal", current_terminal32),
        ("joint_keys", joint_keys32),
        ("projector", projector32),
        ("committed_covariance", committed_covariance32),
        ("regularization", regularization32),
    )
    if any(not isinstance(value, torch.Tensor) for _, value in named_inputs):
        raise ODEBFContractError("residual-reserve layer-step input type differs")
    device = target_proposal32.device
    if (
        target_proposal32.dtype is not torch.float32
        or current_terminal32.dtype is not torch.float32
        or target_proposal32.ndim != 2
        or current_terminal32.ndim != 2
        or target_proposal32.shape != current_terminal32.shape
        or any(value.device != device for _, value in named_inputs)
    ):
        raise ODEBFContractError("residual-reserve layer-step input geometry differs")
    geometry_identity = _validate_geometry(geometry, device=device)
    input_identities = _guard_inputs(named_inputs)
    geometry_guards = _guard_inputs(
        (
            ("geometry_pi", geometry.pi),
            ("geometry_omega", geometry.omega),
            ("geometry_suffix_retention", geometry.suffix_retention),
            ("geometry_beta", geometry.beta),
            ("geometry_rho", geometry.rho),
            ("geometry_mass", geometry.mass),
        )
    )

    residual32 = current_prefix_residual(
        target_proposal32,
        current_terminal32,
    )
    q_solve = solve_residual_independent_alpha_q_fp32(
        projector32,
        joint_keys32,
        committed_covariance32,
        regularization32,
        layer=layer,
        residual_tolerance=q_residual_tolerance,
        condition_max_dimension=q_condition_max_dimension,
    )
    q32 = q_solve.q32
    if (
        residual32.dtype is not torch.float32
        or q32.dtype is not torch.float32
        or residual32.device != device
        or q32.device != device
        or residual32.ndim != 2
        or q32.ndim != 2
        or residual32.shape[1] != q32.shape[1]
    ):
        raise ODEBFContractError(
            "residual-reserve residual/q request axis differs"
        )
    beta32 = geometry.beta[layer_index]
    beta_applied_left32 = (beta32 * residual32).contiguous()
    if (
        beta32.dtype is not torch.float32
        or beta32.device != device
        or beta32.ndim != 0
        or beta_applied_left32.dtype is not torch.float32
        or beta_applied_left32.device != device
        or not bool(torch.isfinite(beta_applied_left32).all())
    ):
        raise ODEBFContractError("residual-reserve beta application differs")
    prepared_update = build_prepared_low_rank_update(
        construction_id=construction_id,
        layer=layer,
        weight_name=weight_name,
        parameter_shape=parameter_shape,
        beta_applied_left32=beta_applied_left32,
        q_side32=q32,
    )
    construction_receipt = prepared_update.receipt
    if (
        construction_receipt.input_beta_application_proof_status
        != INPUT_BETA_APPLICATION_PROOF_STATUS
        or construction_receipt.internal_beta_reapplication_count != 0
        or construction_receipt.raw_left_sha256
        != tensor_sha256(beta_applied_left32)
        or construction_receipt.raw_right_sha256 != q_solve.receipt.q_sha256
        or construction_receipt.dense_construction_count != 1
        or construction_receipt.second_dense_construction_count != 0
    ):
        raise ODEBFStateError("residual-reserve construction binding differs")

    observed_input_identities = _guard_inputs(named_inputs)
    observed_geometry_guards = _guard_inputs(
        (
            ("geometry_pi", geometry.pi),
            ("geometry_omega", geometry.omega),
            ("geometry_suffix_retention", geometry.suffix_retention),
            ("geometry_beta", geometry.beta),
            ("geometry_rho", geometry.rho),
            ("geometry_mass", geometry.mass),
        )
    )
    if (
        observed_input_identities != input_identities
        or observed_geometry_guards != geometry_guards
        or _validate_geometry(geometry, device=device) != geometry_identity
    ):
        raise ODEBFStateError("residual-reserve layer-step input mutated")
    input_pointers = {item.pointer for item in input_identities + geometry_guards}
    intermediates = (
        residual32,
        q32,
        beta_applied_left32,
        prepared_update.factor.left,
        prepared_update.factor.right,
        prepared_update.matched_update32,
    )
    intermediate_pointers = tuple(int(item.data_ptr()) for item in intermediates)
    nonalias = (
        not any(pointer in input_pointers for pointer in intermediate_pointers)
        and len(set(intermediate_pointers)) == len(intermediate_pointers)
    )
    if not nonalias:
        raise ODEBFStateError("residual-reserve layer-step intermediate aliases")

    receipt = ResidualReserveLayerStepReceipt(
        layer=layer,
        layer_index=layer_index,
        geometry_receipt_identity=geometry_identity,
        geometry_beta_scalar=float(beta32),
        geometry_beta_tensor_shape=tuple(beta32.shape),
        geometry_beta_tensor_dtype=str(beta32.dtype),
        geometry_beta_tensor_device=str(beta32.device),
        geometry_beta_tensor_sha256=tensor_sha256(beta32),
        input_identities=input_identities,
        residual_shape=tuple(residual32.shape),
        residual_sha256=tensor_sha256(residual32),
        beta_applied_left_shape=tuple(beta_applied_left32.shape),
        beta_applied_left_sha256=tensor_sha256(beta_applied_left32),
        q_solve_receipt_identity=q_solve.receipt.identity_sha256,
        q_sha256=q_solve.receipt.q_sha256,
        construction_receipt_identity=construction_receipt.identity_sha256,
        construction_factor_identity=prepared_update.factor.identity_sha256,
        construction_update_sha256=construction_receipt.matched_update_sha256,
        construction_orientation=construction_receipt.matched_orientation,
        construction_input_beta_proof_status=(
            construction_receipt.input_beta_application_proof_status
        ),
        beta_proof_status=LAYER_STEP_BETA_PROOF_STATUS,
        beta_application_count=1,
        internal_beta_reapplication_count=0,
        q_solve_count=1,
        dense_construction_count=1,
        second_dense_construction_count=0,
        input_pointer_version_hash_immutability_verified=True,
        intermediate_nonalias_verified=True,
        model_forward_count=0,
        model_backward_count=0,
        semantic_backward_count=0,
        slope_backward_count=0,
        apply_count=0,
        commit_count=0,
        materialization_count=0,
    )
    receipt.identity_sha256
    return ResidualReserveLayerStepPlan(
        residual32=residual32,
        q32=q32,
        beta_applied_left32=beta_applied_left32,
        prepared_update=prepared_update,
        q_solve=q_solve,
        receipt=receipt,
    )


def derive_nominal_reference_factor(
    shadow_prepared_update: PreparedLowRankUpdate,
    geometry: ResidualReserveGeometry,
    *,
    layer: int,
) -> NominalReferenceFactorResult:
    """Convert a prepared shadow construction into quota-normalized A_l."""

    layer_index = _layer_index(layer)
    if not isinstance(shadow_prepared_update, PreparedLowRankUpdate):
        raise ODEBFContractError("nominal reference construction type differs")
    shadow_prepared_update.validate()
    source = shadow_prepared_update.factor
    if (
        source.layer != layer
        or source.source
        is not LowRankFactorSource.AUTHORITATIVE_PREPARED_PRECAST_FP32
    ):
        raise ODEBFContractError("nominal reference factor provenance differs")
    geometry_identity = _validate_geometry(
        geometry,
        device=source.left.device,
    )
    del geometry_identity
    source_guards = (
        source.identity_sha256,
        int(source.left.data_ptr()),
        int(source.left._version),
        tensor_sha256(source.left),
        int(source.right.data_ptr()),
        int(source.right._version),
        tensor_sha256(source.right),
        shadow_prepared_update.receipt.identity_sha256,
    )
    geometry_guards = _guard_inputs(
        (
            ("geometry_pi", geometry.pi),
            ("geometry_mass", geometry.mass),
        )
    )
    pi_reference32 = geometry.pi[layer_index]
    quota32 = geometry.mass * pi_reference32
    if (
        pi_reference32.dtype is not torch.float32
        or quota32.dtype is not torch.float32
        or quota32.device != source.left.device
        or not bool(torch.isfinite(quota32))
        or float(quota32) <= 0.0
    ):
        raise ODEBFContractError("nominal reference quota is not positive FP32")
    divided_left32 = (source.left / quota32).contiguous()
    nominal_factor = LowRankFP32Factor(
        layer=layer,
        parameter_shape=source.parameter_shape,
        left=divided_left32,
        right=source.right,
        source=LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
    )
    observed_source_guards = (
        source.identity_sha256,
        int(source.left.data_ptr()),
        int(source.left._version),
        tensor_sha256(source.left),
        int(source.right.data_ptr()),
        int(source.right._version),
        tensor_sha256(source.right),
        shadow_prepared_update.receipt.identity_sha256,
    )
    if (
        observed_source_guards != source_guards
        or _guard_inputs(
            (
                ("geometry_pi", geometry.pi),
                ("geometry_mass", geometry.mass),
            )
        )
        != geometry_guards
    ):
        raise ODEBFStateError("nominal reference input mutated")
    nonalias = (
        int(nominal_factor.left.data_ptr()) != int(source.left.data_ptr())
        and int(nominal_factor.right.data_ptr()) != int(source.right.data_ptr())
        and int(nominal_factor.left.data_ptr())
        != int(nominal_factor.right.data_ptr())
    )
    if not nonalias:
        raise ODEBFStateError("nominal reference factor aliases source")
    receipt = NominalReferenceFactorReceipt(
        name=NOMINAL_REFERENCE_RECEIPT_NAME,
        layer=layer,
        construction_receipt_identity=(
            shadow_prepared_update.receipt.identity_sha256
        ),
        source_factor_identity=source.identity_sha256,
        source_factor_source=source.source.value,
        nominal_factor_identity=nominal_factor.identity_sha256,
        nominal_factor_source=nominal_factor.source.value,
        shadow_prepared_not_applied=True,
        authoritative_transaction_committed_claim=False,
        mass_scalar=float(geometry.mass),
        pi_reference_scalar=float(pi_reference32),
        quota_scalar=float(quota32),
        quota_tensor_sha256=tensor_sha256(quota32),
        divided_oriented_side="PARAMETER_ORIENTED_LEFT",
        division_count=1,
        dense_factor_materialization_count=0,
        dense_update_construction_count=0,
        second_update_construction_count=0,
        source_tensor_byte_identity_before_division=True,
        input_pointer_version_hash_immutability_verified=True,
        nominal_factor_nonalias_verified=True,
        fp32_equivalence_rtol=NOMINAL_FACTOR_FP32_RTOL,
    )
    receipt.identity_sha256
    return NominalReferenceFactorResult(nominal_factor, receipt)


__all__ = [
    "LAYER_STEP_BETA_PROOF_STATUS",
    "NOMINAL_FACTOR_FP32_RTOL",
    "NOMINAL_REFERENCE_RECEIPT_NAME",
    "LayerStepInputIdentity",
    "NominalReferenceFactorReceipt",
    "NominalReferenceFactorResult",
    "ResidualReserveLayerStepPlan",
    "ResidualReserveLayerStepReceipt",
    "derive_nominal_reference_factor",
    "plan_residual_reserve_layer_step",
]
