"""Committed low-rank structural-P and gross-load ledger for residual reserve."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r52_residual_reserve_fp32_transaction import (
    FP32PreparedCommitReceipt,
    FP32TransactionMode,
    FP32TransactionReceipt,
    OfficialStyleFP32SequentialTransaction,
    RESIDUAL_RESERVE_LAYER_ORDER,
)
from .p1r52_residual_reserve_pc_inventory import (
    LAMBDA_SOURCE,
    LowRankFP32Factor,
    LowRankFactorSource,
    SealedPrevalidatedCovariance,
    _nonnegative_reduction,
    _small_gram_trace,
)
from .p1r52_residual_reserve_pc_router import SOLVER_PRIMAL_TOLERANCE
from .p1r52_residual_reserve_update_binding import (
    EXACT_SINGLE_CONSTRUCTION_STATUS,
    AppliedPreparedLowRankUpdate,
)


FACTOR_UPDATE_EQUIVALENCE_STATUS = EXACT_SINGLE_CONSTRUCTION_STATUS


def _clone_factor(
    factor: LowRankFP32Factor,
    *,
    source: LowRankFactorSource | None = None,
) -> LowRankFP32Factor:
    return LowRankFP32Factor(
        factor.layer,
        factor.parameter_shape,
        factor.left,
        factor.right,
        factor.source if source is None else source,
    )


def _factor_identity_with_source(
    factor: LowRankFP32Factor,
    source: LowRankFactorSource,
) -> str:
    payload = factor.raw_free_payload()
    payload["source"] = source.value
    payload.pop("identity_sha256")
    return canonical_hash(payload)


def _clone_covariance(
    covariance: SealedPrevalidatedCovariance,
) -> SealedPrevalidatedCovariance:
    return SealedPrevalidatedCovariance(
        covariance.layer,
        covariance.value,
        covariance.artifact_identity,
    )


@dataclass(frozen=True, slots=True)
class LayerCommittedStateAnchor:
    layer: int
    covariance: SealedPrevalidatedCovariance
    pretrained_weight_norm_squared: float
    share_sealed_covariance: bool = False

    def __post_init__(self) -> None:
        if (
            self.layer not in RESIDUAL_RESERVE_LAYER_ORDER
            or not isinstance(self.covariance, SealedPrevalidatedCovariance)
            or self.covariance.layer != self.layer
        ):
            raise ODEBFContractError("committed-state layer anchor differs")
        norm_squared = float(self.pretrained_weight_norm_squared)
        if not math.isfinite(norm_squared) or norm_squared <= 0.0:
            raise ODEBFContractError("committed-state pretrained norm is invalid")
        if not isinstance(self.share_sealed_covariance, bool):
            raise ODEBFContractError("committed-state covariance sharing flag differs")
        if not self.share_sealed_covariance:
            object.__setattr__(self, "covariance", _clone_covariance(self.covariance))
        object.__setattr__(self, "pretrained_weight_norm_squared", norm_squared)


@dataclass(frozen=True, slots=True)
class CommittedLayerState:
    layer: int
    covariance: SealedPrevalidatedCovariance
    pretrained_weight_norm_squared: float
    committed_precast_factors: tuple[LowRankFP32Factor, ...]
    structural_p_constant: float
    cumulative_precast_lambda: float
    cumulative_actual_post_storage_load_observation: float
    version: int
    committed_transaction_ids: tuple[str, ...]
    committed_construction_receipt_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.layer not in RESIDUAL_RESERVE_LAYER_ORDER
            or not isinstance(self.covariance, SealedPrevalidatedCovariance)
            or self.covariance.layer != self.layer
        ):
            raise ODEBFContractError("committed layer state geometry differs")
        if (
            not isinstance(self.committed_precast_factors, tuple)
            or any(
                not isinstance(factor, LowRankFP32Factor)
                or factor.layer != self.layer
                or factor.source
                is not LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32
                or factor.parameter_shape[1] != self.covariance.value.shape[0]
                or factor.left.device != self.covariance.value.device
                for factor in self.committed_precast_factors
            )
            or len(
                {factor.parameter_shape for factor in self.committed_precast_factors}
            )
            > 1
        ):
            raise ODEBFContractError("committed layer factor inventory differs")
        values = (
            self.pretrained_weight_norm_squared,
            self.structural_p_constant,
            self.cumulative_precast_lambda,
            self.cumulative_actual_post_storage_load_observation,
        )
        if (
            any(not math.isfinite(float(value)) for value in values)
            or self.pretrained_weight_norm_squared <= 0.0
            or self.structural_p_constant < 0.0
            or self.cumulative_precast_lambda < 0.0
            or self.cumulative_actual_post_storage_load_observation < 0.0
            or isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 0
            or len(self.committed_precast_factors) != self.version
            or not isinstance(self.committed_transaction_ids, tuple)
            or len(self.committed_transaction_ids) != self.version
            or len(set(self.committed_transaction_ids)) != self.version
            or not isinstance(self.committed_construction_receipt_ids, tuple)
            or len(self.committed_construction_receipt_ids) != self.version
            or len(set(self.committed_construction_receipt_ids)) != self.version
            or any(
                not isinstance(item, str) or not item
                for item in (
                    self.committed_transaction_ids
                    + self.committed_construction_receipt_ids
                )
            )
        ):
            raise ODEBFContractError("committed layer state scalar/version differs")

    def decision_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "covariance_identity": self.covariance.identity_sha256,
            "pretrained_weight_norm_squared": self.pretrained_weight_norm_squared,
            "committed_precast_factor_identities": [
                factor.identity_sha256 for factor in self.committed_precast_factors
            ],
            "structural_p_constant": self.structural_p_constant,
            "cumulative_precast_lambda": self.cumulative_precast_lambda,
            "version": self.version,
            "post_storage_decision_influence_count": 0,
        }

    def raw_free_payload(self) -> dict[str, Any]:
        payload = self.decision_payload()
        payload.update(
            {
                "committed_transaction_ids": list(self.committed_transaction_ids),
                "committed_construction_receipt_ids": list(
                    self.committed_construction_receipt_ids
                ),
                "cumulative_actual_post_storage_load_observation": (
                    self.cumulative_actual_post_storage_load_observation
                ),
                "lambda_source": LAMBDA_SOURCE,
                "actual_post_storage_load_role": "OBSERVATION_ONLY",
            }
        )
        payload["decision_identity_sha256"] = canonical_hash(
            self.decision_payload()
        )
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def decision_identity_sha256(self) -> str:
        return canonical_hash(self.decision_payload())

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class CommittedGrossLoadState:
    state_id: str
    layers: tuple[CommittedLayerState, ...]
    version: int
    commit_count: int
    committed_transaction_ids: tuple[str, ...]
    last_transaction_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ODEBFContractError("committed gross-load state identity is empty")
        if (
            not isinstance(self.layers, tuple)
            or not all(isinstance(item, CommittedLayerState) for item in self.layers)
            or tuple(item.layer for item in self.layers)
            != RESIDUAL_RESERVE_LAYER_ORDER
            or isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 0
            or isinstance(self.commit_count, bool)
            or not isinstance(self.commit_count, int)
            or self.commit_count != self.version
            or not isinstance(self.committed_transaction_ids, tuple)
            or len(self.committed_transaction_ids) != self.version
            or len(set(self.committed_transaction_ids)) != self.version
            or any(
                item.version != self.version
                or item.committed_transaction_ids
                != self.committed_transaction_ids
                for item in self.layers
            )
            or self.last_transaction_id
            != (
                None
                if not self.committed_transaction_ids
                else self.committed_transaction_ids[-1]
            )
            or (
                self.last_transaction_id is not None
                and not isinstance(self.last_transaction_id, str)
            )
        ):
            raise ODEBFContractError("committed gross-load state version differs")

    def decision_payload(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "version": self.version,
            "layers": [item.decision_payload() for item in self.layers],
            "decision_inputs": "COMMITTED_PRECAST_FACTORS_P_LAMBDA_ONLY",
            "post_storage_decision_influence_count": 0,
        }

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state_id": self.state_id,
            "version": self.version,
            "commit_count": self.commit_count,
            "committed_transaction_ids": list(self.committed_transaction_ids),
            "last_transaction_id": self.last_transaction_id,
            "layers": [item.raw_free_payload() for item in self.layers],
            "decision_identity_sha256": self.decision_identity_sha256,
            "atomic_case_reset_requires_new_state": True,
            "sequential_persistence_successful_commit_only": True,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def decision_identity_sha256(self) -> str:
        return canonical_hash(self.decision_payload())

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


def initialize_committed_gross_load_state(
    anchors: tuple[LayerCommittedStateAnchor, ...],
    *,
    state_id: str,
) -> CommittedGrossLoadState:
    if not isinstance(state_id, str) or not state_id:
        raise ODEBFContractError("committed-state identity is empty")
    if (
        not isinstance(anchors, tuple)
        or not all(isinstance(item, LayerCommittedStateAnchor) for item in anchors)
        or tuple(item.layer for item in anchors) != RESIDUAL_RESERVE_LAYER_ORDER
    ):
        raise ODEBFContractError("committed-state anchor inventory differs")
    layers = tuple(
        CommittedLayerState(
            layer=anchor.layer,
            covariance=(
                anchor.covariance
                if anchor.share_sealed_covariance
                else _clone_covariance(anchor.covariance)
            ),
            pretrained_weight_norm_squared=anchor.pretrained_weight_norm_squared,
            committed_precast_factors=(),
            structural_p_constant=0.0,
            cumulative_precast_lambda=0.0,
            cumulative_actual_post_storage_load_observation=0.0,
            version=0,
            committed_transaction_ids=(),
            committed_construction_receipt_ids=(),
        )
        for anchor in anchors
    )
    return CommittedGrossLoadState(
        state_id=state_id,
        layers=layers,
        version=0,
        commit_count=0,
        committed_transaction_ids=(),
        last_transaction_id=None,
    )


@dataclass(frozen=True, slots=True)
class CommittedLayerFactorBinding:
    layer: int
    factor: LowRankFP32Factor
    prepared_factor_identity: str
    prepared_factor_source: str
    committed_factor_identity: str
    committed_factor_source: str
    provenance_transition_count: int
    transaction_id: str
    transaction_mode: str
    m3a_layer_receipt_identity: str
    construction_id: str
    construction_receipt_identity: str
    applied_binding_identity: str
    construction_update_pointer: int
    construction_update_version: int
    pre_cast_update_sha256: str
    pre_cast_update_norm: float
    pre_cast_update_energy: float
    actual_post_storage_delta_sha256: str
    actual_post_storage_delta_norm: float
    actual_post_storage_delta_energy: float
    factor_update_equivalence_status: str
    post_storage_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": self.layer,
            "prepared_factor_identity": self.prepared_factor_identity,
            "prepared_factor_source": self.prepared_factor_source,
            "committed_factor_identity": self.committed_factor_identity,
            "committed_factor_source": self.committed_factor_source,
            "provenance_transition_count": self.provenance_transition_count,
            "transaction_id": self.transaction_id,
            "transaction_mode": self.transaction_mode,
            "m3a_layer_receipt_identity": self.m3a_layer_receipt_identity,
            "construction_id": self.construction_id,
            "construction_receipt_identity": (
                self.construction_receipt_identity
            ),
            "applied_binding_identity": self.applied_binding_identity,
            "construction_update_pointer": self.construction_update_pointer,
            "construction_update_version": self.construction_update_version,
            "pre_cast_update_sha256": self.pre_cast_update_sha256,
            "pre_cast_update_norm": self.pre_cast_update_norm,
            "pre_cast_update_energy": self.pre_cast_update_energy,
            "actual_post_storage_delta_sha256": (
                self.actual_post_storage_delta_sha256
            ),
            "actual_post_storage_delta_norm": self.actual_post_storage_delta_norm,
            "actual_post_storage_delta_energy": self.actual_post_storage_delta_energy,
            "factor_update_equivalence_status": (
                self.factor_update_equivalence_status
            ),
            "post_storage_decision_influence_count": (
                self.post_storage_decision_influence_count
            ),
            "second_dense_update_tensor_creation_count": 0,
            "source_transition_tensor_certificate_stage": (
                "COMMITTED_LAYER_TRANSITION_RECEIPT"
            ),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


def bind_authoritative_transaction_factors(
    prepared: FP32PreparedCommitReceipt,
    constructions: tuple[AppliedPreparedLowRankUpdate, ...],
) -> tuple[CommittedLayerFactorBinding, ...]:
    if not isinstance(prepared, FP32PreparedCommitReceipt):
        raise ODEBFContractError("M3A prepared receipt type differs")
    future = prepared.future_final_receipt
    if (
        prepared.mode != FP32TransactionMode.AUTHORITATIVE.value
        or prepared.layer_order != RESIDUAL_RESERVE_LAYER_ORDER
        or len(prepared.layer_receipts) != len(RESIDUAL_RESERVE_LAYER_ORDER)
        or prepared.native_storage_assignment_count
        != len(RESIDUAL_RESERVE_LAYER_ORDER)
        or prepared.storage_cast_boundary_count
        != len(RESIDUAL_RESERVE_LAYER_ORDER)
        or prepared.numeric_storage_cast_count
        != sum(item.numeric_storage_cast_count for item in prepared.layer_receipts)
        or prepared.rounding_observation_count
        != len(RESIDUAL_RESERVE_LAYER_ORDER)
        or prepared.rounding_mismatch_count
        != sum(
            not item.prepared_vs_actual_byte_exact
            for item in prepared.layer_receipts
        )
        or prepared.bf16_path_call_count != 0
        or prepared.bf16_path_decision_influence_count != 0
        or prepared.logical_outer_commit_count != 0
        or prepared.persistent_commit_count != 0
        or prepared.rollback_count != 0
        or prepared.preparation_parameter_mutation_count != 0
        or prepared.postcast_decision_influence_count != 0
        or prepared.model_forward_count != 0
        or prepared.model_backward_count != 0
        or prepared.candidate_materialization_count != 0
        or prepared.external_materializer_call_count != 0
        or future.transaction_id != prepared.transaction_id
        or future.mode != prepared.mode
        or future.layer_order != prepared.layer_order
        or future.weight_names != prepared.weight_names
        or future.layer_receipts != prepared.layer_receipts
        or future.native_storage_assignment_count
        != len(RESIDUAL_RESERVE_LAYER_ORDER)
        or future.storage_cast_boundary_count
        != len(RESIDUAL_RESERVE_LAYER_ORDER)
        or future.numeric_storage_cast_count != prepared.numeric_storage_cast_count
        or future.rounding_observation_count
        != prepared.rounding_observation_count
        or future.rounding_mismatch_count != prepared.rounding_mismatch_count
        or future.bf16_path_call_count != 0
        or future.bf16_path_decision_influence_count != 0
        or future.logical_outer_commit_count != 1
        or future.persistent_commit_count != 1
        or future.rollback_count != 0
        or future.restored_entry_bytes
        or future.restored_entry_pointers
        or future.final_parameter_sha256 != prepared.prepared_parameter_sha256
        or future.final_parameter_pointers != prepared.prepared_parameter_pointers
        or future.postcast_decision_influence_count != 0
        or future.model_forward_count != 0
        or future.model_backward_count != 0
        or future.candidate_materialization_count != 0
        or future.external_materializer_call_count != 0
    ):
        raise ODEBFContractError("M3A authoritative prepare gate differs")
    if (
        not isinstance(constructions, tuple)
        or not all(
            isinstance(item, AppliedPreparedLowRankUpdate)
            for item in constructions
        )
        or tuple(
            item.construction.receipt.layer for item in constructions
        )
        != RESIDUAL_RESERVE_LAYER_ORDER
    ):
        raise ODEBFContractError("committed construction inventory differs")

    bindings: list[CommittedLayerFactorBinding] = []
    for layer, construction, receipt in zip(
        RESIDUAL_RESERVE_LAYER_ORDER,
        constructions,
        prepared.layer_receipts,
        strict=True,
    ):
        construction.validate()
        factor = construction.construction.factor
        construction_receipt = construction.construction.receipt
        if (
            factor.source
            is not LowRankFactorSource.AUTHORITATIVE_PREPARED_PRECAST_FP32
            or receipt.layer != layer
            or construction.m3a_layer_receipt.identity_sha256
            != receipt.identity_sha256
            or construction_receipt.layer != layer
            or construction_receipt.weight_name != receipt.weight_name
            or receipt.parameter_shape != factor.parameter_shape
            or construction_receipt.parameter_shape != factor.parameter_shape
            or construction_receipt.prepared_factor_identity
            != factor.identity_sha256
            or construction_receipt.matched_update_sha256
            != receipt.pre_cast_fp32_update_sha256
            or receipt.prepared_fp32_update_sha256
            != construction_receipt.matched_update_sha256
            or construction_receipt.matched_update_pointer
            != int(construction.construction.matched_update32.data_ptr())
            or receipt.prepared_fp32_update_pointer
            != construction_receipt.matched_update_pointer
            or construction_receipt.matched_update_version
            != int(construction.construction.matched_update32._version)
            or receipt.prepared_fp32_update_version
            != construction_receipt.matched_update_version
            or receipt.prepared_fp32_update_dtype != "torch.float32"
            or receipt.prepared_fp32_update_device
            != str(construction.construction.matched_update32.device)
            or not receipt.official_reference_endpoint_byte_exact
            or receipt.official_reference_endpoint_sha256
            != receipt.post_storage_parameter_sha256
            or receipt.actual_post_storage_delta32_dtype != "torch.float32"
            or receipt.rounding_telemetry_decision_influence_count != 0
            or receipt.storage_assignment_count != 1
            or receipt.storage_cast_boundary_count != 1
            or receipt.numeric_storage_cast_count
            != int(receipt.storage_cast_required)
            or receipt.bf16_path_call_count != 0
            or receipt.bf16_path_decision_influence_count != 0
            or receipt.autocast_count != 0
            or receipt.downcast_count != 0
            or receipt.quantization_count != 0
            or receipt.postcast_decision_influence_count != 0
            or receipt.weight_name != prepared.weight_names[layer - 4]
            or not receipt.pre_cast_fp32_update_sha256
            or not receipt.actual_post_storage_delta32_sha256
        ):
            raise ODEBFContractError("M3A layer/factor binding differs")
        values = (
            receipt.pre_cast_fp32_update_norm,
            receipt.pre_cast_fp32_update_energy,
            receipt.actual_post_storage_delta32_norm,
            receipt.actual_post_storage_delta32_energy,
        )
        if any(not math.isfinite(item) or item < 0.0 for item in values):
            raise ODEBFContractError("M3A factor binding scalar is invalid")
        prepared_factor = _clone_factor(factor)
        committed_factor_identity = _factor_identity_with_source(
            prepared_factor,
            LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32,
        )
        bindings.append(
            CommittedLayerFactorBinding(
                layer=layer,
                factor=prepared_factor,
                prepared_factor_identity=prepared_factor.identity_sha256,
                prepared_factor_source=prepared_factor.source.value,
                committed_factor_identity=committed_factor_identity,
                committed_factor_source=(
                    LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32.value
                ),
                provenance_transition_count=1,
                transaction_id=prepared.transaction_id,
                transaction_mode=prepared.mode,
                m3a_layer_receipt_identity=receipt.identity_sha256,
                construction_id=construction_receipt.construction_id,
                construction_receipt_identity=(
                    construction_receipt.identity_sha256
                ),
                applied_binding_identity=construction.identity_sha256,
                construction_update_pointer=(
                    construction_receipt.matched_update_pointer
                ),
                construction_update_version=(
                    construction_receipt.matched_update_version
                ),
                pre_cast_update_sha256=receipt.pre_cast_fp32_update_sha256,
                pre_cast_update_norm=receipt.pre_cast_fp32_update_norm,
                pre_cast_update_energy=receipt.pre_cast_fp32_update_energy,
                actual_post_storage_delta_sha256=(
                    receipt.actual_post_storage_delta32_sha256
                ),
                actual_post_storage_delta_norm=(
                    receipt.actual_post_storage_delta32_norm
                ),
                actual_post_storage_delta_energy=(
                    receipt.actual_post_storage_delta32_energy
                ),
                factor_update_equivalence_status=FACTOR_UPDATE_EQUIVALENCE_STATUS,
                post_storage_decision_influence_count=0,
            )
        )
    return tuple(bindings)


@dataclass(frozen=True, slots=True)
class CommittedLayerTransitionReceipt:
    layer: int
    before_version: int
    after_version: int
    prepared_factor_identity: str
    prepared_factor_source: str
    committed_factor_identity: str
    committed_factor_source: str
    provenance_transition_count: int
    source_transition_numerical_tensor_byte_identity: bool
    source_transition_numerical_tensor_nonalias: bool
    factor_update_equivalence_status: str
    factor_append_count: int
    committed_factor_count_before: int
    committed_factor_count_after: int
    p_before: float
    p_cross: float
    p_self: float
    p_increment: float
    p_after: float
    lambda_before: float
    lambda_increment: float
    lambda_after: float
    actual_load_observation_before: float
    actual_load_observation_increment: float
    actual_load_observation_after: float
    covariance_right_matmul_count: int
    dense_materialization_count: int
    transaction_id: str
    binding_identity: str
    construction_id: str
    construction_receipt_identity: str
    applied_binding_identity: str
    pre_cast_update_sha256: str
    actual_post_storage_delta_sha256: str
    post_storage_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "before_version": self.before_version,
            "after_version": self.after_version,
            "prepared_factor_identity": self.prepared_factor_identity,
            "prepared_factor_source": self.prepared_factor_source,
            "committed_factor_identity": self.committed_factor_identity,
            "committed_factor_source": self.committed_factor_source,
            "provenance_transition_count": self.provenance_transition_count,
            "source_transition_numerical_tensor_byte_identity": (
                self.source_transition_numerical_tensor_byte_identity
            ),
            "source_transition_numerical_tensor_nonalias": (
                self.source_transition_numerical_tensor_nonalias
            ),
            "factor_update_equivalence_status": (
                self.factor_update_equivalence_status
            ),
            "factor_append_count": self.factor_append_count,
            "committed_factor_count_before": self.committed_factor_count_before,
            "committed_factor_count_after": self.committed_factor_count_after,
            "p_before": self.p_before,
            "p_cross": self.p_cross,
            "p_self": self.p_self,
            "p_increment": self.p_increment,
            "p_after": self.p_after,
            "lambda_before": self.lambda_before,
            "lambda_increment": self.lambda_increment,
            "lambda_after": self.lambda_after,
            "actual_load_observation_before": self.actual_load_observation_before,
            "actual_load_observation_increment": (
                self.actual_load_observation_increment
            ),
            "actual_load_observation_after": self.actual_load_observation_after,
            "covariance_right_matmul_count": self.covariance_right_matmul_count,
            "dense_materialization_count": self.dense_materialization_count,
            "transaction_id": self.transaction_id,
            "binding_identity": self.binding_identity,
            "construction_id": self.construction_id,
            "construction_receipt_identity": (
                self.construction_receipt_identity
            ),
            "applied_binding_identity": self.applied_binding_identity,
            "pre_cast_update_sha256": self.pre_cast_update_sha256,
            "actual_post_storage_delta_sha256": (
                self.actual_post_storage_delta_sha256
            ),
            "post_storage_decision_influence_count": (
                self.post_storage_decision_influence_count
            ),
        }


@dataclass(frozen=True, slots=True)
class LedgerStageReceipt:
    stage_identity: str
    transaction_id: str
    prepared_commit_identity: str
    future_final_transaction_receipt_identity: str
    before_state_identity: str
    before_decision_identity: str
    proposed_after_state_identity: str
    proposed_after_decision_identity: str
    current_state_mutation_count: int
    proposal_decision_influence_count_before_commit: int

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "stage_identity": self.stage_identity,
            "transaction_id": self.transaction_id,
            "prepared_commit_identity": self.prepared_commit_identity,
            "future_final_transaction_receipt_identity": (
                self.future_final_transaction_receipt_identity
            ),
            "before_state_identity": self.before_state_identity,
            "before_decision_identity": self.before_decision_identity,
            "proposed_after_state_identity": self.proposed_after_state_identity,
            "proposed_after_decision_identity": (
                self.proposed_after_decision_identity
            ),
            "current_state_mutation_count": self.current_state_mutation_count,
            "proposal_decision_influence_count_before_commit": (
                self.proposal_decision_influence_count_before_commit
            ),
        }


@dataclass(frozen=True, slots=True)
class LedgerCommitReceipt:
    stage_identity: str
    transaction_id: str
    prepared_commit_identity: str
    final_transaction_receipt_identity: str
    before_version: int
    after_version: int
    before_state_identity: str
    after_state_identity: str
    before_decision_identity: str
    after_decision_identity: str
    factor_append_count: int
    logical_commit_count: int
    layer_receipts: tuple[CommittedLayerTransitionReceipt, ...]
    shadow_commit_count: int
    failed_commit_count: int
    duplicate_commit_count: int
    post_storage_decision_influence_count: int
    model_forward_count: int
    model_backward_count: int
    dense_materialization_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage_identity": self.stage_identity,
            "transaction_id": self.transaction_id,
            "prepared_commit_identity": self.prepared_commit_identity,
            "final_transaction_receipt_identity": (
                self.final_transaction_receipt_identity
            ),
            "before_version": self.before_version,
            "after_version": self.after_version,
            "before_state_identity": self.before_state_identity,
            "after_state_identity": self.after_state_identity,
            "before_decision_identity": self.before_decision_identity,
            "after_decision_identity": self.after_decision_identity,
            "factor_append_count": self.factor_append_count,
            "logical_commit_count": self.logical_commit_count,
            "layer_receipts": [item.raw_free_payload() for item in self.layer_receipts],
            "shadow_commit_count": self.shadow_commit_count,
            "failed_commit_count": self.failed_commit_count,
            "duplicate_commit_count": self.duplicate_commit_count,
            "post_storage_decision_influence_count": (
                self.post_storage_decision_influence_count
            ),
            "model_forward_count": self.model_forward_count,
            "model_backward_count": self.model_backward_count,
            "dense_materialization_count": self.dense_materialization_count,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class LedgerPublicationPreviewReceipt:
    """Primitive-only preview of the exact staged ledger publication."""

    stage_identity: str
    transaction_id: str
    prepared_commit_identity: str
    future_final_transaction_receipt_identity: str
    ledger_commit_receipt_identity: str
    committed_state_id: str
    before_version: int
    after_version: int
    before_state_identity: str
    after_state_identity: str
    before_decision_identity: str
    after_decision_identity: str
    layer_transition_identities: tuple[str, ...]
    layers: tuple[int, ...]
    prepared_factor_identities: tuple[str, ...]
    prepared_factor_sources: tuple[str, ...]
    committed_factor_identities: tuple[str, ...]
    committed_factor_sources: tuple[str, ...]
    provenance_transition_counts: tuple[int, ...]
    factor_update_equivalence_statuses: tuple[str, ...]
    factor_append_counts: tuple[int, ...]
    committed_factor_counts_after: tuple[int, ...]
    factor_append_count: int
    logical_commit_count: int
    shadow_commit_count: int
    failed_commit_count: int
    duplicate_commit_count: int
    post_storage_decision_influence_count: int
    model_forward_count: int
    model_backward_count: int
    dense_materialization_count: int
    native_storage_assignment_count: int
    storage_cast_boundary_count: int
    logical_outer_commit_count: int
    persistent_commit_count: int
    transaction_model_forward_count: int
    transaction_model_backward_count: int
    external_materializer_count: int
    candidate_materialization_count: int
    postcast_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in (
                (field_name, getattr(self, field_name))
                for field_name in self.__dataclass_fields__
            )
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class LedgerAbortReceipt:
    stage_identity: str
    transaction_id: str
    before_state_identity: str
    after_state_identity: str
    before_decision_identity: str
    after_decision_identity: str
    state_mutation_count: int
    rollback_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "stage_identity": self.stage_identity,
            "transaction_id": self.transaction_id,
            "before_state_identity": self.before_state_identity,
            "after_state_identity": self.after_state_identity,
            "before_decision_identity": self.before_decision_identity,
            "after_decision_identity": self.after_decision_identity,
            "state_mutation_count": self.state_mutation_count,
            "rollback_count": self.rollback_count,
        }


@dataclass(frozen=True, slots=True)
class LedgerCommitResult:
    state: CommittedGrossLoadState
    receipt: LedgerCommitReceipt
    transaction_receipt: FP32TransactionReceipt


@dataclass(frozen=True, slots=True)
class _StagedTransition:
    stage_receipt: LedgerStageReceipt
    before_state: CommittedGrossLoadState
    after_state: CommittedGrossLoadState
    precomputed_result: LedgerCommitResult
    publication_preview: LedgerPublicationPreviewReceipt
    transaction: OfficialStyleFP32SequentialTransaction
    prepared_identity: str
    after_state_identity: str
    after_decision_identity: str


def _factor_norm_squared(factor: LowRankFP32Factor) -> float:
    left_gram = factor.left.T @ factor.left
    right_gram = factor.right.T @ factor.right
    raw, absolute, terms = _small_gram_trace(left_gram, right_gram)
    value, _ = _nonnegative_reduction(
        "committed pre-cast factor norm",
        raw,
        absolute,
        terms,
    )
    return value


def _next_layer_state(
    before: CommittedLayerState,
    binding: CommittedLayerFactorBinding,
    *,
    transaction_id: str,
) -> tuple[CommittedLayerState, CommittedLayerTransitionReceipt]:
    factor = binding.factor
    covariance = before.covariance
    if (
        factor.source
        is not LowRankFactorSource.AUTHORITATIVE_PREPARED_PRECAST_FP32
        or binding.prepared_factor_identity != factor.identity_sha256
        or binding.prepared_factor_source != factor.source.value
        or binding.committed_factor_source
        != LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32.value
        or binding.provenance_transition_count != 1
        or binding.factor_update_equivalence_status
        != EXACT_SINGLE_CONSTRUCTION_STATUS
        or not binding.construction_receipt_identity
        or binding.construction_receipt_identity
        in before.committed_construction_receipt_ids
    ):
        raise ODEBFContractError("prepared factor provenance binding differs")
    if before.committed_precast_factors and any(
        historical.parameter_shape != factor.parameter_shape
        for historical in before.committed_precast_factors
    ):
        raise ODEBFContractError("committed factor parameter shape changed")
    guards = tuple(
        (item.identity_sha256 for item in before.committed_precast_factors)
    ) + (factor.identity_sha256, covariance.identity_sha256)
    sigma_right = covariance.value @ factor.right
    left_self = factor.left.T @ factor.left
    right_self = factor.right.T @ sigma_right
    self_raw, self_absolute, self_terms = _small_gram_trace(left_self, right_self)
    p_self, _ = _nonnegative_reduction(
        "committed structural-P self term",
        self_raw,
        self_absolute,
        self_terms,
    )
    cross_values: list[float] = []
    for historical in before.committed_precast_factors:
        left_gram = factor.left.T @ historical.left
        right_gram = historical.right.T @ sigma_right
        cross, _, _ = _small_gram_trace(left_gram, right_gram)
        if not math.isfinite(cross):
            raise ODEBFContractError("committed structural-P cross term is non-finite")
        cross_values.append(cross)
    p_cross = math.fsum(cross_values)
    p_increment = 2.0 * p_cross + p_self
    p_after_raw = before.structural_p_constant + p_increment
    if not math.isfinite(p_after_raw) or p_after_raw < -SOLVER_PRIMAL_TOLERANCE:
        raise ODEBFContractError("recursive structural-P constant is invalid")
    p_after = max(0.0, p_after_raw)

    precast_norm_squared = _factor_norm_squared(factor)
    lambda_increment = (
        precast_norm_squared / before.pretrained_weight_norm_squared
    )
    actual_increment = (
        binding.actual_post_storage_delta_energy
        / before.pretrained_weight_norm_squared
    )
    lambda_after = before.cumulative_precast_lambda + lambda_increment
    actual_after = (
        before.cumulative_actual_post_storage_load_observation + actual_increment
    )
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (lambda_increment, actual_increment, lambda_after, actual_after)
    ):
        raise ODEBFContractError("gross-load recursive scalar is invalid")
    committed_factor = _clone_factor(
        factor,
        source=LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32,
    )
    byte_identity = (
        tensor_sha256(factor.left) == tensor_sha256(committed_factor.left)
        and tensor_sha256(factor.right) == tensor_sha256(committed_factor.right)
    )
    nonalias = (
        int(factor.left.data_ptr()) != int(committed_factor.left.data_ptr())
        and int(factor.right.data_ptr()) != int(committed_factor.right.data_ptr())
    )
    if (
        not byte_identity
        or not nonalias
        or committed_factor.identity_sha256 != binding.committed_factor_identity
    ):
        raise ODEBFStateError("prepared-to-committed factor transition differs")
    after = CommittedLayerState(
        layer=before.layer,
        covariance=before.covariance,
        pretrained_weight_norm_squared=before.pretrained_weight_norm_squared,
        committed_precast_factors=before.committed_precast_factors
        + (committed_factor,),
        structural_p_constant=p_after,
        cumulative_precast_lambda=lambda_after,
        cumulative_actual_post_storage_load_observation=actual_after,
        version=before.version + 1,
        committed_transaction_ids=before.committed_transaction_ids
        + (transaction_id,),
        committed_construction_receipt_ids=(
            before.committed_construction_receipt_ids
            + (binding.construction_receipt_identity,)
        ),
    )
    observed_guards = tuple(
        item.identity_sha256 for item in before.committed_precast_factors
    ) + (factor.identity_sha256, covariance.identity_sha256)
    if observed_guards != guards:
        raise ODEBFStateError("committed-state factor/covariance input mutated")
    receipt = CommittedLayerTransitionReceipt(
        layer=before.layer,
        before_version=before.version,
        after_version=after.version,
        prepared_factor_identity=factor.identity_sha256,
        prepared_factor_source=factor.source.value,
        committed_factor_identity=committed_factor.identity_sha256,
        committed_factor_source=committed_factor.source.value,
        provenance_transition_count=1,
        source_transition_numerical_tensor_byte_identity=byte_identity,
        source_transition_numerical_tensor_nonalias=nonalias,
        factor_update_equivalence_status=FACTOR_UPDATE_EQUIVALENCE_STATUS,
        factor_append_count=1,
        committed_factor_count_before=len(before.committed_precast_factors),
        committed_factor_count_after=len(after.committed_precast_factors),
        p_before=before.structural_p_constant,
        p_cross=p_cross,
        p_self=p_self,
        p_increment=p_increment,
        p_after=p_after,
        lambda_before=before.cumulative_precast_lambda,
        lambda_increment=lambda_increment,
        lambda_after=lambda_after,
        actual_load_observation_before=(
            before.cumulative_actual_post_storage_load_observation
        ),
        actual_load_observation_increment=actual_increment,
        actual_load_observation_after=actual_after,
        covariance_right_matmul_count=1,
        dense_materialization_count=0,
        transaction_id=transaction_id,
        binding_identity=binding.identity_sha256,
        construction_id=binding.construction_id,
        construction_receipt_identity=binding.construction_receipt_identity,
        applied_binding_identity=binding.applied_binding_identity,
        pre_cast_update_sha256=binding.pre_cast_update_sha256,
        actual_post_storage_delta_sha256=(
            binding.actual_post_storage_delta_sha256
        ),
        post_storage_decision_influence_count=0,
    )
    return after, receipt


class CommittedGrossLoadLedger:
    def __init__(self, initial_state: CommittedGrossLoadState) -> None:
        if not isinstance(initial_state, CommittedGrossLoadState):
            raise ODEBFContractError("committed ledger initial state type differs")
        self._state = initial_state
        self._sealed_state_identity = initial_state.identity_sha256
        self._sealed_decision_identity = initial_state.decision_identity_sha256
        self._staged: _StagedTransition | None = None

    @property
    def state(self) -> CommittedGrossLoadState:
        return self._state

    def _verify_state(self) -> None:
        if (
            self._state.identity_sha256 != self._sealed_state_identity
            or self._state.decision_identity_sha256 != self._sealed_decision_identity
        ):
            raise ODEBFStateError("committed ledger state mutated outside commit")

    def stage_authoritative_commit(
        self,
        transaction: OfficialStyleFP32SequentialTransaction,
        prepared: FP32PreparedCommitReceipt,
        constructions: tuple[AppliedPreparedLowRankUpdate, ...],
    ) -> LedgerStageReceipt:
        if not isinstance(transaction, OfficialStyleFP32SequentialTransaction):
            raise ODEBFContractError("staged M3A transaction object type differs")
        try:
            self._verify_state()
            if self._staged is not None:
                raise ODEBFStateError(
                    "committed ledger already has a staged transition"
                )
            if not isinstance(prepared, FP32PreparedCommitReceipt):
                raise ODEBFContractError("staged M3A receipt is not prepared")
            transaction_prepared = transaction.prepared_receipt
            if (
                transaction.finalized
                or transaction_prepared is None
                or transaction_prepared.identity_sha256
                != prepared.identity_sha256
                or transaction.transaction_id != prepared.transaction_id
            ):
                raise ODEBFStateError("staged prepared transaction identity differs")
            if prepared.transaction_id in self._state.committed_transaction_ids:
                raise ODEBFStateError("M3A transaction identity was already committed")
            bindings = bind_authoritative_transaction_factors(
                prepared,
                constructions,
            )
            if (
                tuple(item.layer for item in bindings)
                != RESIDUAL_RESERVE_LAYER_ORDER
                or any(
                    item.transaction_id != prepared.transaction_id
                    or item.transaction_mode
                    != FP32TransactionMode.AUTHORITATIVE.value
                    or item.prepared_factor_identity
                    != item.factor.identity_sha256
                    or item.prepared_factor_source
                    != LowRankFactorSource.AUTHORITATIVE_PREPARED_PRECAST_FP32.value
                    or item.committed_factor_source
                    != LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32.value
                    or item.provenance_transition_count != 1
                    or item.factor_update_equivalence_status
                    != EXACT_SINGLE_CONSTRUCTION_STATUS
                    or not item.construction_receipt_identity
                    or item.post_storage_decision_influence_count != 0
                    for item in bindings
                )
            ):
                raise ODEBFContractError("staged factor binding order/identity differs")

            after_layers: list[CommittedLayerState] = []
            layer_receipts: list[CommittedLayerTransitionReceipt] = []
            for before, binding in zip(self._state.layers, bindings, strict=True):
                if before.layer != binding.layer:
                    raise ODEBFContractError("committed state/binding layer differs")
                after, layer_receipt = _next_layer_state(
                    before,
                    binding,
                    transaction_id=prepared.transaction_id,
                )
                after_layers.append(after)
                layer_receipts.append(layer_receipt)
            after_state = CommittedGrossLoadState(
                state_id=self._state.state_id,
                layers=tuple(after_layers),
                version=self._state.version + 1,
                commit_count=self._state.commit_count + 1,
                committed_transaction_ids=self._state.committed_transaction_ids
                + (prepared.transaction_id,),
                last_transaction_id=prepared.transaction_id,
            )
            before_state_identity = self._state.identity_sha256
            before_decision_identity = self._state.decision_identity_sha256
            after_state_identity = after_state.identity_sha256
            after_decision_identity = after_state.decision_identity_sha256
            prepared_identity = prepared.identity_sha256
            final_identity = prepared.future_final_receipt.identity_sha256
            stage_identity = canonical_hash(
                {
                    "before_state_identity": before_state_identity,
                    "prepared_commit_identity": prepared_identity,
                    "future_final_transaction_receipt_identity": final_identity,
                    "binding_identities": [
                        item.identity_sha256 for item in bindings
                    ],
                    "proposed_after_state_identity": after_state_identity,
                }
            )
            stage_receipt = LedgerStageReceipt(
                stage_identity=stage_identity,
                transaction_id=prepared.transaction_id,
                prepared_commit_identity=prepared_identity,
                future_final_transaction_receipt_identity=final_identity,
                before_state_identity=before_state_identity,
                before_decision_identity=before_decision_identity,
                proposed_after_state_identity=after_state_identity,
                proposed_after_decision_identity=after_decision_identity,
                current_state_mutation_count=0,
                proposal_decision_influence_count_before_commit=0,
            )
            commit_receipt = LedgerCommitReceipt(
                stage_identity=stage_identity,
                transaction_id=prepared.transaction_id,
                prepared_commit_identity=prepared_identity,
                final_transaction_receipt_identity=final_identity,
                before_version=self._state.version,
                after_version=after_state.version,
                before_state_identity=before_state_identity,
                after_state_identity=after_state_identity,
                before_decision_identity=before_decision_identity,
                after_decision_identity=after_decision_identity,
                factor_append_count=len(layer_receipts),
                logical_commit_count=1,
                layer_receipts=tuple(layer_receipts),
                shadow_commit_count=0,
                failed_commit_count=0,
                duplicate_commit_count=0,
                post_storage_decision_influence_count=0,
                model_forward_count=0,
                model_backward_count=0,
                dense_materialization_count=0,
            )
            commit_receipt.identity_sha256
            result = LedgerCommitResult(
                state=after_state,
                receipt=commit_receipt,
                transaction_receipt=prepared.future_final_receipt,
            )
            publication_preview = LedgerPublicationPreviewReceipt(
                stage_identity=stage_identity,
                transaction_id=prepared.transaction_id,
                prepared_commit_identity=prepared_identity,
                future_final_transaction_receipt_identity=final_identity,
                ledger_commit_receipt_identity=commit_receipt.identity_sha256,
                committed_state_id=after_state.state_id,
                before_version=self._state.version,
                after_version=after_state.version,
                before_state_identity=before_state_identity,
                after_state_identity=after_state_identity,
                before_decision_identity=before_decision_identity,
                after_decision_identity=after_decision_identity,
                layer_transition_identities=tuple(
                    canonical_hash(item.raw_free_payload())
                    for item in layer_receipts
                ),
                layers=tuple(item.layer for item in layer_receipts),
                prepared_factor_identities=tuple(
                    item.prepared_factor_identity for item in layer_receipts
                ),
                prepared_factor_sources=tuple(
                    item.prepared_factor_source for item in layer_receipts
                ),
                committed_factor_identities=tuple(
                    item.committed_factor_identity for item in layer_receipts
                ),
                committed_factor_sources=tuple(
                    item.committed_factor_source for item in layer_receipts
                ),
                provenance_transition_counts=tuple(
                    item.provenance_transition_count for item in layer_receipts
                ),
                factor_update_equivalence_statuses=tuple(
                    item.factor_update_equivalence_status
                    for item in layer_receipts
                ),
                factor_append_counts=tuple(
                    item.factor_append_count for item in layer_receipts
                ),
                committed_factor_counts_after=tuple(
                    item.committed_factor_count_after for item in layer_receipts
                ),
                factor_append_count=commit_receipt.factor_append_count,
                logical_commit_count=commit_receipt.logical_commit_count,
                shadow_commit_count=commit_receipt.shadow_commit_count,
                failed_commit_count=commit_receipt.failed_commit_count,
                duplicate_commit_count=commit_receipt.duplicate_commit_count,
                post_storage_decision_influence_count=(
                    commit_receipt.post_storage_decision_influence_count
                ),
                model_forward_count=commit_receipt.model_forward_count,
                model_backward_count=commit_receipt.model_backward_count,
                dense_materialization_count=(
                    commit_receipt.dense_materialization_count
                ),
                native_storage_assignment_count=(
                    prepared.future_final_receipt.native_storage_assignment_count
                ),
                storage_cast_boundary_count=(
                    prepared.future_final_receipt.storage_cast_boundary_count
                ),
                logical_outer_commit_count=(
                    prepared.future_final_receipt.logical_outer_commit_count
                ),
                persistent_commit_count=(
                    prepared.future_final_receipt.persistent_commit_count
                ),
                transaction_model_forward_count=(
                    prepared.future_final_receipt.model_forward_count
                ),
                transaction_model_backward_count=(
                    prepared.future_final_receipt.model_backward_count
                ),
                external_materializer_count=(
                    prepared.future_final_receipt.external_materializer_call_count
                ),
                candidate_materialization_count=(
                    prepared.future_final_receipt.candidate_materialization_count
                ),
                postcast_decision_influence_count=(
                    prepared.future_final_receipt.postcast_decision_influence_count
                ),
            )
            publication_preview.identity_sha256
            self._staged = _StagedTransition(
                stage_receipt=stage_receipt,
                before_state=self._state,
                after_state=after_state,
                precomputed_result=result,
                publication_preview=publication_preview,
                transaction=transaction,
                prepared_identity=prepared_identity,
                after_state_identity=after_state_identity,
                after_decision_identity=after_decision_identity,
            )
            self._verify_state()
            return stage_receipt
        except BaseException:
            staged_on_error = self._staged
            self._staged = None
            if (
                staged_on_error is not None
                and staged_on_error.transaction is not transaction
                and not staged_on_error.transaction.finalized
            ):
                staged_on_error.transaction.abort_and_rollback()
            if not transaction.finalized:
                transaction.abort_and_rollback()
            raise

    def preview_staged_publication(
        self,
        stage_identity: str,
        expected_prepare_identity: str,
    ) -> LedgerPublicationPreviewReceipt:
        """Return the sealed primitive-only result of a valid staged commit."""

        self._verify_state()
        staged = self._staged
        if (
            staged is None
            or staged.stage_receipt.stage_identity != stage_identity
            or staged.prepared_identity != expected_prepare_identity
            or staged.before_state is not self._state
            or staged.transaction.finalized
            or staged.transaction.prepared_receipt is None
            or staged.transaction.prepared_receipt.identity_sha256
            != expected_prepare_identity
            or staged.publication_preview.stage_identity != stage_identity
            or staged.publication_preview.prepared_commit_identity
            != expected_prepare_identity
        ):
            raise ODEBFStateError("committed-ledger publication preview differs")
        staged.publication_preview.identity_sha256
        return staged.publication_preview

    def commit_staged(self, stage_identity: str) -> LedgerCommitResult:
        if self._staged is not None:
            transaction = self._staged.transaction
            self._staged = None
            if not transaction.finalized:
                transaction.abort_and_rollback()
        raise ODEBFStateError(
            "ledger publication requires commit_staged_with_transaction"
        )

    def commit_staged_with_transaction(
        self,
        transaction: OfficialStyleFP32SequentialTransaction,
        stage_identity: str,
        expected_prepare_identity: str,
        expected_preview_identity: str,
    ) -> LedgerCommitResult:
        try:
            self._verify_state()
            staged = self._staged
            if (
                staged is None
                or staged.stage_receipt.stage_identity != stage_identity
                or staged.transaction is not transaction
                or staged.prepared_identity != expected_prepare_identity
                or staged.publication_preview.identity_sha256
                != expected_preview_identity
                or staged.before_state is not self._state
                or transaction.prepared_receipt is None
                or transaction.prepared_receipt.identity_sha256
                != expected_prepare_identity
            ):
                raise ODEBFStateError(
                    "coordinated committed-ledger stage identity differs"
                )
            result = staged.precomputed_result
            after_state_identity = staged.after_state_identity
            after_decision_identity = staged.after_decision_identity
            transaction.commit_prepared(expected_prepare_identity)
            self._state = result.state
            self._sealed_state_identity = after_state_identity
            self._sealed_decision_identity = after_decision_identity
            self._staged = None
            return result
        except BaseException:
            staged_on_error = self._staged
            self._staged = None
            if (
                staged_on_error is not None
                and staged_on_error.transaction is not transaction
                and not staged_on_error.transaction.finalized
            ):
                staged_on_error.transaction.abort_and_rollback()
            if (
                isinstance(transaction, OfficialStyleFP32SequentialTransaction)
                and not transaction.finalized
            ):
                transaction.abort_and_rollback()
            raise

    def abort_staged(self, stage_identity: str) -> LedgerAbortReceipt:
        self._verify_state()
        if self._staged is None or self._staged.stage_receipt.stage_identity != stage_identity:
            raise ODEBFStateError("committed ledger abort stage identity differs")
        staged = self._staged
        before_identity = self._state.identity_sha256
        before_decision = self._state.decision_identity_sha256
        if staged.before_state is not self._state:
            raise ODEBFStateError("committed ledger state changed before abort")
        self._staged = None
        if not staged.transaction.finalized:
            staged.transaction.abort_and_rollback()
        self._verify_state()
        return LedgerAbortReceipt(
            stage_identity=stage_identity,
            transaction_id=staged.stage_receipt.transaction_id,
            before_state_identity=before_identity,
            after_state_identity=self._state.identity_sha256,
            before_decision_identity=before_decision,
            after_decision_identity=self._state.decision_identity_sha256,
            state_mutation_count=0,
            rollback_count=1,
        )


__all__ = [
    "CommittedGrossLoadLedger",
    "CommittedGrossLoadState",
    "CommittedLayerFactorBinding",
    "CommittedLayerState",
    "LayerCommittedStateAnchor",
    "CommittedLayerTransitionReceipt",
    "FACTOR_UPDATE_EQUIVALENCE_STATUS",
    "LedgerAbortReceipt",
    "LedgerCommitReceipt",
    "LedgerCommitResult",
    "LedgerPublicationPreviewReceipt",
    "LedgerStageReceipt",
    "bind_authoritative_transaction_factors",
    "initialize_committed_gross_load_state",
]
