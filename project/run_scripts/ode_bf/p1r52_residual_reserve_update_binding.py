"""Single-construction FP32 update and prepared-factor binding for residual reserve."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import warnings

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r52_residual_reserve_fp32_transaction import (
    FP32LayerApplicationReceipt,
    OfficialStyleFP32SequentialTransaction,
    RESIDUAL_RESERVE_LAYER_ORDER,
)
from .p1r52_residual_reserve_pc_inventory import (
    LowRankFP32Factor,
    LowRankFactorSource,
)


EXACT_SINGLE_CONSTRUCTION_STATUS = "EXACT_BY_SINGLE_FP32_CONSTRUCTION"
OFFICIAL_MATCHER = (
    "easyeditor.models.alphaedit.AlphaEdit_main.upd_matrix_match_shape"
)
INPUT_FORMULA = (
    "left32=CALLER_SUPPLIED_ALREADY_BETA_APPLIED_E32;"
    "right32=Psi32_or_q_side32"
)
INPUT_BETA_APPLICATION_PROOF_STATUS = "DEFERRED_TO_M3D_B"
OFFICIAL_IMPORT_WARNING_MESSAGE = (
    r"^Importing from timm\.models\.hub is deprecated, "
    r"please import via timm\.models$"
)
OFFICIAL_IMPORT_WARNING_MODULE = r"^timm\.models\.hub$"


class UpdateMatchOrientation(str, Enum):
    DIRECT = "DIRECT"
    TRANSPOSE_VIEW = "TRANSPOSE_VIEW"


@dataclass(frozen=True, slots=True)
class PreparedUpdateConstructionReceipt:
    construction_id: str
    layer: int
    weight_name: str
    parameter_shape: tuple[int, int]
    raw_update_shape: tuple[int, int]
    matched_update_shape: tuple[int, int]
    matched_orientation: str
    raw_left_shape: tuple[int, int]
    raw_right_shape: tuple[int, int]
    raw_left_sha256: str
    raw_right_sha256: str
    prepared_factor_left_sha256: str
    prepared_factor_right_sha256: str
    prepared_factor_identity: str
    prepared_factor_source: str
    raw_update_sha256: str
    matched_update_sha256: str
    raw_update_pointer: int
    matched_update_pointer: int
    raw_update_version: int
    matched_update_version: int
    matched_update_dtype: str
    matched_update_device: str
    matched_update_stride: tuple[int, int]
    official_matcher: str
    official_matcher_call_count: int
    dense_construction_count: int
    second_dense_construction_count: int
    fp64_algorithm_tensor_count: int
    input_beta_application_proof_status: str
    internal_beta_reapplication_count: int
    input_formula: str
    factor_input_immutability_verified: bool
    factor_input_nonalias_verified: bool
    postcast_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "construction_id": self.construction_id,
            "layer": self.layer,
            "weight_name": self.weight_name,
            "parameter_shape": list(self.parameter_shape),
            "raw_update_shape": list(self.raw_update_shape),
            "matched_update_shape": list(self.matched_update_shape),
            "matched_orientation": self.matched_orientation,
            "raw_left_shape": list(self.raw_left_shape),
            "raw_right_shape": list(self.raw_right_shape),
            "raw_left_sha256": self.raw_left_sha256,
            "raw_right_sha256": self.raw_right_sha256,
            "prepared_factor_left_sha256": self.prepared_factor_left_sha256,
            "prepared_factor_right_sha256": self.prepared_factor_right_sha256,
            "prepared_factor_identity": self.prepared_factor_identity,
            "prepared_factor_source": self.prepared_factor_source,
            "raw_update_sha256": self.raw_update_sha256,
            "matched_update_sha256": self.matched_update_sha256,
            "raw_update_pointer": self.raw_update_pointer,
            "matched_update_pointer": self.matched_update_pointer,
            "raw_update_version": self.raw_update_version,
            "matched_update_version": self.matched_update_version,
            "matched_update_dtype": self.matched_update_dtype,
            "matched_update_device": self.matched_update_device,
            "matched_update_stride": list(self.matched_update_stride),
            "official_matcher": self.official_matcher,
            "official_matcher_call_count": self.official_matcher_call_count,
            "dense_construction_count": self.dense_construction_count,
            "second_dense_construction_count": (
                self.second_dense_construction_count
            ),
            "fp64_algorithm_tensor_count": self.fp64_algorithm_tensor_count,
            "input_beta_application_proof_status": (
                self.input_beta_application_proof_status
            ),
            "internal_beta_reapplication_count": (
                self.internal_beta_reapplication_count
            ),
            "input_formula": self.input_formula,
            "factor_input_immutability_verified": (
                self.factor_input_immutability_verified
            ),
            "factor_input_nonalias_verified": self.factor_input_nonalias_verified,
            "postcast_decision_influence_count": (
                self.postcast_decision_influence_count
            ),
            "equivalence_status": EXACT_SINGLE_CONSTRUCTION_STATUS,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class PreparedLowRankUpdate:
    matched_update32: torch.Tensor
    factor: LowRankFP32Factor
    receipt: PreparedUpdateConstructionReceipt

    def validate(self) -> None:
        update = self.matched_update32
        factor = self.factor
        receipt = self.receipt
        if (
            not isinstance(update, torch.Tensor)
            or update.dtype is not torch.float32
            or update.ndim != 2
            or not bool(torch.isfinite(update).all())
            or not isinstance(factor, LowRankFP32Factor)
            or not isinstance(receipt, PreparedUpdateConstructionReceipt)
        ):
            raise ODEBFContractError("prepared update binding type/dtype differs")
        if (
            receipt.layer != factor.layer
            or receipt.layer not in RESIDUAL_RESERVE_LAYER_ORDER
            or receipt.weight_name == ""
            or not receipt.weight_name.endswith(".weight")
            or receipt.parameter_shape != factor.parameter_shape
            or receipt.matched_update_shape != tuple(update.shape)
            or receipt.parameter_shape != tuple(update.shape)
            or receipt.prepared_factor_identity != factor.identity_sha256
            or receipt.prepared_factor_source
            != LowRankFactorSource.AUTHORITATIVE_PREPARED_PRECAST_FP32.value
            or factor.source
            is not LowRankFactorSource.AUTHORITATIVE_PREPARED_PRECAST_FP32
            or receipt.prepared_factor_left_sha256 != tensor_sha256(factor.left)
            or receipt.prepared_factor_right_sha256 != tensor_sha256(factor.right)
            or receipt.matched_update_sha256 != tensor_sha256(update)
            or receipt.matched_update_pointer != int(update.data_ptr())
            or receipt.matched_update_version != int(update._version)
            or receipt.matched_update_dtype != str(update.dtype)
            or receipt.matched_update_device != str(update.device)
            or receipt.matched_update_stride != tuple(update.stride())
        ):
            raise ODEBFStateError("prepared update/factor identity differs")
        if (
            receipt.official_matcher != OFFICIAL_MATCHER
            or receipt.official_matcher_call_count != 1
            or receipt.dense_construction_count != 1
            or receipt.second_dense_construction_count != 0
            or receipt.fp64_algorithm_tensor_count != 0
            or receipt.input_beta_application_proof_status
            != INPUT_BETA_APPLICATION_PROOF_STATUS
            or receipt.internal_beta_reapplication_count != 0
            or receipt.input_formula != INPUT_FORMULA
            or not receipt.factor_input_immutability_verified
            or not receipt.factor_input_nonalias_verified
            or receipt.postcast_decision_influence_count != 0
        ):
            raise ODEBFContractError("prepared construction receipt differs")

        try:
            orientation = UpdateMatchOrientation(receipt.matched_orientation)
        except ValueError as error:
            raise ODEBFContractError("prepared update orientation is invalid") from error
        raw_shape = receipt.raw_update_shape
        parameter_shape = receipt.parameter_shape
        if orientation is UpdateMatchOrientation.DIRECT:
            expected_factor_shapes = (
                receipt.raw_left_shape,
                receipt.raw_right_shape,
            )
            orientation_valid = raw_shape == parameter_shape
            side_hashes = (
                receipt.raw_left_sha256,
                receipt.raw_right_sha256,
            )
        else:
            expected_factor_shapes = (
                receipt.raw_right_shape,
                receipt.raw_left_shape,
            )
            orientation_valid = raw_shape[::-1] == parameter_shape
            side_hashes = (
                receipt.raw_right_sha256,
                receipt.raw_left_sha256,
            )
        if (
            not orientation_valid
            or factor.left.shape != expected_factor_shapes[0]
            or factor.right.shape != expected_factor_shapes[1]
            or (
                receipt.prepared_factor_left_sha256,
                receipt.prepared_factor_right_sha256,
            )
            != side_hashes
            or receipt.raw_update_pointer != receipt.matched_update_pointer
            or receipt.raw_update_version != receipt.matched_update_version
        ):
            raise ODEBFStateError("prepared update orientation binding differs")

    @property
    def construction_identity(self) -> str:
        return self.receipt.identity_sha256


@dataclass(frozen=True, slots=True)
class AppliedPreparedLowRankUpdate:
    """Typed proof that M3A consumed the construction's exact update object."""

    construction: PreparedLowRankUpdate
    m3a_layer_receipt: FP32LayerApplicationReceipt
    exact_object_apply_count: int
    input_pointer_unchanged: bool
    input_version_unchanged: bool

    def validate(self) -> None:
        if (
            not isinstance(self.construction, PreparedLowRankUpdate)
            or not isinstance(
                self.m3a_layer_receipt,
                FP32LayerApplicationReceipt,
            )
        ):
            raise ODEBFContractError("applied construction binding type differs")
        self.construction.validate()
        update = self.construction.matched_update32
        receipt = self.construction.receipt
        application = self.m3a_layer_receipt
        if (
            self.exact_object_apply_count != 1
            or not self.input_pointer_unchanged
            or not self.input_version_unchanged
            or application.layer != receipt.layer
            or application.weight_name != receipt.weight_name
            or application.parameter_shape != receipt.parameter_shape
            or application.pre_cast_fp32_update_sha256
            != receipt.matched_update_sha256
            or application.pre_cast_fp32_update_sha256
            != tensor_sha256(update)
            or application.prepared_fp32_update_sha256
            != receipt.matched_update_sha256
            or application.prepared_fp32_update_pointer
            != receipt.matched_update_pointer
            or application.prepared_fp32_update_version
            != receipt.matched_update_version
            or application.prepared_fp32_update_dtype != "torch.float32"
            or application.prepared_fp32_update_device != str(update.device)
            or not application.official_reference_endpoint_byte_exact
            or application.official_reference_endpoint_sha256
            != application.post_storage_parameter_sha256
            or application.official_assignment_equation
            != "parameter[...] = parameter + matched_update32.float()"
            or application.actual_post_storage_delta32_dtype != "torch.float32"
            or application.rounding_telemetry_decision_influence_count != 0
            or int(update.data_ptr()) != receipt.matched_update_pointer
            or int(update._version) != receipt.matched_update_version
            or application.storage_assignment_count != 1
            or application.storage_cast_boundary_count != 1
            or application.numeric_storage_cast_count
            != int(application.storage_cast_required)
            or application.bf16_path_call_count != 0
            or application.bf16_path_decision_influence_count != 0
            or application.autocast_count != 0
            or application.downcast_count != 0
            or application.quantization_count != 0
            or application.postcast_decision_influence_count != 0
        ):
            raise ODEBFStateError("M3A exact-object construction binding differs")

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "construction_receipt_identity": (
                self.construction.receipt.identity_sha256
            ),
            "construction_id": self.construction.receipt.construction_id,
            "layer": self.construction.receipt.layer,
            "weight_name": self.construction.receipt.weight_name,
            "prepared_factor_identity": self.construction.factor.identity_sha256,
            "matched_update_sha256": (
                self.construction.receipt.matched_update_sha256
            ),
            "matched_update_pointer": (
                self.construction.receipt.matched_update_pointer
            ),
            "matched_update_version": (
                self.construction.receipt.matched_update_version
            ),
            "m3a_layer_receipt_identity": (
                self.m3a_layer_receipt.identity_sha256
            ),
            "exact_object_apply_count": self.exact_object_apply_count,
            "input_pointer_unchanged": self.input_pointer_unchanged,
            "input_version_unchanged": self.input_version_unchanged,
            "equivalence_status": EXACT_SINGLE_CONSTRUCTION_STATUS,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


def _official_upd_matrix_match_shape(
    matrix: torch.Tensor,
    shape: torch.Size,
) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=OFFICIAL_IMPORT_WARNING_MESSAGE,
            category=FutureWarning,
            module=OFFICIAL_IMPORT_WARNING_MODULE,
        )
        from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    matcher = alpha_main.upd_matrix_match_shape
    matcher_identity = f"{matcher.__module__}.{matcher.__name__}"
    if matcher_identity != OFFICIAL_MATCHER:
        raise ODEBFStateError("Official update matcher identity differs")
    return matcher(matrix, shape)


def build_prepared_low_rank_update(
    *,
    construction_id: str,
    layer: int,
    weight_name: str,
    parameter_shape: tuple[int, int],
    beta_applied_left32: torch.Tensor,
    q_side32: torch.Tensor,
) -> PreparedLowRankUpdate:
    if not isinstance(construction_id, str) or not construction_id:
        raise ODEBFContractError("prepared construction identity is empty")
    if layer not in RESIDUAL_RESERVE_LAYER_ORDER:
        raise ODEBFContractError("prepared construction layer is invalid")
    if not isinstance(weight_name, str) or not weight_name.endswith(".weight"):
        raise ODEBFContractError("prepared construction weight name is invalid")
    if (
        not isinstance(parameter_shape, tuple)
        or len(parameter_shape) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in parameter_shape
        )
    ):
        raise ODEBFContractError("prepared parameter shape is invalid")
    for name, side in (
        ("beta-applied left", beta_applied_left32),
        ("q side", q_side32),
    ):
        if (
            not isinstance(side, torch.Tensor)
            or side.dtype is not torch.float32
            or side.ndim != 2
            or not bool(torch.isfinite(side).all())
        ):
            raise ODEBFContractError(f"{name} is not finite FP32 rank-2")
    if (
        beta_applied_left32.device != q_side32.device
        or beta_applied_left32.shape[1] != q_side32.shape[1]
        or beta_applied_left32.shape[1] <= 0
    ):
        raise ODEBFContractError("prepared low-rank side geometry differs")

    input_guards = (
        int(beta_applied_left32.data_ptr()),
        int(beta_applied_left32._version),
        tensor_sha256(beta_applied_left32),
        int(q_side32.data_ptr()),
        int(q_side32._version),
        tensor_sha256(q_side32),
    )
    raw_update32 = beta_applied_left32 @ q_side32.T
    if raw_update32.dtype is not torch.float32:
        raise ODEBFStateError("single dense update construction is not FP32")
    raw_pointer = int(raw_update32.data_ptr())
    raw_version = int(raw_update32._version)
    raw_sha256 = tensor_sha256(raw_update32)
    matched_update32 = _official_upd_matrix_match_shape(
        raw_update32,
        torch.Size(parameter_shape),
    )
    if matched_update32 is raw_update32:
        orientation = UpdateMatchOrientation.DIRECT
        factor_left = beta_applied_left32
        factor_right = q_side32
    elif (
        tuple(raw_update32.T.shape) == parameter_shape
        and int(matched_update32.data_ptr()) == raw_pointer
        and tuple(matched_update32.stride()) == tuple(raw_update32.T.stride())
    ):
        orientation = UpdateMatchOrientation.TRANSPOSE_VIEW
        factor_left = q_side32
        factor_right = beta_applied_left32
    else:
        raise ODEBFStateError("Official matcher returned a copied/unknown update")
    if (
        matched_update32.dtype is not torch.float32
        or matched_update32.device != raw_update32.device
        or tuple(matched_update32.shape) != parameter_shape
        or int(matched_update32.data_ptr()) != raw_pointer
        or int(matched_update32._version) != raw_version
    ):
        raise ODEBFStateError("Official matcher changed FP32 update storage")

    factor = LowRankFP32Factor(
        layer,
        parameter_shape,
        factor_left,
        factor_right,
        LowRankFactorSource.AUTHORITATIVE_PREPARED_PRECAST_FP32,
    )
    observed_guards = (
        int(beta_applied_left32.data_ptr()),
        int(beta_applied_left32._version),
        tensor_sha256(beta_applied_left32),
        int(q_side32.data_ptr()),
        int(q_side32._version),
        tensor_sha256(q_side32),
    )
    if observed_guards != input_guards:
        raise ODEBFStateError("prepared low-rank side input mutated")
    nonalias = (
        int(factor.left.data_ptr()) not in (input_guards[0], input_guards[3])
        and int(factor.right.data_ptr()) not in (input_guards[0], input_guards[3])
        and int(factor.left.data_ptr()) != int(factor.right.data_ptr())
    )
    if not nonalias:
        raise ODEBFStateError("prepared factor sides alias construction inputs")

    receipt = PreparedUpdateConstructionReceipt(
        construction_id=construction_id,
        layer=layer,
        weight_name=weight_name,
        parameter_shape=parameter_shape,
        raw_update_shape=tuple(raw_update32.shape),
        matched_update_shape=tuple(matched_update32.shape),
        matched_orientation=orientation.value,
        raw_left_shape=tuple(beta_applied_left32.shape),
        raw_right_shape=tuple(q_side32.shape),
        raw_left_sha256=input_guards[2],
        raw_right_sha256=input_guards[5],
        prepared_factor_left_sha256=tensor_sha256(factor.left),
        prepared_factor_right_sha256=tensor_sha256(factor.right),
        prepared_factor_identity=factor.identity_sha256,
        prepared_factor_source=factor.source.value,
        raw_update_sha256=raw_sha256,
        matched_update_sha256=tensor_sha256(matched_update32),
        raw_update_pointer=raw_pointer,
        matched_update_pointer=int(matched_update32.data_ptr()),
        raw_update_version=raw_version,
        matched_update_version=int(matched_update32._version),
        matched_update_dtype=str(matched_update32.dtype),
        matched_update_device=str(matched_update32.device),
        matched_update_stride=tuple(matched_update32.stride()),
        official_matcher=OFFICIAL_MATCHER,
        official_matcher_call_count=1,
        dense_construction_count=1,
        second_dense_construction_count=0,
        fp64_algorithm_tensor_count=0,
        input_beta_application_proof_status=(
            INPUT_BETA_APPLICATION_PROOF_STATUS
        ),
        internal_beta_reapplication_count=0,
        input_formula=INPUT_FORMULA,
        factor_input_immutability_verified=True,
        factor_input_nonalias_verified=True,
        postcast_decision_influence_count=0,
    )
    result = PreparedLowRankUpdate(matched_update32, factor, receipt)
    result.validate()
    return result


def apply_prepared_low_rank_update(
    transaction: OfficialStyleFP32SequentialTransaction,
    construction: PreparedLowRankUpdate,
) -> AppliedPreparedLowRankUpdate:
    """Apply the exact constructed object once through the audited M3A API."""

    if not isinstance(transaction, OfficialStyleFP32SequentialTransaction):
        raise ODEBFContractError("M3A transaction type differs")
    if not isinstance(construction, PreparedLowRankUpdate):
        raise ODEBFContractError("prepared construction type differs")
    try:
        construction.validate()
        update = construction.matched_update32
        pointer_before = int(update.data_ptr())
        version_before = int(update._version)
        application = transaction.apply_layer_fp32(
            construction.receipt.layer,
            update,
        )
        pointer_unchanged = int(update.data_ptr()) == pointer_before
        version_unchanged = int(update._version) == version_before
        applied = AppliedPreparedLowRankUpdate(
            construction=construction,
            m3a_layer_receipt=application,
            exact_object_apply_count=1,
            input_pointer_unchanged=pointer_unchanged,
            input_version_unchanged=version_unchanged,
        )
        applied.validate()
        return applied
    except BaseException:
        if not transaction.finalized:
            transaction.abort_and_rollback()
        raise


__all__ = [
    "AppliedPreparedLowRankUpdate",
    "EXACT_SINGLE_CONSTRUCTION_STATUS",
    "INPUT_BETA_APPLICATION_PROOF_STATUS",
    "INPUT_FORMULA",
    "OFFICIAL_MATCHER",
    "OFFICIAL_IMPORT_WARNING_MESSAGE",
    "OFFICIAL_IMPORT_WARNING_MODULE",
    "PreparedLowRankUpdate",
    "PreparedUpdateConstructionReceipt",
    "UpdateMatchOrientation",
    "apply_prepared_low_rank_update",
    "build_prepared_low_rank_update",
]
