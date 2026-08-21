"""Residual-independent FP32 Alpha q solve for residual reserve."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256


ALPHA_Q_SOLVE_LAYERS = (4, 5, 6, 7, 8)
ALPHA_Q_SOLVE_DTYPE = torch.float32
ALPHA_Q_SOLVE_REFERENCE = "Q_ONLY_RESIDUAL_INDEPENDENT_ALPHA_SOLVE"
NATIVE_DENSE_BYTE_EQUIVALENCE = "NOT_CLAIMED"
ALPHA_Q_SOLVE_EXPRESSION = (
    "A32=P32@(K32@K32.T+C32)+lambda32*I32;"
    "rhs32=P32@K32;q32=torch.linalg.solve(A32,rhs32)"
)
ALPHA_Q_SOLVE_CONDITION_MAX_DIMENSION = 256


@dataclass(frozen=True, slots=True)
class AlphaQInputIdentity:
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
class AlphaQOnlySolveReceipt:
    layer: int
    reference: str
    native_dense_byte_equivalence: str
    expression_order: str
    input_identities: tuple[AlphaQInputIdentity, ...]
    system_shape: tuple[int, int]
    rhs_shape: tuple[int, int]
    q_shape: tuple[int, int]
    system_sha256: str
    rhs_sha256: str
    q_sha256: str
    key_rank: int
    required_key_rank: int
    relative_solve_residual: float
    residual_tolerance: float
    q_norm: float
    condition_estimate: float | None
    condition_kind: str
    solve_dtype: str
    solve_device: str
    input_pointer_version_hash_immutability_verified: bool
    logical_q_solve_count: int
    dense_writer_update_construction_count: int
    fp64_algorithm_tensor_count: int
    fp64_scalar_reduction_count: int
    model_forward_count: int
    model_backward_count: int
    semantic_backward_count: int
    slope_backward_count: int
    materialization_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": self.layer,
            "reference": self.reference,
            "native_dense_byte_equivalence": (
                self.native_dense_byte_equivalence
            ),
            "expression_order": self.expression_order,
            "input_identities": [
                item.raw_free_payload() for item in self.input_identities
            ],
            "system_shape": list(self.system_shape),
            "rhs_shape": list(self.rhs_shape),
            "q_shape": list(self.q_shape),
            "system_sha256": self.system_sha256,
            "rhs_sha256": self.rhs_sha256,
            "q_sha256": self.q_sha256,
            "key_rank": self.key_rank,
            "required_key_rank": self.required_key_rank,
            "relative_solve_residual": self.relative_solve_residual,
            "residual_tolerance": self.residual_tolerance,
            "q_norm": self.q_norm,
            "condition_estimate": self.condition_estimate,
            "condition_kind": self.condition_kind,
            "solve_dtype": self.solve_dtype,
            "solve_device": self.solve_device,
            "input_pointer_version_hash_immutability_verified": (
                self.input_pointer_version_hash_immutability_verified
            ),
            "logical_q_solve_count": self.logical_q_solve_count,
            "dense_writer_update_construction_count": (
                self.dense_writer_update_construction_count
            ),
            "fp64_algorithm_tensor_count": self.fp64_algorithm_tensor_count,
            "fp64_scalar_reduction_count": self.fp64_scalar_reduction_count,
            "model_forward_count": self.model_forward_count,
            "model_backward_count": self.model_backward_count,
            "semantic_backward_count": self.semantic_backward_count,
            "slope_backward_count": self.slope_backward_count,
            "materialization_count": self.materialization_count,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class AlphaQOnlySolveResult:
    q32: torch.Tensor
    receipt: AlphaQOnlySolveReceipt


def _validate_layer(layer: int) -> None:
    if isinstance(layer, bool) or layer not in ALPHA_Q_SOLVE_LAYERS:
        raise ODEBFContractError("Alpha q-solve layer is invalid")


def _validate_input_tensor(name: str, value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype is not ALPHA_Q_SOLVE_DTYPE
        or value.ndim != 2
        or value.device.type not in ("cpu", "cuda")
        or value.requires_grad
    ):
        raise ODEBFContractError(f"Alpha q-solve {name} contract differs")
    if not bool(torch.isfinite(value).all()):
        raise ODEBFContractError(f"Alpha q-solve {name} is non-finite")


def _validate_regularization(value: torch.Tensor, device: torch.device) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype is not ALPHA_Q_SOLVE_DTYPE
        or value.device != device
        or value.ndim != 0
        or value.requires_grad
        or not bool(torch.isfinite(value))
        or float(value.detach().to(device="cpu")) <= 0.0
    ):
        raise ODEBFContractError("Alpha q-solve regularization32 is invalid")


def _input_identity(name: str, value: torch.Tensor) -> AlphaQInputIdentity:
    return AlphaQInputIdentity(
        name=name,
        shape=tuple(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        sha256=tensor_sha256(value),
        pointer=int(value.data_ptr()),
        version=int(value._version),
        requires_grad=bool(value.requires_grad),
    )


def _relative_residual_scalar(
    system32: torch.Tensor,
    q32: torch.Tensor,
    rhs32: torch.Tensor,
) -> float:
    system64 = system32.detach().to(device="cpu", dtype=torch.float64)
    q64 = q32.detach().to(device="cpu", dtype=torch.float64)
    rhs64 = rhs32.detach().to(device="cpu", dtype=torch.float64)
    denominator = (
        torch.linalg.norm(system64) * torch.linalg.norm(q64)
        + torch.linalg.norm(rhs64)
    )
    denominator = torch.clamp(
        denominator,
        min=torch.finfo(torch.float64).tiny,
    )
    return float(torch.linalg.norm(system64 @ q64 - rhs64) / denominator)


def solve_residual_independent_alpha_q_fp32(
    projector32: torch.Tensor,
    joint_keys32: torch.Tensor,
    committed_covariance32: torch.Tensor,
    regularization32: torch.Tensor,
    *,
    layer: int,
    residual_tolerance: float,
    condition_max_dimension: int = ALPHA_Q_SOLVE_CONDITION_MAX_DIMENSION,
) -> AlphaQOnlySolveResult:
    """Solve q without accepting or constructing a residual-bound update."""

    _validate_layer(layer)
    for name, value in (
        ("projector", projector32),
        ("joint_keys", joint_keys32),
        ("committed_covariance", committed_covariance32),
    ):
        _validate_input_tensor(name, value)
    device = projector32.device
    if (
        joint_keys32.device != device
        or committed_covariance32.device != device
    ):
        raise ODEBFContractError("Alpha q-solve input devices differ")
    _validate_regularization(regularization32, device)
    dimension = int(projector32.shape[0])
    if (
        projector32.shape != (dimension, dimension)
        or committed_covariance32.shape != (dimension, dimension)
        or joint_keys32.shape[0] != dimension
        or joint_keys32.shape[1] <= 0
        or joint_keys32.shape[1] > dimension
    ):
        raise ODEBFContractError("Alpha q-solve geometry differs")
    tolerance = float(residual_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ODEBFContractError("Alpha q-solve residual tolerance is invalid")
    if (
        isinstance(condition_max_dimension, bool)
        or not isinstance(condition_max_dimension, int)
        or condition_max_dimension < 0
    ):
        raise ODEBFContractError("Alpha q-solve condition dimension is invalid")

    inputs = (
        ("projector", projector32),
        ("joint_keys", joint_keys32),
        ("committed_covariance", committed_covariance32),
        ("regularization", regularization32),
    )
    input_identities = tuple(
        _input_identity(name, value) for name, value in inputs
    )
    identity32 = torch.eye(
        dimension,
        dtype=ALPHA_Q_SOLVE_DTYPE,
        device=device,
    )

    # Exact Official AlphaEdit A-matrix multiplication and addition order.
    try:
        with torch.no_grad():
            system32 = projector32 @ (
                joint_keys32 @ joint_keys32.T + committed_covariance32
            ) + regularization32 * identity32
            rhs32 = projector32 @ joint_keys32
            q32 = torch.linalg.solve(system32, rhs32)
    except RuntimeError as error:
        raise ODEBFContractError("Alpha q-solve backend failed") from error
    if (
        system32.dtype is not ALPHA_Q_SOLVE_DTYPE
        or rhs32.dtype is not ALPHA_Q_SOLVE_DTYPE
        or q32.dtype is not ALPHA_Q_SOLVE_DTYPE
        or system32.device != device
        or rhs32.device != device
        or q32.device != device
        or system32.shape != (dimension, dimension)
        or rhs32.shape != joint_keys32.shape
        or q32.shape != joint_keys32.shape
        or not bool(torch.isfinite(system32).all())
        or not bool(torch.isfinite(rhs32).all())
        or not bool(torch.isfinite(q32).all())
    ):
        raise ODEBFContractError("Alpha q-solve output contract differs")

    key_rank = int(torch.linalg.matrix_rank(joint_keys32).item())
    required_key_rank = int(joint_keys32.shape[1])
    if key_rank != required_key_rank:
        raise ODEBFContractError("Alpha q-solve joint keys are rank deficient")
    relative_residual = _relative_residual_scalar(system32, q32, rhs32)
    if (
        not math.isfinite(relative_residual)
        or relative_residual > tolerance
    ):
        raise ODEBFContractError("Alpha q-solve residual certificate failed")
    q_norm = float(
        torch.linalg.norm(
            q32.detach().to(device="cpu", dtype=torch.float64)
        )
    )
    if not math.isfinite(q_norm):
        raise ODEBFContractError("Alpha q-solve norm is non-finite")

    condition_estimate: float | None = None
    condition_kind = "OMITTED_LARGE_PRODUCTION_SYSTEM"
    fp64_scalar_reduction_count = 2
    if dimension <= condition_max_dimension:
        condition_estimate = float(
            torch.linalg.cond(
                system32.detach().to(device="cpu", dtype=torch.float64)
            )
        )
        if not math.isfinite(condition_estimate):
            raise ODEBFContractError("Alpha q-solve condition is non-finite")
        condition_kind = "FP64_SMALL_FIXTURE_SCALAR_DIAGNOSTIC"
        fp64_scalar_reduction_count += 1

    observed_inputs = tuple(
        _input_identity(name, value) for name, value in inputs
    )
    if observed_inputs != input_identities:
        raise ODEBFStateError("Alpha q-solve mutated or rebound an input")
    receipt = AlphaQOnlySolveReceipt(
        layer=layer,
        reference=ALPHA_Q_SOLVE_REFERENCE,
        native_dense_byte_equivalence=NATIVE_DENSE_BYTE_EQUIVALENCE,
        expression_order=ALPHA_Q_SOLVE_EXPRESSION,
        input_identities=input_identities,
        system_shape=tuple(system32.shape),
        rhs_shape=tuple(rhs32.shape),
        q_shape=tuple(q32.shape),
        system_sha256=tensor_sha256(system32),
        rhs_sha256=tensor_sha256(rhs32),
        q_sha256=tensor_sha256(q32),
        key_rank=key_rank,
        required_key_rank=required_key_rank,
        relative_solve_residual=relative_residual,
        residual_tolerance=tolerance,
        q_norm=q_norm,
        condition_estimate=condition_estimate,
        condition_kind=condition_kind,
        solve_dtype=str(ALPHA_Q_SOLVE_DTYPE),
        solve_device=str(device),
        input_pointer_version_hash_immutability_verified=True,
        logical_q_solve_count=1,
        dense_writer_update_construction_count=0,
        fp64_algorithm_tensor_count=0,
        fp64_scalar_reduction_count=fp64_scalar_reduction_count,
        model_forward_count=0,
        model_backward_count=0,
        semantic_backward_count=0,
        slope_backward_count=0,
        materialization_count=0,
    )
    receipt.identity_sha256
    return AlphaQOnlySolveResult(q32=q32, receipt=receipt)


__all__ = [
    "ALPHA_Q_SOLVE_CONDITION_MAX_DIMENSION",
    "ALPHA_Q_SOLVE_DTYPE",
    "ALPHA_Q_SOLVE_EXPRESSION",
    "ALPHA_Q_SOLVE_REFERENCE",
    "AlphaQInputIdentity",
    "AlphaQOnlySolveReceipt",
    "AlphaQOnlySolveResult",
    "NATIVE_DENSE_BYTE_EQUIVALENCE",
    "solve_residual_independent_alpha_q_fp32",
]
