"""Low-rank FP32 P/C proxy inventory for the residual-reserve router."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r52_residual_reserve_geometry import RESIDUAL_RESERVE_LAYER_COUNT
from .p1r52_residual_reserve_pc_router import (
    SOLVER_PRIMAL_TOLERANCE,
    QuadraticProxy,
)


RESIDUAL_RESERVE_INVENTORY_LAYERS = (4, 5, 6, 7, 8)
P_PROXY_NAME = "PRETRAINED_DISTRIBUTION_STRUCTURAL_DRIFT_PROXY"
C_PROXY_NAME = "ACCUMULATED_GROSS_UPDATE_LOAD_PROXY"
LAMBDA_SOURCE = "SUCCESSFUL_COMMITTED_PRECAST_FP32_ONLY"
NOMINAL_FACTOR_SCALE = "CALLER_SUPPLIED_ALREADY_DIVIDED_BY_MASS_TIMES_PI_REF"


class LowRankFactorSource(str, Enum):
    NOMINAL_REFERENCE_PRECAST_FP32 = "NOMINAL_REFERENCE_PRECAST_FP32"
    SUCCESSFUL_COMMITTED_PRECAST_FP32 = "SUCCESSFUL_COMMITTED_PRECAST_FP32"


@dataclass(frozen=True, slots=True)
class LowRankFP32Factor:
    layer: int
    parameter_shape: tuple[int, int]
    left: torch.Tensor
    right: torch.Tensor
    source: LowRankFactorSource
    rank: int = field(init=False)

    def __post_init__(self) -> None:
        if self.layer not in RESIDUAL_RESERVE_INVENTORY_LAYERS:
            raise ODEBFContractError("low-rank factor layer is invalid")
        if (
            not isinstance(self.parameter_shape, tuple)
            or len(self.parameter_shape) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in self.parameter_shape
            )
        ):
            raise ODEBFContractError("low-rank factor parameter shape is invalid")
        if not isinstance(self.source, LowRankFactorSource):
            raise ODEBFContractError("low-rank factor source is invalid")
        if (
            not isinstance(self.left, torch.Tensor)
            or not isinstance(self.right, torch.Tensor)
            or self.left.dtype is not torch.float32
            or self.right.dtype is not torch.float32
            or self.left.device != self.right.device
            or self.left.ndim != 2
            or self.right.ndim != 2
            or self.left.shape[1] != self.right.shape[1]
            or self.left.shape[1] <= 0
            or self.left.shape[0] != self.parameter_shape[0]
            or self.right.shape[0] != self.parameter_shape[1]
        ):
            raise ODEBFContractError("low-rank FP32 factor contract differs")
        if not bool(torch.isfinite(self.left).all()) or not bool(
            torch.isfinite(self.right).all()
        ):
            raise ODEBFContractError("low-rank FP32 factor is non-finite")
        left = self.left.detach().clone().contiguous()
        right = self.right.detach().clone().contiguous()
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "rank", int(left.shape[1]))

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": self.layer,
            "parameter_shape": list(self.parameter_shape),
            "rank": self.rank,
            "source": self.source.value,
            "left_shape": list(self.left.shape),
            "right_shape": list(self.right.shape),
            "left_sha256": tensor_sha256(self.left),
            "right_sha256": tensor_sha256(self.right),
            "dtype": "torch.float32",
            "device": str(self.left.device),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class SealedPrevalidatedCovariance:
    layer: int
    value: torch.Tensor
    artifact_identity: str

    def __post_init__(self) -> None:
        if self.layer not in RESIDUAL_RESERVE_INVENTORY_LAYERS:
            raise ODEBFContractError("covariance layer is invalid")
        if not isinstance(self.artifact_identity, str) or not self.artifact_identity:
            raise ODEBFContractError("covariance artifact identity is missing")
        if (
            not isinstance(self.value, torch.Tensor)
            or self.value.dtype is not torch.float32
            or self.value.ndim != 2
            or self.value.shape[0] != self.value.shape[1]
            or not bool(torch.isfinite(self.value).all())
        ):
            raise ODEBFContractError("prevalidated covariance contract differs")
        object.__setattr__(
            self,
            "value",
            self.value.detach().clone().contiguous(),
        )

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": self.layer,
            "artifact_identity": self.artifact_identity,
            "shape": list(self.value.shape),
            "sha256": tensor_sha256(self.value),
            "dtype": "torch.float32",
            "device": str(self.value.device),
            "caller_prevalidated": True,
            "production_symmetry_psd_eigendecomposition_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class LayerPCInventoryInput:
    layer: int
    nominal_factor: LowRankFP32Factor
    committed_drift_factors: tuple[LowRankFP32Factor, ...]
    covariance: SealedPrevalidatedCovariance
    committed_p_constant: float
    cumulative_precast_lambda: float
    pretrained_weight_norm_squared: float
    p_constant_numerical_tolerance: float = field(init=False)

    def __post_init__(self) -> None:
        if self.layer not in RESIDUAL_RESERVE_INVENTORY_LAYERS:
            raise ODEBFContractError("P/C inventory layer is invalid")
        if not isinstance(self.nominal_factor, LowRankFP32Factor) or (
            self.nominal_factor.layer != self.layer
            or self.nominal_factor.source
            is not LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32
        ):
            raise ODEBFContractError("nominal factor provenance differs")
        if not isinstance(self.committed_drift_factors, tuple):
            raise ODEBFContractError("committed drift inventory must be immutable")
        if any(
            not isinstance(factor, LowRankFP32Factor)
            or factor.layer != self.layer
            or factor.source
            is not LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32
            or factor.parameter_shape != self.nominal_factor.parameter_shape
            or factor.left.device != self.nominal_factor.left.device
            for factor in self.committed_drift_factors
        ):
            raise ODEBFContractError("committed drift factor provenance differs")
        if not isinstance(self.covariance, SealedPrevalidatedCovariance) or (
            self.covariance.layer != self.layer
            or self.covariance.value.shape[0]
            != self.nominal_factor.parameter_shape[1]
            or self.covariance.value.device != self.nominal_factor.left.device
        ):
            raise ODEBFContractError("covariance and factor geometry differ")

        p_constant = float(self.committed_p_constant)
        cumulative_lambda = float(self.cumulative_precast_lambda)
        anchor = float(self.pretrained_weight_norm_squared)
        for name, value in (
            ("committed P constant", p_constant),
            ("cumulative pre-cast Lambda", cumulative_lambda),
            ("pretrained weight norm squared", anchor),
        ):
            if not math.isfinite(value):
                raise ODEBFContractError(f"{name} is non-finite")
        p_tolerance = SOLVER_PRIMAL_TOLERANCE
        if p_constant < -p_tolerance:
            raise ODEBFContractError("committed P constant is negative")
        if cumulative_lambda < 0.0:
            raise ODEBFContractError("cumulative pre-cast Lambda is negative")
        if anchor <= 0.0:
            raise ODEBFContractError("pretrained weight norm squared is not positive")
        object.__setattr__(self, "committed_p_constant", max(0.0, p_constant))
        object.__setattr__(self, "cumulative_precast_lambda", cumulative_lambda)
        object.__setattr__(self, "pretrained_weight_norm_squared", anchor)
        object.__setattr__(self, "p_constant_numerical_tolerance", p_tolerance)


@dataclass(frozen=True, slots=True)
class LayerPCInventoryReceipt:
    layer: int
    parameter_shape: tuple[int, int]
    nominal_factor_identity: str
    nominal_rank: int
    nominal_factor_scale: str
    nominal_redivision_count: int
    committed_factor_count: int
    committed_factor_ranks: tuple[int, ...]
    committed_factor_identities: tuple[str, ...]
    covariance_identity: str
    covariance_artifact_identity: str
    covariance_shape: tuple[int, int]
    covariance_right_matmul_count: int
    committed_p_constant: float
    committed_p_constant_numerical_tolerance: float
    cross_term: float
    p_linear: float
    p_quadratic_diagonal: float
    p_quadratic_nonnegative_tolerance: float
    cumulative_precast_lambda: float
    lambda_source: str
    pretrained_weight_norm_squared: float
    nominal_update_norm_squared: float
    nominal_norm_nonnegative_tolerance: float
    c_quadratic_diagonal: float
    dense_drift_materialization_count: int
    dense_nominal_materialization_count: int
    dense_candidate_materialization_count: int
    post_storage_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "parameter_shape": list(self.parameter_shape),
            "nominal_factor_identity": self.nominal_factor_identity,
            "nominal_rank": self.nominal_rank,
            "nominal_factor_scale": self.nominal_factor_scale,
            "nominal_redivision_count": self.nominal_redivision_count,
            "committed_factor_count": self.committed_factor_count,
            "committed_factor_ranks": list(self.committed_factor_ranks),
            "committed_factor_identities": list(self.committed_factor_identities),
            "covariance_identity": self.covariance_identity,
            "covariance_artifact_identity": self.covariance_artifact_identity,
            "covariance_shape": list(self.covariance_shape),
            "covariance_right_matmul_count": self.covariance_right_matmul_count,
            "committed_p_constant": self.committed_p_constant,
            "committed_p_constant_numerical_tolerance": (
                self.committed_p_constant_numerical_tolerance
            ),
            "cross_term": self.cross_term,
            "p_linear": self.p_linear,
            "p_quadratic_diagonal": self.p_quadratic_diagonal,
            "p_quadratic_nonnegative_tolerance": (
                self.p_quadratic_nonnegative_tolerance
            ),
            "cumulative_precast_lambda": self.cumulative_precast_lambda,
            "lambda_source": self.lambda_source,
            "pretrained_weight_norm_squared": self.pretrained_weight_norm_squared,
            "nominal_update_norm_squared": self.nominal_update_norm_squared,
            "nominal_norm_nonnegative_tolerance": (
                self.nominal_norm_nonnegative_tolerance
            ),
            "c_quadratic_diagonal": self.c_quadratic_diagonal,
            "dense_drift_materialization_count": self.dense_drift_materialization_count,
            "dense_nominal_materialization_count": (
                self.dense_nominal_materialization_count
            ),
            "dense_candidate_materialization_count": (
                self.dense_candidate_materialization_count
            ),
            "post_storage_decision_influence_count": (
                self.post_storage_decision_influence_count
            ),
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_hash(self.raw_free_payload())


@dataclass(frozen=True, slots=True)
class PCInventoryReceipt:
    mass: float
    mass_sha256: str
    layer_receipts: tuple[LayerPCInventoryReceipt, ...]
    p_proxy_identity: str
    c_proxy_identity: str
    uniform_p_gradient: tuple[float, ...]
    uniform_c_gradient: tuple[float, ...]
    uniform_p_gradient_variance: float
    uniform_c_gradient_variance: float
    uniform_gradient_dot: float
    uniform_gradient_cosine: float | None
    uniform_gradient_alignment: str
    p_q_rank: int
    c_q_rank: int
    simplex_degrees_of_freedom: int
    input_immutability_verified: bool
    output_tensor_dtype: str
    output_tensor_device: str
    model_forward_count: int
    model_backward_count: int
    dense_materialization_count: int
    post_storage_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mass": self.mass,
            "mass_sha256": self.mass_sha256,
            "layer_receipts": [
                item.raw_free_payload() for item in self.layer_receipts
            ],
            "p_proxy_identity": self.p_proxy_identity,
            "c_proxy_identity": self.c_proxy_identity,
            "uniform_p_gradient": list(self.uniform_p_gradient),
            "uniform_c_gradient": list(self.uniform_c_gradient),
            "uniform_p_gradient_variance": self.uniform_p_gradient_variance,
            "uniform_c_gradient_variance": self.uniform_c_gradient_variance,
            "uniform_gradient_dot": self.uniform_gradient_dot,
            "uniform_gradient_cosine": self.uniform_gradient_cosine,
            "uniform_gradient_alignment": self.uniform_gradient_alignment,
            "p_q_rank": self.p_q_rank,
            "c_q_rank": self.c_q_rank,
            "simplex_degrees_of_freedom": self.simplex_degrees_of_freedom,
            "input_immutability_verified": self.input_immutability_verified,
            "output_tensor_dtype": self.output_tensor_dtype,
            "output_tensor_device": self.output_tensor_device,
            "model_forward_count": self.model_forward_count,
            "model_backward_count": self.model_backward_count,
            "dense_materialization_count": self.dense_materialization_count,
            "post_storage_decision_influence_count": (
                self.post_storage_decision_influence_count
            ),
            "lambda_source": LAMBDA_SOURCE,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class PCInventoryResult:
    p_proxy: QuadraticProxy
    c_proxy: QuadraticProxy
    receipt: PCInventoryReceipt


@dataclass(frozen=True, slots=True)
class _TensorGuard:
    tensor: torch.Tensor
    pointer: int
    version: int
    sha256: str


def _tensor_guards(
    mass: torch.Tensor,
    layers: Iterable[LayerPCInventoryInput],
) -> tuple[_TensorGuard, ...]:
    tensors: list[torch.Tensor] = [mass]
    for item in layers:
        tensors.extend((item.nominal_factor.left, item.nominal_factor.right))
        for factor in item.committed_drift_factors:
            tensors.extend((factor.left, factor.right))
        tensors.append(item.covariance.value)
    return tuple(
        _TensorGuard(
            tensor=value,
            pointer=int(value.data_ptr()),
            version=int(value._version),
            sha256=tensor_sha256(value),
        )
        for value in tensors
    )


def _verify_guards(guards: tuple[_TensorGuard, ...]) -> None:
    if any(
        int(item.tensor.data_ptr()) != item.pointer
        or int(item.tensor._version) != item.version
        or tensor_sha256(item.tensor) != item.sha256
        for item in guards
    ):
        raise ODEBFStateError("P/C inventory input tensor mutated")


def _small_gram_trace(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[float, float, int]:
    product = left.to(dtype=torch.float64) * right.T.to(dtype=torch.float64)
    return float(product.sum()), float(product.abs().sum()), product.numel()


def _nonnegative_reduction(
    name: str,
    value: float,
    absolute_sum: float,
    term_count: int,
) -> tuple[float, float]:
    tolerance = (
        torch.finfo(torch.float32).eps
        * max(1, term_count)
        * max(1.0, absolute_sum)
    )
    if not math.isfinite(value) or value < -tolerance:
        raise ODEBFContractError(f"{name} is negative or non-finite")
    return max(0.0, value), tolerance


def _uniform_gradient_observation(
    p_proxy: QuadraticProxy,
    c_proxy: QuadraticProxy,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    float,
    float,
    float,
    float | None,
    str,
]:
    uniform = torch.full(
        (RESIDUAL_RESERVE_LAYER_COUNT,),
        1.0 / RESIDUAL_RESERVE_LAYER_COUNT,
        dtype=torch.float32,
    )
    p_gradient = (p_proxy.linear + 2.0 * (p_proxy.quadratic @ uniform)).contiguous()
    c_gradient = (c_proxy.linear + 2.0 * (c_proxy.quadratic @ uniform)).contiguous()
    p64 = p_gradient.to(dtype=torch.float64)
    c64 = c_gradient.to(dtype=torch.float64)
    p_variance = float(torch.var(p64, correction=0))
    c_variance = float(torch.var(c64, correction=0))
    dot = float(torch.dot(p64, c64))
    denominator = float(torch.linalg.vector_norm(p64) * torch.linalg.vector_norm(c64))
    cosine = None if denominator == 0.0 else dot / denominator
    alignment = "ZERO" if dot == 0.0 else "POSITIVE" if dot > 0.0 else "NEGATIVE"
    return (
        tuple(float(item) for item in p_gradient),
        tuple(float(item) for item in c_gradient),
        p_variance,
        c_variance,
        dot,
        cosine,
        alignment,
    )


def build_residual_reserve_pc_inventory(
    layers: tuple[LayerPCInventoryInput, ...],
    *,
    mass: torch.Tensor,
) -> PCInventoryResult:
    """Build P/C quadratic coefficients using only low-rank Gram contractions."""

    if (
        not isinstance(layers, tuple)
        or not all(isinstance(item, LayerPCInventoryInput) for item in layers)
        or tuple(item.layer for item in layers) != RESIDUAL_RESERVE_INVENTORY_LAYERS
    ):
        raise ODEBFContractError("P/C layer inventory is incomplete or out of order")
    if (
        not isinstance(mass, torch.Tensor)
        or mass.dtype is not torch.float32
        or mass.ndim != 0
        or not bool(torch.isfinite(mass))
        or not (0.0 < float(mass) < 1.0)
    ):
        raise ODEBFContractError("P/C inventory mass contract differs")
    guards = _tensor_guards(mass, layers)
    mass_value = float(mass.detach().to(device="cpu"))

    p_constants: list[float] = []
    p_linear_values: list[float] = []
    p_diagonal_values: list[float] = []
    c_diagonal_values: list[float] = []
    layer_receipts: list[LayerPCInventoryReceipt] = []
    for item in layers:
        nominal = item.nominal_factor
        covariance = item.covariance
        sigma_right = covariance.value @ nominal.right

        nominal_left_gram = nominal.left.T @ nominal.left
        nominal_right_sigma_gram = nominal.right.T @ sigma_right
        p_energy_raw, p_energy_absolute, p_energy_terms = _small_gram_trace(
            nominal_left_gram,
            nominal_right_sigma_gram,
        )
        p_energy, p_energy_tolerance = _nonnegative_reduction(
            "nominal P quadratic contraction",
            p_energy_raw,
            p_energy_absolute,
            p_energy_terms,
        )

        nominal_right_gram = nominal.right.T @ nominal.right
        nominal_norm_raw, nominal_norm_absolute, nominal_norm_terms = _small_gram_trace(
            nominal_left_gram,
            nominal_right_gram,
        )
        nominal_norm_squared, nominal_norm_tolerance = _nonnegative_reduction(
            "nominal update norm contraction",
            nominal_norm_raw,
            nominal_norm_absolute,
            nominal_norm_terms,
        )

        cross_terms: list[float] = []
        for historical in item.committed_drift_factors:
            left_gram = nominal.left.T @ historical.left
            right_gram = historical.right.T @ sigma_right
            cross, _, _ = _small_gram_trace(left_gram, right_gram)
            if not math.isfinite(cross):
                raise ODEBFContractError("historical P cross contraction is non-finite")
            cross_terms.append(cross)
        cross_total = math.fsum(cross_terms)
        p_linear = 2.0 * mass_value * cross_total
        p_diagonal = mass_value * mass_value * p_energy
        c_diagonal = (
            (1.0 + item.cumulative_precast_lambda)
            * mass_value
            * mass_value
            * nominal_norm_squared
            / item.pretrained_weight_norm_squared
        )
        if not all(
            math.isfinite(value)
            for value in (cross_total, p_linear, p_diagonal, c_diagonal)
        ):
            raise ODEBFContractError("P/C inventory coefficient is non-finite")

        p_constants.append(item.committed_p_constant)
        p_linear_values.append(p_linear)
        p_diagonal_values.append(p_diagonal)
        c_diagonal_values.append(c_diagonal)
        layer_receipts.append(
            LayerPCInventoryReceipt(
                layer=item.layer,
                parameter_shape=nominal.parameter_shape,
                nominal_factor_identity=nominal.identity_sha256,
                nominal_rank=nominal.rank,
                nominal_factor_scale=NOMINAL_FACTOR_SCALE,
                nominal_redivision_count=0,
                committed_factor_count=len(item.committed_drift_factors),
                committed_factor_ranks=tuple(
                    factor.rank for factor in item.committed_drift_factors
                ),
                committed_factor_identities=tuple(
                    factor.identity_sha256
                    for factor in item.committed_drift_factors
                ),
                covariance_identity=covariance.identity_sha256,
                covariance_artifact_identity=covariance.artifact_identity,
                covariance_shape=tuple(covariance.value.shape),
                covariance_right_matmul_count=1,
                committed_p_constant=item.committed_p_constant,
                committed_p_constant_numerical_tolerance=(
                    item.p_constant_numerical_tolerance
                ),
                cross_term=cross_total,
                p_linear=p_linear,
                p_quadratic_diagonal=p_diagonal,
                p_quadratic_nonnegative_tolerance=(
                    mass_value * mass_value * p_energy_tolerance
                ),
                cumulative_precast_lambda=item.cumulative_precast_lambda,
                lambda_source=LAMBDA_SOURCE,
                pretrained_weight_norm_squared=(
                    item.pretrained_weight_norm_squared
                ),
                nominal_update_norm_squared=nominal_norm_squared,
                nominal_norm_nonnegative_tolerance=nominal_norm_tolerance,
                c_quadratic_diagonal=c_diagonal,
                dense_drift_materialization_count=0,
                dense_nominal_materialization_count=0,
                dense_candidate_materialization_count=0,
                post_storage_decision_influence_count=0,
            )
        )

    p_constant = math.fsum(p_constants)
    if not math.isfinite(p_constant):
        raise ODEBFContractError("total committed P constant is non-finite")
    p_linear = torch.tensor(p_linear_values, dtype=torch.float32, device="cpu")
    p_quadratic = torch.diag(
        torch.tensor(p_diagonal_values, dtype=torch.float32, device="cpu")
    ).contiguous()
    c_linear = torch.zeros(RESIDUAL_RESERVE_LAYER_COUNT, dtype=torch.float32)
    c_quadratic = torch.diag(
        torch.tensor(c_diagonal_values, dtype=torch.float32, device="cpu")
    ).contiguous()
    p_proxy = QuadraticProxy(P_PROXY_NAME, p_constant, p_linear, p_quadratic)
    c_proxy = QuadraticProxy(C_PROXY_NAME, 0.0, c_linear, c_quadratic)
    observation = _uniform_gradient_observation(p_proxy, c_proxy)
    _verify_guards(guards)
    receipt = PCInventoryReceipt(
        mass=mass_value,
        mass_sha256=tensor_sha256(mass),
        layer_receipts=tuple(layer_receipts),
        p_proxy_identity=p_proxy.raw_free_payload()["identity_sha256"],
        c_proxy_identity=c_proxy.raw_free_payload()["identity_sha256"],
        uniform_p_gradient=observation[0],
        uniform_c_gradient=observation[1],
        uniform_p_gradient_variance=observation[2],
        uniform_c_gradient_variance=observation[3],
        uniform_gradient_dot=observation[4],
        uniform_gradient_cosine=observation[5],
        uniform_gradient_alignment=observation[6],
        p_q_rank=p_proxy.rank,
        c_q_rank=c_proxy.rank,
        simplex_degrees_of_freedom=RESIDUAL_RESERVE_LAYER_COUNT - 1,
        input_immutability_verified=True,
        output_tensor_dtype="torch.float32",
        output_tensor_device="cpu",
        model_forward_count=0,
        model_backward_count=0,
        dense_materialization_count=0,
        post_storage_decision_influence_count=0,
    )
    return PCInventoryResult(p_proxy, c_proxy, receipt)


__all__ = [
    "C_PROXY_NAME",
    "LAMBDA_SOURCE",
    "LayerPCInventoryInput",
    "LayerPCInventoryReceipt",
    "LowRankFP32Factor",
    "LowRankFactorSource",
    "NOMINAL_FACTOR_SCALE",
    "PCInventoryReceipt",
    "PCInventoryResult",
    "P_PROXY_NAME",
    "RESIDUAL_RESERVE_INVENTORY_LAYERS",
    "SealedPrevalidatedCovariance",
    "build_residual_reserve_pc_inventory",
]
