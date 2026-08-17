"""CPU-only preparation primitives for P1R29 sequential Historical editing.

This module deliberately contains no model, evaluator, GPU, Slurm, or result-
root entry point.  It prepares the state, exact historical Woodbury, incremental
Structural-H, cumulative Structural-P, and adapter-supplied routing contracts
that may be connected only after a viable Atomic handoff is authorized.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import minimize

from .contracts import BATCH_SIZE, FIXED_K, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1_state import (
    P1HistoryLedger,
    ProspectiveP1HistoryBatch,
)
from .progress_simplex_routing import (
    SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
    SIMPLEX_ENERGY_RELATIVE_TOLERANCE,
    SIMPLEX_PRIMAL_TOLERANCE,
    SIMPLEX_XI_TIE_TOLERANCE,
)
from .woodbury import ProjectorCertificate, solve_alpha_woodbury


P1R29_SEQUENTIAL_PREPARATION_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R29-SEQUENTIAL-HISTORICAL-PREPARATION-V1"
)
P1R29_SEQUENTIAL_PREPARATION_METHOD_ID = (
    "P1R29-SEQUENTIAL-HISTORICAL-PREPARATION-NO-ATOMIC-WINNER-V1"
)
P1R29_BACKEND_HARDENING_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R29-INDEPENDENT-SEQUENTIAL-BACKEND-HARDENING-V1"
)
P1R29_BACKEND_METHOD_ID = "ADAPTER-NEUTRAL-SEQUENTIAL-HISTORICAL-BACKEND-V1"
P1R30_BACKEND_ADAPTER_AMENDMENT_ID = (
    "P1R30-DEBT-PRIORITY-A0-RELATIVE-BARRIER-BACKEND-ADAPTER-V1"
)
FIXED_H = 1.0 / FIXED_K
SEQUENTIAL_ROUNDS = 10
MAXIMUM_HISTORY_RECORDS = BATCH_SIZE * SEQUENTIAL_ROUNDS
WOODBURY_RESIDUAL_TOLERANCE = 1.0e-8
SOFT_HP_SOLVER_FTOL = 1.0e-12
SOFT_HP_SOLVER_MAXITER = 500
NORMALIZATION_EPSILON = 1.0e-12


def _finite_tensor(name: str, value: torch.Tensor, *, ndim: int = 2) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != ndim
        or value.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(value).all()
    ):
        raise ODEBFContractError(f"{name} is not a finite FP32/FP64 tensor")
    return value.detach()


@dataclass(frozen=True, slots=True)
class AtomicAdapterPreprocessingIdentity:
    """Identity supplied by the selected Atomic adapter, never rebuilt here."""

    adapter_id: str
    preprocessing_sha256: str
    request_order_sha256: str
    tokenizer_target_normalization_sha256: str

    def __post_init__(self) -> None:
        if not self.adapter_id or any(
            not isinstance(value, str) or len(value) != 64
            for value in (
                self.preprocessing_sha256,
                self.request_order_sha256,
                self.tokenizer_target_normalization_sha256,
            )
        ):
            raise ODEBFContractError("Atomic adapter preprocessing identity differs")


@dataclass(frozen=True, slots=True)
class AtomicAdapterConstraintCertificate:
    """Raw-free pointer to an adapter-owned feasibility/certificate receipt."""

    constraint_family: str
    status: str
    constraint_receipt_sha256: str
    certificate_receipt_sha256: str
    reference_feasible: bool
    fallback_to_reference: bool

    def __post_init__(self) -> None:
        if not self.constraint_family or not self.status:
            raise ODEBFContractError("Atomic adapter constraint certificate differs")
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in (
                self.constraint_receipt_sha256,
                self.certificate_receipt_sha256,
            )
        ):
            raise ODEBFContractError("Atomic adapter constraint receipt identity differs")


@dataclass(frozen=True, slots=True)
class AtomicAdapterRoutingReference:
    """Adapter-owned action geometry; the backend imposes no universal rho equality."""

    reference_coefficients: tuple[float, ...]
    reference_mean_progress: float
    reference_debt_priority_progress: float
    reference_energy: float
    selected_coefficients: tuple[float, ...]
    constraint_certificate: AtomicAdapterConstraintCertificate

    def validate_for_layers(self, layer_order: Sequence[int]) -> None:
        layer_count = len(layer_order)
        if (
            len(self.reference_coefficients) != layer_count
            or len(self.selected_coefficients) != layer_count
        ):
            raise ODEBFContractError("Atomic adapter coefficient geometry differs")
        if not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in self.reference_coefficients + self.selected_coefficients
        ):
            raise ODEBFContractError("Atomic adapter coefficients are not finite nonnegative")
        if not all(
            math.isfinite(float(value))
            for value in (
                self.reference_mean_progress,
                self.reference_debt_priority_progress,
            )
        ) or not (
            math.isfinite(float(self.reference_energy))
            and float(self.reference_energy) >= 0.0
        ):
            raise ODEBFContractError("Atomic adapter reference geometry differs")

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "reference_coefficients": self.reference_coefficients,
            "reference_mean_progress": self.reference_mean_progress,
            "reference_debt_priority_progress": self.reference_debt_priority_progress,
            "reference_energy": self.reference_energy,
            "selected_coefficients": self.selected_coefficients,
            "constraint_certificate": asdict(self.constraint_certificate),
            "universal_semantic_rho_equality_required": False,
            "universal_rho_over_slope_transform_required": False,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class AtomicSequentialAdapterFrame:
    """Adapter-neutral current-state input for one future sequential field."""

    preprocessing: AtomicAdapterPreprocessingIdentity
    layer_order: tuple[int, ...]
    current_w_by_layer: Mapping[int, torch.Tensor]
    current_z: torch.Tensor
    proposals_by_layer: Mapping[int, torch.Tensor]
    physical_signed_slopes: tuple[float, ...]
    routing_reference: AtomicAdapterRoutingReference
    current_raw_keys_by_layer: Mapping[int, torch.Tensor]
    accepted_state_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.accepted_state_sha256, str) or len(self.accepted_state_sha256) != 64:
            raise ODEBFContractError("Atomic adapter current-state identity differs")
        if len(self.layer_order) < 2 or len(set(self.layer_order)) != len(self.layer_order):
            raise ODEBFContractError("Atomic adapter layer order differs")
        if set(self.current_w_by_layer) != set(self.layer_order) or set(
            self.proposals_by_layer
        ) != set(self.layer_order) or set(
            self.current_raw_keys_by_layer
        ) != set(self.layer_order):
            raise ODEBFContractError("Atomic adapter layer payload is incomplete")
        if len(self.physical_signed_slopes) != len(self.layer_order) or not all(
            math.isfinite(float(value)) for value in self.physical_signed_slopes
        ):
            raise ODEBFContractError("Atomic adapter physical slope geometry differs")
        self.routing_reference.validate_for_layers(self.layer_order)
        _finite_tensor("Atomic adapter current z", self.current_z)
        for layer in self.layer_order:
            weight = self.current_w_by_layer[layer]
            if weight.dtype != torch.bfloat16 or weight.ndim != 2 or not torch.isfinite(weight).all():
                raise ODEBFContractError("Atomic adapter current W is not finite BF16")
            _finite_tensor("Atomic adapter proposal", self.proposals_by_layer[layer])
            key = _finite_tensor("Atomic adapter current raw key", self.current_raw_keys_by_layer[layer])
            if key.shape[1] != BATCH_SIZE:
                raise ODEBFContractError("Atomic adapter current key is not B10")

    def raw_free_receipt(self) -> dict[str, Any]:
        payload = {
            "adapter_id": self.preprocessing.adapter_id,
            "preprocessing_sha256": self.preprocessing.preprocessing_sha256,
            "request_order_sha256": self.preprocessing.request_order_sha256,
            "tokenizer_target_normalization_sha256": (
                self.preprocessing.tokenizer_target_normalization_sha256
            ),
            "current_weight_sha256": tuple(
                (layer, tensor_sha256(self.current_w_by_layer[layer]))
                for layer in self.layer_order
            ),
            "current_z_sha256": tensor_sha256(self.current_z),
            "accepted_state_sha256": self.accepted_state_sha256,
            "layer_order": self.layer_order,
            "proposal_sha256": tuple(
                (layer, tensor_sha256(self.proposals_by_layer[layer]))
                for layer in self.layer_order
            ),
            "physical_signed_slopes": self.physical_signed_slopes,
            "routing_reference": self.routing_reference.raw_free_payload(),
            "current_raw_key_sha256": tuple(
                (layer, tensor_sha256(self.current_raw_keys_by_layer[layer]))
                for layer in self.layer_order
            ),
            "target_or_debt_preprocessing_implemented_by_backend": False,
            "shared_preprocessing_owned_by_selected_adapter": True,
            "universal_semantic_rho_equality_required": False,
            "universal_rho_over_slope_transform_required": False,
            "backend_preprocessing_model_forward_count": 0,
            "backend_preprocessing_backward_count": 0,
            "backend_h_p_model_forward_count": 0,
            "backend_h_p_backward_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class TerminalPhysicalKeyIdentity:
    committed_bf16_weight_sha256: str
    request_order_sha256: str
    tokenizer_target_normalization_sha256: str
    layer_set_sha256: str
    capture_method_sha256: str
    hook_disabled_physical_state_sha256: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in asdict(self).values()
        ):
            raise ODEBFContractError("terminal physical-key identity differs")

    @property
    def digest(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class AtomicSequentialTerminalFrame:
    preprocessing: AtomicAdapterPreprocessingIdentity
    terminal_keys_by_layer: Mapping[int, torch.Tensor]
    physical_identity: TerminalPhysicalKeyIdentity

    def __post_init__(self) -> None:
        if (
            self.physical_identity.request_order_sha256
            != self.preprocessing.request_order_sha256
            or self.physical_identity.tokenizer_target_normalization_sha256
            != self.preprocessing.tokenizer_target_normalization_sha256
        ):
            raise ODEBFContractError("terminal keys do not share Atomic preprocessing")
        if not self.terminal_keys_by_layer:
            raise ODEBFContractError("Atomic adapter terminal keys are absent")
        for value in self.terminal_keys_by_layer.values():
            key = _finite_tensor("Atomic adapter terminal key", value)
            if key.shape[1] != BATCH_SIZE:
                raise ODEBFContractError("Atomic adapter terminal key is not B10")


@dataclass(frozen=True, slots=True)
class HistoryFactorization:
    """History-version factorization of ``lambda I + K_H^T Z_H``."""

    history_version: int
    history_columns: int
    active_record_sha256: str
    regularization: float
    raw_history_sha256: str
    projected_history_sha256: str
    gram_sha256: str
    projector_sha256: str
    factor_method: str
    condition: float
    factor: torch.Tensor
    pivots: torch.Tensor

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        if self.history_columns == 0:
            return torch.empty_like(rhs)
        return torch.linalg.lu_solve(self.factor, self.pivots, rhs)


def build_history_factorization(
    projector: torch.Tensor,
    raw_history_keys: torch.Tensor,
    projected_history_keys: torch.Tensor,
    *,
    history_version: int,
    active_record_sha256: str,
    regularization: float,
) -> HistoryFactorization:
    """Validate paired history and cache its exact general-projector block."""

    matrix = _finite_tensor("history projector", projector)
    raw = _finite_tensor("raw historical solve keys", raw_history_keys)
    projected = _finite_tensor("projected historical risk keys", projected_history_keys)
    if matrix.shape[0] != matrix.shape[1] or raw.shape != projected.shape:
        raise ODEBFContractError("historical raw/projected key geometry differs")
    if raw.shape[0] != matrix.shape[0]:
        raise ODEBFContractError("historical key dimension differs from projector")
    if (
        history_version < 0
        or raw.shape[1] < 0
        or raw.shape[1] > MAXIMUM_HISTORY_RECORDS
        or raw.shape[1] % BATCH_SIZE != 0
        or not isinstance(active_record_sha256, str)
        or len(active_record_sha256) != 64
    ):
        raise ODEBFContractError("active history count or record identity differs")
    lam = float(regularization)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ODEBFContractError("historical Woodbury regularization is not positive")

    raw64 = raw.to(dtype=torch.float64)
    expected = (matrix @ raw.to(dtype=matrix.dtype)).to(dtype=torch.float64)
    projected64 = projected.to(dtype=torch.float64)
    drift = float(torch.max(torch.abs(expected - projected64))) if projected64.numel() else 0.0
    if drift != 0.0:
        raise ODEBFContractError("precomputed projected history has nonzero semantic drift")

    columns = raw.shape[1]
    if columns:
        gram = lam * torch.eye(columns, dtype=torch.float64) + raw64.T @ projected64
        factor, pivots, info = torch.linalg.lu_factor_ex(gram)
        if int(info.max()) != 0:
            raise ODEBFContractError("historical general-projector factorization failed")
        condition = float(torch.linalg.cond(gram))
        if not math.isfinite(condition):
            raise ODEBFContractError("historical factorization condition is nonfinite")
    else:
        gram = torch.empty((0, 0), dtype=torch.float64)
        factor = torch.empty((0, 0), dtype=torch.float64)
        pivots = torch.empty((0,), dtype=torch.int32)
        condition = 1.0
    return HistoryFactorization(
        history_version=history_version,
        history_columns=columns,
        active_record_sha256=active_record_sha256,
        regularization=lam,
        raw_history_sha256=tensor_sha256(raw64),
        projected_history_sha256=tensor_sha256(projected64),
        gram_sha256=tensor_sha256(gram),
        projector_sha256=tensor_sha256(matrix),
        factor_method="general-projector-lu-history-block",
        condition=condition,
        factor=factor,
        pivots=pivots,
    )


@dataclass(frozen=True, slots=True)
class CachedWoodburyReceipt:
    history_version: int
    history_columns: int
    current_columns: int
    small_dimension: int
    history_cache_hit: bool
    cache_mismatch_exact_backend_fallback: bool
    cache_key_sha256: str
    block_method: str
    alpha_residual: float
    exact_backend_max_abs: float
    parity_passed: bool
    model_forward_count: int = 0
    backward_count: int = 0


def solve_alpha_woodbury_cached(
    projector: torch.Tensor,
    current_keys: torch.Tensor,
    raw_history_keys: torch.Tensor,
    projected_history_keys: torch.Tensor,
    *,
    history_version: int,
    active_record_sha256: str,
    regularization: float,
    projector_certificate: ProjectorCertificate,
    cached_factorization: HistoryFactorization | None,
    residual_tolerance: float = WOODBURY_RESIDUAL_TOLERANCE,
) -> tuple[torch.Tensor, HistoryFactorization, CachedWoodburyReceipt]:
    """Exact block/Schur Woodbury solve with an independently checked backend."""

    matrix = _finite_tensor("projector", projector)
    current = _finite_tensor("current keys", current_keys)
    raw = _finite_tensor("raw history keys", raw_history_keys)
    projected = _finite_tensor("projected history keys", projected_history_keys)
    if current.shape != (matrix.shape[0], BATCH_SIZE):
        raise ODEBFContractError("current Alpha keys are not joint B10")
    if raw.shape != projected.shape or raw.shape[0] != matrix.shape[0]:
        raise ODEBFContractError("paired historical key geometry differs")

    factor = cached_factorization
    hit = factor is not None
    exact_fallback = False
    expected_identity = (
        history_version,
        raw.shape[1],
        active_record_sha256,
        float(regularization),
        tensor_sha256(raw.to(dtype=torch.float64)),
        tensor_sha256(projected.to(dtype=torch.float64)),
        tensor_sha256(matrix),
    )
    if factor is None:
        factor = build_history_factorization(
            matrix,
            raw,
            projected,
            history_version=history_version,
            active_record_sha256=active_record_sha256,
            regularization=regularization,
        )
    observed_identity = (
        factor.history_version,
        factor.history_columns,
        factor.active_record_sha256,
        factor.regularization,
        factor.raw_history_sha256,
        factor.projected_history_sha256,
        factor.projector_sha256,
    )
    if observed_identity != expected_identity:
        # A stale/mismatched cache is never allowed to alter the solve.  Use the
        # established exact backend for this field and rebuild the history-only
        # factor for a future state with the current exact identities.
        exact_fallback = True
        exact = solve_alpha_woodbury(
            matrix,
            current,
            history_keys=raw,
            regularization=float(regularization),
            projector_certificate=projector_certificate,
            residual_tolerance=residual_tolerance,
        )
        factor = build_history_factorization(
            matrix,
            raw,
            projected,
            history_version=history_version,
            active_record_sha256=active_record_sha256,
            regularization=regularization,
        )
        cache_key = canonical_hash(
            {
                "version": history_version,
                "active_records": active_record_sha256,
                "raw": factor.raw_history_sha256,
                "projected": factor.projected_history_sha256,
                "projector": factor.projector_sha256,
            }
        )
        return exact.q, factor, CachedWoodburyReceipt(
            history_version,
            raw.shape[1],
            BATCH_SIZE,
            BATCH_SIZE + raw.shape[1],
            False,
            True,
            cache_key,
            "exact-backend-cache-identity-fallback",
            exact.certificate.alpha_linear_residual,
            0.0,
            True,
        )

    lam = factor.regularization
    current64 = current.to(dtype=torch.float64)
    raw64 = raw.to(dtype=torch.float64)
    projected64 = projected.to(dtype=torch.float64)
    g = (matrix @ current.to(dtype=matrix.dtype)).to(dtype=torch.float64)
    current_block = lam * torch.eye(BATCH_SIZE, dtype=torch.float64) + current64.T @ g
    rhs_current = current64.T @ g
    if factor.history_columns:
        upper = current64.T @ projected64
        lower = raw64.T @ g
        d_inv_lower = factor.solve(lower)
        d_inv_rhs = factor.solve(raw64.T @ g)
        schur = current_block - upper @ d_inv_lower
        schur_rhs = rhs_current - upper @ d_inv_rhs
        solve_current, info = torch.linalg.solve_ex(schur, schur_rhs)
        if int(info.max()) != 0:
            raise ODEBFContractError("current Woodbury Schur solve failed")
        solve_history = factor.solve(raw64.T @ g - lower @ solve_current)
        combined_solution = torch.cat((solve_current, solve_history), dim=0)
        combined_raw = torch.cat((current64, raw64), dim=1)
        combined_projected = torch.cat((g, projected64), dim=1)
    else:
        solve_current, info = torch.linalg.solve_ex(current_block, rhs_current)
        if int(info.max()) != 0:
            raise ODEBFContractError("empty-history current Woodbury solve failed")
        combined_solution = solve_current
        combined_raw = current64
        combined_projected = g

    q = (g - combined_projected @ combined_solution) / lam
    action = lam * q + combined_projected @ (combined_raw.T @ q)
    denominator = max(float(torch.linalg.norm(g)), torch.finfo(torch.float64).eps)
    alpha_residual = float(torch.linalg.norm(action - g)) / denominator
    exact = solve_alpha_woodbury(
        matrix,
        current,
        history_keys=raw,
        regularization=lam,
        projector_certificate=projector_certificate,
        residual_tolerance=residual_tolerance,
    )
    exact_max_abs = float(torch.max(torch.abs(q - exact.q)))
    parity = (
        alpha_residual <= residual_tolerance
        and exact_max_abs <= residual_tolerance
        and exact.certificate.passed
    )
    if not parity:
        raise ODEBFContractError("cached Woodbury differs from exact backend; cache disabled")
    receipt = CachedWoodburyReceipt(
        history_version,
        raw.shape[1],
        BATCH_SIZE,
        BATCH_SIZE + raw.shape[1],
        hit,
        exact_fallback,
        canonical_hash(
            {
                "version": history_version,
                "active_records": active_record_sha256,
                "raw": factor.raw_history_sha256,
                "projected": factor.projected_history_sha256,
                "projector": factor.projector_sha256,
            }
        ),
        "general-projector-history-lu-current-schur",
        alpha_residual,
        exact_max_abs,
        True,
    )
    return q, factor, receipt


@dataclass(frozen=True, slots=True)
class LowRankUpdate:
    """One accepted low-rank update and its cached covariance action."""

    left: torch.Tensor
    right: torch.Tensor
    covariance_right: torch.Tensor
    coefficient: float
    identity_sha256: str

    @classmethod
    def build(
        cls,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        coefficient: float,
        covariance_action: Callable[[torch.Tensor], torch.Tensor],
    ) -> "LowRankUpdate":
        lhs = _finite_tensor("low-rank left", left)
        rhs = _finite_tensor("low-rank right", right)
        if lhs.shape[1] != rhs.shape[1]:
            raise ODEBFContractError("low-rank factor ranks differ")
        cov = _finite_tensor("covariance action", covariance_action(rhs))
        if cov.shape != rhs.shape:
            raise ODEBFContractError("covariance action shape differs")
        scale = float(coefficient)
        if not math.isfinite(scale):
            raise ODEBFContractError("low-rank coefficient is nonfinite")
        identity = canonical_hash(
            {
                "left": tensor_sha256(lhs),
                "right": tensor_sha256(rhs),
                "covariance_right": tensor_sha256(cov),
                "coefficient": scale,
            }
        )
        return cls(lhs.clone(), rhs.clone(), cov.clone(), scale, identity)

    def scaled(self, multiplier: float) -> "LowRankUpdate":
        scale = self.coefficient * float(multiplier)
        if not math.isfinite(scale):
            raise ODEBFContractError("scaled low-rank coefficient is nonfinite")
        identity = canonical_hash(
            {
                "source": self.identity_sha256,
                "coefficient": scale,
            }
        )
        return LowRankUpdate(
            self.left,
            self.right,
            self.covariance_right,
            scale,
            identity,
        )


def _pretrained_inner(left: LowRankUpdate, right: LowRankUpdate) -> float:
    value = torch.trace(
        (right.left.T.to(dtype=torch.float64) @ left.left.to(dtype=torch.float64))
        @ (left.right.T.to(dtype=torch.float64) @ right.covariance_right.to(dtype=torch.float64))
    )
    return float(value) * left.coefficient * right.coefficient


@dataclass(frozen=True, slots=True)
class QuadraticRoutingRisk:
    offset: float
    linear: np.ndarray
    gram: np.ndarray
    normalization: float
    label: str

    def raw_value(self, velocity: np.ndarray) -> float:
        value = np.asarray(velocity, dtype=np.float64)
        return float(self.offset + self.linear @ value + value @ self.gram @ value)

    def marginal_value(self, velocity: np.ndarray) -> float:
        value = np.asarray(velocity, dtype=np.float64)
        return float(self.linear @ value + value @ self.gram @ value)

    def normalized_value(self, velocity: np.ndarray) -> float:
        return self.raw_value(velocity) / self.normalization


@dataclass(frozen=True, slots=True)
class CumulativePCheckpoint:
    contributions: tuple[tuple[int, tuple[LowRankUpdate, ...]], ...]
    prior_left: tuple[tuple[int, torch.Tensor], ...]
    prior_right: tuple[tuple[int, torch.Tensor], ...]
    prior_covariance_right: tuple[tuple[int, torch.Tensor], ...]
    committed_load: tuple[tuple[int, float], ...]
    prior_value: float
    accepted_update_count: int
    batched_cross_call_count: int
    prior_pairwise_replay_count: int


@dataclass(slots=True)
class CumulativeStructuralPState:
    """Layer-wise accepted low-rank contributions; scalar load is telemetry only."""

    layer_order: tuple[int, ...]
    contributions: dict[int, list[LowRankUpdate]] = field(init=False)
    prior_left: dict[int, torch.Tensor] = field(init=False)
    prior_right: dict[int, torch.Tensor] = field(init=False)
    prior_covariance_right: dict[int, torch.Tensor] = field(init=False)
    committed_load: dict[int, float] = field(init=False)
    prior_value: float = field(default=0.0, init=False)
    accepted_update_count: int = field(default=0, init=False)
    batched_cross_call_count: int = field(default=0, init=False)
    prior_pairwise_replay_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if len(self.layer_order) < 2 or len(set(self.layer_order)) != len(self.layer_order):
            raise ODEBFContractError("cumulative-P layer order differs")
        self.contributions = {layer: [] for layer in self.layer_order}
        self.prior_left = {layer: torch.empty((0, 0), dtype=torch.float64) for layer in self.layer_order}
        self.prior_right = {layer: torch.empty((0, 0), dtype=torch.float64) for layer in self.layer_order}
        self.prior_covariance_right = {
            layer: torch.empty((0, 0), dtype=torch.float64) for layer in self.layer_order
        }
        self.committed_load = {layer: 0.0 for layer in self.layer_order}

    def value(self) -> float:
        return self.prior_value

    def checkpoint(self) -> CumulativePCheckpoint:
        return CumulativePCheckpoint(
            tuple((layer, tuple(self.contributions[layer])) for layer in self.layer_order),
            tuple((layer, self.prior_left[layer].clone()) for layer in self.layer_order),
            tuple((layer, self.prior_right[layer].clone()) for layer in self.layer_order),
            tuple((layer, self.prior_covariance_right[layer].clone()) for layer in self.layer_order),
            tuple(sorted(self.committed_load.items())),
            self.prior_value,
            self.accepted_update_count,
            self.batched_cross_call_count,
            self.prior_pairwise_replay_count,
        )

    def restore(self, checkpoint: CumulativePCheckpoint) -> None:
        self.contributions = {
            layer: list(values) for layer, values in checkpoint.contributions
        }
        self.prior_left = {layer: value.clone() for layer, value in checkpoint.prior_left}
        self.prior_right = {layer: value.clone() for layer, value in checkpoint.prior_right}
        self.prior_covariance_right = {
            layer: value.clone() for layer, value in checkpoint.prior_covariance_right
        }
        self.committed_load = dict(checkpoint.committed_load)
        self.prior_value = checkpoint.prior_value
        self.accepted_update_count = checkpoint.accepted_update_count
        self.batched_cross_call_count = checkpoint.batched_cross_call_count
        self.prior_pairwise_replay_count = checkpoint.prior_pairwise_replay_count

    def fork(self) -> "CumulativeStructuralPState":
        result = CumulativeStructuralPState(self.layer_order)
        result.restore(self.checkpoint())
        return result

    def identity(self) -> str:
        return canonical_hash(
            {
                "prior_value": self.prior_value,
                "accepted_update_count": self.accepted_update_count,
                "batched_cross_call_count": self.batched_cross_call_count,
                "prior_pairwise_replay_count": self.prior_pairwise_replay_count,
                "factors": tuple(
                    (
                        layer,
                        tuple(item.identity_sha256 for item in self.contributions[layer]),
                    )
                    for layer in self.layer_order
                ),
                "load": tuple(sorted(self.committed_load.items())),
            }
        )

    def _candidate_cross(self, layer: int, candidate: LowRankUpdate) -> float:
        prior_left = self.prior_left[layer]
        if prior_left.numel() == 0:
            return 0.0
        self.batched_cross_call_count += 1
        value = torch.trace(
            (
                candidate.left.T.to(dtype=torch.float64)
                @ prior_left
            )
            @ (
                self.prior_right[layer].T
                @ candidate.covariance_right.to(dtype=torch.float64)
            )
        )
        return float(value) * candidate.coefficient

    def candidate_risk(
        self,
        candidates: Mapping[int, LowRankUpdate],
        *,
        h: float = FIXED_H,
    ) -> QuadraticRoutingRisk:
        if set(candidates) != set(self.layer_order) or h != FIXED_H:
            raise ODEBFContractError("cumulative-P candidate set or h differs")
        offset = self.prior_value
        linear = np.zeros(len(self.layer_order), dtype=np.float64)
        gram = np.zeros((len(self.layer_order), len(self.layer_order)), dtype=np.float64)
        for index, layer in enumerate(self.layer_order):
            candidate = candidates[layer]
            cross = self._candidate_cross(layer, candidate)
            linear[index] = 2.0 * h * cross
            gram[index, index] = h * h * _pretrained_inner(candidate, candidate)
        return QuadraticRoutingRisk(
            offset,
            linear,
            gram,
            NORMALIZATION_EPSILON,
            "cumulative-structural-p-marginal-routing",
        )

    def finalize(
        self,
        candidates: Mapping[int, LowRankUpdate],
        velocity: Sequence[float],
        *,
        candidate_risk: QuadraticRoutingRisk | None = None,
    ) -> None:
        values = np.asarray(velocity, dtype=np.float64)
        if values.shape != (len(self.layer_order),) or not np.all(np.isfinite(values)):
            raise ODEBFContractError("cumulative-P finalized velocity differs")
        if set(candidates) != set(self.layer_order):
            raise ODEBFContractError("cumulative-P finalized candidates differ")
        if candidate_risk is not None and (
            candidate_risk.offset != self.prior_value
            or candidate_risk.linear.shape != values.shape
            or candidate_risk.gram.shape != (values.size, values.size)
        ):
            raise ODEBFStateError("cumulative-P staged quadratic is stale")
        for index, layer in enumerate(self.layer_order):
            applied = candidates[layer].scaled(FIXED_H * float(values[index]))
            if candidate_risk is None:
                cross = self._candidate_cross(layer, applied)
                self_term = _pretrained_inner(applied, applied)
            else:
                # candidate_risk is expressed for unscaled v.  Convert the
                # staged linear/self coefficients to this already-applied item.
                cross = (
                    0.0
                    if values[index] == 0.0
                    else candidate_risk.linear[index] * values[index] / 2.0
                )
                self_term = candidate_risk.gram[index, index] * values[index] ** 2
            self.prior_value += 2.0 * cross + self_term
            self.contributions[layer].append(applied)
            self.committed_load[layer] += max(self_term, 0.0)
            weighted_left = applied.left.to(dtype=torch.float64) * applied.coefficient
            right = applied.right.to(dtype=torch.float64)
            covariance_right = applied.covariance_right.to(dtype=torch.float64)
            if self.prior_left[layer].numel() == 0:
                self.prior_left[layer] = weighted_left.clone()
                self.prior_right[layer] = right.clone()
                self.prior_covariance_right[layer] = covariance_right.clone()
            else:
                self.prior_left[layer] = torch.cat((self.prior_left[layer], weighted_left), dim=1)
                self.prior_right[layer] = torch.cat((self.prior_right[layer], right), dim=1)
                self.prior_covariance_right[layer] = torch.cat(
                    (self.prior_covariance_right[layer], covariance_right), dim=1
                )
        self.accepted_update_count += 1


@dataclass(frozen=True, slots=True)
class RoundHCheckpoint:
    history_version: int
    history_columns: int
    accumulated_movement: tuple[tuple[int, torch.Tensor], ...]
    accepted_step_count: int


@dataclass(slots=True)
class RoundStructuralHState:
    """Within-round accumulated historical-key response movement."""

    layer_order: tuple[int, ...]
    history_version: int
    history_columns: int
    accumulated_movement: dict[int, torch.Tensor] = field(init=False)
    accepted_step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if (
            self.history_version < 0
            or self.history_columns < 0
            or self.history_columns > MAXIMUM_HISTORY_RECORDS
            or self.history_columns % BATCH_SIZE != 0
        ):
            raise ODEBFContractError("round Structural-H history identity differs")
        self.accumulated_movement = {
            layer: torch.empty((0, self.history_columns), dtype=torch.float64)
            for layer in self.layer_order
        }

    def checkpoint(self) -> RoundHCheckpoint:
        return RoundHCheckpoint(
            self.history_version,
            self.history_columns,
            tuple(
                (layer, self.accumulated_movement[layer].clone())
                for layer in self.layer_order
            ),
            self.accepted_step_count,
        )

    def restore(self, checkpoint: RoundHCheckpoint) -> None:
        self.history_version = checkpoint.history_version
        self.history_columns = checkpoint.history_columns
        self.accumulated_movement = {
            layer: value.clone() for layer, value in checkpoint.accumulated_movement
        }
        self.accepted_step_count = checkpoint.accepted_step_count

    def candidate_risk(
        self,
        factors: Mapping[int, torch.Tensor],
        projected_history: Mapping[int, torch.Tensor],
        *,
        h: float = FIXED_H,
    ) -> QuadraticRoutingRisk:
        if set(factors) != set(self.layer_order) or set(projected_history) != set(
            self.layer_order
        ) or h != FIXED_H:
            raise ODEBFContractError("cumulative Structural-H inputs or h differ")
        offset = 0.0
        linear = np.zeros(len(self.layer_order), dtype=np.float64)
        gram = np.zeros((len(self.layer_order), len(self.layer_order)), dtype=np.float64)
        if self.history_columns == 0:
            return QuadraticRoutingRisk(
                0.0, linear, gram, NORMALIZATION_EPSILON, "round1-empty-structural-h"
            )
        for index, layer in enumerate(self.layer_order):
            factor = _finite_tensor("H candidate factor", factors[layer])
            keys = _finite_tensor("H projected history", projected_history[layer])
            if keys.shape[1] != self.history_columns or factor.shape[1] != keys.shape[0]:
                raise ODEBFContractError("cumulative Structural-H geometry differs")
            movement = factor.to(dtype=torch.float64) @ keys.to(dtype=torch.float64)
            accumulated = self.accumulated_movement[layer]
            if accumulated.shape != movement.shape:
                if self.accepted_step_count == 0 and accumulated.shape[0] == 0:
                    accumulated = torch.zeros_like(movement)
                else:
                    raise ODEBFStateError("round Structural-H accumulated shape differs")
            offset += float(torch.sum(accumulated * accumulated))
            linear[index] = 2.0 * h * float(torch.sum(accumulated * movement))
            gram[index, index] = h * h * float(torch.sum(movement * movement))
        return QuadraticRoutingRisk(
            offset,
            linear,
            gram,
            NORMALIZATION_EPSILON,
            "round-cumulative-structural-h",
        )

    def accept(
        self,
        factors: Mapping[int, torch.Tensor],
        projected_history: Mapping[int, torch.Tensor],
        velocity: Sequence[float],
    ) -> None:
        values = np.asarray(velocity, dtype=np.float64)
        if values.shape != (len(self.layer_order),) or not np.all(np.isfinite(values)):
            raise ODEBFContractError("round Structural-H accepted velocity differs")
        risk = self.candidate_risk(factors, projected_history)
        del risk
        if self.history_columns:
            for index, layer in enumerate(self.layer_order):
                movement = (
                    FIXED_H
                    * float(values[index])
                    * (
                        factors[layer].to(dtype=torch.float64)
                        @ projected_history[layer].to(dtype=torch.float64)
                    )
                )
                if self.accumulated_movement[layer].shape[0] == 0:
                    self.accumulated_movement[layer] = torch.zeros_like(movement)
                self.accumulated_movement[layer] += movement
        self.accepted_step_count += 1


def incremental_structural_h(
    factors: Mapping[int, torch.Tensor],
    projected_history: Mapping[int, torch.Tensor],
    *,
    layer_order: Sequence[int],
    h: float = FIXED_H,
) -> QuadraticRoutingRisk:
    """Compatibility wrapper for the k0 cumulative Structural-H quadratic."""

    columns = {value.shape[1] for value in projected_history.values()}
    if len(columns) != 1:
        raise ODEBFContractError("incremental-H layers have different history counts")
    state = RoundStructuralHState(tuple(layer_order), 0, next(iter(columns)))
    return state.candidate_risk(factors, projected_history, h=h)


@dataclass(frozen=True, slots=True)
class ExactStrengthRoutingResult:
    arm: str
    status: str
    velocity: tuple[float, ...]
    neutral_velocity: tuple[float, ...]
    requested_strength: float
    predicted_strength: float
    exact_strength_residual: float
    neutral_energy: float
    selected_energy: float
    neutral_relative_trust_limit: float
    trust_bound_active: bool
    h_score: float
    p_score: float
    worst_hp_score: float
    h_raw: float
    p_raw_cumulative: float
    p_marginal_delta: float
    p_neutral_marginal_delta: float
    h_neutral_normalization: float
    p_neutral_normalization: float
    capacity: float
    active_layer_count: int
    routing_dof: int
    allocation_influence: bool
    fallback_to_neutral: bool
    fallback_count: int
    retry_count: int
    backtracking_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def _quadratic_energy(value: np.ndarray, metric: np.ndarray) -> float:
    return float(value @ metric @ value)


def solve_exact_strength_soft_hp(
    signed_slopes: Sequence[float],
    requested_strength: float,
    trust_metric: np.ndarray,
    capacity_metric: np.ndarray,
    historical_risk: QuadraticRoutingRisk,
    pretrained_risk: QuadraticRoutingRisk,
    *,
    arm: str,
    inject_numerical_failure: bool = False,
) -> ExactStrengthRoutingResult:
    """Strength first, normalized H/P minimax second, capacity third."""

    slopes = np.asarray(signed_slopes, dtype=np.float64)
    trust = np.asarray(trust_metric, dtype=np.float64)
    capacity = np.asarray(capacity_metric, dtype=np.float64)
    rho = float(requested_strength)
    selected_arm = str(arm).upper()
    if selected_arm not in ("NEUTRAL", "SOFT"):
        raise ODEBFContractError("SoftHP arm differs")
    if (
        slopes.ndim != 1
        or trust.shape != (slopes.size, slopes.size)
        or capacity.shape != trust.shape
        or not all(np.all(np.isfinite(item)) for item in (slopes, trust, capacity))
        or not math.isfinite(rho)
        or rho < 0.0
    ):
        raise ODEBFContractError("SoftHP routing geometry differs")
    active = np.flatnonzero(slopes > 0.0)
    if rho == 0.0:
        zero = np.zeros_like(slopes)
        return ExactStrengthRoutingResult(
            selected_arm, "SEMANTIC_NO_WRITE", tuple(zero), tuple(zero), rho, 0.0, 0.0,
            0.0, 0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            NORMALIZATION_EPSILON, NORMALIZATION_EPSILON, 0.0,
            int(active.size), max(int(active.size)-1, 0),
            False, False, 0, 0, 0,
        )
    if active.size == 0:
        raise ODEBFContractError("positive semantic demand has no positive layer direction")

    q = float(slopes[active].sum())
    neutral = np.zeros_like(slopes)
    neutral[active] = rho / q
    neutral_energy = _quadratic_energy(neutral, trust)
    energy_limit = (
        neutral_energy * (1.0 + SIMPLEX_ENERGY_RELATIVE_TOLERANCE)
        + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
    )
    dof = int(active.size) - 1

    h_neutral_normalization = max(
        abs(historical_risk.raw_value(neutral)), NORMALIZATION_EPSILON
    )
    p_neutral_marginal = pretrained_risk.marginal_value(neutral)
    p_neutral_normalization = max(abs(p_neutral_marginal), NORMALIZATION_EPSILON)

    def score(risk: QuadraticRoutingRisk, value: np.ndarray) -> float:
        if risk is historical_risk:
            return risk.raw_value(value) / h_neutral_normalization
        return risk.marginal_value(value) / p_neutral_normalization

    def receipt(value: np.ndarray, status: str, fallback: bool) -> ExactStrengthRoutingResult:
        predicted = float(slopes @ value)
        h_score = score(historical_risk, value)
        p_score = score(pretrained_risk, value)
        selected_energy = _quadratic_energy(value, trust)
        h_raw = historical_risk.raw_value(value)
        p_raw = pretrained_risk.raw_value(value)
        p_marginal = pretrained_risk.marginal_value(value)
        return ExactStrengthRoutingResult(
            selected_arm,
            status,
            tuple(float(item) for item in value),
            tuple(float(item) for item in neutral),
            rho,
            predicted,
            abs(predicted-rho),
            neutral_energy,
            selected_energy,
            energy_limit,
            bool(
                selected_energy
                >= energy_limit
                - max(SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE, SIMPLEX_PRIMAL_TOLERANCE)
            ),
            h_score,
            p_score,
            max(h_score, p_score),
            h_raw,
            p_raw,
            p_marginal,
            p_neutral_marginal,
            h_neutral_normalization,
            p_neutral_normalization,
            0.5 * _quadratic_energy(value, capacity),
            int(active.size),
            dof,
            bool(np.max(np.abs(value-neutral), initial=0.0) > SIMPLEX_PRIMAL_TOLERANCE),
            fallback,
            int(fallback),
            0,
            0,
        )

    if selected_arm == "NEUTRAL" or dof == 0:
        return receipt(neutral, "NO_ROUTING_DOF" if dof == 0 else "NEUTRAL_CONTROL", False)

    transform = rho / slopes[active]
    neutral_pi = slopes[active] / q

    def expand(pi: np.ndarray) -> np.ndarray:
        value = np.zeros_like(slopes)
        value[active] = transform * pi
        return value

    def energy(pi: np.ndarray) -> float:
        return _quadratic_energy(expand(pi), trust)

    def risks(pi: np.ndarray) -> tuple[float, float]:
        value = expand(pi)
        return score(historical_risk, value), score(pretrained_risk, value)

    initial_xi = max(risks(neutral_pi))
    initial = np.concatenate((neutral_pi, [initial_xi]))
    constraints = [
        {"type": "eq", "fun": lambda value: float(value[:-1].sum()-1.0)},
        {"type": "ineq", "fun": lambda value: float(energy_limit-energy(value[:-1]))},
        {"type": "ineq", "fun": lambda value: float(value[-1]-risks(value[:-1])[0])},
        {"type": "ineq", "fun": lambda value: float(value[-1]-risks(value[:-1])[1])},
    ]
    stage1 = minimize(
        lambda value: float(value[-1]),
        initial,
        method="SLSQP",
        bounds=tuple((0.0, None) for _ in active)+((None, None),),
        constraints=tuple(constraints),
        options={"ftol": SOFT_HP_SOLVER_FTOL, "maxiter": SOFT_HP_SOLVER_MAXITER, "disp": False},
    )
    stage1_value = np.asarray(stage1.x, dtype=np.float64)
    stage1_valid = bool(
        stage1.success
        and np.all(np.isfinite(stage1_value))
        and abs(float(stage1_value[:-1].sum())-1.0) <= SIMPLEX_PRIMAL_TOLERANCE
        and float(np.min(stage1_value[:-1])) >= -SIMPLEX_PRIMAL_TOLERANCE
        and energy(stage1_value[:-1]) <= energy_limit + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
    )
    if inject_numerical_failure or not stage1_valid:
        return receipt(neutral, "SOFT_NUMERICAL_NEUTRAL_FALLBACK", True)

    xi_tie = float(stage1_value[-1]) + SIMPLEX_XI_TIE_TOLERANCE
    constraints2 = [
        {"type": "eq", "fun": lambda value: float(value.sum()-1.0)},
        {"type": "ineq", "fun": lambda value: float(energy_limit-energy(value))},
        {"type": "ineq", "fun": lambda value: float(xi_tie-risks(value)[0])},
        {"type": "ineq", "fun": lambda value: float(xi_tie-risks(value)[1])},
    ]

    def capacity_pi(pi: np.ndarray) -> float:
        return 0.5 * _quadratic_energy(expand(pi), capacity)

    stage2 = minimize(
        capacity_pi,
        stage1_value[:-1],
        method="SLSQP",
        bounds=tuple((0.0, None) for _ in active),
        constraints=tuple(constraints2),
        options={"ftol": SOFT_HP_SOLVER_FTOL, "maxiter": SOFT_HP_SOLVER_MAXITER, "disp": False},
    )
    pi = np.asarray(stage2.x, dtype=np.float64)
    stage2_valid = bool(
        stage2.success
        and np.all(np.isfinite(pi))
        and abs(float(pi.sum())-1.0) <= SIMPLEX_PRIMAL_TOLERANCE
        and float(np.min(pi)) >= -SIMPLEX_PRIMAL_TOLERANCE
        and energy(pi) <= energy_limit + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
        and max(risks(pi)) <= xi_tie + SIMPLEX_PRIMAL_TOLERANCE
    )
    if not stage2_valid:
        return receipt(neutral, "SOFT_NUMERICAL_NEUTRAL_FALLBACK", True)
    result = receipt(expand(pi), "SOFT_HP_JOINT_WRITE", False)
    if result.exact_strength_residual > SIMPLEX_PRIMAL_TOLERANCE:
        raise ODEBFContractError("SoftHP lowered semantic strength")
    return result


@dataclass(frozen=True, slots=True)
class TerminalKeyReuseReceipt:
    reused_terminal_keys: bool
    explicit_recapture_fallback: bool
    recapture_count: int
    committed_identity_sha256: str
    captured_identity_sha256: str
    identity_components_matched: bool
    keys_sha256: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class StagedAcceptedStepReceipt:
    step: int
    p_prior: float
    p_marginal_delta: float
    p_after: float
    h_prior: float
    h_cross: float
    h_self: float
    h_after: float
    p_identity_sha256: str


@dataclass(slots=True)
class SequentialRoundStaging:
    """Uncommitted K8 P/H state; it cannot mutate the persistent arm."""

    round_index: int
    history_version: int
    active_record_sha256: str
    projected_history_by_layer: dict[int, torch.Tensor]
    cumulative_p: CumulativeStructuralPState
    round_h: RoundStructuralHState
    receipts: list[StagedAcceptedStepReceipt] = field(default_factory=list)

    def stage_accepted_step(
        self,
        p_candidates: Mapping[int, LowRankUpdate],
        h_factors: Mapping[int, torch.Tensor],
        velocity: Sequence[float],
    ) -> StagedAcceptedStepReceipt:
        if len(self.receipts) >= FIXED_K:
            raise ODEBFStateError("round staging exceeds fixed K8")
        values = np.asarray(velocity, dtype=np.float64)
        p_risk = self.cumulative_p.candidate_risk(p_candidates)
        h_risk = self.round_h.candidate_risk(
            h_factors, self.projected_history_by_layer
        )
        p_prior = self.cumulative_p.value()
        p_delta = p_risk.marginal_value(values)
        h_prior = h_risk.offset
        h_cross = float(h_risk.linear @ values)
        h_self = float(values @ h_risk.gram @ values)
        self.cumulative_p.finalize(
            p_candidates, values, candidate_risk=p_risk
        )
        self.round_h.accept(h_factors, self.projected_history_by_layer, values)
        receipt = StagedAcceptedStepReceipt(
            len(self.receipts) + 1,
            p_prior,
            p_delta,
            self.cumulative_p.value(),
            h_prior,
            h_cross,
            h_self,
            h_prior + h_cross + h_self,
            self.cumulative_p.identity(),
        )
        self.receipts.append(receipt)
        return receipt


@dataclass(frozen=True, slots=True)
class BackendRoundCommitReceipt:
    transaction_id: str
    before_state_sha256: str
    after_state_sha256: str
    appended_count: int
    completed_round: int
    history_version: int
    cumulative_p: float
    terminal_round_h: float
    cache_invalidated: bool
    idempotent_replay: bool
    verified_scope: str
    evaluator_metric_gate_count: int
    identity_sha256: str


@dataclass(slots=True)
class SequentialArmState:
    """Independent arm state; no object is shared across Neutral/Soft arms."""

    arm_id: str
    layer_order: tuple[int, ...]
    maximum_history_records: int = MAXIMUM_HISTORY_RECORDS
    history_batch_size: int = BATCH_SIZE
    ledger: P1HistoryLedger = field(init=False)
    cumulative_p: CumulativeStructuralPState = field(init=False)
    persistent_weights: dict[int, torch.Tensor] = field(init=False)
    factorization_cache: dict[str, HistoryFactorization] = field(init=False)
    completed_rounds: int = field(default=0, init=False)
    terminal_key_recapture_count: int = field(default=0, init=False)
    completed_transactions: dict[str, str] = field(init=False)
    transaction_receipts: dict[str, BackendRoundCommitReceipt] = field(init=False)
    round_h_state: RoundStructuralHState = field(init=False)

    def __post_init__(self) -> None:
        self.ledger = P1HistoryLedger(
            layer_order=self.layer_order,
            maximum_records=self.maximum_history_records,
            batch_size=self.history_batch_size,
        )
        self.cumulative_p = CumulativeStructuralPState(self.layer_order)
        self.persistent_weights = {}
        self.factorization_cache = {}
        self.completed_transactions = {}
        self.transaction_receipts = {}
        self.round_h_state = RoundStructuralHState(self.layer_order, 0, 0)

    def set_initial_weights(self, weights: Mapping[int, torch.Tensor]) -> None:
        if self.persistent_weights or set(weights) != set(self.layer_order):
            raise ODEBFStateError("persistent BF16 W initialization differs")
        for layer in self.layer_order:
            value = weights[layer]
            if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ODEBFContractError("persistent sequential W must be finite BF16")
            self.persistent_weights[layer] = value.detach().clone()

    def history_entry_count(self) -> int:
        return len(self.ledger.snapshot().active_records)

    def active_record_sha256(self) -> str:
        records = self.ledger.snapshot().active_records
        return canonical_hash(
            tuple(
                (
                    item.request_sha256,
                    item.collision_sha256,
                    item.target_sha256,
                    item.version,
                )
                for item in records
            )
        )

    def state_identity(self) -> str:
        return canonical_hash(
            {
                "arm_id": self.arm_id,
                "weights": tuple(
                    (layer, tensor_sha256(self.persistent_weights[layer]))
                    for layer in self.layer_order
                ),
                "history": self.ledger.snapshot().digest,
                "p": self.cumulative_p.identity(),
                "round_h": tuple(
                    (
                        layer,
                        tensor_sha256(self.round_h_state.accumulated_movement[layer]),
                    )
                    for layer in self.layer_order
                ),
                "cache": tuple(sorted(self.factorization_cache)),
                "completed_rounds": self.completed_rounds,
                "transactions": tuple(sorted(self.completed_transactions.items())),
            }
        )

    def begin_round_staging(
        self,
        projected_history_by_layer: Mapping[int, torch.Tensor],
    ) -> SequentialRoundStaging:
        if set(projected_history_by_layer) != set(self.layer_order):
            raise ODEBFContractError("round staging projected history layer set differs")
        count = self.history_entry_count()
        if any(value.shape[1] != count for value in projected_history_by_layer.values()):
            raise ODEBFContractError("round staging history columns differ from active ledger")
        return SequentialRoundStaging(
            self.completed_rounds + 1,
            self.ledger.version,
            self.active_record_sha256(),
            {
                layer: _finite_tensor(
                    "round staging projected history", projected_history_by_layer[layer]
                ).clone()
                for layer in self.layer_order
            },
            self.cumulative_p.fork(),
            RoundStructuralHState(self.layer_order, self.ledger.version, count),
        )

    def solve_historical_alpha(
        self,
        layer: int,
        projector: torch.Tensor,
        current_raw_keys: torch.Tensor,
        *,
        regularization: float,
        projector_certificate: ProjectorCertificate,
    ) -> tuple[torch.Tensor, CachedWoodburyReceipt]:
        """Resolve history columns from the active ledger and cache by full identity."""

        if layer not in self.layer_order:
            raise ODEBFContractError("historical Alpha layer is absent")
        snapshot = self.ledger.snapshot()
        raw = self.ledger.solve_keys(layer)
        projected = self.ledger.risk_keys(layer)
        if not snapshot.active_records and raw.numel() == 0 and projected.numel() == 0:
            raw = torch.empty(
                (current_raw_keys.shape[0], 0),
                dtype=current_raw_keys.dtype,
                device=current_raw_keys.device,
            )
            projected = raw.clone()
        else:
            raw = raw.to(device=current_raw_keys.device, dtype=current_raw_keys.dtype)
            projected = projected.to(
                device=current_raw_keys.device, dtype=current_raw_keys.dtype
            )
        active_digest = self.active_record_sha256()
        if raw.shape[1] != len(snapshot.active_records) or projected.shape[1] != len(
            snapshot.active_records
        ):
            raise ODEBFStateError("ledger key columns differ from active records")
        lookup_key = canonical_hash(
            {
                "version": snapshot.version,
                "active_records": active_digest,
                "raw": tensor_sha256(raw.to(dtype=torch.float64)),
                "projected": tensor_sha256(projected.to(dtype=torch.float64)),
                "projector": tensor_sha256(projector),
            }
        )
        q, factor, receipt = solve_alpha_woodbury_cached(
            projector,
            current_raw_keys,
            raw,
            projected,
            history_version=snapshot.version,
            active_record_sha256=active_digest,
            regularization=regularization,
            projector_certificate=projector_certificate,
            cached_factorization=self.factorization_cache.get(lookup_key),
        )
        if receipt.cache_key_sha256 != lookup_key:
            raise ODEBFStateError("historical Alpha cache key construction differs")
        self.factorization_cache[lookup_key] = factor
        return q, receipt

    def stage_terminal_keys(
        self,
        terminal_keys_by_layer: Mapping[int, torch.Tensor],
        projectors_by_layer: Mapping[int, torch.Tensor],
        *,
        captured_identity: TerminalPhysicalKeyIdentity,
        committed_identity: TerminalPhysicalKeyIdentity,
        recapture: Callable[[], Mapping[int, torch.Tensor]] | None = None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], TerminalKeyReuseReceipt]:
        if set(terminal_keys_by_layer) != set(self.layer_order) or set(projectors_by_layer) != set(self.layer_order):
            raise ODEBFContractError("terminal key layer set differs")
        same_state = captured_identity == committed_identity
        keys = terminal_keys_by_layer
        recaptured = False
        if not same_state:
            if recapture is None:
                raise ODEBFStateError("terminal key identity differs and no recapture is available")
            keys = recapture()
            if set(keys) != set(self.layer_order):
                raise ODEBFContractError("recaptured terminal key layer set differs")
            recaptured = True
            self.terminal_key_recapture_count += 1
        raw: dict[int, torch.Tensor] = {}
        projected: dict[int, torch.Tensor] = {}
        for layer in self.layer_order:
            key = _finite_tensor("terminal key", keys[layer])
            if key.shape[1] != BATCH_SIZE:
                raise ODEBFContractError("terminal key is not B10")
            projector = _finite_tensor("terminal projector", projectors_by_layer[layer])
            raw[layer] = key.detach().to(device="cpu", dtype=torch.float32).clone()
            projected[layer] = (projector @ key.to(dtype=projector.dtype)).detach().to(
                device="cpu", dtype=torch.float32
            )
        receipt = TerminalKeyReuseReceipt(
            not recaptured,
            recaptured,
            self.terminal_key_recapture_count,
            committed_identity.digest,
            captured_identity.digest,
            same_state,
            tuple((layer, tensor_sha256(raw[layer])) for layer in self.layer_order),
        )
        return raw, projected, receipt

    def stage_adapter_terminal_frame(
        self,
        frame: AtomicSequentialTerminalFrame,
        projectors_by_layer: Mapping[int, torch.Tensor],
        *,
        committed_identity: TerminalPhysicalKeyIdentity,
        recapture: Callable[[], Mapping[int, torch.Tensor]] | None = None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], TerminalKeyReuseReceipt]:
        if set(frame.terminal_keys_by_layer) != set(self.layer_order):
            raise ODEBFContractError("Atomic terminal frame layer set differs")
        return self.stage_terminal_keys(
            frame.terminal_keys_by_layer,
            projectors_by_layer,
            captured_identity=frame.physical_identity,
            committed_identity=committed_identity,
            recapture=recapture,
        )

    def commit_verified_round(
        self,
        transaction_id: str,
        prospective: ProspectiveP1HistoryBatch,
        staging: SequentialRoundStaging,
        committed_weights: Mapping[int, torch.Tensor],
        *,
        post_commit_verified: bool,
        load_increment_by_layer: Mapping[int, float],
        fault_phase: str | None = None,
    ) -> BackendRoundCommitReceipt | None:
        """Atomically commit W/history/P/H/cache/version after physical verification."""

        if not post_commit_verified:
            return None
        if fault_phase not in (
            None,
            "after_weights",
            "after_history",
            "after_p",
            "after_h",
            "after_cache",
            "after_round",
        ):
            raise ODEBFContractError("backend transaction fault phase differs")
        if (
            transaction_id != prospective.transaction_id
            or len(staging.receipts) != FIXED_K
            or set(committed_weights) != set(self.layer_order)
        ):
            raise ODEBFStateError("backend transaction scope or K8 staging differs")
        for layer in self.layer_order:
            value = committed_weights[layer]
            if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ODEBFContractError("backend committed W must be finite BF16")
        payload_sha256 = canonical_hash(
            {
                "transaction_id": transaction_id,
                "prospective": prospective.payload_sha256,
                "round": staging.round_index,
                "weights": tuple(
                    (layer, tensor_sha256(committed_weights[layer]))
                    for layer in self.layer_order
                ),
                "p": staging.cumulative_p.identity(),
                "h": tuple(
                    (
                        layer,
                        tensor_sha256(staging.round_h.accumulated_movement[layer]),
                    )
                    for layer in self.layer_order
                ),
            }
        )
        prior_payload = self.completed_transactions.get(transaction_id)
        if prior_payload is not None:
            if prior_payload != payload_sha256:
                raise ODEBFStateError("backend transaction identity was reused")
            prior = self.transaction_receipts[transaction_id]
            current = self.state_identity()
            payload = {
                **asdict(prior),
                "before_state_sha256": current,
                "after_state_sha256": current,
                "appended_count": 0,
                "idempotent_replay": True,
            }
            payload.pop("identity_sha256")
            payload["identity_sha256"] = canonical_hash(payload)
            return BackendRoundCommitReceipt(**payload)
        if (
            staging.round_index != self.completed_rounds + 1
            or staging.history_version != self.ledger.version
            or staging.active_record_sha256 != self.active_record_sha256()
        ):
            raise ODEBFStateError("backend transaction history/round identity differs")

        before_identity = self.state_identity()
        weights_before = {layer: value.clone() for layer, value in self.persistent_weights.items()}
        ledger_before = self.ledger.transaction_checkpoint()
        p_before = self.cumulative_p.checkpoint()
        h_before = self.round_h_state.checkpoint()
        cache_before = dict(self.factorization_cache)
        rounds_before = self.completed_rounds
        transactions_before = dict(self.completed_transactions)
        receipts_before = dict(self.transaction_receipts)
        try:
            self.persistent_weights = {
                layer: committed_weights[layer].detach().clone()
                for layer in self.layer_order
            }
            if fault_phase == "after_weights":
                raise RuntimeError("injected backend transaction fault after weights")
            history_receipt = self.ledger.finalize(
                prospective,
                post_commit_verified=True,
                load_increment_by_layer=load_increment_by_layer,
            )
            if history_receipt.appended_count != BATCH_SIZE:
                raise ODEBFStateError("backend transaction did not append one B10")
            if fault_phase == "after_history":
                raise RuntimeError("injected backend transaction fault after history")
            self.cumulative_p.restore(staging.cumulative_p.checkpoint())
            if fault_phase == "after_p":
                raise RuntimeError("injected backend transaction fault after P")
            terminal_h = sum(
                float(torch.sum(value * value))
                for value in staging.round_h.accumulated_movement.values()
            )
            self.round_h_state = RoundStructuralHState(
                self.layer_order,
                history_receipt.after_version,
                len(self.ledger.snapshot().active_records),
            )
            if fault_phase == "after_h":
                raise RuntimeError("injected backend transaction fault after H")
            self.factorization_cache.clear()
            if fault_phase == "after_cache":
                raise RuntimeError("injected backend transaction fault after cache")
            self.completed_rounds = staging.round_index
            self.completed_transactions[transaction_id] = payload_sha256
            if fault_phase == "after_round":
                raise RuntimeError("injected backend transaction fault after round")
            after_identity = self.state_identity()
            base_payload = {
                "transaction_id": transaction_id,
                "before_state_sha256": before_identity,
                "after_state_sha256": after_identity,
                "appended_count": BATCH_SIZE,
                "completed_round": self.completed_rounds,
                "history_version": self.ledger.version,
                "cumulative_p": self.cumulative_p.value(),
                "terminal_round_h": terminal_h,
                "cache_invalidated": True,
                "idempotent_replay": False,
                "verified_scope": "NUMERICAL_PROVENANCE_PHYSICAL_ENDPOINT_ONLY",
                "evaluator_metric_gate_count": 0,
            }
            receipt = BackendRoundCommitReceipt(
                **base_payload,
                identity_sha256=canonical_hash(base_payload),
            )
            self.transaction_receipts[transaction_id] = receipt
            return receipt
        except Exception:
            self.persistent_weights = weights_before
            self.ledger.restore_transaction_checkpoint(ledger_before)
            self.cumulative_p.restore(p_before)
            self.round_h_state = RoundStructuralHState(
                self.layer_order, h_before.history_version, h_before.history_columns
            )
            self.round_h_state.restore(h_before)
            self.factorization_cache = cache_before
            self.completed_rounds = rounds_before
            self.completed_transactions = transactions_before
            self.transaction_receipts = receipts_before
            if self.state_identity() != before_identity:
                raise ODEBFStateError("backend transaction rollback was not exact")
            raise


@dataclass(frozen=True, slots=True)
class RoundPlan:
    round_index: int
    history_count_at_entry: int
    debt_reset_value: float
    inner_k: int
    inner_evaluator_access_count: int
    post_commit_current_evaluations: int
    post_commit_historical_evaluations: int


@dataclass(frozen=True, slots=True)
class AcceptedStepTelemetrySchema:
    """Raw-free schema required from a later frozen Atomic integration."""

    required_fields: tuple[str, ...] = (
        "rho_nominal",
        "semantic_debt",
        "rho_commanded",
        "predicted_w_only_progress",
        "actual_w_only_progress",
        "exact_strength_residual",
        "active_layer_count",
        "routing_dof",
        "neutral_allocation",
        "soft_allocation",
        "historical_normalized_score",
        "pretrained_normalized_score",
        "historical_raw_offset_cross_self",
        "pretrained_raw_cumulative_p",
        "pretrained_marginal_delta_p",
        "pretrained_neutral_marginal_delta_p",
        "allocation_influence",
        "neutral_fallback",
        "trust_bound_active",
        "history_count",
        "history_version",
        "woodbury_small_dimension",
        "history_cache_hit",
        "woodbury_cache_exact_fallback",
        "field_identity_sha256",
        "key_identity_sha256",
        "target_refresh_identity_sha256",
    )
    model_forward_count_added_by_telemetry: int = 0
    backward_count_added_by_telemetry: int = 0


@dataclass(slots=True)
class SequentialOuterRuntimeSkeleton:
    """Model-free B10x10 clock/transaction skeleton for later Stage B wiring."""

    arm_state: SequentialArmState
    active_round: int | None = field(default=None, init=False)
    accepted_steps: int = field(default=0, init=False)
    debt: float = field(default=0.0, init=False)
    inner_evaluator_access_count: int = field(default=0, init=False)
    cumulative_evaluation_count: int = field(default=0, init=False)

    def begin_round(self, round_index: int) -> RoundPlan:
        if self.active_round is not None or round_index != self.arm_state.completed_rounds + 1:
            raise ODEBFStateError("sequential outer round order differs")
        plan = build_b10x10_outer_plan()[round_index - 1]
        if self.arm_state.history_entry_count() != plan.history_count_at_entry:
            raise ODEBFStateError("current B10 entered its own history or history count drifted")
        self.active_round = round_index
        self.accepted_steps = 0
        self.debt = 0.0
        return plan

    def record_accepted_step(self) -> None:
        if self.active_round is None or self.accepted_steps >= FIXED_K:
            raise ODEBFStateError("sequential K8 accepted-step clock differs")
        self.accepted_steps += 1

    def record_inner_evaluator_access(self) -> None:
        self.inner_evaluator_access_count += 1
        raise ODEBFContractError("inner heldout evaluator is inaccessible")

    def close_round_after_history_finalize(
        self,
        receipt: BackendRoundCommitReceipt,
    ) -> int:
        if self.active_round is None or self.accepted_steps != FIXED_K:
            raise ODEBFStateError("sequential round lacks a complete K8")
        if receipt.appended_count != BATCH_SIZE or receipt.history_version != self.active_round:
            raise ODEBFStateError("sequential round did not append exactly one B10")
        evaluations = self.active_round
        self.cumulative_evaluation_count += evaluations
        self.active_round = None
        self.accepted_steps = 0
        return evaluations


def build_b10x10_outer_plan() -> tuple[RoundPlan, ...]:
    return tuple(
        RoundPlan(
            round_index=round_index,
            history_count_at_entry=(round_index-1)*BATCH_SIZE,
            debt_reset_value=0.0,
            inner_k=FIXED_K,
            inner_evaluator_access_count=0,
            post_commit_current_evaluations=1,
            post_commit_historical_evaluations=round_index-1,
        )
        for round_index in range(1, SEQUENTIAL_ROUNDS+1)
    )


def round1_empty_history_equivalence_payload() -> dict[str, Any]:
    payload = {
        "history_count": 0,
        "history_version": 0,
        "historical_solve_extra_columns": 0,
        "structural_h_value": 0.0,
        "historical_router_influence": 0,
        "wrapper_only_differences": (
            "persistent-arm-state",
            "post-commit-history-transaction",
            "outer-evaluation-ledger",
        ),
        "atomic_winner_checkpoint": None,
        "atomic_endpoint_equivalence_gate_status": "WAITING_FROZEN_HANDOFF",
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def assert_round1_empty_history_equivalence(
    atomic_payload: Mapping[str, Any],
    sequential_payload: Mapping[str, Any],
    *,
    wrapper_only_keys: Sequence[str],
) -> str:
    """Require exact scientific payload identity after removing wrapper metadata."""

    excluded = set(wrapper_only_keys)
    atomic = {key: value for key, value in atomic_payload.items() if key not in excluded}
    sequential = {key: value for key, value in sequential_payload.items() if key not in excluded}
    if atomic != sequential:
        raise ODEBFContractError("round1 empty-history endpoint differs from frozen Atomic")
    return canonical_hash(atomic)


def stage_a_dry_plan() -> dict[str, Any]:
    rounds = build_b10x10_outer_plan()
    payload = {
        "instruction_id": P1R29_BACKEND_HARDENING_INSTRUCTION_ID,
        "scaffold_instruction_id": P1R29_SEQUENTIAL_PREPARATION_INSTRUCTION_ID,
        "method_id": P1R29_BACKEND_METHOD_ID,
        "status": "BACKEND_HARDENED_WAITING_VIABLE_ATOMIC_ADAPTER",
        "sequential_backend_implementation": "CONTINUE_COMPLETE_MODEL_FREE",
        "scientific_sequential_execution": "HOLD_UNTIL_VIABLE_ATOMIC_ADAPTER",
        "stage": "BACKEND_CPU_SOURCE_TEST_REPORT_ONLY",
        "models": 0,
        "gpu_allocations": 0,
        "slurm_jobs": 0,
        "result_roots": 0,
        "atomic_winner_imports": 0,
        "failed_p1r29_atomic_imports": 0,
        "actual_b10x2_model_smoke_receipt": "NOT_RECORDED_NO_ADAPTER_SELECTED",
        "adapter_neutral_dry_integration_contract": {
            "current_w": True,
            "current_z": True,
            "layer_proposals": True,
            "physical_signed_slopes": True,
            "reference_allocation_c0": True,
            "reference_mean_progress": True,
            "reference_debt_priority_progress": True,
            "reference_energy": True,
            "selected_coefficients": True,
            "constraint_and_certificate_receipts": True,
            "universal_semantic_rho_equality": False,
            "universal_rho_over_slope_transform": False,
            "current_raw_keys": True,
            "terminal_physical_state_keys": True,
            "backend_target_or_debt_preprocessing": False,
            "shared_preprocessing_required_from_selected_adapter": True,
        },
        "adapter_amendment_id": P1R30_BACKEND_ADAPTER_AMENDMENT_ID,
        "server1_janghj_gpu_cap": 4,
        "server2_janghj_gpu_cap_independent": 4,
        "future_stage_throttle_max": 2,
        "rounds": SEQUENTIAL_ROUNDS,
        "inner_k": FIXED_K,
        "history_counts_at_entry": [item.history_count_at_entry for item in rounds],
        "cumulative_b10_evaluations": sum(
            item.post_commit_current_evaluations + item.post_commit_historical_evaluations
            for item in rounds
        ),
        "inner_evaluator_access_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "history_append_before_verified_commit": 0,
        "debt_cross_batch_carry_count": 0,
        "accepted_step_telemetry_schema": asdict(AcceptedStepTelemetrySchema()),
        "p1r28_action_count": 0,
        "sh2_live_p1r29_action_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload
