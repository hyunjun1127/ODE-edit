"""CPU-only preparation primitives for P1R29 sequential Historical editing.

This module deliberately contains no model, evaluator, GPU, Slurm, or result-
root entry point.  It prepares the state, exact historical Woodbury, incremental
Structural-H, cumulative Structural-P, and exact-strength SoftHP contracts that
may be connected only after a frozen P1R29 Atomic handoff is authorized.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import minimize

from .contracts import BATCH_SIZE, FIXED_K, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1_state import (
    P1HistoryFinalizeReceipt,
    P1HistoryLedger,
    P1HistoryRecord,
    ProspectiveP1HistoryBatch,
)
from .progress_simplex_routing import (
    SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
    SIMPLEX_ENERGY_RELATIVE_TOLERANCE,
    SIMPLEX_PRIMAL_TOLERANCE,
    SIMPLEX_XI_TIE_TOLERANCE,
)
from .woodbury import ProjectorCertificate, WoodburyMethod, solve_alpha_woodbury


P1R29_SEQUENTIAL_PREPARATION_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R29-SEQUENTIAL-HISTORICAL-PREPARATION-V1"
)
P1R29_SEQUENTIAL_PREPARATION_METHOD_ID = (
    "P1R29-SEQUENTIAL-HISTORICAL-PREPARATION-NO-ATOMIC-WINNER-V1"
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
class HistoryFactorization:
    """History-version factorization of ``lambda I + K_H^T Z_H``."""

    history_version: int
    history_columns: int
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
    if history_version < 0 or raw.shape[1] != history_version * BATCH_SIZE:
        raise ODEBFContractError("history columns are not 0,10,...,90 for this version")
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
    expected_identity = (
        history_version,
        raw.shape[1],
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
            regularization=regularization,
        )
    observed_identity = (
        factor.history_version,
        factor.history_columns,
        factor.regularization,
        factor.raw_history_sha256,
        factor.projected_history_sha256,
        factor.projector_sha256,
    )
    if observed_identity != expected_identity:
        raise ODEBFStateError("historical factorization cache identity is stale")

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

    def normalized_value(self, velocity: np.ndarray) -> float:
        value = np.asarray(velocity, dtype=np.float64)
        return float(self.offset + self.linear @ value + value @ self.gram @ value) / self.normalization


@dataclass(slots=True)
class CumulativeStructuralPState:
    """Layer-wise accepted low-rank contributions; scalar load is telemetry only."""

    layer_order: tuple[int, ...]
    contributions: dict[int, list[LowRankUpdate]] = field(init=False)
    committed_load: dict[int, float] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.layer_order) < 2 or len(set(self.layer_order)) != len(self.layer_order):
            raise ODEBFContractError("cumulative-P layer order differs")
        self.contributions = {layer: [] for layer in self.layer_order}
        self.committed_load = {layer: 0.0 for layer in self.layer_order}

    def value(self) -> float:
        total = 0.0
        for layer in self.layer_order:
            for left in self.contributions[layer]:
                for right in self.contributions[layer]:
                    total += _pretrained_inner(left, right)
        return total

    def candidate_risk(
        self,
        candidates: Mapping[int, LowRankUpdate],
        *,
        h: float = FIXED_H,
    ) -> QuadraticRoutingRisk:
        if set(candidates) != set(self.layer_order) or h != FIXED_H:
            raise ODEBFContractError("cumulative-P candidate set or h differs")
        offset = self.value()
        linear = np.zeros(len(self.layer_order), dtype=np.float64)
        gram = np.zeros((len(self.layer_order), len(self.layer_order)), dtype=np.float64)
        for index, layer in enumerate(self.layer_order):
            candidate = candidates[layer]
            cross = sum(
                _pretrained_inner(prior, candidate)
                for prior in self.contributions[layer]
            )
            linear[index] = 2.0 * h * cross
            gram[index, index] = h * h * _pretrained_inner(candidate, candidate)
        scale = max(offset, float(np.trace(gram)), NORMALIZATION_EPSILON)
        return QuadraticRoutingRisk(offset, linear, gram, scale, "cumulative-structural-p")

    def finalize(self, candidates: Mapping[int, LowRankUpdate], velocity: Sequence[float]) -> None:
        values = np.asarray(velocity, dtype=np.float64)
        if values.shape != (len(self.layer_order),) or not np.all(np.isfinite(values)):
            raise ODEBFContractError("cumulative-P finalized velocity differs")
        if set(candidates) != set(self.layer_order):
            raise ODEBFContractError("cumulative-P finalized candidates differ")
        for index, layer in enumerate(self.layer_order):
            applied = candidates[layer].scaled(FIXED_H * float(values[index]))
            self.contributions[layer].append(applied)
            self.committed_load[layer] += max(_pretrained_inner(applied, applied), 0.0)


def incremental_structural_h(
    factors: Mapping[int, torch.Tensor],
    projected_history: Mapping[int, torch.Tensor],
    *,
    layer_order: Sequence[int],
    h: float = FIXED_H,
) -> QuadraticRoutingRisk:
    """Return ``sum_l ||h v_l B_l Z_H,l||_F^2`` with no baseline term."""

    layers = tuple(layer_order)
    if set(factors) != set(layers) or set(projected_history) != set(layers) or h != FIXED_H:
        raise ODEBFContractError("incremental-H inputs or h differ")
    gram = np.zeros((len(layers), len(layers)), dtype=np.float64)
    history_columns = set()
    for index, layer in enumerate(layers):
        factor = _finite_tensor("H candidate factor", factors[layer])
        keys = _finite_tensor("H projected history", projected_history[layer])
        if factor.shape[1] != keys.shape[0]:
            raise ODEBFContractError("incremental-H factor/key orientation differs")
        history_columns.add(keys.shape[1])
        movement = factor.to(dtype=torch.float64) @ keys.to(dtype=torch.float64)
        gram[index, index] = h * h * float(torch.sum(movement * movement))
    if len(history_columns) != 1:
        raise ODEBFContractError("incremental-H layers have different history counts")
    scale = max(float(np.trace(gram)), NORMALIZATION_EPSILON)
    return QuadraticRoutingRisk(
        0.0,
        np.zeros(len(layers), dtype=np.float64),
        gram,
        scale,
        "incremental-structural-h",
    )


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
    h_score: float
    p_score: float
    worst_hp_score: float
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
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, int(active.size), max(int(active.size)-1, 0),
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

    def score(risk: QuadraticRoutingRisk, value: np.ndarray) -> float:
        return risk.normalized_value(value)

    def receipt(value: np.ndarray, status: str, fallback: bool) -> ExactStrengthRoutingResult:
        predicted = float(slopes @ value)
        h_score = score(historical_risk, value)
        p_score = score(pretrained_risk, value)
        return ExactStrengthRoutingResult(
            selected_arm,
            status,
            tuple(float(item) for item in value),
            tuple(float(item) for item in neutral),
            rho,
            predicted,
            abs(predicted-rho),
            neutral_energy,
            _quadratic_energy(value, trust),
            h_score,
            p_score,
            max(h_score, p_score),
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
    endpoint_weight_sha256: str
    terminal_virtual_weight_sha256: str
    keys_sha256: tuple[tuple[int, str], ...]


@dataclass(slots=True)
class SequentialArmState:
    """Independent arm state; no object is shared across Neutral/Soft arms."""

    arm_id: str
    layer_order: tuple[int, ...]
    ledger: P1HistoryLedger = field(init=False)
    cumulative_p: CumulativeStructuralPState = field(init=False)
    persistent_weights: dict[int, torch.Tensor] = field(init=False)
    factorization_cache: dict[tuple[int, int], HistoryFactorization] = field(init=False)
    completed_rounds: int = field(default=0, init=False)
    terminal_key_recapture_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.ledger = P1HistoryLedger(
            layer_order=self.layer_order,
            maximum_records=MAXIMUM_HISTORY_RECORDS,
        )
        self.cumulative_p = CumulativeStructuralPState(self.layer_order)
        self.persistent_weights = {}
        self.factorization_cache = {}

    def set_initial_weights(self, weights: Mapping[int, torch.Tensor]) -> None:
        if self.persistent_weights or set(weights) != set(self.layer_order):
            raise ODEBFStateError("persistent BF16 W initialization differs")
        for layer in self.layer_order:
            value = weights[layer]
            if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ODEBFContractError("persistent sequential W must be finite BF16")
            self.persistent_weights[layer] = value.detach().clone()

    def replace_persistent_weights_after_verified_commit(
        self,
        weights: Mapping[int, torch.Tensor],
        *,
        verified: bool,
    ) -> None:
        if not verified:
            return
        if set(weights) != set(self.layer_order):
            raise ODEBFContractError("persistent committed W layer set differs")
        replacement: dict[int, torch.Tensor] = {}
        for layer in self.layer_order:
            value = weights[layer]
            if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ODEBFContractError("persistent committed W must be finite BF16")
            replacement[layer] = value.detach().clone()
        self.persistent_weights = replacement

    def history_entry_count(self) -> int:
        return len(self.ledger.snapshot().active_records)

    def stage_terminal_keys(
        self,
        terminal_keys_by_layer: Mapping[int, torch.Tensor],
        projectors_by_layer: Mapping[int, torch.Tensor],
        *,
        terminal_virtual_weight_sha256: str,
        endpoint_weight_sha256: str,
        recapture: Callable[[], Mapping[int, torch.Tensor]] | None = None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], TerminalKeyReuseReceipt]:
        if set(terminal_keys_by_layer) != set(self.layer_order) or set(projectors_by_layer) != set(self.layer_order):
            raise ODEBFContractError("terminal key layer set differs")
        same_state = terminal_virtual_weight_sha256 == endpoint_weight_sha256
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
            endpoint_weight_sha256,
            terminal_virtual_weight_sha256,
            tuple((layer, tensor_sha256(raw[layer])) for layer in self.layer_order),
        )
        return raw, projected, receipt

    def finalize_history_after_commit(
        self,
        prospective: ProspectiveP1HistoryBatch,
        *,
        post_commit_verified: bool,
        load_increment_by_layer: Mapping[int, float],
        fault_phase: str | None = None,
    ) -> P1HistoryFinalizeReceipt | None:
        """Append exactly once; verification/rollback failure has append count zero."""

        if not post_commit_verified:
            return None
        receipt = self.ledger.finalize(
            prospective,
            post_commit_verified=True,
            load_increment_by_layer=load_increment_by_layer,
            fault_phase=fault_phase,
        )
        if receipt.appended_count:
            self.completed_rounds += 1
            self.factorization_cache.clear()
        return receipt


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
        "allocation_influence",
        "neutral_fallback",
        "history_count",
        "history_version",
        "woodbury_small_dimension",
        "history_cache_hit",
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
        receipt: P1HistoryFinalizeReceipt,
    ) -> int:
        if self.active_round is None or self.accepted_steps != FIXED_K:
            raise ODEBFStateError("sequential round lacks a complete K8")
        if receipt.appended_count != BATCH_SIZE or receipt.after_version != self.active_round:
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
        "instruction_id": P1R29_SEQUENTIAL_PREPARATION_INSTRUCTION_ID,
        "method_id": P1R29_SEQUENTIAL_PREPARATION_METHOD_ID,
        "status": "PREPARED_WAITING_P1R29_ATOMIC_GATE",
        "stage": "A_CPU_SOURCE_TEST_REPORT_ONLY",
        "models": 0,
        "gpu_allocations": 0,
        "slurm_jobs": 0,
        "result_roots": 0,
        "atomic_winner_imports": 0,
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
