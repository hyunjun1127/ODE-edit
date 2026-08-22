"""Joint structural-P/capacity routing for the P1R52 one-shot writer.

The five coordinates are semantic contribution shares.  This module is
model-free and deliberately implements exactly one epigraph minimax solve;
the existing P1R52 target, field construction, and writer implementations
remain outside this boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch
from scipy import __version__ as SCIPY_VERSION
from scipy.optimize import minimize, nnls

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import FIXED_E8_KKT_TOLERANCE
from .p1r52_residual_reserve_pc_router import (
    FLOAT64_EPSILON,
    ROUTER_DIMENSION,
    SOLVER_FTOL,
    SOLVER_MAX_ITERATIONS,
    SOLVER_PRIMAL_TOLERANCE,
    QuadraticProxy,
)
from .routing import RoutingProblem


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-JOINT-PC-C1-C2-INDEPENDENT-B100-V1"
METHOD_ID = "P1R52-JOINT-PC-EPIGRAPH-MINIMAX-V1"
REFERENCE_PI = (0.2, 0.2, 0.2, 0.2, 0.2)
REFERENCE_EPSILON = FLOAT64_EPSILON


class JointPCReferenceDegenerate(ODEBFStateError):
    """Typed formulation boundary for a numerical-zero reference scale."""


class JointPCNumericalCertificateError(ODEBFStateError):
    """Raw-free technical certificate failure with exact residual facts."""

    def __init__(self, receipt: dict[str, Any]) -> None:
        super().__init__("joint P/C epigraph numerical certificate failed")
        self.raw_free_receipt = receipt


@dataclass(frozen=True, slots=True)
class JointPCRouterReceipt:
    p_proxy: dict[str, Any]
    c_proxy: dict[str, Any]
    pi_reference: tuple[float, ...]
    p_zero: float
    c_zero: float
    p_reference: float
    c_reference: float
    p_reference_denominator_raw: float
    c_reference_denominator_raw: float
    normalization_epsilon: float
    selected_pi: tuple[float, ...]
    selected_p: float
    selected_c: float
    selected_p_normalized: float
    selected_c_normalized: float
    minimax_t: float
    simplex_sum_residual: float
    simplex_minimum: float
    p_constraint_slack: float
    c_constraint_slack: float
    stationarity_residual: float
    complementarity_residual: float
    solver_success: bool
    solver_status: int
    solver_message_sha256: str
    solver_iterations: int
    solver_backend: str
    solver_backend_version: str
    weighted_sum_count: int
    lexicographic_stage_count: int
    hard_budget_count: int
    fallback_count: int
    model_forward_count: int
    model_backward_count: int
    materialization_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema": "ode-edit-s05-p1r52-joint-pc-router/v1",
                "instruction_id": INSTRUCTION_ID,
                "method_id": METHOD_ID,
                "solver_ftol": SOLVER_FTOL,
                "solver_primal_tolerance": SOLVER_PRIMAL_TOLERANCE,
                "solver_kkt_tolerance": FIXED_E8_KKT_TOLERANCE,
                "kkt_active_set_tolerance": SOLVER_PRIMAL_TOLERANCE,
                "solver_max_iterations": SOLVER_MAX_ITERATIONS,
            }
        )
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class JointPCRoutingResult:
    pi: torch.Tensor
    p_proxy: QuadraticProxy
    c_proxy: QuadraticProxy
    receipt: JointPCRouterReceipt


def _proxy_value(proxy: QuadraticProxy, pi: np.ndarray) -> float:
    linear = proxy.linear.to(dtype=torch.float64).numpy()
    quadratic = proxy.quadratic.to(dtype=torch.float64).numpy()
    value = float(proxy.constant + linear @ pi + pi @ quadratic @ pi)
    if not math.isfinite(value):
        raise ODEBFContractError("joint P/C proxy evaluation is nonfinite")
    return value


def _proxy_gradient(proxy: QuadraticProxy, pi: np.ndarray) -> np.ndarray:
    linear = proxy.linear.to(dtype=torch.float64).numpy()
    quadratic = proxy.quadratic.to(dtype=torch.float64).numpy()
    return linear + 2.0 * (quadratic @ pi)


def proxies_from_entry_problem(
    problem: RoutingProblem,
) -> tuple[QuadraticProxy, QuadraticProxy]:
    """Bind the existing authoritative P and capacity quadratics.

    ``QuadraticBarrier.value`` uses ``offset + 2*l^T*pi + pi^T*G*pi``;
    the inherited capacity definition is ``0.5*pi^T*C*pi``.
    """

    if not isinstance(problem, RoutingProblem) or problem.signed_progress.size != ROUTER_DIMENSION:
        raise ODEBFContractError("joint P/C entry routing problem differs")
    p = problem.pretrained
    p_proxy = QuadraticProxy(
        "AUTHORITATIVE_STRUCTURAL_P",
        float(p.offset),
        torch.tensor(2.0 * p.linear, dtype=torch.float32),
        torch.tensor(p.gram, dtype=torch.float32),
    )
    c_proxy = QuadraticProxy(
        "AUTHORITATIVE_CAPACITY",
        0.0,
        torch.zeros(ROUTER_DIMENSION, dtype=torch.float32),
        torch.tensor(0.5 * problem.capacity_metric, dtype=torch.float32),
    )
    return p_proxy, c_proxy


def _kkt_residual(
    pi: np.ndarray,
    t: float,
    *,
    p_slack: float,
    c_slack: float,
    p_gradient: np.ndarray,
    c_gradient: np.ndarray,
) -> tuple[float, float]:
    # For g(x)>=0, grad f = J_g^T lambda + J_eq^T mu.  Represent the
    # unrestricted equality multiplier as the difference of two nonnegative
    # coordinates so NNLS yields a deterministic certificate.
    objective_gradient = np.concatenate(
        (np.zeros(ROUTER_DIMENSION, dtype=np.float64), np.ones(1, dtype=np.float64))
    )
    gradients: list[np.ndarray] = []
    slacks: list[float] = []
    # Inactive inequalities have exactly zero KKT multipliers. Including
    # their columns in NNLS can assign a spurious positive multiplier to a
    # slack constraint and falsely fail complementarity.
    for index, value in enumerate(pi):
        if float(value) <= SOLVER_PRIMAL_TOLERANCE:
            row = np.zeros(ROUTER_DIMENSION + 1, dtype=np.float64)
            row[index] = 1.0
            gradients.append(row)
            slacks.append(float(value))
    for slack, gradient in ((p_slack, p_gradient), (c_slack, c_gradient)):
        if float(slack) <= SOLVER_PRIMAL_TOLERANCE:
            row = np.concatenate((-gradient, np.ones(1, dtype=np.float64)))
            gradients.append(row)
            slacks.append(float(slack))
    equality = np.concatenate(
        (np.ones(ROUTER_DIMENSION, dtype=np.float64), np.zeros(1, dtype=np.float64))
    )
    matrix = np.stack((*gradients, equality, -equality), axis=1)
    multipliers, _ = nnls(matrix, objective_gradient)
    reconstructed = matrix @ multipliers
    stationarity = float(
        np.linalg.norm(objective_gradient - reconstructed)
        / max(1.0, float(np.linalg.norm(objective_gradient)))
    )
    inequality_multipliers = multipliers[: len(gradients)]
    complementarity = float(
        np.max(np.abs(inequality_multipliers * np.asarray(slacks)), initial=0.0)
    )
    return stationarity, complementarity


def solve_joint_pc_router(
    p_proxy: QuadraticProxy,
    c_proxy: QuadraticProxy,
) -> JointPCRoutingResult:
    """Solve the exact one-stage base-relative epigraph minimax."""

    if not isinstance(p_proxy, QuadraticProxy) or not isinstance(c_proxy, QuadraticProxy):
        raise ODEBFContractError("joint P/C proxy type differs")
    zero = np.zeros(ROUTER_DIMENSION, dtype=np.float64)
    reference = np.asarray(REFERENCE_PI, dtype=np.float64)
    p_zero = _proxy_value(p_proxy, zero)
    c_zero = _proxy_value(c_proxy, zero)
    p_reference = _proxy_value(p_proxy, reference)
    c_reference = _proxy_value(c_proxy, reference)
    p_raw = p_reference - p_zero
    c_raw = c_reference - c_zero
    p_scale = max(1.0, abs(p_zero), abs(p_reference), p_proxy.coefficient_scale)
    c_scale = max(1.0, abs(c_zero), abs(c_reference), c_proxy.coefficient_scale)
    if (
        not math.isfinite(p_raw)
        or not math.isfinite(c_raw)
        or abs(p_raw) <= SOLVER_PRIMAL_TOLERANCE * p_scale
        or abs(c_raw) <= SOLVER_PRIMAL_TOLERANCE * c_scale
    ):
        raise JointPCReferenceDegenerate("JOINT_PC_REFERENCE_DEGENERATE")
    p_denom = p_raw + REFERENCE_EPSILON
    c_denom = c_raw + REFERENCE_EPSILON

    def normalized(proxy: QuadraticProxy, base: float, denominator: float, pi: np.ndarray) -> float:
        return (_proxy_value(proxy, pi) - base) / denominator

    def normalized_gradient(proxy: QuadraticProxy, denominator: float, pi: np.ndarray) -> np.ndarray:
        return _proxy_gradient(proxy, pi) / denominator

    start_pi = reference.copy()
    start_t = max(
        normalized(p_proxy, p_zero, p_denom, start_pi),
        normalized(c_proxy, c_zero, c_denom, start_pi),
    )
    start = np.concatenate((start_pi, np.asarray([start_t], dtype=np.float64)))

    def objective(value: np.ndarray) -> float:
        return float(value[-1])

    def objective_jacobian(_value: np.ndarray) -> np.ndarray:
        return np.concatenate((np.zeros(ROUTER_DIMENSION), np.ones(1)))

    constraints = (
        {
            "type": "eq",
            "fun": lambda value: float(np.sum(value[:ROUTER_DIMENSION]) - 1.0),
            "jac": lambda _value: np.concatenate((np.ones(ROUTER_DIMENSION), np.zeros(1))),
        },
        {
            "type": "ineq",
            "fun": lambda value: float(
                value[-1]
                - normalized(p_proxy, p_zero, p_denom, value[:ROUTER_DIMENSION])
            ),
            "jac": lambda value: np.concatenate(
                (-normalized_gradient(p_proxy, p_denom, value[:ROUTER_DIMENSION]), np.ones(1))
            ),
        },
        {
            "type": "ineq",
            "fun": lambda value: float(
                value[-1]
                - normalized(c_proxy, c_zero, c_denom, value[:ROUTER_DIMENSION])
            ),
            "jac": lambda value: np.concatenate(
                (-normalized_gradient(c_proxy, c_denom, value[:ROUTER_DIMENSION]), np.ones(1))
            ),
        },
    )
    result = minimize(
        objective,
        start,
        jac=objective_jacobian,
        method="SLSQP",
        bounds=tuple((0.0, None) for _ in range(ROUTER_DIMENSION)) + ((None, None),),
        constraints=constraints,
        options={"ftol": SOLVER_FTOL, "maxiter": SOLVER_MAX_ITERATIONS, "disp": False},
    )
    value = np.asarray(result.x, dtype=np.float64)
    if not bool(result.success) or value.shape != (ROUTER_DIMENSION + 1,) or not np.all(np.isfinite(value)):
        raise ODEBFStateError("joint P/C epigraph solver failed")
    pi = value[:ROUTER_DIMENSION]
    t = float(value[-1])
    p_value = _proxy_value(p_proxy, pi)
    c_value = _proxy_value(c_proxy, pi)
    p_normalized = (p_value - p_zero) / p_denom
    c_normalized = (c_value - c_zero) / c_denom
    simplex_residual = abs(float(np.sum(pi)) - 1.0)
    p_slack = t - p_normalized
    c_slack = t - c_normalized
    stationarity, complementarity = _kkt_residual(
        pi,
        t,
        p_slack=p_slack,
        c_slack=c_slack,
        p_gradient=normalized_gradient(p_proxy, p_denom, pi),
        c_gradient=normalized_gradient(c_proxy, c_denom, pi),
    )
    if (
        simplex_residual > SOLVER_PRIMAL_TOLERANCE
        or float(np.min(pi)) < -SOLVER_PRIMAL_TOLERANCE
        or min(p_slack, c_slack) < -SOLVER_PRIMAL_TOLERANCE
        or not all(math.isfinite(item) for item in (p_value, c_value, t, stationarity, complementarity))
        or stationarity > FIXED_E8_KKT_TOLERANCE
        or complementarity > FIXED_E8_KKT_TOLERANCE
    ):
        raise JointPCNumericalCertificateError(
            {
                "schema": "ode-edit-s05-p1r52-joint-pc-certificate-failure/v1",
                "simplex_sum_residual": simplex_residual,
                "simplex_minimum": float(np.min(pi)),
                "p_constraint_slack": p_slack,
                "c_constraint_slack": c_slack,
                "stationarity_residual": stationarity,
                "complementarity_residual": complementarity,
                "solver_primal_tolerance": SOLVER_PRIMAL_TOLERANCE,
                "solver_kkt_tolerance": FIXED_E8_KKT_TOLERANCE,
                "kkt_active_set_tolerance": SOLVER_PRIMAL_TOLERANCE,
                "solver_success": bool(result.success),
                "solver_status": int(result.status),
                "solver_iterations": int(result.nit),
            }
        )
    pi32 = torch.tensor(pi, dtype=torch.float32)
    if abs(float(pi32.sum()) - 1.0) > 5 * torch.finfo(torch.float32).eps:
        raise ODEBFStateError("joint P/C FP32 simplex preservation failed")
    import hashlib

    receipt = JointPCRouterReceipt(
        p_proxy=p_proxy.raw_free_payload(),
        c_proxy=c_proxy.raw_free_payload(),
        pi_reference=REFERENCE_PI,
        p_zero=p_zero,
        c_zero=c_zero,
        p_reference=p_reference,
        c_reference=c_reference,
        p_reference_denominator_raw=p_raw,
        c_reference_denominator_raw=c_raw,
        normalization_epsilon=REFERENCE_EPSILON,
        selected_pi=tuple(float(item) for item in pi32),
        selected_p=p_value,
        selected_c=c_value,
        selected_p_normalized=p_normalized,
        selected_c_normalized=c_normalized,
        minimax_t=t,
        simplex_sum_residual=simplex_residual,
        simplex_minimum=float(np.min(pi)),
        p_constraint_slack=p_slack,
        c_constraint_slack=c_slack,
        stationarity_residual=stationarity,
        complementarity_residual=complementarity,
        solver_success=bool(result.success),
        solver_status=int(result.status),
        solver_message_sha256=hashlib.sha256(str(result.message).encode()).hexdigest(),
        solver_iterations=int(result.nit),
        solver_backend="scipy-slsqp-float64",
        solver_backend_version=SCIPY_VERSION,
        weighted_sum_count=0,
        lexicographic_stage_count=0,
        hard_budget_count=0,
        fallback_count=0,
        model_forward_count=0,
        model_backward_count=0,
        materialization_count=0,
    )
    return JointPCRoutingResult(pi32, p_proxy, c_proxy, receipt)


__all__ = [
    "INSTRUCTION_ID",
    "JointPCReferenceDegenerate",
    "JointPCNumericalCertificateError",
    "JointPCRouterReceipt",
    "JointPCRoutingResult",
    "METHOD_ID",
    "REFERENCE_EPSILON",
    "REFERENCE_PI",
    "proxies_from_entry_problem",
    "solve_joint_pc_router",
]
