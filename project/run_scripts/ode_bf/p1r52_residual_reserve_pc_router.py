"""Model-free P/C quadratic router for residual-reserve layer allocation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch
from scipy.optimize import minimize as _scipy_minimize

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .p1r52_residual_reserve_geometry import (
    FP32_SIMPLEX_SUM_TOLERANCE,
    RESIDUAL_RESERVE_LAYER_COUNT,
)


ROUTER_DIMENSION = RESIDUAL_RESERVE_LAYER_COUNT
FLOAT64_EPSILON = np.finfo(np.float64).eps
SOLVER_FTOL = FLOAT64_EPSILON ** 0.75
SOLVER_PRIMAL_TOLERANCE = math.sqrt(FLOAT64_EPSILON)
SOLVER_TIE_ENVELOPE = SOLVER_FTOL
SOLVER_MAX_ITERATIONS = 200 * ROUTER_DIMENSION
TIE_BREAK_ORDER = (
    "MINIMUM_MAX_NORMALIZED_REGRET",
    "MINIMUM_NORMALIZED_P_PLUS_C_REGRET",
    "MINIMUM_EUCLIDEAN_DISTANCE_TO_UNIFORM",
    "DETERMINISTIC_LAYER_ORDER_IF_PRIOR_KEYS_EXACTLY_TIED",
)


@dataclass(frozen=True, slots=True)
class QuadraticProxy:
    """A validated FP32 quadratic ``c + l^T pi + pi^T Q pi``."""

    name: str
    constant: float
    linear: torch.Tensor
    quadratic: torch.Tensor
    rank: int = field(init=False)
    condition: float | None = field(init=False)
    minimum_eigenvalue: float = field(init=False)
    coefficient_scale: float = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ODEBFContractError("P/C quadratic name is empty")
        constant = float(self.constant)
        if not math.isfinite(constant):
            raise ODEBFContractError("P/C quadratic constant is non-finite")
        if (
            not isinstance(self.linear, torch.Tensor)
            or not isinstance(self.quadratic, torch.Tensor)
            or self.linear.dtype is not torch.float32
            or self.quadratic.dtype is not torch.float32
            or self.linear.shape != (ROUTER_DIMENSION,)
            or self.quadratic.shape != (ROUTER_DIMENSION, ROUTER_DIMENSION)
            or self.linear.device.type != "cpu"
            or self.quadratic.device.type != "cpu"
            or not bool(torch.isfinite(self.linear).all())
            or not bool(torch.isfinite(self.quadratic).all())
        ):
            raise ODEBFContractError("P/C quadratic tensor contract differs")
        if not torch.equal(self.quadratic, self.quadratic.T):
            raise ODEBFContractError("P/C quadratic matrix is not symmetric")

        linear = self.linear.detach().clone().contiguous()
        quadratic = self.quadratic.detach().clone().contiguous()
        q64 = quadratic.to(dtype=torch.float64).numpy()
        eigenvalues = np.linalg.eigvalsh(q64)
        spectral_scale = max(
            float(np.linalg.norm(q64, ord=2)),
            np.finfo(np.float64).tiny,
        )
        psd_tolerance = FLOAT64_EPSILON * ROUTER_DIMENSION * spectral_scale
        minimum = float(eigenvalues.min())
        if minimum < -psd_tolerance:
            raise ODEBFContractError("P/C quadratic matrix is not PSD")
        rank = int(np.linalg.matrix_rank(q64))
        condition = float(np.linalg.cond(q64)) if rank == ROUTER_DIMENSION else None
        if condition is not None and not math.isfinite(condition):
            raise ODEBFContractError("P/C quadratic condition is non-finite")
        coefficient_scale = max(
            float(torch.max(torch.abs(linear))),
            float(torch.max(torch.abs(quadratic))),
        )
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadratic", quadratic)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "minimum_eigenvalue", minimum)
        object.__setattr__(self, "coefficient_scale", coefficient_scale)

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "constant": self.constant,
            "linear": [float(item) for item in self.linear],
            "quadratic": [
                [float(item) for item in row] for row in self.quadratic
            ],
            "dtype": "torch.float32",
            "rank": self.rank,
            "condition": self.condition,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "coefficient_scale": self.coefficient_scale,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class SolverStageReceipt:
    stage: str
    status: str
    iterations: int
    message: str
    objective: float
    simplex_sum_residual: float
    simplex_minimum: float
    minimum_constraint_slack: float

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "iterations": self.iterations,
            "message": self.message,
            "objective": self.objective,
            "simplex_sum_residual": self.simplex_sum_residual,
            "simplex_minimum": self.simplex_minimum,
            "minimum_constraint_slack": self.minimum_constraint_slack,
        }


@dataclass(frozen=True, slots=True)
class RouteCandidateReceipt:
    role: str
    pi: tuple[float, ...]
    p_value: float
    c_value: float
    p_normalized_regret: float | None
    c_normalized_regret: float | None
    maximum_normalized_regret: float
    normalized_regret_sum: float
    uniform_distance: float
    solver_status: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "pi": list(self.pi),
            "p_value": self.p_value,
            "c_value": self.c_value,
            "p_normalized_regret": self.p_normalized_regret,
            "c_normalized_regret": self.c_normalized_regret,
            "maximum_normalized_regret": self.maximum_normalized_regret,
            "normalized_regret_sum": self.normalized_regret_sum,
            "uniform_distance": self.uniform_distance,
            "solver_status": self.solver_status,
        }


@dataclass(frozen=True, slots=True)
class ResidualReservePCRouterReceipt:
    p_proxy: dict[str, Any]
    c_proxy: dict[str, Any]
    p_scale: float
    c_scale: float
    p_flat_tolerance: float
    c_flat_tolerance: float
    p_flat: bool
    c_flat: bool
    candidates: tuple[RouteCandidateReceipt, ...]
    solver_stages: tuple[SolverStageReceipt, ...]
    selected_status: str
    tie_break_order: tuple[str, ...]

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r52-residual-reserve-pc-router/v1",
            "p_proxy": self.p_proxy,
            "c_proxy": self.c_proxy,
            "sP": self.p_scale,
            "sC": self.c_scale,
            "p_flat_tolerance": self.p_flat_tolerance,
            "c_flat_tolerance": self.c_flat_tolerance,
            "p_flat": self.p_flat,
            "c_flat": self.c_flat,
            "candidates": [item.raw_free_payload() for item in self.candidates],
            "solver_stages": [item.raw_free_payload() for item in self.solver_stages],
            "selected_status": self.selected_status,
            "tie_break_order": list(self.tie_break_order),
            "solver_backend": "SCIPY_SLSQP_SMALL_5D_INDEPENDENT",
            "solver_ftol": SOLVER_FTOL,
            "solver_primal_tolerance": SOLVER_PRIMAL_TOLERANCE,
            "solver_tie_envelope": SOLVER_TIE_ENVELOPE,
            "solver_max_iterations": SOLVER_MAX_ITERATIONS,
            "manual_weighted_sum_count": 0,
            "hard_budget_count": 0,
            "fallback_retry_backtracking_count": 0,
            "model_forward_backward_materialization": [0, 0, 0],
            "algorithmic_tensor_dtype": "torch.float32",
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class ResidualReservePCRoutingResult:
    pi_p: torch.Tensor
    pi_c: torch.Tensor
    pi_balanced: torch.Tensor
    pi_uniform: torch.Tensor
    receipt: ResidualReservePCRouterReceipt


def _evaluate(proxy: QuadraticProxy, pi: np.ndarray) -> float:
    linear = proxy.linear.to(dtype=torch.float64).numpy()
    quadratic = proxy.quadratic.to(dtype=torch.float64).numpy()
    value = proxy.constant + float(linear @ pi) + float(pi @ quadratic @ pi)
    if not math.isfinite(value):
        raise ODEBFContractError("P/C quadratic evaluation is non-finite")
    return value


def evaluate_quadratic_proxy(proxy: QuadraticProxy, pi: torch.Tensor) -> float:
    """Evaluate a proxy at a validated FP32 simplex without mutation."""

    allocation = _validated_output_pi(pi.detach().to(device="cpu", dtype=pi.dtype))
    return _evaluate(proxy, allocation.to(dtype=torch.float64).numpy())


def _gradient(proxy: QuadraticProxy, pi: np.ndarray) -> np.ndarray:
    linear = proxy.linear.to(dtype=torch.float64).numpy()
    quadratic = proxy.quadratic.to(dtype=torch.float64).numpy()
    return linear + 2.0 * (quadratic @ pi)


def _simplex_constraint(size: int) -> dict[str, Any]:
    return {
        "type": "eq",
        "fun": lambda value: float(np.sum(value[:ROUTER_DIMENSION]) - 1.0),
        "jac": lambda value: np.concatenate(
            (np.ones(ROUTER_DIMENSION, dtype=np.float64), np.zeros(size - ROUTER_DIMENSION))
        ),
    }


def _run_slsqp(
    *,
    stage: str,
    objective: Callable[[np.ndarray], float],
    jacobian: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    constraints: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, SolverStageReceipt]:
    result = _scipy_minimize(
        objective,
        start.copy(),
        jac=jacobian,
        method="SLSQP",
        bounds=tuple(bounds),
        constraints=tuple(constraints),
        options={"ftol": SOLVER_FTOL, "maxiter": SOLVER_MAX_ITERATIONS, "disp": False},
    )
    value = np.asarray(result.x, dtype=np.float64)
    if not bool(result.success) or value.shape != start.shape or not np.all(np.isfinite(value)):
        raise ODEBFStateError(f"P/C router solver failed at {stage}: {result.message}")
    pi = value[:ROUTER_DIMENSION]
    simplex_residual = abs(float(np.sum(pi)) - 1.0)
    simplex_minimum = float(np.min(pi))
    slacks: list[float] = []
    for constraint in constraints:
        observed = np.asarray(constraint["fun"](value), dtype=np.float64)
        if constraint["type"] == "eq":
            slacks.append(-float(np.max(np.abs(observed))))
        else:
            slacks.append(float(np.min(observed)))
    minimum_slack = min(slacks, default=0.0)
    if (
        simplex_residual > SOLVER_PRIMAL_TOLERANCE
        or simplex_minimum < -SOLVER_PRIMAL_TOLERANCE
        or minimum_slack < -SOLVER_PRIMAL_TOLERANCE
    ):
        raise ODEBFStateError(f"P/C router solver certificate failed at {stage}")
    receipt = SolverStageReceipt(
        stage=stage,
        status="PASS",
        iterations=int(result.nit),
        message=str(result.message),
        objective=float(objective(value)),
        simplex_sum_residual=simplex_residual,
        simplex_minimum=simplex_minimum,
        minimum_constraint_slack=minimum_slack,
    )
    return value, receipt


def _validated_output_pi(value: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype is not torch.float32
        or value.shape != (ROUTER_DIMENSION,)
        or value.device.type != "cpu"
        or not bool(torch.isfinite(value).all())
        or bool((value < 0.0).any())
        or abs(float(value.sum(dtype=torch.float32)) - 1.0)
        > FP32_SIMPLEX_SUM_TOLERANCE
    ):
        raise ODEBFContractError("P/C router output simplex differs")
    return value.detach().clone().contiguous()


def _fp32_pi(value: np.ndarray) -> torch.Tensor:
    return _validated_output_pi(torch.tensor(value, dtype=torch.float32))


def _solve_axis(
    proxy: QuadraticProxy,
    *,
    stage: str,
) -> tuple[torch.Tensor, SolverStageReceipt]:
    uniform = np.full(ROUTER_DIMENSION, 1.0 / ROUTER_DIMENSION, dtype=np.float64)
    if proxy.coefficient_scale == 0.0:
        receipt = SolverStageReceipt(
            stage=stage,
            status="CONSTANT_AXIS_UNIFORM",
            iterations=0,
            message="exact constant quadratic",
            objective=proxy.constant,
            simplex_sum_residual=0.0,
            simplex_minimum=float(uniform.min()),
            minimum_constraint_slack=0.0,
        )
        return _fp32_pi(uniform), receipt
    scale = proxy.coefficient_scale
    constraint = _simplex_constraint(ROUTER_DIMENSION)
    value, receipt = _run_slsqp(
        stage=stage,
        objective=lambda pi: (_evaluate(proxy, pi) - proxy.constant) / scale,
        jacobian=lambda pi: _gradient(proxy, pi) / scale,
        start=uniform,
        bounds=((0.0, 1.0),) * ROUTER_DIMENSION,
        constraints=(constraint,),
    )
    return _fp32_pi(value), receipt


def _flat_tolerance(proxy: QuadraticProxy) -> float:
    return SOLVER_PRIMAL_TOLERANCE * proxy.coefficient_scale


def _candidate_receipt(
    *,
    role: str,
    pi: torch.Tensor,
    p_proxy: QuadraticProxy,
    c_proxy: QuadraticProxy,
    p_minimum: float,
    c_minimum: float,
    p_scale: float,
    c_scale: float,
    p_flat: bool,
    c_flat: bool,
    solver_status: str,
) -> RouteCandidateReceipt:
    array = pi.to(dtype=torch.float64).numpy()
    p_value = _evaluate(p_proxy, array)
    c_value = _evaluate(c_proxy, array)
    p_regret = None if p_flat else (p_value - p_minimum) / p_scale
    c_regret = None if c_flat else (c_value - c_minimum) / c_scale
    regrets = [item for item in (p_regret, c_regret) if item is not None]
    uniform = np.full(ROUTER_DIMENSION, 1.0 / ROUTER_DIMENSION, dtype=np.float64)
    return RouteCandidateReceipt(
        role=role,
        pi=tuple(float(item) for item in pi),
        p_value=p_value,
        c_value=c_value,
        p_normalized_regret=p_regret,
        c_normalized_regret=c_regret,
        maximum_normalized_regret=max(regrets, default=0.0),
        normalized_regret_sum=math.fsum(regrets),
        uniform_distance=float(np.linalg.norm(array - uniform)),
        solver_status=solver_status,
    )


def solve_residual_reserve_pc_router(
    p_proxy: QuadraticProxy,
    c_proxy: QuadraticProxy,
) -> ResidualReservePCRoutingResult:
    """Solve the P-only, C-only, and weight-free balanced simplex routes."""

    if not isinstance(p_proxy, QuadraticProxy) or not isinstance(c_proxy, QuadraticProxy):
        raise ODEBFContractError("P/C router proxy type differs")
    pi_p, p_stage = _solve_axis(p_proxy, stage="P_ONLY")
    pi_c, c_stage = _solve_axis(c_proxy, stage="C_ONLY")
    pi_uniform = _fp32_pi(
        np.full(ROUTER_DIMENSION, 1.0 / ROUTER_DIMENSION, dtype=np.float64)
    )
    p_minimum = evaluate_quadratic_proxy(p_proxy, pi_p)
    c_minimum = evaluate_quadratic_proxy(c_proxy, pi_c)
    p_scale = evaluate_quadratic_proxy(p_proxy, pi_c) - p_minimum
    c_scale = evaluate_quadratic_proxy(c_proxy, pi_p) - c_minimum
    p_flat_tolerance = _flat_tolerance(p_proxy)
    c_flat_tolerance = _flat_tolerance(c_proxy)
    if p_scale < -p_flat_tolerance or c_scale < -c_flat_tolerance:
        raise ODEBFStateError("P/C axis minimum certificate differs")
    p_flat = p_scale <= p_flat_tolerance
    c_flat = c_scale <= c_flat_tolerance
    stages: list[SolverStageReceipt] = [p_stage, c_stage]

    if p_flat and c_flat:
        pi_balanced = pi_uniform.clone()
        selected_status = "BOTH_AXES_FLAT_UNIFORM"
    elif p_flat:
        pi_balanced = pi_c.clone()
        selected_status = "P_AXIS_FLAT_C_ONLY"
    elif c_flat:
        pi_balanced = pi_p.clone()
        selected_status = "C_AXIS_FLAT_P_ONLY"
    else:
        def p_regret(pi: np.ndarray) -> float:
            return (_evaluate(p_proxy, pi) - p_minimum) / p_scale

        def c_regret(pi: np.ndarray) -> float:
            return (_evaluate(c_proxy, pi) - c_minimum) / c_scale

        def p_regret_gradient(pi: np.ndarray) -> np.ndarray:
            return _gradient(p_proxy, pi) / p_scale

        def c_regret_gradient(pi: np.ndarray) -> np.ndarray:
            return _gradient(c_proxy, pi) / c_scale

        starts = (
            pi_uniform.to(dtype=torch.float64).numpy(),
            pi_p.to(dtype=torch.float64).numpy(),
            pi_c.to(dtype=torch.float64).numpy(),
        )
        start_pi = min(
            starts,
            key=lambda value: (
                max(p_regret(value), c_regret(value)),
                p_regret(value) + c_regret(value),
                float(np.linalg.norm(value - starts[0])),
                tuple(float(item) for item in value),
            ),
        )
        start_t = max(p_regret(start_pi), c_regret(start_pi), 0.0)
        start_epigraph = np.concatenate((start_pi, np.array([start_t])))
        epigraph_constraints = (
            _simplex_constraint(ROUTER_DIMENSION + 1),
            {
                "type": "ineq",
                "fun": lambda value: float(value[-1] - p_regret(value[:-1])),
                "jac": lambda value: np.concatenate(
                    (-p_regret_gradient(value[:-1]), np.ones(1))
                ),
            },
            {
                "type": "ineq",
                "fun": lambda value: float(value[-1] - c_regret(value[:-1])),
                "jac": lambda value: np.concatenate(
                    (-c_regret_gradient(value[:-1]), np.ones(1))
                ),
            },
        )
        stage1, receipt1 = _run_slsqp(
            stage="BALANCED_MINMAX",
            objective=lambda value: float(value[-1]),
            jacobian=lambda value: np.concatenate(
                (np.zeros(ROUTER_DIMENSION), np.ones(1))
            ),
            start=start_epigraph,
            bounds=((0.0, 1.0),) * ROUTER_DIMENSION + ((0.0, None),),
            constraints=epigraph_constraints,
        )
        stages.append(receipt1)
        maximum_optimum = float(stage1[-1])
        numerical_envelope = SOLVER_TIE_ENVELOPE
        stage2_constraints = (
            _simplex_constraint(ROUTER_DIMENSION),
            {
                "type": "ineq",
                "fun": lambda pi: maximum_optimum + numerical_envelope - p_regret(pi),
                "jac": lambda pi: -p_regret_gradient(pi),
            },
            {
                "type": "ineq",
                "fun": lambda pi: maximum_optimum + numerical_envelope - c_regret(pi),
                "jac": lambda pi: -c_regret_gradient(pi),
            },
        )
        stage2, receipt2 = _run_slsqp(
            stage="BALANCED_MINIMUM_REGRET_SUM_TIE",
            objective=lambda pi: p_regret(pi) + c_regret(pi),
            jacobian=lambda pi: p_regret_gradient(pi) + c_regret_gradient(pi),
            start=stage1[:-1],
            bounds=((0.0, 1.0),) * ROUTER_DIMENSION,
            constraints=stage2_constraints,
        )
        stages.append(receipt2)
        regret_sum_optimum = p_regret(stage2) + c_regret(stage2)
        uniform64 = pi_uniform.to(dtype=torch.float64).numpy()
        stage3_constraints = stage2_constraints + (
            {
                "type": "ineq",
                "fun": lambda pi: regret_sum_optimum
                + numerical_envelope
                - p_regret(pi)
                - c_regret(pi),
                "jac": lambda pi: -p_regret_gradient(pi) - c_regret_gradient(pi),
            },
        )
        stage3, receipt3 = _run_slsqp(
            stage="BALANCED_UNIFORM_DISTANCE_TIE",
            objective=lambda pi: float(np.sum((pi - uniform64) ** 2)),
            jacobian=lambda pi: 2.0 * (pi - uniform64),
            start=stage2,
            bounds=((0.0, 1.0),) * ROUTER_DIMENSION,
            constraints=stage3_constraints,
        )
        stages.append(receipt3)
        pi_balanced = _fp32_pi(stage3)
        selected_status = "BALANCED_BOTH_AXES_ACTIVE"

    candidate_specs = (
        ("P_ONLY", pi_p, p_stage.status),
        ("C_ONLY", pi_c, c_stage.status),
        ("BALANCED", pi_balanced, selected_status),
        ("UNIFORM", pi_uniform, "FIXED_UNIFORM"),
    )
    candidates = tuple(
        _candidate_receipt(
            role=role,
            pi=pi,
            p_proxy=p_proxy,
            c_proxy=c_proxy,
            p_minimum=p_minimum,
            c_minimum=c_minimum,
            p_scale=p_scale,
            c_scale=c_scale,
            p_flat=p_flat,
            c_flat=c_flat,
            solver_status=status,
        )
        for role, pi, status in candidate_specs
    )
    receipt = ResidualReservePCRouterReceipt(
        p_proxy=p_proxy.raw_free_payload(),
        c_proxy=c_proxy.raw_free_payload(),
        p_scale=p_scale,
        c_scale=c_scale,
        p_flat_tolerance=p_flat_tolerance,
        c_flat_tolerance=c_flat_tolerance,
        p_flat=p_flat,
        c_flat=c_flat,
        candidates=candidates,
        solver_stages=tuple(stages),
        selected_status=selected_status,
        tie_break_order=TIE_BREAK_ORDER,
    )
    return ResidualReservePCRoutingResult(
        pi_p=pi_p.detach().clone(),
        pi_c=pi_c.detach().clone(),
        pi_balanced=pi_balanced.detach().clone(),
        pi_uniform=pi_uniform.detach().clone(),
        receipt=receipt,
    )


__all__ = [
    "FLOAT64_EPSILON",
    "QuadraticProxy",
    "ResidualReservePCRouterReceipt",
    "ResidualReservePCRoutingResult",
    "RouteCandidateReceipt",
    "SOLVER_FTOL",
    "SOLVER_MAX_ITERATIONS",
    "SOLVER_PRIMAL_TOLERANCE",
    "SOLVER_TIE_ENVELOPE",
    "SolverStageReceipt",
    "TIE_BREAK_ORDER",
    "evaluate_quadratic_proxy",
    "solve_residual_reserve_pc_router",
]
