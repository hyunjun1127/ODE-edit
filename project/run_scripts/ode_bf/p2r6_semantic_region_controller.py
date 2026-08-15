"""P2R6 semantic-region writer controller.

The module is deliberately model-free.  It consumes the current in-memory
request/layer response and the sealed P2R5 cumulative quadratics, then returns
one certified allocation.  Target construction, response VJP, materialization,
transaction, and evaluation remain owned by the protected P2R1/P2R2/P2R4/P2R5
runtime interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linprog

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p2r5_sdrt_writer import (
    P2R5RoutingTechnicalError,
    SDRTCalibration,
    SDRTQuadratics,
    solve_sdrt_routing,
)
from .p2r6_certified_convex_qp import (
    ConvexQuadraticTie,
    P2R6CertifiedQPError,
    solve_certified_semantic_region_qp,
)
from .progress_simplex_routing import (
    SIMPLEX_PRIMAL_TOLERANCE,
    SIMPLEX_XI_TIE_TOLERANCE,
)


P2R6_INSTRUCTION_ID = "ODEEDIT-S05-P2R6-RED-R2-FINAL-SCIENTIFIC-RUN-V1"
P2R6_METHOD_ID = "P2R6-RED-R2-FINAL-SEMANTIC-REGION-CONTROLLER-V1"
P2R6_CAP_ARMS = ("A0-CAP", "AETA-CAP", "AR-CAP", "AS-CAP")
P2R6_STRUCTP_ARMS = ("AR-STRUCTP", "AS-STRUCTP")
P2R6_ARMS = (*P2R6_CAP_ARMS, *P2R6_STRUCTP_ARMS)
P2R6_SHADOW_ARMS = P2R6_CAP_ARMS
P2R6_NUMERICAL_EPSILON = SIMPLEX_PRIMAL_TOLERANCE
# HiGHS documents 1e-10 as the smallest accepted feasibility tolerance.  This
# is backend accuracy, strictly tighter than the unchanged scientific/primal
# certificate above.
P2R6_HIGHS_INTERNAL_TOLERANCE = 1.0e-10
P2R6_E1_XI_AUTHORITY = "FP64_RECOMPUTED_FROM_RETURNED_ALPHA"


class P2R6RoutingTechnicalError(P2R5RoutingTechnicalError):
    """Typed fail-close error with raw-free numerical observability."""


@dataclass(frozen=True, slots=True)
class P2R6RoutingResult:
    arm: str
    controller: str
    preservation_objective: str
    scale_policy: str
    allocation: torch.Tensor
    deficit: torch.Tensor
    entry_deficit: torch.Tensor
    semantic_scale: torch.Tensor
    semantic_lower_bound: torch.Tensor
    raw_predicted_response: torch.Tensor
    calibrated_predicted_response: torch.Tensor
    semantic_response_star: torch.Tensor
    eta_observation: float
    eta_decision: float
    eta_decision_influence_count: int
    e1_xi: float
    e2_enabled: bool
    exact_response_face_enabled: bool
    cumulative_capacity: float
    marginal_capacity: float
    cumulative_structural_p: float
    marginal_structural_p: float
    coefficient_mass_by_request: tuple[float, ...]
    active_mass_constraints: tuple[int, ...]
    active_semantic_constraints: tuple[int, ...]
    feasible_rank: int
    feasible_nullity: int
    semantic_region_max_violation: float
    solver_receipts: tuple[Mapping[str, Any], ...]
    status: str
    identity_sha256: str

    @property
    def predicted_progress(self) -> torch.Tensor:
        return self.raw_predicted_response

    def raw_free_payload(self) -> dict[str, Any]:
        allocation = self.allocation
        semantic_slack = (
            self.calibrated_predicted_response - self.semantic_lower_bound
        )
        return {
            "schema": "ode-edit-s05-p2r6-semantic-region-routing/v1",
            "arm": self.arm,
            "controller": self.controller,
            "preservation_objective": self.preservation_objective,
            "scale_policy": self.scale_policy,
            "allocation_shape": list(allocation.shape),
            "allocation_sha256": tensor_sha256(allocation),
            "allocation_by_layer_request": allocation.tolist(),
            "deficit_by_request": self.deficit.tolist(),
            "entry_deficit_by_request": self.entry_deficit.tolist(),
            "semantic_scale_by_request": self.semantic_scale.tolist(),
            "semantic_lower_bound_by_request": self.semantic_lower_bound.tolist(),
            "semantic_slack_by_request": semantic_slack.tolist(),
            "raw_predicted_response_by_request": self.raw_predicted_response.tolist(),
            "calibrated_predicted_response_by_request": (
                self.calibrated_predicted_response.tolist()
            ),
            "semantic_response_star_by_request": self.semantic_response_star.tolist(),
            "eta_observation": self.eta_observation,
            "eta_decision": self.eta_decision,
            "eta_decision_influence_count": self.eta_decision_influence_count,
            "e1_xi": self.e1_xi,
            "e2_enabled": self.e2_enabled,
            "exact_response_face_enabled": self.exact_response_face_enabled,
            "cumulative_capacity": self.cumulative_capacity,
            "marginal_capacity": self.marginal_capacity,
            "cumulative_structural_p": self.cumulative_structural_p,
            "marginal_structural_p": self.marginal_structural_p,
            "coefficient_mass_by_request": list(self.coefficient_mass_by_request),
            "coefficient_mass_min": min(self.coefficient_mass_by_request),
            "coefficient_mass_max": max(self.coefficient_mass_by_request),
            "active_mass_constraints": list(self.active_mass_constraints),
            "active_semantic_constraints": list(self.active_semantic_constraints),
            "feasible_rank": self.feasible_rank,
            "feasible_nullity": self.feasible_nullity,
            "semantic_region_max_violation": self.semantic_region_max_violation,
            "solver_receipts": [dict(item) for item in self.solver_receipts],
            "status": self.status,
            "neutral_fallback_count": 0,
            "hard_p_budget_influence_count": 0,
            "functional_p_veto_count": 0,
            "strength_attenuation_by_preservation_count": 0,
            "candidate_forward_count": 0,
            "candidate_materialization_count": 0,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class P2R6ShadowArmResult:
    arm: str
    selected: bool
    route: P2R6RoutingResult | None
    technical_error: Mapping[str, Any] | None
    status: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "selected": self.selected,
            "status": self.status,
            "route": self.route.raw_free_payload() if self.route is not None else None,
            "technical_error": (
                dict(self.technical_error)
                if self.technical_error is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class P2R6ShadowPanel:
    selected_arm: str
    outer_step: int
    arm_results: tuple[P2R6ShadowArmResult, ...]
    routes: tuple[P2R6RoutingResult, ...]
    pairwise_allocation_l2: Mapping[str, float]
    eta_on_off_allocation_l2: float | None
    failing_shadow_arms: tuple[str, ...]
    status: str
    solver_call_count: int
    certificate_count: int
    model_forward_count: int
    model_backward_count: int
    materialization_count: int
    identity_sha256: str

    def route(self, arm: str) -> P2R6RoutingResult:
        for item in self.routes:
            if item.arm == arm:
                return item
        raise ODEBFContractError("P2R6 shadow arm is absent")

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p2r6-same-state-shadow-panel/v1",
            "selected_arm": self.selected_arm,
            "outer_step": self.outer_step,
            "arms": [item.arm for item in self.routes],
            "arm_results": [item.raw_free_payload() for item in self.arm_results],
            "routes": [item.raw_free_payload() for item in self.routes],
            "pairwise_allocation_l2": dict(self.pairwise_allocation_l2),
            "eta_on_off_allocation_l2": self.eta_on_off_allocation_l2,
            "failing_shadow_arms": list(self.failing_shadow_arms),
            "status": self.status,
            "selected_action_unchanged_by_shadow_error": True,
            "shadow_error_decision_influence_count": 0,
            "solver_call_count": self.solver_call_count,
            "certificate_count": self.certificate_count,
            "model_forward_count": self.model_forward_count,
            "model_backward_count": self.model_backward_count,
            "materialization_count": self.materialization_count,
            "identity_sha256": self.identity_sha256,
        }


def _mass_matrix(layer_count: int, request_count: int) -> np.ndarray:
    matrix = np.zeros((request_count, layer_count * request_count), dtype=np.float64)
    for request in range(request_count):
        matrix[request, request::request_count] = 1.0
    return matrix


def _quadratic_value(matrix: np.ndarray, cross: np.ndarray, x: np.ndarray) -> float:
    return float(x @ matrix @ x + 2.0 * cross @ x)


def _eta_one(calibration: SDRTCalibration) -> SDRTCalibration:
    payload = {
        "eta": 1.0,
        "observation_eta": calibration.eta,
        "prior_step_count": calibration.prior_step_count,
        "pair_count": calibration.pair_count,
        "status": "ETA_OBSERVATION_ONLY_DECISION_ONE",
    }
    return SDRTCalibration(
        1.0,
        calibration.prior_step_count,
        calibration.pair_count,
        calibration.numerator,
        calibration.denominator,
        "ETA_OBSERVATION_ONLY_DECISION_ONE",
        canonical_hash(payload),
    )


def _wrap_legacy_route(
    route: Any,
    *,
    arm: str,
    controller: str,
    calibration: SDRTCalibration,
    entry_deficit: torch.Tensor,
) -> P2R6RoutingResult:
    allocation = route.allocation.detach().to(torch.float64).cpu()
    predicted = route.raw_predicted_response.detach().to(torch.float64).cpu()
    response_star = route.semantic_response_star.detach().to(torch.float64).cpu()
    deficit = route.deficit.detach().to(torch.float64).cpu()
    if controller == "A0":
        eta_decision = calibration.eta
        eta_influence = 1
        calibrated = route.calibrated_predicted_response.detach().to(torch.float64).cpu()
    else:
        eta_decision = 1.0
        eta_influence = 0
        calibrated = predicted.clone()
    semantic_lower = response_star.clone()
    payload = {
        "arm": arm,
        "controller": controller,
        "allocation_sha256": tensor_sha256(allocation),
        "legacy_route_sha256": route.identity_sha256,
        "eta_observation": calibration.eta,
        "eta_decision": eta_decision,
        "exact_response_face_enabled": True,
    }
    return P2R6RoutingResult(
        arm=arm,
        controller=controller,
        preservation_objective="CAPACITY",
        scale_policy="P2R5_NORMALIZED_EXACT_RESPONSE",
        allocation=allocation,
        deficit=deficit,
        entry_deficit=entry_deficit.detach().to(torch.float64).cpu(),
        semantic_scale=deficit + P2R6_NUMERICAL_EPSILON,
        semantic_lower_bound=semantic_lower,
        raw_predicted_response=predicted,
        calibrated_predicted_response=calibrated,
        semantic_response_star=response_star,
        eta_observation=calibration.eta,
        eta_decision=eta_decision,
        eta_decision_influence_count=eta_influence,
        e1_xi=route.e1_xi,
        e2_enabled=True,
        exact_response_face_enabled=True,
        cumulative_capacity=route.cumulative_capacity,
        marginal_capacity=route.marginal_capacity,
        cumulative_structural_p=route.cumulative_structural_p,
        marginal_structural_p=route.marginal_structural_p,
        coefficient_mass_by_request=route.coefficient_mass_by_request,
        active_mass_constraints=route.active_mass_constraints,
        active_semantic_constraints=tuple(range(int(deficit.numel()))),
        feasible_rank=route.feasible_rank,
        feasible_nullity=route.feasible_nullity,
        semantic_region_max_violation=route.semantic_face_max_abs_residual,
        solver_receipts=route.solver_receipts,
        status=f"P2R6_{controller}_CAP_CERTIFIED",
        identity_sha256=canonical_hash(payload),
    )


def _semantic_region_optimum(
    response: np.ndarray,
    deficit: np.ndarray,
    entry_deficit: np.ndarray,
    *,
    scale_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    request_count, alpha_count = response.shape
    layer_count = alpha_count // request_count
    mass = _mass_matrix(layer_count, request_count)
    if scale_policy == "CURRENT_DEFICIT":
        scale = deficit + P2R6_NUMERICAL_EPSILON
    elif scale_policy == "ENTRY_ANCHORED_DEFICIT":
        scale = np.maximum(
            np.maximum(entry_deficit, deficit), P2R6_NUMERICAL_EPSILON
        )
    else:
        raise ODEBFContractError("P2R6 semantic scale policy differs")
    objective = np.zeros(alpha_count + 1, dtype=np.float64)
    objective[-1] = 1.0
    # The scientific objective is a dimensionless per-request ratio.  Encode
    # its LP rows in that coordinate as well, so HiGHS feasibility is not
    # applied to heterogeneous raw response units before certification.
    normalized_response = response / scale[:, None]
    normalized_deficit = deficit / scale
    a_ub = np.concatenate(
        (
            np.concatenate((mass, np.zeros((request_count, 1))), axis=1),
            np.concatenate(
                (-normalized_response, -np.ones((request_count, 1))), axis=1
            ),
            np.concatenate(
                (-normalized_response, np.zeros((request_count, 1))), axis=1
            ),
        ),
        axis=0,
    )
    b_ub = np.concatenate(
        (
            np.ones(request_count),
            -normalized_deficit,
            np.zeros(request_count),
        )
    )
    solved = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * alpha_count + [(0.0, None)],
        method="highs",
        options={
            "primal_feasibility_tolerance": P2R6_HIGHS_INTERNAL_TOLERANCE,
            "dual_feasibility_tolerance": P2R6_HIGHS_INTERNAL_TOLERANCE,
            "ipm_optimality_tolerance": P2R6_HIGHS_INTERNAL_TOLERANCE,
        },
    )
    if not solved.success or not np.all(np.isfinite(solved.x)):
        raise P2R6RoutingTechnicalError(
            "P2R6 E1 semantic-region solve failed",
            {
                "schema": "ode-edit-s05-p2r6-routing-technical/v1",
                "stage": "E1_SEMANTIC_REGION",
                "solver_status": int(solved.status),
                "message_sha256": canonical_hash(str(solved.message)),
                "scale_policy": scale_policy,
            },
        )
    allocation = np.asarray(solved.x[:alpha_count], dtype=np.float64)
    xi_solver = max(0.0, float(solved.x[-1]))
    semantic_response = np.asarray(response @ allocation, dtype=np.float64)
    # The contract defines xi as the max-ratio objective of alpha, not as the
    # epigraph helper variable returned by the LP backend.  Recompute the one
    # authoritative value from the returned primal allocation in FP64.  The
    # backend helper remains observation-only because HiGHS postsolve may leave
    # it below the objective of the returned allocation.
    xi_recomputed = max(
        0.0,
        float(np.max((deficit - semantic_response) / scale)),
    )
    xi_recertification_delta = max(0.0, xi_recomputed - xi_solver)
    xi = xi_recomputed
    lower = np.maximum(
        deficit - (xi + P2R6_NUMERICAL_EPSILON) * scale,
        0.0,
    )
    mass_violation = max(0.0, float(np.max(mass @ allocation - 1.0)))
    semantic_violation = max(0.0, float(np.max(lower - semantic_response)))
    semantic_nonnegative_violation = max(
        0.0, -float(np.min(semantic_response))
    )
    ratio_objective_violation = max(
        0.0,
        float(np.max((deficit - semantic_response) / scale - xi)),
    )
    negative_violation = max(0.0, -float(np.min(allocation)))
    certified = (
        max(
            mass_violation,
            semantic_nonnegative_violation,
            semantic_violation,
            ratio_objective_violation,
            negative_violation,
        )
        <= P2R6_NUMERICAL_EPSILON
    )
    receipt = {
        "schema": "ode-edit-s05-p2r6-e1-semantic-region/v1",
        "stage": "E1_SEMANTIC_REGION",
        "scale_policy": scale_policy,
        "solver_success": bool(solved.success),
        "solver_status": int(solved.status),
        "message_sha256": canonical_hash(str(solved.message)),
        "iterations": int(getattr(solved, "nit", -1)),
        "highs_internal_tolerance": P2R6_HIGHS_INTERNAL_TOLERANCE,
        "constraint_row_normalization": "DIVIDE_BY_SEMANTIC_SCALE",
        "semantic_scale_min": float(np.min(scale)),
        "semantic_scale_max": float(np.max(scale)),
        "xi_solver": xi_solver,
        "xi_solver_decision_influence_count": 0,
        "xi_recomputed": xi_recomputed,
        "xi_recertification_delta": xi_recertification_delta,
        "e1_xi": xi,
        "e1_xi_authority": P2R6_E1_XI_AUTHORITY,
        "mass_violation": mass_violation,
        "semantic_region_violation": semantic_violation,
        "semantic_response_nonnegative_violation": semantic_nonnegative_violation,
        "e1_semantic_nonnegative_constraint_enabled": True,
        "post_b_max_violation": semantic_violation,
        "e1_start_in_semantic_region": (
            semantic_violation <= P2R6_NUMERICAL_EPSILON
        ),
        "ratio_objective_violation": ratio_objective_violation,
        "negative_violation": negative_violation,
        "certificate_pass": certified,
        "primal_tolerance": P2R6_NUMERICAL_EPSILON,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    if not certified:
        raise P2R6RoutingTechnicalError(
            "P2R6 E1 semantic-region certificate failed", receipt
        )
    return allocation, scale, lower, xi, receipt


def _solve_region_quadratic(
    start: np.ndarray,
    objective_matrix: np.ndarray,
    objective_cross: np.ndarray,
    response: np.ndarray,
    lower: np.ndarray,
    *,
    stage: str,
    p_matrix: np.ndarray | None = None,
    p_cross: np.ndarray | None = None,
    p_limit: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dispatch the exact alpha-coordinate problem to the certified backend."""

    request_count, alpha_count = response.shape
    mass = _mass_matrix(alpha_count // request_count, request_count)
    tie = None
    if p_limit is not None:
        if p_matrix is None or p_cross is None:
            raise ODEBFContractError("P2R6 P tie geometry is absent")
        tie = ConvexQuadraticTie(
            np.asarray(p_matrix, dtype=np.float64),
            np.asarray(p_cross, dtype=np.float64),
            float(p_limit),
        )
    try:
        solved = solve_certified_semantic_region_qp(
            np.asarray(start, dtype=np.float64),
            np.asarray(objective_matrix, dtype=np.float64),
            np.asarray(objective_cross, dtype=np.float64),
            np.asarray(response, dtype=np.float64),
            np.asarray(lower, dtype=np.float64),
            mass,
            stage=stage,
            tie=tie,
        )
    except P2R6CertifiedQPError as exc:
        raise P2R6RoutingTechnicalError(
            f"P2R6 {stage} certified convex QP failed",
            exc.raw_free_receipt,
        ) from exc
    return solved.value, dict(solved.receipt)

def _region_geometry(
    allocation: np.ndarray,
    response: np.ndarray,
    lower: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    request_count, alpha_count = response.shape
    mass = _mass_matrix(alpha_count // request_count, request_count)
    mass_slack = 1.0 - mass @ allocation
    semantic_slack = response @ allocation - lower
    active_mass = tuple(
        int(i) for i in np.flatnonzero(mass_slack <= P2R6_NUMERICAL_EPSILON)
    )
    active_semantic = tuple(
        int(i) for i in np.flatnonzero(semantic_slack <= P2R6_NUMERICAL_EPSILON)
    )
    active_zero = tuple(
        int(i) for i in np.flatnonzero(allocation <= P2R6_NUMERICAL_EPSILON)
    )
    rows: list[np.ndarray] = [np.eye(alpha_count)[i] for i in active_zero]
    rows.extend(mass[i] for i in active_mass)
    rows.extend(response[i] for i in active_semantic)
    matrix = np.stack(rows) if rows else np.empty((0, alpha_count), dtype=np.float64)
    rank = int(np.linalg.matrix_rank(matrix, tol=P2R6_NUMERICAL_EPSILON))
    return active_mass, active_semantic, rank, alpha_count - rank


def solve_p2r6_routing(
    response: torch.Tensor,
    deficit: torch.Tensor,
    entry_deficit: torch.Tensor,
    calibration: SDRTCalibration,
    quadratics: SDRTQuadratics,
    *,
    arm: str,
) -> P2R6RoutingResult:
    """Solve one P2R6 causal arm with strict numerical certification."""

    if arm not in P2R6_ARMS:
        raise ODEBFContractError("P2R6 arm differs")
    observed = response.detach().to(device="cpu", dtype=torch.float64).numpy()
    deficit_np = deficit.detach().to(device="cpu", dtype=torch.float64).numpy()
    entry_np = entry_deficit.detach().to(device="cpu", dtype=torch.float64).numpy()
    if observed.shape != (10, 50) or deficit_np.shape != (10,) or entry_np.shape != (10,):
        raise ODEBFContractError("P2R6 response/deficit geometry differs")
    if not (
        np.all(np.isfinite(observed))
        and np.all(np.isfinite(deficit_np))
        and np.all(np.isfinite(entry_np))
    ):
        raise ODEBFContractError("P2R6 response/deficit is nonfinite")

    if arm == "A0-CAP":
        legacy = solve_sdrt_routing(response, deficit, calibration, quadratics, arm="SDRT-CAP")
        return _wrap_legacy_route(
            legacy,
            arm=arm,
            controller="A0",
            calibration=calibration,
            entry_deficit=entry_deficit,
        )
    if arm == "AETA-CAP":
        legacy = solve_sdrt_routing(
            response, deficit, _eta_one(calibration), quadratics, arm="SDRT-CAP"
        )
        return _wrap_legacy_route(
            legacy,
            arm=arm,
            controller="AETA",
            calibration=calibration,
            entry_deficit=entry_deficit,
        )

    controller = "AR" if arm.startswith("AR-") else "AS"
    preservation = "STRUCTP" if arm.endswith("STRUCTP") else "CAPACITY"
    scale_policy = (
        "CURRENT_DEFICIT" if controller == "AR" else "ENTRY_ANCHORED_DEFICIT"
    )
    semantic_start, scale, lower, xi, e1_receipt = _semantic_region_optimum(
        observed,
        deficit_np,
        entry_np,
        scale_policy=scale_policy,
    )
    capacity = quadratics.capacity_gram.detach().cpu().numpy()
    capacity_cross = quadratics.capacity_cross.detach().cpu().numpy()
    structural = quadratics.structural_p_gram.detach().cpu().numpy()
    structural_cross = quadratics.structural_p_cross.detach().cpu().numpy()
    if preservation == "CAPACITY":
        selected, objective_receipt = _solve_region_quadratic(
            semantic_start,
            capacity,
            capacity_cross,
            observed,
            lower,
            stage=f"{controller}_CAPACITY_IN_SEMANTIC_REGION",
        )
        receipts: tuple[Mapping[str, Any], ...] = (e1_receipt, objective_receipt)
        status = f"P2R6_{controller}_CAP_CERTIFIED"
    else:
        p_selected, p_receipt = _solve_region_quadratic(
            semantic_start,
            structural,
            structural_cross,
            observed,
            lower,
            stage=f"{controller}_STRUCTURAL_P_IN_SEMANTIC_REGION",
        )
        p_star = _quadratic_value(structural, structural_cross, p_selected)
        p_tie = SIMPLEX_XI_TIE_TOLERANCE * max(float(np.trace(structural)), 1.0e-12)
        selected, capacity_receipt = _solve_region_quadratic(
            p_selected,
            capacity,
            capacity_cross,
            observed,
            lower,
            stage=f"{controller}_CAPACITY_TIE_IN_STRUCTURAL_P_REGION",
            p_matrix=structural,
            p_cross=structural_cross,
            p_limit=p_star + p_tie,
        )
        receipts = (e1_receipt, p_receipt, capacity_receipt)
        status = f"P2R6_{controller}_STRUCTP_CERTIFIED"

    predicted = observed @ selected
    mass = _mass_matrix(5, 10) @ selected
    semantic_slack = predicted - lower
    negative_violation = max(0.0, -float(np.min(selected)))
    mass_violation = max(0.0, float(np.max(mass - 1.0)))
    semantic_violation = max(0.0, -float(np.min(semantic_slack)))
    if max(negative_violation, mass_violation, semantic_violation) > P2R6_NUMERICAL_EPSILON:
        raise P2R6RoutingTechnicalError(
            "P2R6 final routing certificate failed",
            {
                "schema": "ode-edit-s05-p2r6-routing-technical/v1",
                "stage": "FINAL_CERTIFICATE",
                "negative_violation": negative_violation,
                "mass_violation": mass_violation,
                "semantic_region_violation": semantic_violation,
                "primal_tolerance": P2R6_NUMERICAL_EPSILON,
            },
        )
    marginal_capacity = _quadratic_value(capacity, capacity_cross, selected)
    cumulative_capacity = max(0.0, quadratics.prior_capacity + marginal_capacity)
    marginal_p = _quadratic_value(structural, structural_cross, selected)
    cumulative_p = max(0.0, quadratics.prior_structural_p + marginal_p)
    active_mass, active_semantic, rank, nullity = _region_geometry(
        selected, observed, lower
    )
    allocation = torch.from_numpy(selected.reshape(5, 10)).to(torch.float64)
    predicted_tensor = torch.from_numpy(predicted).to(torch.float64)
    lower_tensor = torch.from_numpy(lower).to(torch.float64)
    scale_tensor = torch.from_numpy(scale).to(torch.float64)
    payload = {
        "arm": arm,
        "controller": controller,
        "preservation": preservation,
        "allocation_sha256": tensor_sha256(allocation),
        "deficit_sha256": tensor_sha256(deficit),
        "entry_deficit_sha256": tensor_sha256(entry_deficit),
        "lower_sha256": tensor_sha256(lower_tensor),
        "predicted_sha256": tensor_sha256(predicted_tensor),
        "eta_observation": calibration.eta,
        "eta_decision": 1.0,
        "e1_xi": xi,
        "cumulative_capacity": cumulative_capacity,
        "cumulative_p": cumulative_p,
        "status": status,
    }
    return P2R6RoutingResult(
        arm=arm,
        controller=controller,
        preservation_objective=preservation,
        scale_policy=scale_policy,
        allocation=allocation,
        deficit=deficit.detach().to(torch.float64).cpu(),
        entry_deficit=entry_deficit.detach().to(torch.float64).cpu(),
        semantic_scale=scale_tensor,
        semantic_lower_bound=lower_tensor,
        raw_predicted_response=predicted_tensor,
        calibrated_predicted_response=predicted_tensor.clone(),
        semantic_response_star=predicted_tensor.clone(),
        eta_observation=calibration.eta,
        eta_decision=1.0,
        eta_decision_influence_count=0,
        e1_xi=xi,
        e2_enabled=False,
        exact_response_face_enabled=False,
        cumulative_capacity=cumulative_capacity,
        marginal_capacity=marginal_capacity,
        cumulative_structural_p=cumulative_p,
        marginal_structural_p=marginal_p,
        coefficient_mass_by_request=tuple(float(item) for item in mass),
        active_mass_constraints=active_mass,
        active_semantic_constraints=active_semantic,
        feasible_rank=rank,
        feasible_nullity=nullity,
        semantic_region_max_violation=semantic_violation,
        solver_receipts=receipts,
        status=status,
        identity_sha256=canonical_hash(payload),
    )


def solve_p2r6_shadow_panel(
    response: torch.Tensor,
    deficit: torch.Tensor,
    entry_deficit: torch.Tensor,
    calibration: SDRTCalibration,
    quadratics: SDRTQuadratics,
    *,
    selected_arm: str,
    outer_step: int,
) -> P2R6ShadowPanel:
    """Solve the physical arm strictly and isolate every shadow failure."""

    if selected_arm not in P2R6_SHADOW_ARMS or outer_step < 0 or outer_step >= 8:
        raise ODEBFContractError("P2R6 selected shadow identity differs")
    solved_by_arm: dict[str, P2R6RoutingResult] = {}
    errors_by_arm: dict[str, Mapping[str, Any]] = {}

    # The selected physical solve is authoritative and fail-close.
    solved_by_arm[selected_arm] = solve_p2r6_routing(
        response,
        deficit,
        entry_deficit,
        calibration,
        quadratics,
        arm=selected_arm,
    )
    for shadow_arm in P2R6_SHADOW_ARMS:
        if shadow_arm == selected_arm:
            continue
        try:
            solved_by_arm[shadow_arm] = solve_p2r6_routing(
                response,
                deficit,
                entry_deficit,
                calibration,
                quadratics,
                arm=shadow_arm,
            )
        except Exception as exc:
            observability = getattr(exc, "raw_free_receipt", None)
            error = {
                "schema": "ode-edit-s05-p2r6-shadow-technical-error/v1",
                "selected_arm": selected_arm,
                "failing_shadow_arm": shadow_arm,
                "outer_step": outer_step,
                "exception_class": type(exc).__name__,
                "message_sha256": canonical_hash(str(exc)),
                "selected_action_decision_influence_count": 0,
            }
            if isinstance(observability, Mapping):
                error["technical_observability"] = dict(observability)
            error["identity_sha256"] = canonical_hash(error)
            errors_by_arm[shadow_arm] = error

    arm_results = tuple(
        P2R6ShadowArmResult(
            arm=arm,
            selected=arm == selected_arm,
            route=solved_by_arm.get(arm),
            technical_error=errors_by_arm.get(arm),
            status=(
                "SELECTED_RESULT"
                if arm == selected_arm
                else ("SHADOW_RESULT" if arm in solved_by_arm else "SHADOW_TECHNICAL_ERROR")
            ),
        )
        for arm in P2R6_SHADOW_ARMS
    )
    routes = tuple(
        solved_by_arm[arm] for arm in P2R6_SHADOW_ARMS if arm in solved_by_arm
    )
    pairwise: dict[str, float] = {}
    for left_index, left in enumerate(routes):
        for right in routes[left_index + 1 :]:
            pairwise[f"{left.arm}__{right.arm}"] = float(
                torch.linalg.vector_norm(left.allocation - right.allocation)
            )
    eta_distance = pairwise.get("A0-CAP__AETA-CAP")
    failing = tuple(arm for arm in P2R6_SHADOW_ARMS if arm in errors_by_arm)
    status = "SHADOW_TECHNICAL_INVALID" if failing else "SHADOW_PANEL_CERTIFIED"
    solver_call_count = sum(
        max(1, len(route.solver_receipts)) for route in solved_by_arm.values()
    ) + len(errors_by_arm)
    certificate_count = sum(
        sum(bool(item.get("certificate_pass", True)) for item in route.solver_receipts)
        for route in solved_by_arm.values()
    )
    payload = {
        "selected_arm": selected_arm,
        "outer_step": outer_step,
        "arm_status": {item.arm: item.status for item in arm_results},
        "route_sha256": [route.identity_sha256 for route in routes],
        "error_sha256": [
            errors_by_arm[arm]["identity_sha256"] for arm in sorted(errors_by_arm)
        ],
        "pairwise_allocation_l2": pairwise,
        "eta_on_off_allocation_l2": eta_distance,
        "failing_shadow_arms": list(failing),
        "status": status,
        "solver_call_count": solver_call_count,
        "certificate_count": certificate_count,
        "model_forward_count": 0,
        "model_backward_count": 0,
        "materialization_count": 0,
    }
    return P2R6ShadowPanel(
        selected_arm=selected_arm,
        outer_step=outer_step,
        arm_results=arm_results,
        routes=routes,
        pairwise_allocation_l2=pairwise,
        eta_on_off_allocation_l2=eta_distance,
        failing_shadow_arms=failing,
        status=status,
        solver_call_count=solver_call_count,
        certificate_count=certificate_count,
        model_forward_count=0,
        model_backward_count=0,
        materialization_count=0,
        identity_sha256=canonical_hash(payload),
    )

def p2r6_forbidden_influence_receipt() -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-s05-p2r6-forbidden-influence/v1",
        "clamp_on_decision_influence_count_per_target_microstep": 1,
        "clamp_off_access_count": 0,
        "eta_decision_influence_count_for_aeta_ar_as": 0,
        "semantic_debt_input_count": 0,
        "explicit_lag_input_count": 0,
        "remaining_horizon_division_count": 0,
        "rho_trust_strength_decision_count": 0,
        "hard_p_budget_influence_count": 0,
        "functional_p_veto_count": 0,
        "candidate_forward_count": 0,
        "candidate_materialization_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "historical_h_influence_count": 0,
        "sequential_state_count": 0,
        "shadow_model_forward_count": 0,
        "shadow_model_backward_count": 0,
        "shadow_materialization_count": 0,
        "outer_joint_materialization_count_per_step": 1,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "P2R6_ARMS",
    "P2R6_CAP_ARMS",
    "P2R6_INSTRUCTION_ID",
    "P2R6_METHOD_ID",
    "P2R6RoutingResult",
    "P2R6RoutingTechnicalError",
    "P2R6ShadowPanel",
    "p2r6_forbidden_influence_receipt",
    "solve_p2r6_routing",
    "solve_p2r6_shadow_panel",
]
