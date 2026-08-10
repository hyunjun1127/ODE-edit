"""P1R16 coupled-demand routing and per-request G-dual target control.

The controller keeps the P1R15 per-request Dynamic-z coordinate.  The writer
uses the coupled physical demand and one semantic W-only edit floor.  Soft
transport, P, and capacity are strictly lexicographic tie-breakers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .integrated_physical_writer import (
    PHYSICAL_WRITER_EPSILON_P,
    PHYSICAL_WRITER_GRID_COUNT,
    PHYSICAL_WRITER_H,
    PHYSICAL_WRITER_LAYER_ORDER,
)
from .integrated_physical_writer_solver import (
    PHYSICAL_SOLVER_FTOL,
    PHYSICAL_SOLVER_KKT_TOLERANCE,
    PHYSICAL_SOLVER_PRIMAL_TOLERANCE,
    PhysicalWriterConstraint,
    PhysicalWriterSolverCertificate,
    physical_writer_technical_constraints,
    solve_certified_physical_qp,
)
from .routing import RoutingProblem


P1R16_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-COUPLED-DEMAND-SOFT-TRANSPORT-P1R16-V1"
)
P1R16_AMENDMENT_ID = f"{P1R16_INSTRUCTION_ID}-A1"
P1R16_METHOD_ID = (
    "FIXED_E8_PERREQUEST_GDUAL_COUPLED_DEMAND_SOFT_TRANSPORT_V1"
)
P1R16_SCHEMA = "ode-edit-s05-p1r16-coupled-demand-soft-transport/v1"

P1R16_EPS_BASE = 1.0e-12
P1R16_EPS_LOSS = 1.0e-12
P1R16_EPS_DUAL = 1.0e-12
P1R16_EPS_ALPHA = 1.0e-12
P1R16_EPS_SPEED = 1.0e-8
P1R16_EPS_TRANSPORT_SCALE = 1.0e-12
P1R16_EPS_TRANSPORT_TIE = 1.0e-8
P1R16_EPS_P = PHYSICAL_WRITER_EPSILON_P
P1R16_G_PATH_TOLERANCE = 1.0e-8


class TargetStepStatus(str, Enum):
    TARGET_STEP = "TARGET_STEP"
    TARGET_GOAL_MET = "TARGET_GOAL_MET"
    TARGET_NO_DESCENT_DIRECTION = "TARGET_NO_DESCENT_DIRECTION"


class EditFloorComponent(str, Enum):
    NONE = "NONE"
    EDIT_NO_DIRECTION = "EDIT_NO_DIRECTION"


class CoupledStepStatus(str, Enum):
    JOINT_WRITE = "JOINT_WRITE"
    ZERO_WRITE_GOAL_MET = "ZERO_WRITE_GOAL_MET"
    COLD_TARGET_ONLY_RECOVERY = "COLD_TARGET_ONLY_RECOVERY"
    EDIT_NO_DIRECTION = "EDIT_NO_DIRECTION"


@dataclass(frozen=True, slots=True)
class PerRequestTargetStep:
    status: TargetStepStatus
    step_index: int
    target_state_sha256: str
    desired_target_sha256: str
    z_base_sha256: str
    loss_by_request: tuple[float, ...]
    base_norm_by_request: tuple[float, ...]
    gradient_norm_by_request: tuple[float, ...]
    dual_norm_by_request: tuple[float, ...]
    rho_by_request: tuple[float, ...]
    unit_g_norm_by_request: tuple[float, ...]
    step_g_path_by_request: tuple[float, ...]
    cumulative_g_path_before: tuple[float, ...]
    cumulative_g_path_after: tuple[float, ...]
    predicted_reduction_by_request: tuple[float, ...]
    controller_requested_reduction: float
    goal_met_mask: tuple[bool, ...]
    no_descent_mask: tuple[bool, ...]
    target_clock_advance_count: int
    target_clock_advance_eligibility_count: int
    h_application_count: int
    old_target_access_count: int
    heldout_access_count: int
    native_or_direct_z_access_count: int
    desired_target: torch.Tensor
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("desired_target")
        payload["status"] = self.status.value
        return payload


def build_per_request_gdual_target_step(
    *,
    target_state: torch.Tensor,
    z_base: torch.Tensor,
    loss_by_request: Sequence[float] | torch.Tensor,
    gradient_by_request: torch.Tensor,
    step_index: int,
    cumulative_g_path_before: Sequence[float],
) -> PerRequestTargetStep:
    """Build the frozen request-wise G-dual target step in FP64/FP32."""

    if step_index < 0 or step_index >= PHYSICAL_WRITER_GRID_COUNT:
        raise ODEBFContractError("P1R16 target step index differs")
    if (
        not isinstance(target_state, torch.Tensor)
        or not isinstance(z_base, torch.Tensor)
        or not isinstance(gradient_by_request, torch.Tensor)
        or target_state.ndim != 2
        or target_state.shape[1] != BATCH_SIZE
        or z_base.shape != target_state.shape
        or gradient_by_request.shape != target_state.shape
    ):
        raise ODEBFContractError("P1R16 target geometry differs")
    target64 = target_state.detach().to(device="cpu", dtype=torch.float64)
    base64 = z_base.detach().to(device="cpu", dtype=torch.float64)
    gradient64 = gradient_by_request.detach().to(
        device="cpu", dtype=torch.float64
    )
    losses = torch.as_tensor(
        tuple(float(item) for item in loss_by_request), dtype=torch.float64
    )
    cumulative_before = torch.as_tensor(
        tuple(float(item) for item in cumulative_g_path_before),
        dtype=torch.float64,
    )
    if (
        losses.shape != (BATCH_SIZE,)
        or cumulative_before.shape != (BATCH_SIZE,)
        or not bool(torch.isfinite(target64).all())
        or not bool(torch.isfinite(base64).all())
        or not bool(torch.isfinite(gradient64).all())
        or not bool(torch.isfinite(losses).all())
        or not bool(torch.isfinite(cumulative_before).all())
        or bool(torch.any(losses < 0.0))
        or bool(torch.any(cumulative_before < 0.0))
    ):
        raise ODEBFContractError("P1R16 target value is malformed")
    base_norm = torch.linalg.vector_norm(base64, dim=0)
    if not bool(torch.isfinite(base_norm).all()) or bool(
        torch.any(base_norm <= P1R16_EPS_BASE)
    ):
        raise ODEBFContractError("P1R16 target base metric is degenerate")
    gradient_norm = torch.linalg.vector_norm(gradient64, dim=0)
    dual_norm = base_norm * gradient_norm
    if not bool(torch.isfinite(dual_norm).all()):
        raise ODEBFContractError("P1R16 target dual norm is non-finite")

    goal_met = losses <= P1R16_EPS_LOSS
    no_descent = (~goal_met) & (dual_norm <= P1R16_EPS_DUAL)
    unit = torch.zeros_like(gradient64)
    rho = torch.zeros(BATCH_SIZE, dtype=torch.float64)
    active = (~goal_met) & (~no_descent)
    remaining = PHYSICAL_WRITER_GRID_COUNT - step_index
    for index in range(BATCH_SIZE):
        if not bool(active[index]):
            continue
        unit[:, index] = (
            -(base_norm[index] ** 2) * gradient64[:, index]
            / dual_norm[index]
        )
        rho[index] = min(
            1.0,
            float(
                losses[index]
                / (PHYSICAL_WRITER_H * remaining * dual_norm[index])
            ),
        )
    unit_g_norm = torch.linalg.vector_norm(unit, dim=0) / base_norm
    if bool(torch.any(torch.abs(unit_g_norm[active] - 1.0) > 1.0e-10)):
        raise ODEBFContractError("P1R16 target G-unit certificate differs")
    if bool(torch.any(unit_g_norm[~active] != 0.0)):
        raise ODEBFContractError("P1R16 inactive target unit differs")

    desired64 = target64 + PHYSICAL_WRITER_H * rho.unsqueeze(0) * unit
    desired = desired64.to(dtype=torch.float32).contiguous()
    step_path = PHYSICAL_WRITER_H * rho
    cumulative_after = cumulative_before + step_path
    if (
        bool(torch.any(step_path > PHYSICAL_WRITER_H + P1R16_G_PATH_TOLERANCE))
        or bool(torch.any(cumulative_after > 1.0 + P1R16_G_PATH_TOLERANCE))
    ):
        raise ODEBFContractError("P1R16 target G-path trust differs")
    predicted = PHYSICAL_WRITER_H * rho * dual_norm
    requested = float(torch.mean(predicted))
    status = (
        TargetStepStatus.TARGET_NO_DESCENT_DIRECTION
        if bool(torch.any(no_descent))
        else (
            TargetStepStatus.TARGET_GOAL_MET
            if bool(torch.all(goal_met))
            else TargetStepStatus.TARGET_STEP
        )
    )
    eligibility = 0 if status is TargetStepStatus.TARGET_NO_DESCENT_DIRECTION else 1
    payload = {
        "schema": f"{P1R16_SCHEMA}-target-step",
        "status": status.value,
        "step_index": step_index,
        "target_state_sha256": tensor_sha256(target_state),
        "desired_target_sha256": tensor_sha256(desired),
        "z_base_sha256": tensor_sha256(z_base),
        "loss_by_request": [float(item) for item in losses],
        "base_norm_by_request": [float(item) for item in base_norm],
        "gradient_norm_by_request": [float(item) for item in gradient_norm],
        "dual_norm_by_request": [float(item) for item in dual_norm],
        "rho_by_request": [float(item) for item in rho],
        "unit_g_norm_by_request": [float(item) for item in unit_g_norm],
        "step_g_path_by_request": [float(item) for item in step_path],
        "cumulative_g_path_before": [float(item) for item in cumulative_before],
        "cumulative_g_path_after": [float(item) for item in cumulative_after],
        "predicted_reduction_by_request": [float(item) for item in predicted],
        "controller_requested_reduction": requested,
        "goal_met_mask": [bool(item) for item in goal_met],
        "no_descent_mask": [bool(item) for item in no_descent],
        "target_clock_advance_count": 0,
        "target_clock_advance_eligibility_count": eligibility,
        "h_application_count": 1,
        "old_target_access_count": 0,
        "heldout_access_count": 0,
        "native_or_direct_z_access_count": 0,
    }
    return PerRequestTargetStep(
        status,
        step_index,
        payload["target_state_sha256"],
        payload["desired_target_sha256"],
        payload["z_base_sha256"],
        tuple(payload["loss_by_request"]),
        tuple(payload["base_norm_by_request"]),
        tuple(payload["gradient_norm_by_request"]),
        tuple(payload["dual_norm_by_request"]),
        tuple(payload["rho_by_request"]),
        tuple(payload["unit_g_norm_by_request"]),
        tuple(payload["step_g_path_by_request"]),
        tuple(payload["cumulative_g_path_before"]),
        tuple(payload["cumulative_g_path_after"]),
        tuple(payload["predicted_reduction_by_request"]),
        requested,
        tuple(payload["goal_met_mask"]),
        tuple(payload["no_descent_mask"]),
        0,
        eligibility,
        1,
        0,
        0,
        0,
        desired,
        canonical_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class CoupledDemandRoutingResult:
    status: CoupledStepStatus
    failure_component: EditFloorComponent
    step_index: int
    a_edit: tuple[float, ...]
    a_transport: tuple[float, ...]
    p_edit_max: float
    current_nohook_loss: float
    controller_requested_reduction: float
    delta_edit: float
    transport_loss_current: float
    transport_predicted_remaining: float | None
    transport_bar_selected: float | None
    transport_bar_star: float | None
    transport_active: bool
    velocity: tuple[float, ...]
    applied_theta: tuple[float, ...]
    speed: float
    alpha: tuple[float, ...] | None
    alpha_status: str
    stage1_velocity: tuple[float, ...] | None
    stage2_velocity: tuple[float, ...] | None
    stage3_velocity: tuple[float, ...] | None
    stage1_speed: float | None
    stage2_speed: float | None
    stage3_speed: float | None
    p_scale: float
    p_bar_star: float | None
    p_active: bool
    transport_allocation_influence: bool
    p_allocation_influence: bool
    speed_tie_slack: float | None
    transport_tie_slack: float | None
    p_tie_slack: float | None
    cold_entry_eligible: bool
    cold_target_only_recovery_count: int
    target_clock_advance_count: int
    weight_clock_advance_count: int
    candidate_count: int
    transport_guard_veto_termination_influence_count: int
    hard_h_p_influence_count: int
    solver_certificates: tuple[PhysicalWriterSolverCertificate, ...]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["failure_component"] = self.failure_component.value
        payload["solver_certificates"] = [
            item.raw_free_payload() for item in self.solver_certificates
        ]
        return payload


def _vector(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(tuple(float(item) for item in value), dtype=np.float64)
    if result.shape != (5,) or not np.all(np.isfinite(result)):
        raise ODEBFContractError(f"P1R16 {label} differs")
    return result


def _matrix(value: Sequence[Sequence[float]], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (5, 5)
        or not np.all(np.isfinite(result))
        or not np.allclose(result, result.T, rtol=0.0, atol=1.0e-12)
        or float(np.min(np.linalg.eigvalsh(result))) < -1.0e-10
    ):
        raise ODEBFContractError(f"P1R16 {label} differs")
    return result


def _alpha(value: np.ndarray) -> tuple[tuple[float, ...] | None, str]:
    speed = float(np.sum(value))
    if speed <= P1R16_EPS_ALPHA:
        return None, "UNDEFINED_ZERO_SPEED"
    alpha = value / speed
    if np.any(alpha < 0.0) or not math.isclose(
        float(np.sum(alpha)), 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ODEBFContractError("P1R16 alpha simplex differs")
    return tuple(float(item) for item in alpha), "DEFINED_POSITIVE_SPEED"


def _constraint(
    name: str,
    function: Callable[[np.ndarray], float],
    jacobian: Callable[[np.ndarray], np.ndarray],
    hessian: Callable[[np.ndarray], np.ndarray],
) -> PhysicalWriterConstraint:
    return PhysicalWriterConstraint(name, function, jacobian, hessian)


def _routing_result(
    *,
    status: CoupledStepStatus,
    failure: EditFloorComponent,
    step_index: int,
    edit: np.ndarray,
    transport: np.ndarray,
    p_edit: float,
    current_loss: float,
    requested: float,
    delta_edit: float,
    transport_loss_current: float,
    cold_eligible: bool,
    certificates: Sequence[PhysicalWriterSolverCertificate],
) -> CoupledDemandRoutingResult:
    zero = (0.0,) * 5
    recovery = int(status is CoupledStepStatus.COLD_TARGET_ONLY_RECOVERY)
    payload: dict[str, Any] = {
        "status": status,
        "failure_component": failure,
        "step_index": step_index,
        "a_edit": tuple(float(item) for item in edit),
        "a_transport": tuple(float(item) for item in transport),
        "p_edit_max": p_edit,
        "current_nohook_loss": current_loss,
        "controller_requested_reduction": requested,
        "delta_edit": delta_edit,
        "transport_loss_current": transport_loss_current,
        "transport_predicted_remaining": None,
        "transport_bar_selected": None,
        "transport_bar_star": None,
        "transport_active": bool(
            transport_loss_current > P1R16_EPS_TRANSPORT_SCALE
        ),
        "velocity": zero,
        "applied_theta": zero,
        "speed": 0.0,
        "alpha": None,
        "alpha_status": "UNDEFINED_ZERO_SPEED",
        "stage1_velocity": None,
        "stage2_velocity": None,
        "stage3_velocity": None,
        "stage1_speed": None,
        "stage2_speed": None,
        "stage3_speed": None,
        "p_scale": 0.0,
        "p_bar_star": None,
        "p_active": False,
        "transport_allocation_influence": False,
        "p_allocation_influence": False,
        "speed_tie_slack": None,
        "transport_tie_slack": None,
        "p_tie_slack": None,
        "cold_entry_eligible": cold_eligible,
        "cold_target_only_recovery_count": recovery,
        "target_clock_advance_count": recovery,
        "weight_clock_advance_count": 0,
        "candidate_count": 0,
        "transport_guard_veto_termination_influence_count": 0,
        "hard_h_p_influence_count": 0,
        "solver_certificates": tuple(certificates),
    }
    identity_payload = {
        **{
            key: value.value if isinstance(value, Enum) else value
            for key, value in payload.items()
            if key != "solver_certificates"
        },
        "solver_certificates": [item.raw_free_payload() for item in certificates],
    }
    return CoupledDemandRoutingResult(
        **payload, identity_sha256=canonical_hash(identity_payload)
    )


def solve_coupled_demand_soft_transport(
    problem: RoutingProblem,
    *,
    a_edit: Sequence[float],
    a_transport: Sequence[float],
    current_nohook_loss: float,
    controller_requested_reduction: float,
    transport_loss_current: float,
    step_index: int,
    functional_p_derivative: Sequence[float],
    structural_p_matrix_raw: Sequence[Sequence[float]],
    cold_entry: bool,
    target_equals_z_base: bool,
    old_residual_exact_zero: bool,
    pre_guard_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> CoupledDemandRoutingResult:
    """Solve edit floor then speed -> soft transport -> P -> capacity."""

    if step_index < 0 or step_index >= PHYSICAL_WRITER_GRID_COUNT:
        raise ODEBFContractError("P1R16 routing step differs")
    if (
        not math.isfinite(float(current_nohook_loss))
        or float(current_nohook_loss) < 0.0
        or not math.isfinite(float(controller_requested_reduction))
        or float(controller_requested_reduction) < 0.0
        or not math.isfinite(float(transport_loss_current))
        or float(transport_loss_current) < 0.0
    ):
        raise ODEBFContractError("P1R16 routing loss differs")
    edit = _vector(a_edit, "edit slope")
    transport = _vector(a_transport, "transport slope")
    p_derivative = _vector(functional_p_derivative, "P derivative")
    p_raw = _matrix(structural_p_matrix_raw, "structural P")
    edit_problem = replace(problem, signed_progress=edit)
    caps = np.minimum(problem.layer_caps, 1.0)
    lower = np.zeros(5, dtype=np.float64)
    technical = physical_writer_technical_constraints(
        edit_problem, requested_progress=None
    )
    zero_hessian = lambda value: np.zeros((5, 5), dtype=np.float64)  # noqa: E731
    certificates: list[PhysicalWriterSolverCertificate] = []

    max_edit_v, max_edit_cert = solve_certified_physical_qp(
        phase="p1r16-maximum-edit",
        problem=edit_problem,
        objective=lambda value: float(-edit @ value),
        jacobian=lambda value: -edit.copy(),
        hessian=zero_hessian,
        constraints=technical,
        lower=lower,
        upper=caps,
        initial=np.zeros(5, dtype=np.float64),
        p_max=0.0,
        requested_progress=0.0,
        authority_role="AUTHORITATIVE_EDIT_MAXIMUM",
    )
    certificates.append(max_edit_cert)
    p_edit = max(float(edit @ max_edit_v), 0.0)
    requested = float(controller_requested_reduction)
    delta_edit = min(PHYSICAL_WRITER_H * p_edit, requested) if requested > 0.0 else 0.0
    cold_eligible = bool(
        cold_entry
        and step_index == 0
        and target_equals_z_base
        and old_residual_exact_zero
        and requested > 0.0
        and p_edit <= 0.0
    )
    pre_guard = {
        "schema": f"{P1R16_SCHEMA}-pre-guard",
        "step_index": step_index,
        "a_edit": edit.tolist(),
        "a_transport": transport.tolist(),
        "p_edit_max": p_edit,
        "current_nohook_loss": float(current_nohook_loss),
        "controller_requested_reduction": requested,
        "transport_loss_current": float(transport_loss_current),
        "delta_edit": delta_edit,
        "remaining_step_division_count": 0,
        "old_residual_exact_zero": old_residual_exact_zero,
        "cold_entry_eligible": cold_eligible,
        "all_five_layer_domain": True,
        "negative_coefficient_allowed": False,
        "transport_hard_floor_count": 0,
        "transport_guard_veto_termination_influence_count": 0,
        "gamma_access_count": 0,
        "overlay_access_count": 0,
        "hard_h_p_influence_count": 0,
    }
    if pre_guard_observer is not None:
        pre_guard_observer(pre_guard)

    if requested > 0.0 and p_edit <= 0.0:
        return _routing_result(
            status=(
                CoupledStepStatus.COLD_TARGET_ONLY_RECOVERY
                if cold_eligible
                else CoupledStepStatus.EDIT_NO_DIRECTION
            ),
            failure=EditFloorComponent.EDIT_NO_DIRECTION,
            step_index=step_index,
            edit=edit,
            transport=transport,
            p_edit=p_edit,
            current_loss=float(current_nohook_loss),
            requested=requested,
            delta_edit=delta_edit,
            transport_loss_current=float(transport_loss_current),
            cold_eligible=cold_eligible,
            certificates=certificates,
        )

    floor_constraints: list[PhysicalWriterConstraint] = []
    if delta_edit > 0.0:
        floor_constraints.append(
            _constraint(
                "semantic_physical_edit_floor",
                lambda value: float(PHYSICAL_WRITER_H * edit @ value - delta_edit),
                lambda value: PHYSICAL_WRITER_H * edit.copy(),
                zero_hessian,
            )
        )
    feasible = [*technical, *floor_constraints]
    ones = np.ones(5, dtype=np.float64)
    stage1, speed_cert = solve_certified_physical_qp(
        phase="p1r16-stage1-minimum-global-speed",
        problem=edit_problem,
        objective=lambda value: float(ones @ value),
        jacobian=lambda value: ones.copy(),
        hessian=zero_hessian,
        constraints=feasible,
        lower=lower,
        upper=caps,
        initial=max_edit_v if delta_edit > 0.0 else np.zeros(5, dtype=np.float64),
        p_max=p_edit,
        requested_progress=delta_edit,
        authority_role="AUTHORITATIVE_MINIMUM_L1_SPEED",
    )
    certificates.append(speed_cert)
    speed_star = float(np.sum(stage1))
    speed_tie = _constraint(
        "global_speed_tie",
        lambda value: float(speed_star + P1R16_EPS_SPEED - np.sum(value)),
        lambda value: -ones.copy(),
        zero_hessian,
    )

    transport_loss = float(transport_loss_current)
    transport_active = transport_loss > P1R16_EPS_TRANSPORT_SCALE

    def transport_bar(value: np.ndarray) -> float:
        if not transport_active:
            return 0.0
        predicted = transport_loss - PHYSICAL_WRITER_H * float(transport @ value)
        return float((predicted / transport_loss) ** 2)

    def transport_gradient(value: np.ndarray) -> np.ndarray:
        if not transport_active:
            return np.zeros(5, dtype=np.float64)
        predicted = transport_loss - PHYSICAL_WRITER_H * float(transport @ value)
        return (
            -2.0
            * PHYSICAL_WRITER_H
            * predicted
            * transport
            / (transport_loss**2)
        )

    def transport_hessian(value: np.ndarray) -> np.ndarray:
        del value
        if not transport_active:
            return np.zeros((5, 5), dtype=np.float64)
        return (
            2.0
            * PHYSICAL_WRITER_H**2
            * np.outer(transport, transport)
            / (transport_loss**2)
        )

    stage2, transport_cert = solve_certified_physical_qp(
        phase="p1r16-stage2-soft-transport-within-speed-tie",
        problem=edit_problem,
        objective=transport_bar,
        jacobian=transport_gradient,
        hessian=transport_hessian,
        constraints=[*feasible, speed_tie],
        lower=lower,
        upper=caps,
        initial=stage1,
        p_max=p_edit,
        requested_progress=delta_edit,
        authority_role="AUTHORITATIVE_SOFT_TRANSPORT_TIE_STAGE",
    )
    certificates.append(transport_cert)
    transport_star = transport_bar(stage2)
    transport_tie = _constraint(
        "dimensionless_soft_transport_tie",
        lambda value: float(
            transport_star + P1R16_EPS_TRANSPORT_TIE - transport_bar(value)
        ),
        lambda value: -transport_gradient(value),
        lambda value: -transport_hessian(value),
    )

    pi = PHYSICAL_WRITER_H * np.maximum(p_derivative, 0.0)
    m_step = PHYSICAL_WRITER_H**2 * p_raw
    p_scale = float(pi @ caps + caps @ np.abs(m_step) @ caps)
    p_active = p_scale > 0.0

    def p_bar(value: np.ndarray) -> float:
        return 0.0 if not p_active else float((pi @ value + value @ m_step @ value) / p_scale)

    def p_gradient(value: np.ndarray) -> np.ndarray:
        return np.zeros_like(value) if not p_active else (pi + 2.0 * m_step @ value) / p_scale

    def p_hessian(value: np.ndarray) -> np.ndarray:
        del value
        return np.zeros_like(m_step) if not p_active else 2.0 * m_step / p_scale

    stage3, p_cert = solve_certified_physical_qp(
        phase="p1r16-stage3-minimum-P-within-speed-transport-ties",
        problem=edit_problem,
        objective=p_bar,
        jacobian=p_gradient,
        hessian=p_hessian,
        constraints=[*feasible, speed_tie, transport_tie],
        lower=lower,
        upper=caps,
        initial=stage2,
        p_max=p_edit,
        requested_progress=delta_edit,
        authority_role="AUTHORITATIVE_P_TIE_STAGE",
    )
    certificates.append(p_cert)
    p_star = p_bar(stage3)
    p_tie = _constraint(
        "dimensionless_P_tie",
        lambda value: float(p_star + P1R16_EPS_P - p_bar(value)),
        lambda value: -p_gradient(value),
        lambda value: -p_hessian(value),
    )
    capacity = problem.capacity_metric
    stage4, capacity_cert = solve_certified_physical_qp(
        phase="p1r16-stage4-minimum-capacity-within-all-ties",
        problem=edit_problem,
        objective=lambda value: float(0.5 * value @ capacity @ value),
        jacobian=lambda value: capacity @ value,
        hessian=lambda value: capacity.copy(),
        constraints=[*feasible, speed_tie, transport_tie, p_tie],
        lower=lower,
        upper=caps,
        initial=stage3,
        p_max=p_edit,
        requested_progress=delta_edit,
        authority_role="AUTHORITATIVE_CAPACITY_TIE_STAGE",
    )
    certificates.append(capacity_cert)
    speed = float(np.sum(stage4))
    alpha, alpha_status = _alpha(stage4)
    alpha1, _ = _alpha(stage1)
    alpha2, _ = _alpha(stage2)
    alpha3, _ = _alpha(stage3)
    transport_influence = bool(
        alpha1 is not None
        and alpha2 is not None
        and float(np.sum(np.abs(np.asarray(alpha2) - np.asarray(alpha1))))
        > P1R16_EPS_ALPHA
    )
    p_influence = bool(
        p_active
        and alpha2 is not None
        and alpha3 is not None
        and float(np.sum(np.abs(np.asarray(alpha3) - np.asarray(alpha2))))
        > P1R16_EPS_ALPHA
    )
    theta = PHYSICAL_WRITER_H * stage4
    zero_write = bool(np.count_nonzero(stage4) == 0)
    if zero_write and requested > 0.0:
        raise ODEBFContractError("P1R16 positive semantic request selected zero write")
    status = CoupledStepStatus.ZERO_WRITE_GOAL_MET if zero_write else CoupledStepStatus.JOINT_WRITE
    payload: dict[str, Any] = {
        "status": status,
        "failure_component": EditFloorComponent.NONE,
        "step_index": step_index,
        "a_edit": tuple(float(item) for item in edit),
        "a_transport": tuple(float(item) for item in transport),
        "p_edit_max": p_edit,
        "current_nohook_loss": float(current_nohook_loss),
        "controller_requested_reduction": requested,
        "delta_edit": delta_edit,
        "transport_loss_current": transport_loss,
        "transport_predicted_remaining": (
            transport_loss - PHYSICAL_WRITER_H * float(transport @ stage4)
        ),
        "transport_bar_selected": transport_bar(stage4),
        "transport_bar_star": transport_star,
        "transport_active": transport_active,
        "velocity": tuple(float(item) for item in stage4),
        "applied_theta": tuple(float(item) for item in theta),
        "speed": speed,
        "alpha": alpha,
        "alpha_status": alpha_status,
        "stage1_velocity": tuple(float(item) for item in stage1),
        "stage2_velocity": tuple(float(item) for item in stage2),
        "stage3_velocity": tuple(float(item) for item in stage3),
        "stage1_speed": speed_star,
        "stage2_speed": float(np.sum(stage2)),
        "stage3_speed": float(np.sum(stage3)),
        "p_scale": p_scale,
        "p_bar_star": p_star,
        "p_active": p_active,
        "transport_allocation_influence": transport_influence,
        "p_allocation_influence": p_influence,
        "speed_tie_slack": float(speed_star + P1R16_EPS_SPEED - speed),
        "transport_tie_slack": float(
            transport_star + P1R16_EPS_TRANSPORT_TIE - transport_bar(stage4)
        ),
        "p_tie_slack": float(p_star + P1R16_EPS_P - p_bar(stage4)),
        "cold_entry_eligible": False,
        "cold_target_only_recovery_count": 0,
        "target_clock_advance_count": 1,
        "weight_clock_advance_count": 1,
        "candidate_count": 0 if zero_write else 1,
        "transport_guard_veto_termination_influence_count": 0,
        "hard_h_p_influence_count": 0,
        "solver_certificates": tuple(certificates),
    }
    identity_payload = {
        **{
            key: value.value if isinstance(value, Enum) else value
            for key, value in payload.items()
            if key != "solver_certificates"
        },
        "solver_certificates": [item.raw_free_payload() for item in certificates],
    }
    return CoupledDemandRoutingResult(
        **payload, identity_sha256=canonical_hash(identity_payload)
    )


def p1r16_source_contract() -> dict[str, Any]:
    payload = {
        "schema": f"{P1R16_SCHEMA}-source-contract",
        "instruction_id": P1R16_INSTRUCTION_ID,
        "method_id": P1R16_METHOD_ID,
        "grid_count": PHYSICAL_WRITER_GRID_COUNT,
        "h": PHYSICAL_WRITER_H,
        "target_metric": "G_i=I/||z_base_i||_2^2",
        "target_direction": "-G_i^-1*g_i/||g_i||_(G_i^-1)",
        "eps_base": P1R16_EPS_BASE,
        "eps_loss": P1R16_EPS_LOSS,
        "eps_dual": P1R16_EPS_DUAL,
        "eps_alpha": P1R16_EPS_ALPHA,
        "eps_speed": P1R16_EPS_SPEED,
        "eps_transport_scale": P1R16_EPS_TRANSPORT_SCALE,
        "eps_transport_tie": P1R16_EPS_TRANSPORT_TIE,
        "eps_P": P1R16_EPS_P,
        "coupled_factor_demand": "(z_des-H_T(W_k))/h",
        "coefficient_coordinate": "W(theta)=W+sum_l(theta_l*B_l)",
        "applied_coefficient": "theta=h*v",
        "edit_floor": "min(h*p_edit_max,controller_requested_reduction)",
        "remaining_step_division_count_in_writer_floor": 0,
        "transport_predicted_remaining": "T0-h*a_transport^T*v",
        "transport_objective": "(T_pred/T0)^2",
        "transport_zero_scale": "T0<=eps_transport_scale_implies_zero_inactive",
        "transport_hard_floor_count": 0,
        "lexicographic_stages": [
            "MIN_SPEED",
            "SOFT_TRANSPORT",
            "MIN_P_BAR",
            "MIN_CAPACITY",
        ],
        "online_logical_groups_per_step": 13,
        "online_logical_groups_k8": 104,
        "model_forwards_per_full_field": 18,
        "backwards_per_full_field": 22,
        "stage_b_material_count": 0,
        "native_or_direct_z_controller_access_count": 0,
        "hard_h_p_veto_retry_count": 0,
        "solver_ftol": PHYSICAL_SOLVER_FTOL,
        "solver_primal_tolerance": PHYSICAL_SOLVER_PRIMAL_TOLERANCE,
        "solver_kkt_tolerance": PHYSICAL_SOLVER_KKT_TOLERANCE,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "CoupledDemandRoutingResult",
    "CoupledStepStatus",
    "EditFloorComponent",
    "P1R16_EPS_ALPHA",
    "P1R16_EPS_BASE",
    "P1R16_EPS_DUAL",
    "P1R16_EPS_LOSS",
    "P1R16_EPS_P",
    "P1R16_EPS_SPEED",
    "P1R16_EPS_TRANSPORT_SCALE",
    "P1R16_EPS_TRANSPORT_TIE",
    "P1R16_AMENDMENT_ID",
    "P1R16_INSTRUCTION_ID",
    "P1R16_METHOD_ID",
    "P1R16_SCHEMA",
    "PerRequestTargetStep",
    "TargetStepStatus",
    "build_per_request_gdual_target_step",
    "p1r16_source_contract",
    "solve_coupled_demand_soft_transport",
]
