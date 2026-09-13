"""FzCB predictor-corrector and initial comparator arm execution."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from typing import Any, Callable

import torch

from .contracts import Arm, NumericalLock, ScientificBoundary
from .geometry import FullModelControlOperator, MEMITGeometryFactory
from .hashing import canonical_hash, tensor_sha256
from .linear import (
    dimension_relative_floor,
    equality_null_projection,
    scalar_rectification,
    solve_minimum_action,
    suffix_value,
)
from .transaction import AtomicEditTransaction, WeightSnapshot


class CandidateFailure(ScientificBoundary):
    """Finite predictor/corrector candidate may be retried at a smaller step."""


def _seed(namespace: str, arm: Arm, axis: int) -> int:
    body = f"{NumericalLock().seed_namespace}|{namespace}|{arm.value}|{axis}"
    return int(hashlib.sha256(body.encode()).hexdigest()[:16], 16) % (2**31)


def _same_basis(model: Any, source: FullModelControlOperator) -> FullModelControlOperator:
    return FullModelControlOperator(
        model, source.batch, source.z_layer, source.weight_names,
        source.right_factors, source.direct_gram_floor,
    )


def _solve(operator: FullModelControlOperator, rhs: torch.Tensor, tolerance: float) -> Any:
    lock = NumericalLock()
    return solve_minimum_action(
        operator, rhs.detach().float(), tolerance=tolerance,
        maximum_iterations=lock.cg_max_iterations,
        refinement_count=lock.cg_internal_refinement_count,
        preconditioner=operator.output_preconditioner,
    )


def _orthonormal_null_axes(
    operator: FullModelControlOperator, count: int, namespace: str, arm: Arm, tolerance: float
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    lock = NumericalLock()
    axes: list[torch.Tensor] = []
    receipts = []
    attempts = 0
    while len(axes) < count and attempts < count * 4:
        generator = torch.Generator().manual_seed(_seed(namespace, arm, attempts))
        seed = torch.randn(operator.coefficient_dimension, generator=generator, dtype=torch.float32).to(operator.primals[0].device)
        projected, receipt = equality_null_projection(
            operator, seed, tolerance=tolerance,
            maximum_iterations=lock.cg_max_iterations,
            refinement_count=lock.cg_internal_refinement_count,
            preconditioner=operator.output_preconditioner,
        )
        for previous in axes:
            projected = projected - torch.dot(previous, projected) * previous
        norm = torch.linalg.vector_norm(projected)
        if float(norm.item()) > tolerance:
            projected = (projected / norm).detach()
            residual = float(torch.linalg.vector_norm(operator.apply(projected)).item())
            if residual <= tolerance:
                axes.append(projected)
                receipts.append({"axis": len(axes) - 1, "seed": _seed(namespace, arm, attempts), **receipt, "post_orthogonal_null_residual": residual})
        attempts += 1
    if len(axes) != count:
        raise ScientificBoundary(f"effective equality-null authority below reduced prototype dimension: {len(axes)}/{count}")
    return axes, receipts


def _suffix_at_perturbation(
    *, model: Any, factory: MEMITGeometryFactory, snapshot: WeightSnapshot,
    entry_operator: FullModelControlOperator, direction: torch.Tensor, scale: float,
    target: torch.Tensor, remaining_time: float, tolerance: float,
) -> tuple[float, dict[str, Any]]:
    snapshot.restore(model)
    snapshot.set_displacement(model, entry_operator.coefficient_to_tangents(direction), scale)
    perturbed, geometry = factory.build()
    residual = target - perturbed.phi()
    solution = _solve(perturbed, residual, tolerance)
    value = suffix_value(solution, remaining_time)
    snapshot.restore(model)
    return value, {
        "scale": scale,
        "geometry_identity": geometry["identity"],
        "range_residual": solution.range_residual,
        "suffix_value": value,
        "operator_counts": perturbed.receipt(),
    }


def barrier_control(
    *, model: Any, factory: MEMITGeometryFactory, operator: FullModelControlOperator,
    snapshot: WeightSnapshot, equality: Any, target: torch.Tensor, s: float,
    spent_action: float, initial_budget: float, namespace: str, tolerance: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    lock = NumericalLock()
    remaining = 1.0 - s
    if remaining <= 0:
        raise ScientificBoundary("barrier requested at terminal state")
    residual = target - operator.phi()
    suffix_solution = _solve(operator, residual, tolerance)
    value = suffix_value(suffix_solution, remaining)
    slack = initial_budget - spent_action - value
    axes, axis_receipts = _orthonormal_null_axes(
        operator, lock.reduced_null_axes, namespace, Arm.FZCB, tolerance
    )
    epsilon = lock.finite_difference_step
    plus, plus_receipt = _suffix_at_perturbation(
        model=model, factory=factory, snapshot=snapshot, entry_operator=operator,
        direction=equality.coefficient, scale=epsilon, target=target,
        remaining_time=remaining, tolerance=tolerance,
    )
    minus, minus_receipt = _suffix_at_perturbation(
        model=model, factory=factory, snapshot=snapshot, entry_operator=operator,
        direction=equality.coefficient, scale=-epsilon, target=target,
        remaining_time=remaining, tolerance=tolerance,
    )
    g_eq = (plus - minus) / (2.0 * epsilon)
    g_free_values = []
    finite_difference_receipts = []
    for ordinal, axis in enumerate(axes):
        positive, positive_receipt = _suffix_at_perturbation(
            model=model, factory=factory, snapshot=snapshot, entry_operator=operator,
            direction=axis, scale=epsilon, target=target,
            remaining_time=remaining, tolerance=tolerance,
        )
        negative, negative_receipt = _suffix_at_perturbation(
            model=model, factory=factory, snapshot=snapshot, entry_operator=operator,
            direction=axis, scale=-epsilon, target=target,
            remaining_time=remaining, tolerance=tolerance,
        )
        derivative = (positive - negative) / (2.0 * epsilon)
        g_free_values.append(derivative)
        finite_difference_receipts.append({
            "axis": ordinal, "positive": positive_receipt,
            "negative": negative_receipt, "derivative": derivative,
        })
    snapshot.restore(model)
    g_free = torch.tensor(g_free_values, dtype=torch.float32, device=equality.coefficient.device)
    partial_s = value / remaining
    psi0 = 0.5 * equality.action_squared + g_eq + partial_s - lock.kappa * slack
    y, rectification = scalar_rectification(psi0, g_free, tolerance=tolerance)
    corrected = equality.coefficient.clone()
    for scalar, axis in zip(y, axes, strict=True):
        corrected = corrected + scalar * axis
    equality_residual = float(torch.linalg.vector_norm(operator.apply(corrected) - operator.apply(equality.coefficient)).item())
    if equality_residual > tolerance:
        raise ScientificBoundary("barrier correction left equality-null space")
    return corrected.detach(), {
        "sensitivity_method": lock.sensitivity_method,
        "remaining_time": remaining,
        "suffix_value": value,
        "spent_action": spent_action,
        "initial_budget": initial_budget,
        "h_cc": slack,
        "partial_s_suffix": partial_s,
        "g_eq": g_eq,
        "g_free": g_free_values,
        "axis_receipts": axis_receipts,
        "finite_difference_equality_direction": {"positive": plus_receipt, "negative": minus_receipt},
        "finite_difference_axes": finite_difference_receipts,
        "rectification": rectification,
        "equality_null_residual": equality_residual,
        "correction_h_norm": float(torch.linalg.vector_norm(corrected - equality.coefficient).item()),
    }


def _corrected_candidate(
    *, model: Any, operator: FullModelControlOperator, entry: WeightSnapshot,
    coefficient: torch.Tensor, target_waypoint: torch.Tensor, delta_s: float,
    tolerance: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    lock = NumericalLock()
    value = coefficient.detach().clone()
    correctors = []
    for ordinal in range(lock.corrector_iterations):
        entry.set_displacement(model, operator.coefficient_to_tangents(value), delta_s)
        candidate_operator = _same_basis(model, operator)
        residual = target_waypoint - candidate_operator.phi()
        absolute_before = float(torch.linalg.vector_norm(residual).item())
        if absolute_before <= tolerance * (1.0 + float(torch.linalg.vector_norm(target_waypoint).item())):
            correctors.append({"iteration": ordinal + 1, "residual_before": absolute_before, "skipped_zero_within_lock": True})
            continue
        try:
            correction = _solve(candidate_operator, residual / delta_s, tolerance)
        except ScientificBoundary as exc:
            raise CandidateFailure(f"candidate corrector range failure: {exc}") from exc
        value = (value + correction.coefficient).detach()
        correctors.append({
            "iteration": ordinal + 1,
            "residual_before": absolute_before,
            "correction_action_norm": math.sqrt(correction.action_squared),
            "range_residual": correction.range_residual,
            "solver": asdict(correction.receipt),
            "operator_counts": candidate_operator.receipt(),
        })
    entry.set_displacement(model, operator.coefficient_to_tangents(value), delta_s)
    final_operator = _same_basis(model, operator)
    phi = final_operator.phi()
    absolute = float(torch.linalg.vector_norm(phi - target_waypoint).item())
    relative = absolute / (float(torch.linalg.vector_norm(target_waypoint).item()) + torch.finfo(torch.float32).eps)
    if absolute > tolerance * (1.0 + float(torch.linalg.vector_norm(target_waypoint).item())):
        raise CandidateFailure(f"nonlinear corrector closure failure absolute={absolute} relative={relative}")
    return value, {
        "iterations": correctors,
        "final_absolute_closure": absolute,
        "final_relative_closure": relative,
        "corrected_coefficient_sha256": tensor_sha256(value),
        "corrected_action_squared": float(torch.dot(value, value).item()),
        "corrector_action_included": True,
        "operator_counts": final_operator.receipt(),
    }


def _initial_budget(operator: FullModelControlOperator, target: torch.Tensor, tolerance: float) -> tuple[float, dict[str, Any]]:
    residual = target - operator.phi()
    solution = _solve(operator, residual, tolerance)
    value = suffix_value(solution, 1.0)
    return value, {
        "A0": value,
        "range_residual": solution.range_residual,
        "action_squared": solution.action_squared,
        "solver": asdict(solution.receipt),
    }


def run_path_arm(
    *, arm: Arm, model: Any, factory: MEMITGeometryFactory,
    w0: WeightSnapshot, target: torch.Tensor, namespace: str,
) -> dict[str, Any]:
    if arm not in {Arm.FROZEN_SPLIT, Arm.REFRESHED_EQUALITY_ONLY, Arm.FZCB, Arm.STATIC_PATH}:
        raise ScientificBoundary(f"controller received unsupported arm {arm}")
    lock = NumericalLock()
    w0.restore(model)
    transaction = AtomicEditTransaction(model, w0)
    initial_operator, initial_geometry = factory.build()
    phi0 = initial_operator.phi()
    displacement = target - phi0
    tolerance = dimension_relative_floor(initial_operator.output_dimension)
    initial_budget, budget_receipt = _initial_budget(initial_operator, target, tolerance)
    frozen_basis = initial_operator if arm in {Arm.FROZEN_SPLIT, Arm.STATIC_PATH} else None
    schedule = [(0.0, 1.0)] if arm is Arm.STATIC_PATH else list(zip(lock.progress_grid[:-1], lock.progress_grid[1:], strict=True))
    spent = 0.0
    waypoints = []
    accepted_s = 0.0
    try:
        while schedule:
            scheduled_start, scheduled_end = schedule.pop(0)
            if abs(scheduled_start - accepted_s) > 1e-12:
                raise ScientificBoundary("accepted progress/schedule identity mismatch")
            proposed = scheduled_end - scheduled_start
            attempt = 0
            while True:
                delta_s = proposed * (lock.backtrack_factor ** attempt)
                next_s = accepted_s + delta_s
                if next_s > scheduled_end + 1e-12:
                    raise ScientificBoundary("backtracking progress overflow")
                entry = WeightSnapshot.capture(model, w0.names)
                operator, geometry = (
                    (_same_basis(model, frozen_basis), initial_geometry)
                    if frozen_basis is not None
                    else factory.build()
                )
                phi_entry = operator.phi()
                waypoint = phi0 + next_s * displacement
                rhs = (waypoint - phi_entry) / delta_s
                equality = _solve(operator, rhs, tolerance)
                barrier_receipt = None
                coefficient = equality.coefficient
                if arm is Arm.FZCB:
                    coefficient, barrier_receipt = barrier_control(
                        model=model, factory=factory, operator=operator, snapshot=entry,
                        equality=equality, target=target, s=accepted_s,
                        spent_action=spent, initial_budget=initial_budget,
                        namespace=f"{namespace}|s={accepted_s:.8f}", tolerance=tolerance,
                    )
                try:
                    corrected, corrector = _corrected_candidate(
                        model=model, operator=operator, entry=entry,
                        coefficient=coefficient, target_waypoint=waypoint,
                        delta_s=delta_s, tolerance=tolerance,
                    )
                    step_action = 0.5 * delta_s * float(torch.dot(corrected, corrected).item())
                    candidate_snapshot = WeightSnapshot.capture(model, w0.names)
                    refreshed, refreshed_geometry = factory.build()
                    refreshed_phi = refreshed.phi()
                    next_viability = None
                    if next_s < 1.0 - 1e-12:
                        try:
                            suffix_solution = _solve(refreshed, target - refreshed_phi, tolerance)
                        except ScientificBoundary as exc:
                            raise CandidateFailure(f"candidate suffix range failure: {exc}") from exc
                        candidate_suffix = suffix_value(suffix_solution, 1.0 - next_s)
                    else:
                        terminal_absolute = float(torch.linalg.vector_norm(refreshed_phi - target).item())
                        if terminal_absolute > tolerance * (1.0 + float(torch.linalg.vector_norm(target).item())):
                            raise CandidateFailure("terminal fixed-z closure failed")
                        suffix_solution = None
                        candidate_suffix = 0.0
                    candidate_total = spent + step_action + candidate_suffix
                    budget_error = candidate_total - initial_budget
                    if arm is Arm.FZCB and budget_error > tolerance * (1.0 + initial_budget):
                        raise CandidateFailure(f"actual completion budget failure: {budget_error}")
                    if arm is Arm.FZCB and next_s < 1.0 - 1e-12:
                        try:
                            next_equality = _solve(refreshed, displacement, tolerance)
                            _, next_viability = barrier_control(
                                model=model, factory=factory, operator=refreshed,
                                snapshot=candidate_snapshot, equality=next_equality,
                                target=target, s=next_s,
                                spent_action=spent + step_action,
                                initial_budget=initial_budget,
                                namespace=f"{namespace}|next-s={next_s:.8f}",
                                tolerance=tolerance,
                            )
                            candidate_snapshot.restore(model)
                        except ScientificBoundary as exc:
                            candidate_snapshot.restore(model)
                            raise CandidateFailure(f"candidate next-state local viability failure: {exc}") from exc
                    waypoints.append({
                        "ordinal": len(waypoints), "scheduled_start": scheduled_start,
                        "scheduled_end": scheduled_end, "s_entry": accepted_s,
                        "s_accepted": next_s, "delta_s": delta_s, "backtrack_count": attempt,
                        "phi_entry_sha256": tensor_sha256(phi_entry),
                        "target_waypoint_sha256": tensor_sha256(waypoint),
                        "geometry": geometry, "refreshed_geometry": refreshed_geometry,
                        "equality": {
                            "range_residual": equality.range_residual,
                            "action_squared": equality.action_squared,
                            "solver": asdict(equality.receipt),
                        },
                        "barrier": barrier_receipt,
                        "next_state_viability": next_viability,
                        "corrector": corrector,
                        "step_action": step_action,
                        "spent_action_after": spent + step_action,
                        "actual_suffix_after": candidate_suffix,
                        "spent_plus_suffix": candidate_total,
                        "budget_error": budget_error,
                        "candidate_root": candidate_snapshot.root,
                        "candidate_operator_counts": refreshed.receipt(),
                    })
                    spent += step_action
                    accepted_s = next_s
                    break
                except CandidateFailure:
                    entry.restore(model)
                    if attempt >= lock.maximum_backtracks:
                        raise
                    attempt += 1
            if accepted_s < scheduled_end - 1e-12:
                # Finish a scheduled interval after a reduced accepted step.
                schedule.insert(0, (accepted_s, scheduled_end))
        if abs(accepted_s - 1.0) > 1e-12:
            raise ScientificBoundary(f"terminal progress is not s=1: {accepted_s}")
        terminal = WeightSnapshot.capture(model, w0.names)
        terminal_operator, terminal_geometry = factory.build()
        terminal_phi = terminal_operator.phi()
        absolute_closure = float(torch.linalg.vector_norm(terminal_phi - target).item())
        if absolute_closure > tolerance * (1.0 + float(torch.linalg.vector_norm(target).item())):
            raise ScientificBoundary("terminal fixed-z closure contract failed")
        if arm is Arm.FZCB and spent > initial_budget + tolerance * (1.0 + initial_budget):
            raise ScientificBoundary("terminal completion budget contract failed")
        w0.restore(model)
        commit = transaction.commit_terminal(terminal)
        return {
            "arm": arm.value,
            "status": "TERMINAL_VALID",
            "phi0_sha256": tensor_sha256(phi0),
            "target_sha256": tensor_sha256(target),
            "tolerance": tolerance,
            "initial_geometry": initial_geometry,
            "initial_budget": budget_receipt,
            "waypoints": waypoints,
            "terminal": {
                "s": accepted_s, "absolute_closure": absolute_closure,
                "spent_action": spent, "budget": initial_budget,
                "budget_error": spent - initial_budget,
                "weight_root": terminal.root,
                "geometry": terminal_geometry,
                "operator_counts": terminal_operator.receipt(),
            },
            "atomic_commit": commit,
            "direct_z_recompute_count": 0,
            "controller_output_metric_access_count": 0,
        }
    except Exception:
        transaction.rollback()
        raise
