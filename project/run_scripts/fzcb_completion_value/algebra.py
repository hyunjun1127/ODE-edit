"""Matrix-free minimum-action, suffix-value, and equality-null algebra."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Protocol

import torch

from .contracts import ScientificBoundary


Vector = torch.Tensor


class WhitenedControlOperator(Protocol):
    """A = C H^{-1/2}; neither C nor a Kronecker matrix is materialized."""

    coefficient_dimension: int
    output_dimension: int

    def apply(self, value: Vector) -> Vector: ...
    def adjoint(self, value: Vector) -> Vector: ...


@dataclass(frozen=True, slots=True)
class CGReceipt:
    iterations: int
    residual_norm: float
    relative_residual: float
    converged: bool
    residual_history: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MinimumActionSolution:
    whitened_coefficient: Vector
    coefficient: Vector
    action_norm_squared: float
    range_residual: float
    receipt: CGReceipt


def _finite_vector(value: Vector, name: str) -> Vector:
    if value.dtype != torch.float32 or value.ndim != 1 or value.requires_grad or not bool(torch.isfinite(value).all()):
        raise ScientificBoundary(f"{name} must be detached finite FP32 vector")
    return value


def dimension_relative_floor(dimension: int) -> float:
    if dimension <= 0:
        raise ScientificBoundary("dimension must be positive")
    return float(64.0 * torch.finfo(torch.float32).eps * math.sqrt(dimension))


def conjugate_gradient(
    matvec: Callable[[Vector], Vector],
    rhs: Vector,
    *,
    relative_tolerance: float,
    max_iterations: int,
    preconditioner: Callable[[Vector], Vector] | None = None,
) -> tuple[Vector, CGReceipt]:
    b = _finite_vector(rhs, "cg rhs")
    x = torch.zeros_like(b)
    r = b.clone()
    z = r.clone() if preconditioner is None else preconditioner(r)
    p = z.clone()
    rz = torch.dot(r, z)
    b_norm = float(torch.linalg.vector_norm(b).item())
    history = [float(torch.linalg.vector_norm(r).item())]
    if b_norm == 0.0:
        return x, CGReceipt(0, 0.0, 0.0, True, tuple(history))
    converged = False
    completed = 0
    for index in range(max_iterations):
        ap = _finite_vector(matvec(p).detach(), "cg matvec")
        denominator = torch.dot(p, ap)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
            break
        alpha = rz / denominator
        x = x + alpha * p
        r = r - alpha * ap
        completed = index + 1
        norm = float(torch.linalg.vector_norm(r).item())
        history.append(norm)
        if norm <= relative_tolerance * (b_norm + torch.finfo(torch.float32).eps):
            converged = True
            break
        z = r.clone() if preconditioner is None else preconditioner(r)
        next_rz = torch.dot(r, z)
        if not bool(torch.isfinite(next_rz)):
            break
        p = z + (next_rz / rz) * p
        rz = next_rz
    residual = float(torch.linalg.vector_norm(r).item())
    return x.detach(), CGReceipt(
        completed, residual, residual / (b_norm + torch.finfo(torch.float32).eps),
        converged, tuple(history),
    )


def solve_minimum_action(
    operator: WhitenedControlOperator,
    rhs: Vector,
    h_sqrt: Vector,
    *,
    relative_tolerance: float,
    max_iterations: int,
    output_preconditioner: Callable[[Vector], Vector] | None = None,
) -> MinimumActionSolution:
    b = _finite_vector(rhs, "equality rhs")
    scale = _finite_vector(h_sqrt, "H sqrt")
    if scale.numel() != operator.coefficient_dimension or bool((scale <= 0).any()):
        raise ScientificBoundary("H sqrt closure failed")

    def gram(value: Vector) -> Vector:
        return operator.apply(operator.adjoint(value))

    multiplier, receipt = conjugate_gradient(
        gram, b, relative_tolerance=relative_tolerance, max_iterations=max_iterations,
        preconditioner=output_preconditioner,
    )
    x = operator.adjoint(multiplier).detach()
    observed = operator.apply(x).detach()
    range_residual = float(torch.linalg.vector_norm(observed - b).div(torch.linalg.vector_norm(b) + torch.finfo(torch.float32).eps).item())
    if range_residual > relative_tolerance or not receipt.converged:
        raise ScientificBoundary(
            f"typed range failure: residual={range_residual:.9g}, cg={asdict(receipt)}"
        )
    coefficient = x / scale
    return MinimumActionSolution(
        whitened_coefficient=x, coefficient=coefficient.detach(),
        action_norm_squared=float(torch.dot(x, x).item()), range_residual=range_residual,
        receipt=receipt,
    )


def project_equality_null(
    operator: WhitenedControlOperator,
    seed: Vector,
    *,
    relative_tolerance: float,
    max_iterations: int,
    refinement_iterations: int = 2,
    output_preconditioner: Callable[[Vector], Vector] | None = None,
) -> tuple[Vector, dict[str, float | int | bool]]:
    value = _finite_vector(seed, "null seed")
    if value.numel() != operator.coefficient_dimension:
        raise ScientificBoundary("null seed dimension differs")
    image = operator.apply(value).detach()

    def gram(vector: Vector) -> Vector:
        return operator.apply(operator.adjoint(vector))

    # The null gate is ||A p|| / ||p|| <= tau.  Stopping the Gram solve at
    # tau relative to ||A seed|| is not sufficient when A has large gain.
    # Keep the scientific tau unchanged and tighten only the internal linear
    # solve deterministically so its truncation cannot masquerade as absent
    # null authority.
    solver_relative_tolerance = min(relative_tolerance, relative_tolerance * relative_tolerance)
    multiplier, receipt = conjugate_gradient(
        gram, image, relative_tolerance=solver_relative_tolerance,
        max_iterations=max_iterations, preconditioner=output_preconditioner,
    )
    projected = (value - operator.adjoint(multiplier)).detach()
    refinements = []
    for ordinal in range(refinement_iterations):
        image_residual = operator.apply(projected).detach()
        correction_multiplier, correction_receipt = conjugate_gradient(
            gram, image_residual, relative_tolerance=solver_relative_tolerance,
            max_iterations=max_iterations, preconditioner=output_preconditioner,
        )
        projected = (projected - operator.adjoint(correction_multiplier)).detach()
        refinements.append({
            "ordinal": ordinal + 1,
            "image_residual_before": float(torch.linalg.vector_norm(image_residual).item()),
            "cg_iterations": correction_receipt.iterations,
            "cg_converged": correction_receipt.converged,
            "cg_relative_residual": correction_receipt.relative_residual,
        })
    norm = float(torch.linalg.vector_norm(projected).item())
    residual = float(torch.linalg.vector_norm(operator.apply(projected)).item() / (norm + torch.finfo(torch.float32).eps))
    return projected, {
        "norm": norm, "null_residual": residual, "cg_iterations": receipt.iterations,
        "cg_converged": receipt.converged,
        "solver_relative_tolerance": solver_relative_tolerance,
        "external_null_tolerance": relative_tolerance,
        "refinement_iterations": refinement_iterations,
        "refinements": refinements,
    }


def scale_null_direction(projected: Vector, equality_action_norm: float, rho: float) -> Vector:
    value = _finite_vector(projected, "projected null")
    norm = torch.linalg.vector_norm(value)
    if not math.isfinite(equality_action_norm) or equality_action_norm <= 0 or rho <= 0 or float(norm) <= 0:
        raise ScientificBoundary("null direction has no effective authority")
    return (value * (rho * equality_action_norm / norm)).detach()


def suffix_value(solution: MinimumActionSolution, remaining_time: float) -> float:
    if not 0.0 < remaining_time <= 1.0:
        raise ScientificBoundary("remaining time must be in (0,1]")
    return float(solution.action_norm_squared / (2.0 * remaining_time))


def macro_action_receipt(coefficient_action_squared: float, delta_s: float) -> dict[str, float]:
    """Account for the final corrected coefficient, including all corrections."""
    if not math.isfinite(coefficient_action_squared) or coefficient_action_squared < 0:
        raise ScientificBoundary("corrected coefficient action is invalid")
    if not math.isfinite(delta_s) or delta_s <= 0:
        raise ScientificBoundary("macro step must be positive")
    step_action = delta_s * 0.5 * coefficient_action_squared
    displacement_metric_action = delta_s * delta_s * coefficient_action_squared
    equivalent = displacement_metric_action / (2.0 * delta_s)
    return {
        "step_action": step_action,
        "displacement_metric_action": displacement_metric_action,
        "equivalent_step_action": equivalent,
        "absolute_error": abs(step_action - equivalent),
    }


def frozen_geometry_identity(action0: float, progress: tuple[float, ...]) -> dict[str, object]:
    if not math.isfinite(action0) or action0 <= 0:
        raise ScientificBoundary("frozen action must be positive")
    rows = []
    for s in progress:
        spent = s * action0
        suffix = (1.0 - s) * action0
        rows.append({"s": s, "spent": spent, "suffix": suffix, "sum": spent + suffix})
    error = max(abs(row["sum"] - action0) for row in rows)
    return {"A0": action0, "rows": rows, "maximum_absolute_error": error}
