"""Matrix-free equality/suffix solves and the analytic one-barrier rectifier."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Protocol

import torch

from .contracts import ScientificBoundary, SubspaceInconclusive


Vector = torch.Tensor


class ControlOperator(Protocol):
    coefficient_dimension: int
    output_dimension: int

    def apply(self, value: Vector) -> Vector: ...
    def adjoint(self, value: Vector) -> Vector: ...


@dataclass(frozen=True, slots=True)
class IterativeReceipt:
    iterations: int
    converged: bool
    relative_residual: float
    residual_history: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EqualitySolution:
    coefficient: Vector
    action_squared: float
    range_residual: float
    receipt: IterativeReceipt


def dimension_relative_floor(dimension: int) -> float:
    if dimension <= 0:
        raise ScientificBoundary("output dimension must be positive")
    return float(64.0 * torch.finfo(torch.float32).eps * math.sqrt(dimension))


def _finite_vector(value: Vector, name: str) -> Vector:
    if value.dtype != torch.float32 or value.ndim != 1 or value.requires_grad or not bool(torch.isfinite(value).all()):
        raise ScientificBoundary(f"{name} must be detached finite FP32 vector")
    return value


def conjugate_gradient(
    matvec: Callable[[Vector], Vector],
    rhs: Vector,
    *,
    tolerance: float,
    maximum_iterations: int,
    preconditioner: Callable[[Vector], Vector] | None = None,
) -> tuple[Vector, IterativeReceipt]:
    b = _finite_vector(rhs, "CG rhs")
    x = torch.zeros_like(b)
    r = b.clone()
    z = r.clone() if preconditioner is None else preconditioner(r)
    p = z.clone()
    rz = torch.dot(r, z)
    b_norm = float(torch.linalg.vector_norm(b).item())
    history = [float(torch.linalg.vector_norm(r).item())]
    if b_norm == 0.0:
        return x, IterativeReceipt(0, True, 0.0, tuple(history))
    converged = False
    completed = 0
    for ordinal in range(maximum_iterations):
        ap = _finite_vector(matvec(p).detach(), "CG matvec")
        denominator = torch.dot(p, ap)
        if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0.0:
            break
        alpha = rz / denominator
        x = x + alpha * p
        r = r - alpha * ap
        completed = ordinal + 1
        norm = float(torch.linalg.vector_norm(r).item())
        history.append(norm)
        if norm <= tolerance * (b_norm + torch.finfo(torch.float32).eps):
            converged = True
            break
        z = r.clone() if preconditioner is None else preconditioner(r)
        next_rz = torch.dot(r, z)
        if not bool(torch.isfinite(next_rz)) or float(rz.abs().item()) == 0.0:
            break
        p = z + (next_rz / rz) * p
        rz = next_rz
    relative = float(torch.linalg.vector_norm(r).item()) / (b_norm + torch.finfo(torch.float32).eps)
    return x.detach(), IterativeReceipt(completed, converged, relative, tuple(history))


def solve_minimum_action(
    operator: ControlOperator,
    rhs: Vector,
    *,
    tolerance: float,
    maximum_iterations: int,
    refinement_count: int,
    preconditioner: Callable[[Vector], Vector] | None = None,
) -> EqualitySolution:
    b = _finite_vector(rhs, "equality rhs")

    def gram(value: Vector) -> Vector:
        return operator.apply(operator.adjoint(value))

    multiplier, receipt = conjugate_gradient(
        gram, b, tolerance=min(tolerance, tolerance * tolerance),
        maximum_iterations=maximum_iterations, preconditioner=preconditioner,
    )
    coefficient = operator.adjoint(multiplier).detach()
    refinement_rows = []
    for _ in range(refinement_count):
        residual = (b - operator.apply(coefficient)).detach()
        relative = float(torch.linalg.vector_norm(residual).div(torch.linalg.vector_norm(b) + torch.finfo(torch.float32).eps).item())
        refinement_rows.append(relative)
        if relative <= tolerance:
            break
        correction_multiplier, _ = conjugate_gradient(
            gram, residual, tolerance=min(tolerance, tolerance * tolerance),
            maximum_iterations=maximum_iterations, preconditioner=preconditioner,
        )
        coefficient = (coefficient + operator.adjoint(correction_multiplier)).detach()
    observed = operator.apply(coefficient).detach()
    range_residual = float(torch.linalg.vector_norm(observed - b).div(torch.linalg.vector_norm(b) + torch.finfo(torch.float32).eps).item())
    if range_residual > tolerance:
        raise ScientificBoundary(
            f"typed equality range failure residual={range_residual:.9g} tolerance={tolerance:.9g} receipt={asdict(receipt)} refinement={refinement_rows}"
        )
    return EqualitySolution(coefficient, float(torch.dot(coefficient, coefficient).item()), range_residual, receipt)


def suffix_value(solution: EqualitySolution, remaining_time: float) -> float:
    if not 0.0 < remaining_time <= 1.0:
        raise ScientificBoundary("suffix time must be in (0,1]")
    return float(solution.action_squared / (2.0 * remaining_time))


def equality_null_projection(
    operator: ControlOperator,
    seed: Vector,
    *,
    tolerance: float,
    maximum_iterations: int,
    refinement_count: int,
    preconditioner: Callable[[Vector], Vector] | None = None,
) -> tuple[Vector, dict[str, float | int | bool]]:
    value = _finite_vector(seed, "null seed")
    image = operator.apply(value).detach()
    input_norm = float(torch.linalg.vector_norm(value).item())
    image_norm = float(torch.linalg.vector_norm(image).item())
    operator_scale = image_norm / (input_norm + torch.finfo(torch.float32).eps)

    def gram(vector: Vector) -> Vector:
        return operator.apply(operator.adjoint(vector))

    multiplier, receipt = conjugate_gradient(
        gram, image, tolerance=min(tolerance, tolerance * tolerance),
        maximum_iterations=maximum_iterations, preconditioner=preconditioner,
    )
    projected = (value - operator.adjoint(multiplier)).detach()
    for _ in range(refinement_count):
        residual = operator.apply(projected).detach()
        correction, _ = conjugate_gradient(
            gram, residual, tolerance=min(tolerance, tolerance * tolerance),
            maximum_iterations=maximum_iterations, preconditioner=preconditioner,
        )
        projected = (projected - operator.adjoint(correction)).detach()
    norm = torch.linalg.vector_norm(projected)
    if float(norm.item()) <= tolerance:
        raise ScientificBoundary("equality-null guide authority absent")
    projected = (projected / norm).detach()
    residual = float(torch.linalg.vector_norm(operator.apply(projected)).item())
    relative_residual = residual / (operator_scale + torch.finfo(torch.float32).eps)
    if relative_residual > tolerance:
        raise ScientificBoundary(
            f"equality-null normalized residual exceeds lock: {relative_residual}"
        )
    return projected, {
        "norm_before_normalization": float(norm.item()),
        "input_coefficient_norm": input_norm,
        "input_image_norm": image_norm,
        "operator_scale": operator_scale,
        "null_residual_absolute": residual,
        "null_residual": relative_residual,
        "cg_iterations": receipt.iterations,
        "cg_converged": receipt.converged,
    }


def full_projected_sensitivity(
    operator: ControlOperator,
    gradient: Vector,
    *,
    tau_range: float,
    tau_null: float,
    maximum_iterations: int,
    refinement_count: int,
    preconditioner: Callable[[Vector], Vector] | None = None,
) -> tuple[Vector, dict[str, float | int | bool]]:
    """Project a supplied full coefficient gradient into ker(A), matrix-free."""

    value = _finite_vector(gradient, "full coefficient gradient")
    image = operator.apply(value).detach()

    def gram(vector: Vector) -> Vector:
        return operator.apply(operator.adjoint(vector))

    multiplier, receipt = conjugate_gradient(
        gram, image, tolerance=min(tau_range, tau_range * tau_range),
        maximum_iterations=maximum_iterations, preconditioner=preconditioner,
    )
    projected = (value - operator.adjoint(multiplier)).detach()
    for _ in range(refinement_count):
        residual = operator.apply(projected).detach()
        relative = float(torch.linalg.vector_norm(residual).item()) / (
            float(torch.linalg.vector_norm(value).item()) + torch.finfo(torch.float32).eps
        )
        if relative <= tau_null:
            break
        correction, _ = conjugate_gradient(
            gram, residual, tolerance=min(tau_range, tau_range * tau_range),
            maximum_iterations=maximum_iterations, preconditioner=preconditioner,
        )
        projected = (projected - operator.adjoint(correction)).detach()
    projected_image_norm = float(torch.linalg.vector_norm(operator.apply(projected)).item())
    input_image_norm = float(torch.linalg.vector_norm(image).item())
    null_residual = projected_image_norm / (input_image_norm + torch.finfo(torch.float32).eps)
    if null_residual > tau_null:
        raise ScientificBoundary(f"full projected sensitivity null residual={null_residual}")
    return projected, {
        "authority": "FULL_PROJECTED_SENSITIVITY",
        "coefficient_dimension": operator.coefficient_dimension,
        "output_dimension": operator.output_dimension,
        "null_dimension": max(0, operator.coefficient_dimension - operator.output_dimension),
        "g_free_norm_squared": float(torch.dot(projected, projected).item()),
        "input_image_norm": input_image_norm,
        "projected_image_norm": projected_image_norm,
        "null_residual": null_residual,
        "cg_iterations": receipt.iterations,
        "cg_converged": receipt.converged,
    }


def scalar_rectification(
    psi0: float,
    g_free: Vector,
    *,
    tau_cbf: float,
    authority: str = "FULL_PROJECTED_SENSITIVITY",
) -> tuple[Vector, dict[str, float | bool | str]]:
    gradient = _finite_vector(g_free, "projected barrier gradient")
    norm_squared = float(torch.dot(gradient, gradient).item())
    if not math.isfinite(tau_cbf) or tau_cbf <= 0.0:
        raise ScientificBoundary("tau_cbf must be finite positive in action-rate units")
    if psi0 <= tau_cbf:
        return torch.zeros_like(gradient), {
            "barrier_active": False, "psi0": psi0, "g_free_norm_squared": norm_squared,
            "alpha": 0.0, "psi_min": psi0 - 0.5 * norm_squared,
            "tau_cbf": tau_cbf, "authority": authority,
        }
    psi_min = psi0 - 0.5 * norm_squared
    if psi_min > tau_cbf:
        if authority != "FULL_PROJECTED_SENSITIVITY":
            raise SubspaceInconclusive(
                f"sketched authority cannot establish full infeasibility psi0={psi0:.9g} "
                f"g2={norm_squared:.9g} psi_min={psi_min:.9g} tau_cbf={tau_cbf:.9g}"
            )
        raise ScientificBoundary(
            f"typed full-space local barrier infeasibility psi0={psi0:.9g} "
            f"g2={norm_squared:.9g} psi_min={psi_min:.9g} tau_cbf={tau_cbf:.9g}"
        )
    radicand = 1.0 - 2.0 * psi0 / norm_squared
    dimensionless_floor = 64.0 * torch.finfo(torch.float32).eps
    if radicand < -dimensionless_floor:
        raise ScientificBoundary("scalar rectification has negative radicand")
    alpha = 1.0 - math.sqrt(max(0.0, radicand))
    value = (-alpha * gradient).detach()
    residual = 0.5 * float(torch.dot(value, value).item()) + float(torch.dot(gradient, value).item()) + psi0
    if residual > tau_cbf:
        raise ScientificBoundary(f"scalar rectification residual exceeds lock: {residual}")
    return value, {
        "barrier_active": True, "psi0": psi0, "g_free_norm_squared": norm_squared,
        "alpha": alpha, "psi_min": psi_min, "constraint_residual": residual,
        "tau_cbf": tau_cbf, "authority": authority,
    }
