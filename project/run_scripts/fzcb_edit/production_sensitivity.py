"""Scale-neutral finite-difference sensitivity with explicit uncertainty.

The old TECH-R1 absolute cross-epsilon spread is intentionally not imported.
Every derivative here is taken in the dimensionless coordinate
``c(eta)=c0+eta*sqrt(A0)*d`` against ``Vbar=V/A0``.  Cross-epsilon variation is
reported as uncertainty, never as a universal method-infeasibility gate.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from .contracts import ScientificBoundary


class NumericalInconclusive(ScientificBoundary):
    """The uncertainty interval does not support a fixed gate decision."""

    def __init__(self, message: str, receipt: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.receipt = receipt or {}


@dataclass(frozen=True, slots=True)
class UncertainDerivative:
    estimate: float
    uncertainty: float
    lower: float
    upper: float
    rows: tuple[dict[str, Any], ...]
    uncertainty_components: dict[str, float]

    def payload(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "uncertainty": self.uncertainty,
            "interval": [self.lower, self.upper],
            "rows": list(self.rows),
            "uncertainty_components": dict(self.uncertainty_components),
            "absolute_cross_epsilon_hard_gate_count": 0,
            "coordinate": "c(eta)=c0+eta*sqrt(A0)*unit_direction",
            "value_coordinate": "Vbar=V/A0",
        }


def _fp32_ulp(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32)
    if not bool(torch.isfinite(tensor)):
        return math.inf
    nxt = torch.nextafter(tensor, torch.tensor(math.inf, dtype=torch.float32))
    return abs(float((nxt - tensor).item()))


def _realized_asymmetry(positive: dict[str, Any], negative: dict[str, Any]) -> float:
    plus = float(positive.get("realized_weight_perturbation_norm", 0.0))
    minus = float(negative.get("realized_weight_perturbation_norm", 0.0))
    return abs(plus - minus) / max(plus, minus, torch.finfo(torch.float64).tiny)


def uncertainty_aware_sweep(
    evaluate: Callable[[float, float], tuple[float, dict[str, Any]]],
    *,
    initial_budget: float,
    base_epsilon: float,
    multipliers: Sequence[float],
    repeats: int,
) -> UncertainDerivative:
    """Estimate one normalized directional derivative and its uncertainty.

    ``evaluate`` receives ``(eta, solver_tolerance_scale)`` and returns raw
    suffix action plus a perturbation receipt.  The second repeat reverses the
    +/- call order.  Solver variation is measured at the central epsilon.
    """

    if not math.isfinite(initial_budget) or initial_budget <= 0.0:
        raise ScientificBoundary("normalized FD requires finite positive A0")
    if base_epsilon <= 0.0 or tuple(multipliers) != (0.25, 0.5, 1.0, 2.0) or repeats != 2:
        raise ScientificBoundary("normalized FD sweep differs from sealed pilot lock")
    rows: list[dict[str, Any]] = []
    selected_by_step: list[float] = []
    repeat_noise = 0.0
    order_variation = 0.0
    realized_asymmetry = 0.0
    ulp_bound = 0.0
    for multiplier in multipliers:
        eta = float(base_epsilon * multiplier)
        derivatives: list[float] = []
        repeat_rows: list[dict[str, Any]] = []
        for repeat in range(repeats):
            order = (1.0, -1.0) if repeat == 0 else (-1.0, 1.0)
            observed: dict[float, tuple[float, dict[str, Any]]] = {}
            for sign in order:
                observed[sign] = evaluate(sign * eta, 1.0)
            plus_raw, plus_receipt = observed[1.0]
            minus_raw, minus_receipt = observed[-1.0]
            numerator_raw = float(plus_raw - minus_raw)
            derivative = numerator_raw / (2.0 * eta * initial_budget)
            if not math.isfinite(derivative):
                raise NumericalInconclusive("normalized FD derivative is nonfinite", {
                    "eta": eta, "repeat": repeat, "positive": plus_receipt,
                    "negative": minus_receipt, "numerator_raw": numerator_raw,
                })
            derivatives.append(derivative)
            asymmetry = _realized_asymmetry(plus_receipt, minus_receipt)
            realized_asymmetry = max(realized_asymmetry, asymmetry)
            local_ulp = (
                _fp32_ulp(plus_raw) + _fp32_ulp(minus_raw)
            ) / (2.0 * eta * initial_budget)
            ulp_bound = max(ulp_bound, local_ulp)
            repeat_rows.append({
                "repeat": repeat,
                "evaluation_order": ["plus" if value > 0 else "minus" for value in order],
                "eta": eta,
                "requested_coefficient_perturbation": eta * math.sqrt(initial_budget),
                "positive_raw_suffix": plus_raw,
                "negative_raw_suffix": minus_raw,
                "positive_normalized_suffix": plus_raw / initial_budget,
                "negative_normalized_suffix": minus_raw / initial_budget,
                "fd_numerator_raw": numerator_raw,
                "fd_numerator_normalized": numerator_raw / initial_budget,
                "derivative": derivative,
                "suffix_ulp_derivative_bound": local_ulp,
                "realized_plus_minus_asymmetry": asymmetry,
                "positive": plus_receipt,
                "negative": minus_receipt,
            })
        noise = abs(derivatives[0] - derivatives[1])
        repeat_noise = max(repeat_noise, noise)
        order_variation = max(order_variation, noise)
        chosen = statistics.fmean(derivatives)
        selected_by_step.append(chosen)
        rows.append({
            "multiplier": float(multiplier),
            "eta": eta,
            "selected_derivative": chosen,
            "repeat_noise": noise,
            "repeats": repeat_rows,
        })

    central_eta = float(base_epsilon)
    solver_derivatives: list[dict[str, float]] = []
    for tolerance_scale in (0.5, 2.0):
        plus, _ = evaluate(central_eta, tolerance_scale)
        minus, _ = evaluate(-central_eta, tolerance_scale)
        derivative = float((plus - minus) / (2.0 * central_eta * initial_budget))
        solver_derivatives.append({"solver_tolerance_scale": tolerance_scale, "derivative": derivative})
    estimate = float(statistics.median(selected_by_step))
    adjacent = max(
        (abs(left - right) for left, right in zip(selected_by_step, selected_by_step[1:])),
        default=0.0,
    )
    solver_variation = max(abs(row["derivative"] - estimate) for row in solver_derivatives)
    asymmetry_bound = realized_asymmetry * max(abs(estimate), torch.finfo(torch.float64).eps)
    components = {
        "adjacent_epsilon_or_plateau": adjacent,
        "repeat_noise": repeat_noise,
        "plus_minus_order": order_variation,
        "suffix_ulp_and_cancellation": ulp_bound,
        "solver_tolerance_variation": solver_variation,
        "realized_perturbation_asymmetry": asymmetry_bound,
    }
    uncertainty = max(components.values())
    rows.append({"solver_tolerance_variation": solver_derivatives})
    return UncertainDerivative(
        estimate=estimate,
        uncertainty=uncertainty,
        lower=estimate - uncertainty,
        upper=estimate + uncertainty,
        rows=tuple(rows),
        uncertainty_components=components,
    )


def robust_scalar_rectification(
    *,
    psi_estimate: float,
    psi_uncertainty: float,
    gradient: torch.Tensor,
    gradient_uncertainty: torch.Tensor,
    tau_normalized: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve the one-barrier robust scalar problem along ``-gradient``.

    The uncertainty term is ``sum_j u_j |y_j|``.  Along ``y=-a g`` this is a
    scalar quadratic, so no generic QP or hidden fallback is introduced.
    """

    g = gradient.detach().to(dtype=torch.float64)
    u = gradient_uncertainty.detach().to(dtype=torch.float64)
    if g.ndim != 1 or u.shape != g.shape or bool((u < 0).any()):
        raise ScientificBoundary("robust rectification gradient/uncertainty mismatch")
    if not all(math.isfinite(value) for value in (psi_estimate, psi_uncertainty, tau_normalized)):
        raise ScientificBoundary("robust rectification scalar is nonfinite")
    low = psi_estimate - psi_uncertainty
    high = psi_estimate + psi_uncertainty
    if low <= tau_normalized < high:
        raise NumericalInconclusive("barrier activity changes inside uncertainty interval", {
            "psi_interval": [low, high], "tau_normalized": tau_normalized,
        })
    if high <= tau_normalized:
        return torch.zeros_like(gradient), {
            "barrier_active": False,
            "psi_estimate": psi_estimate,
            "psi_uncertainty": psi_uncertainty,
            "psi_interval": [low, high],
            "tau_normalized": tau_normalized,
            "alpha": 0.0,
            "robust_constraint": high,
            "authority": "UNCERTAINTY_STRICT_SKETCH",
        }
    g2 = float(torch.dot(g, g).item())
    udot = float(torch.dot(u, g.abs()).item())
    if g2 <= 0.0:
        raise NumericalInconclusive("active barrier has zero sketched authority", {
            "psi_interval": [low, high], "tau_normalized": tau_normalized,
        })
    # 0.5*g2*a^2 + (udot-g2)*a + high-tau <= 0.
    linear = udot - g2
    constant = high - tau_normalized
    discriminant = linear * linear - 2.0 * g2 * constant
    if linear >= 0.0 or discriminant < 0.0:
        raise NumericalInconclusive("sketch width has no uncertainty-strict candidate", {
            "psi_interval": [low, high], "tau_normalized": tau_normalized,
            "g2": g2, "uncertainty_l1_slope": udot,
            "discriminant": discriminant,
        })
    alpha = (-linear - math.sqrt(max(0.0, discriminant))) / g2
    if alpha < 0.0:
        raise NumericalInconclusive("robust rectification selected negative alpha")
    y64 = -alpha * g
    robust = 0.5 * float(torch.dot(y64, y64).item()) + float(torch.dot(g, y64).item())
    robust += high + float(torch.dot(u, y64.abs()).item())
    numerical = 64.0 * torch.finfo(torch.float64).eps * max(1.0, abs(robust), abs(tau_normalized))
    if robust > tau_normalized + numerical:
        raise ScientificBoundary("robust scalar closed-form residual exceeds FP64 bound")
    return y64.to(device=gradient.device, dtype=gradient.dtype), {
        "barrier_active": True,
        "psi_estimate": psi_estimate,
        "psi_uncertainty": psi_uncertainty,
        "psi_interval": [low, high],
        "tau_normalized": tau_normalized,
        "g_free_norm_squared_fp64": g2,
        "uncertainty_l1_slope_fp64": udot,
        "alpha": alpha,
        "robust_constraint": robust,
        "authority": "UNCERTAINTY_STRICT_SKETCH",
        "scalar_reduction_dtype": "float64",
    }
