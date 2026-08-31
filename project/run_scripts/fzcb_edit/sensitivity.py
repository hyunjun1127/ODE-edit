"""Full-projector primitive and mandatory finite-difference sketch audit."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from .contracts import NumericalSensitivityFailure, ScientificBoundary


@dataclass(frozen=True, slots=True)
class DirectionSweep:
    selected_derivative: float
    stable: bool
    repeat_noise_max: float
    cross_step_spread: float
    stability_limit: float
    steps: tuple[dict[str, Any], ...]

    def payload(self) -> dict[str, Any]:
        return {
            "selected_derivative": self.selected_derivative,
            "stable": self.stable,
            "repeat_noise_max": self.repeat_noise_max,
            "cross_step_spread": self.cross_step_spread,
            "stability_limit": self.stability_limit,
            "steps": list(self.steps),
        }


def finite_difference_sweep(
    evaluate: Callable[[float], tuple[float, dict[str, Any]]],
    *,
    base_epsilon: float,
    multipliers: Sequence[float],
    repeats: int,
    tau_grad: float,
) -> DirectionSweep:
    if base_epsilon <= 0.0 or repeats != 2 or tuple(multipliers) != (0.25, 0.5, 1.0, 2.0):
        raise ScientificBoundary("finite-difference sweep differs from sealed TECH-R1 lock")
    rows: list[dict[str, Any]] = []
    derivatives: list[float] = []
    repeat_noise = 0.0
    for multiplier in multipliers:
        epsilon = float(base_epsilon * multiplier)
        repeat_derivatives = []
        repeat_receipts = []
        for repeat in range(repeats):
            plus, plus_receipt = evaluate(epsilon)
            minus, minus_receipt = evaluate(-epsilon)
            derivative = float((plus - minus) / (2.0 * epsilon))
            if not math.isfinite(derivative):
                raise NumericalSensitivityFailure(
                    "finite-difference derivative is nonfinite",
                    {
                        "stage": "finite_difference_nonfinite",
                        "multiplier": float(multiplier),
                        "epsilon": epsilon,
                        "repeat": repeat,
                        "positive_value": plus,
                        "negative_value": minus,
                        "positive": plus_receipt,
                        "negative": minus_receipt,
                        "completed_steps": rows,
                    },
                )
            repeat_derivatives.append(derivative)
            repeat_receipts.append({
                "repeat": repeat,
                "positive_value": plus,
                "negative_value": minus,
                "positive": plus_receipt,
                "negative": minus_receipt,
                "derivative": derivative,
            })
        noise = abs(repeat_derivatives[0] - repeat_derivatives[1])
        repeat_noise = max(repeat_noise, noise)
        selected = statistics.fmean(repeat_derivatives)
        derivatives.append(selected)
        rows.append({
            "multiplier": float(multiplier), "epsilon": epsilon,
            "selected_derivative": selected, "repeat_noise": noise,
            "repeats": repeat_receipts,
        })
    selected_derivative = float(statistics.median(derivatives))
    spread = float(max(derivatives) - min(derivatives))
    precision = 64.0 * float(torch.finfo(torch.float32).eps) * max(
        abs(selected_derivative), float(torch.finfo(torch.float32).tiny)
    )
    stability_limit = max(float(tau_grad), 8.0 * repeat_noise, precision)
    stable = spread <= stability_limit
    receipt = DirectionSweep(
        selected_derivative=selected_derivative,
        stable=stable,
        repeat_noise_max=repeat_noise,
        cross_step_spread=spread,
        stability_limit=stability_limit,
        steps=tuple(rows),
    )
    if not stable:
        raise NumericalSensitivityFailure(
            f"FD sweep unstable spread={spread:.9g} limit={stability_limit:.9g}",
            receipt.payload(),
        )
    return receipt


def sketch_ladder_receipt(
    derivatives: Sequence[float],
    *,
    coefficient_dimension: int,
    output_dimension: int,
    ladder: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, Any]:
    if tuple(ladder) != (2, 8, 32, 128) or len(derivatives) < 128 or len(seeds) < 128:
        raise ScientificBoundary("mandatory 2/8/32/128 sketch ladder is incomplete")
    null_dimension = coefficient_dimension - output_dimension
    if null_dimension <= 0:
        raise ScientificBoundary("equality-null dimension is not positive")
    rows = []
    for width in ladder:
        squares = [float(value * value) for value in derivatives[:width]]
        raw = float(sum(squares))
        samples = [null_dimension * value for value in squares]
        estimate = float(statistics.fmean(samples))
        if width > 1:
            standard_error = float(statistics.stdev(samples) / math.sqrt(width))
        else:
            standard_error = 0.0
        rows.append({
            "k": int(width),
            "raw_subspace_g_free_norm_squared": raw,
            "dimension_corrected_g_free_norm_squared": estimate,
            "ci95_lower": max(0.0, estimate - 1.96 * standard_error),
            "ci95_upper": estimate + 1.96 * standard_error,
            "standard_error": standard_error,
            "seed_root": _seed_root(seeds[:width]),
        })
    return {
        "authority": "SKETCH_LADDER_ONLY",
        "full_gradient_available": False,
        "full_gradient_unavailable_reason": "stock_MEMIT_key_capture_is_detached_and_exact_HVP_not_implemented",
        "coefficient_dimension": coefficient_dimension,
        "output_dimension": output_dimension,
        "null_dimension": null_dimension,
        "dimension_correction_rule": "d_eff_times_mean_squared_directional_derivative",
        "ladder": rows,
        "selected_k": int(ladder[-1]),
        "selected_raw_g_free_norm_squared": float(sum(value * value for value in derivatives[: ladder[-1]])),
        "selected_dimension_corrected_g_free_norm_squared": rows[-1]["dimension_corrected_g_free_norm_squared"],
        "seed_root": _seed_root(seeds[: ladder[-1]]),
    }


def _seed_root(seeds: Sequence[int]) -> str:
    import hashlib
    import json

    body = json.dumps(list(seeds), separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()
