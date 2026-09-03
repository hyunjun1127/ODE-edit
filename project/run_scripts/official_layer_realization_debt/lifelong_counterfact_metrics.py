"""Canonical CounterFact NLL-pair metrics for sealed lifelong records."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from project.run_scripts.ode_bf.contracts import canonical_hash

from .lifelong_counterfact_contracts import CounterFactMetricBoundary


def strict_nll_pair_success(
    target_new_nll: float, target_true_nll: float, *, locality: bool
) -> int:
    """Return the canonical strict CounterFact success bit; ties fail."""

    new = float(target_new_nll)
    true = float(target_true_nll)
    if not math.isfinite(new) or not math.isfinite(true):
        raise CounterFactMetricBoundary("nonfinite CounterFact NLL pair")
    return int(true < new) if locality else int(new < true)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise CounterFactMetricBoundary("empty CounterFact distribution")
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * probability
    low = math.floor(location)
    high = math.ceil(location)
    if low == high:
        return ordered[low]
    fraction = location - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def distribution(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise CounterFactMetricBoundary("invalid CounterFact distribution")
    return {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p90": quantile(numeric, 0.9),
        "max": max(numeric),
    }


def paired_prompt_metrics(
    target_new: Sequence[Mapping[str, Any]],
    target_true: Sequence[Mapping[str, Any]],
    *,
    locality: bool,
) -> dict[str, Any]:
    """Aggregate order-preserving prompt pairs and publish their bit root."""

    if not target_new or len(target_new) != len(target_true):
        raise CounterFactMetricBoundary("CounterFact prompt pair cardinality differs")
    bits: list[int] = []
    new_values: list[float] = []
    true_values: list[float] = []
    differences: list[float] = []
    pairs: list[list[float]] = []
    for new_prompt, true_prompt in zip(target_new, target_true, strict=True):
        new = float(new_prompt["nll"])
        true = float(true_prompt["nll"])
        bits.append(strict_nll_pair_success(new, true, locality=locality))
        new_values.append(new)
        true_values.append(true)
        # Positive means the desired side wins for both edit and locality.
        differences.append((new - true) if locality else (true - new))
        pairs.append([new, true])
    return {
        "numerator": sum(bits),
        "denominator": len(bits),
        "rate": sum(bits) / len(bits),
        "bit_vector_sha256": canonical_hash(bits),
        "ordered_nll_pair_sha256": canonical_hash(pairs),
        "target_new_nll": distribution(new_values),
        "target_true_nll": distribution(true_values),
        "desired_minus_undesired_nll_advantage": distribution(differences),
        "tie_count": sum(int(new == true) for new, true in zip(new_values, true_values, strict=True)),
    }


__all__ = [
    "distribution",
    "paired_prompt_metrics",
    "quantile",
    "strict_nll_pair_success",
]
