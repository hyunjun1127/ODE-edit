"""Strict proposal-level CBF, action, closure, and rollback verification."""

from __future__ import annotations

import math
from typing import Any

from .contracts import ScientificBoundary
from .tolerances import ResolvedTolerances


def verify_barrier_state(
    *,
    initial_budget: float,
    spent_action: float,
    predicted_suffix: float,
    tolerances: ResolvedTolerances,
    stage: str,
) -> dict[str, Any]:
    values = (initial_budget, spent_action, predicted_suffix)
    if not all(math.isfinite(value) for value in values):
        raise ScientificBoundary(f"{stage}: nonfinite completion-budget component")
    h_cc = initial_budget - spent_action - predicted_suffix
    receipt = {
        "stage": stage,
        "A0": initial_budget,
        "E": spent_action,
        "predicted_suffix": predicted_suffix,
        "h_cc": h_cc,
        "tau_budget": tolerances.tau_budget,
        "strict_pass": h_cc >= -tolerances.tau_budget,
    }
    if not receipt["strict_pass"]:
        raise ScientificBoundary(
            f"{stage}: strict intermediate completion budget violation h_cc={h_cc:.9g} "
            f"tau_budget={tolerances.tau_budget:.9g}"
        )
    return receipt


def verify_terminal(
    *,
    closure: float,
    spent_action: float,
    initial_budget: float,
    current_viability: bool,
    all_waypoints_strict: bool,
    corrector_action_included: bool,
    rollback_exact: bool,
    tolerances: ResolvedTolerances,
) -> dict[str, Any]:
    receipt = {
        "closure": closure,
        "tau_z": tolerances.tau_z,
        "spent_action": spent_action,
        "A0": initial_budget,
        "tau_budget": tolerances.tau_budget,
        "current_viability": current_viability,
        "all_waypoints_strict": all_waypoints_strict,
        "corrector_action_included": corrector_action_included,
        "rollback_exact": rollback_exact,
    }
    receipt["strict_pass"] = bool(
        math.isfinite(closure)
        and closure <= tolerances.tau_z
        and spent_action <= initial_budget + tolerances.tau_budget
        and current_viability
        and all_waypoints_strict
        and corrector_action_included
        and rollback_exact
    )
    if not receipt["strict_pass"]:
        raise ScientificBoundary(f"TERMINAL_VALID_STRICT verification failed: {receipt}")
    return receipt


def frozen_completion_identity(
    *, initial_budget: float, progress: float, spent_action: float,
    suffix_value: float, tau_budget: float,
) -> dict[str, Any]:
    total = spent_action + suffix_value
    error = total - initial_budget
    receipt = {
        "progress": progress,
        "A0": initial_budget,
        "E": spent_action,
        "predicted_suffix": suffix_value,
        "E_plus_V": total,
        "h_cc": -error,
        "identity_error": error,
        "tau_budget": tau_budget,
        "pass": abs(error) <= tau_budget,
    }
    if not receipt["pass"]:
        raise ScientificBoundary(f"TRUE_FROZEN_C_SPLIT E+V=A0 identity failed: {receipt}")
    return receipt

