"""Typed true-frozen and strong-static same-objective comparator policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .contracts import ScientificBoundary
from .verifier import frozen_completion_identity


def true_frozen_schedule(
    *, initial_budget: float, progress_grid: Iterable[float], tau_budget: float
) -> list[dict[str, Any]]:
    """Return the exact frozen-geometry E+V=A0 negative-control ledger."""

    rows = []
    for progress in progress_grid:
        if progress < 0.0 or progress > 1.0:
            raise ScientificBoundary("frozen progress is outside [0,1]")
        spent = progress * initial_budget
        suffix = (1.0 - progress) * initial_budget
        rows.append(frozen_completion_identity(
            initial_budget=initial_budget, progress=progress,
            spent_action=spent, suffix_value=suffix, tau_budget=tau_budget,
        ))
    return rows


@dataclass(frozen=True, slots=True)
class StaticCandidate:
    candidate_id: str
    closure: float
    action: float
    coefficient_sha256: str
    weight_root: str
    solver_iterations: int
    residual_history: tuple[float, ...]
    feasible_equality: bool
    feasible_budget: bool

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def select_strong_static(
    candidates: Iterable[StaticCandidate], *, tau_z: float, initial_budget: float, tau_budget: float
) -> tuple[StaticCandidate, dict[str, Any]]:
    """Choose minimum native action among actual endpoint-feasible static solves."""

    rows = list(candidates)
    if not rows:
        raise ScientificBoundary("strong static direct-transcription produced no candidates")
    feasible = [
        row for row in rows
        if row.feasible_equality and row.closure <= tau_z
        and row.feasible_budget and row.action <= initial_budget + tau_budget
    ]
    best_observed = min(rows, key=lambda row: (row.closure, row.action, row.candidate_id))
    receipt = {
        "solver": "DIRECT_TRANSCRIPTION_FIXED_BASIS_SQP",
        "candidate_count": len(rows),
        "feasible_count": len(feasible),
        "candidates": [row.payload() for row in rows],
        "best_observed": best_observed.payload(),
    }
    if not feasible:
        raise ScientificBoundary(f"strong static same-objective infeasible: {receipt}")
    selected = min(feasible, key=lambda row: (row.action, row.closure, row.candidate_id))
    receipt["selected_candidate_id"] = selected.candidate_id
    receipt["selected_action"] = selected.action
    return selected, receipt

