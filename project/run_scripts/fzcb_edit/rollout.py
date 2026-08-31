"""Observation-only actual remaining rollout and prediction concordance."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from .contracts import ScientificBoundary


@dataclass(frozen=True, slots=True)
class SuffixCandidate:
    candidate_id: str
    predicted_suffix_after: float
    realized_suffix_action: float
    layer_action: tuple[float, ...]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def run_actual_remaining_rollout(
    *,
    start_progress: float,
    progress_grid: Iterable[float],
    advance: Callable[[float, float], tuple[float, tuple[float, ...], dict[str, Any]]],
) -> tuple[float, tuple[float, ...], dict[str, Any]]:
    """Execute an equality-only virtual rollout; caller owns exact state restore."""

    boundaries = [value for value in progress_grid if value > start_progress + 1e-12]
    current = start_progress
    total = 0.0
    layer_totals: list[float] | None = None
    steps = []
    for end in boundaries:
        action, layer_action, receipt = advance(current, end)
        if not math.isfinite(action) or action < 0.0:
            raise ScientificBoundary("realized suffix rollout action is invalid")
        if layer_totals is None:
            layer_totals = [0.0 for _ in layer_action]
        if len(layer_action) != len(layer_totals):
            raise ScientificBoundary("realized suffix layer-action width changed")
        total += action
        for index, value in enumerate(layer_action):
            layer_totals[index] += float(value)
        steps.append({"s_start": current, "s_end": end, "action": action, "receipt": receipt})
        current = end
    return total, tuple(layer_totals or ()), {
        "method": "ACTUAL_REFRESHED_EQUALITY_ONLY_REMAINING_ROLLOUT",
        "start_progress": start_progress,
        "terminal_progress": current,
        "step_count": len(steps),
        "steps": steps,
        "controller_selection_influence_count": 0,
    }


def concordance(candidates: Iterable[SuffixCandidate]) -> dict[str, Any]:
    rows = list(candidates)
    if not rows:
        return {"candidate_count": 0, "predictive_concordance": "NOT_RECORDED", "predicted_best_regret": "NOT_RECORDED"}
    predicted = sorted(rows, key=lambda row: (row.predicted_suffix_after, row.candidate_id))
    realized = sorted(rows, key=lambda row: (row.realized_suffix_action, row.candidate_id))
    if len(rows) == 1:
        coefficient: float | str = "NOT_IDENTIFIABLE_SINGLE_CANDIDATE"
    else:
        predicted_rank = {row.candidate_id: index for index, row in enumerate(predicted)}
        realized_rank = {row.candidate_id: index for index, row in enumerate(realized)}
        count = len(rows)
        squared = sum((predicted_rank[row.candidate_id] - realized_rank[row.candidate_id]) ** 2 for row in rows)
        coefficient = 1.0 - 6.0 * squared / (count * (count * count - 1.0))
    predicted_best = predicted[0]
    realized_best = realized[0]
    regret = predicted_best.realized_suffix_action - realized_best.realized_suffix_action
    return {
        "candidate_count": len(rows),
        "candidates": [row.payload() for row in rows],
        "predicted_ranking": [row.candidate_id for row in predicted],
        "realized_ranking": [row.candidate_id for row in realized],
        "predictive_concordance": coefficient,
        "predicted_best_regret": regret,
        "controller_selection_influence_count": 0,
    }

