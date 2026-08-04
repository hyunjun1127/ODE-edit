"""Leakage-free absolute rewrite gate over authorized context summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import ODEAllocContractError, canonical_hash, finite, positive


def _panel(name: str, values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ODEAllocContractError(f"{name} panel is empty")
    result: dict[str, float] = {}
    for context_id, value in values.items():
        if not isinstance(context_id, str) or not context_id:
            raise ODEAllocContractError(f"{name} context identity is empty")
        result[context_id] = finite(name, value)
    return result


@dataclass(frozen=True, slots=True)
class RewriteGateConfig:
    rho: float = 0.5
    denominator_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        rho = finite("rewrite rho", self.rho)
        epsilon = positive("rewrite denominator epsilon", self.denominator_epsilon)
        if not 0.0 < rho <= 1.0:
            raise ODEAllocContractError("rewrite rho must be in (0, 1]")
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "denominator_epsilon", epsilon)


@dataclass(frozen=True, slots=True)
class RewriteGateReading:
    mean_margin: float
    oracle_ratio: float
    worst_context_margin: float
    oracle_mean_margin: float
    denominator: float
    feasible: bool
    event_id: str


def evaluate_rewrite_gate(
    *,
    candidate_new: Mapping[str, float],
    candidate_old: Mapping[str, float],
    entry_new: Mapping[str, float],
    direct_z_new: Mapping[str, float],
    direct_z_old: Mapping[str, float],
    config: RewriteGateConfig,
) -> RewriteGateReading:
    """Evaluate M and R_new; worst context remains diagnostic only."""

    panels = {
        "candidate_new": _panel("candidate new logp", candidate_new),
        "candidate_old": _panel("candidate old logp", candidate_old),
        "entry_new": _panel("entry new logp", entry_new),
        "direct_z_new": _panel("direct-z new logp", direct_z_new),
        "direct_z_old": _panel("direct-z old logp", direct_z_old),
    }
    context_sets = {frozenset(value) for value in panels.values()}
    if len(context_sets) != 1:
        raise ODEAllocContractError("rewrite context panels are misaligned")
    context_ids = tuple(sorted(panels["candidate_new"]))
    count = len(context_ids)
    candidate_margins = tuple(
        panels["candidate_new"][context_id]
        - panels["candidate_old"][context_id]
        for context_id in context_ids
    )
    oracle_margins = tuple(
        panels["direct_z_new"][context_id]
        - panels["direct_z_old"][context_id]
        for context_id in context_ids
    )
    mean_margin = sum(candidate_margins) / count
    oracle_mean_margin = sum(oracle_margins) / count
    entry_mean = sum(panels["entry_new"].values()) / count
    candidate_mean = sum(panels["candidate_new"].values()) / count
    direct_z_mean = sum(panels["direct_z_new"].values()) / count
    denominator = direct_z_mean - entry_mean
    if denominator <= config.denominator_epsilon:
        raise ODEAllocContractError("direct-z rewrite denominator is ineligible")
    if oracle_mean_margin <= 0.0:
        raise ODEAllocContractError("direct-z oracle margin is not positive")
    ratio = (candidate_mean - entry_mean) / denominator
    feasible = mean_margin >= 0.0 and ratio >= config.rho
    payload = {
        "schema": "ode-alloc-rewrite-event/v1",
        "mean_margin": mean_margin,
        "oracle_ratio": ratio,
        "oracle_mean_margin": oracle_mean_margin,
        "denominator": denominator,
        "rho": config.rho,
        "context_count": count,
        "context_ids": context_ids,
        "panels": {
            name: {context_id: panel[context_id] for context_id in context_ids}
            for name, panel in panels.items()
        },
        "feasible": feasible,
    }
    return RewriteGateReading(
        mean_margin=mean_margin,
        oracle_ratio=ratio,
        worst_context_margin=min(candidate_margins),
        oracle_mean_margin=oracle_mean_margin,
        denominator=denominator,
        feasible=feasible,
        event_id=canonical_hash(payload),
    )
