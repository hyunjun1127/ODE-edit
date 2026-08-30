"""Immutable contracts for the F2 barrier usefulness screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from project.run_scripts.fixed_z_nonuniqueness.contracts import (
    MODEL_SPECS,
    Method,
    ScientificBoundary,
    TechnicalBoundary,
)


class Phase(str, Enum):
    PHASE_A = "phase-a-f1b-corrected"
    PHASE_B_CTRL = "phase-b-ctrl-only"
    PHASE_B_GATE = "phase-b-gate-open"


class Selector(str, Enum):
    BARRIER_SAFE = "barrier-safe"
    BARRIER_ADVERSE = "barrier-adverse-finite"
    KL_ONLY = "kl-only"
    ACTION_REPRESENTATIVE = "action-representative"
    OFFICIAL = "official"
    GATE_ORACLE_DIAGNOSTIC = "gate-oracle-diagnostic-only"


@dataclass(frozen=True)
class BarrierLock:
    schema: str = "odeedit.s06.barrier-usefulness.numerical-lock.v1"
    dtype: str = "float32"
    rho_tangent: float = 0.05
    phase_a_axis_count: int = 4
    phase_b_axis_count: int = 16
    candidate_signs: tuple[int, int] = (-1, 1)
    ctrl_anchor_count: int = 32
    gate_anchor_count: int = 32
    cvar_alpha: float = 0.875
    fp32_relative_tolerance: float = 3.0517578125e-5
    fp32_absolute_tolerance: float = 3.0517578125e-5
    rank_rule: str = "rtol=max(m,n)*torch.finfo(float32).eps;atol=0"
    dense_inverse_count: int = 0
    candidate_rescale_max_count: int = 1
    seed_namespace: str = "odeedit-s06-barrier-usefulness-f2-v1"
    action_representative_candidate_id: str = "axis-00-minus"
    selector_tie_break: str = "lexical-candidate-id"
    margin_log_floor_rule: str = "max(256*float32_eps,8*cold_replay_abs_margin_drift_max)"
    realized_z_rule: str = "max(256*float32_eps,8*cold_official_context_replay_relative_max)"
    nuisance_rule: str = "max(cold-replay,rebatch,order,fp32-preflight upper bounds)"

    def payload(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "BarrierLock",
    "MODEL_SPECS",
    "Method",
    "Phase",
    "ScientificBoundary",
    "Selector",
    "TechnicalBoundary",
]
