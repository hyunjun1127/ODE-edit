"""Immutable contracts for the atomic-B10 FzCB production pilot.

This module deliberately does not alter the preserved TECH-R1 contract.  The
pilot consumes the same equation-identical controller primitives, but gives
the uncertainty-aware sketch implementation an honest, separate authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ProductionMethod(str, Enum):
    MEMIT = "memit"
    ALPHAEDIT = "alphaedit"


class ProductionArm(str, Enum):
    OFFICIAL_MEMIT = "OFFICIAL_MEMIT"
    REFRESHED_EQUALITY_ONLY_MEMIT = "REFRESHED_EQUALITY_ONLY_MEMIT"
    FZCB_SKETCH_K_MEMIT = "FZCB_SKETCH_K_PROTOTYPE_MEMIT"
    OFFICIAL_ALPHAEDIT = "OFFICIAL_ALPHAEDIT"
    REFRESHED_EQUALITY_ONLY_ALPHAEDIT = "REFRESHED_EQUALITY_ONLY_ALPHAEDIT"
    FZCB_SKETCH_K_ALPHAEDIT = "FZCB_SKETCH_K_PROTOTYPE_ALPHAEDIT"


@dataclass(frozen=True, slots=True)
class ProductionNumericalLock:
    dtype: str = "float32"
    scalar_reduction_dtype: str = "float64"
    progress_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    sketch_ladder: tuple[int, ...] = (2, 8, 32, 128)
    finite_difference_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    finite_difference_repeats: int = 2
    perturbation_coordinate: str = "c(eta)=c0+eta*sqrt(A0)*unit_direction"
    normalized_value: str = "Vbar=V/A0"
    authority: str = "FZCB_SKETCH_K_PROTOTYPE"
    full_suffix_gradient_production_call_count: int = 0
    full_projected_sensitivity_production_call_count: int = 0
    lazy_stop_rule: str = "first_uncertainty_strict_candidate_at_K_2_8_32_128"
    uncertainty_components: tuple[str, ...] = (
        "adjacent_epsilon_or_plateau",
        "suffix_ulp_and_cancellation",
        "solver_tolerance_variation",
        "plus_minus_order",
        "realized_perturbation_asymmetry",
    )
    predictive_waypoint_rule: str = "first_barrier_active_waypoint_per_cohort"
    predictive_candidates: tuple[str, ...] = (
        "EQUALITY_ONLY",
        "FZCB_SELECTED",
        "MATCHED_ACTION_NULL_OPPOSITE",
    )
    stock_compute_z_rule: str = "once_per_request_per_method_cohort_at_W0"
    target_recompute_count: int = 0
    sequential_submit_count: int = 0
    scientific_promotion: bool = False

    def payload(self) -> dict[str, Any]:
        return asdict(self)


CONTRACT_PATH = Path(
    "/data/janghj/ODE-edit/local/state/fzcb-atomic-b10-production-pilot-v1/"
    "authoritative-contract.txt"
)
CONTRACT_SHA256 = "d0f05807e874b84569d3158af70c78e7c6642bf0ab8049b4a2fa4682a49f54e5"
CONTRACT_BYTES = 16676
CONTRACT_LINES = 521
INSTRUCTION_ID = "ODEEDIT-S06-FZCB-ATOMIC-B10-PRODUCTION-PILOT-V1"

OLD_TECH_R1_HEAD = "e9942fc6f8bf6c1d3e799a7eb052582a49c410c6"
OLD_TECH_R1_TREE = "f8fd07b8b171bdf9ce2ff6a17bd0d48a04c4152e"
OLD_TECH_R1_REPORT_SHA256 = "fb9918f54b90f2e11e654f47570bfc47ed3d213d2b7945bec8127df2ff8bd342"
OLD_TECH_R1_EVIDENCE_CLASS = "ENGINEERING_CONTROLLER_AUDIT"


def cohort_arms(method: ProductionMethod) -> tuple[ProductionArm, ...]:
    if method is ProductionMethod.MEMIT:
        return (
            ProductionArm.OFFICIAL_MEMIT,
            ProductionArm.REFRESHED_EQUALITY_ONLY_MEMIT,
            ProductionArm.FZCB_SKETCH_K_MEMIT,
        )
    return (
        ProductionArm.OFFICIAL_ALPHAEDIT,
        ProductionArm.REFRESHED_EQUALITY_ONLY_ALPHAEDIT,
        ProductionArm.FZCB_SKETCH_K_ALPHAEDIT,
    )
