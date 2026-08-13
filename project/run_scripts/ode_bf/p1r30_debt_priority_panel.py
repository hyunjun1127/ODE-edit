"""Closed server2 identities for P1R30 atomic debt-priority stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import MODEL_ALIASES, ODEBFContractError
from .p1_scalable_batched_runtime_panel import forecast_p1r23_panel


P1R30_ROLES = (
    "P1R30_B1_RS_REFERENCE_SOFT",
    "P1R30_B10_RS_DEBT_NEUTRAL",
    "P1R30_B10_RS_DEBT_SOFT",
    "P1R30_B10_BG_PAIR",
)


def expected_p1r30_result_name(alias: str, role: str) -> str:
    if alias not in MODEL_ALIASES or role not in P1R30_ROLES:
        raise ODEBFContractError("P1R30 result identity differs")
    suffix = {
        "P1R30_B1_RS_REFERENCE_SOFT": "b1-rs-a0-reference-soft-smoke",
        "P1R30_B10_RS_DEBT_NEUTRAL": "b10-rs-debt-priority-neutral",
        "P1R30_B10_RS_DEBT_SOFT": "b10-rs-debt-priority-soft",
        "P1R30_B10_BG_PAIR": "b10-bg-debt-priority-neutral-soft-pair",
    }[role]
    return f"s05-p1r30-{alias}-{suffix}-v1"


def forecast_p1r30_panel(
    artifact_lock_path: Path, base_model_lock_path: Path, alias: str
) -> Any:
    return forecast_p1r23_panel(artifact_lock_path, base_model_lock_path, alias)


__all__ = [
    "P1R30_ROLES",
    "expected_p1r30_result_name",
    "forecast_p1r30_panel",
]
