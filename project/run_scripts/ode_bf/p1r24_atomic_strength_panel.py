"""Closed execution identities for P1R24 Atomic strength recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import MODEL_ALIASES, ODEBFContractError
from .p1_scalable_batched_runtime_panel import forecast_p1r23_panel


P1R24_ROLES = (
    "P1R24_B1_RS_NEUTRAL",
    "P1R24_B10_RS_PAIR",
    "P1R24_B10_BG_PAIR",
)


def expected_p1r24_result_name(alias: str, role: str) -> str:
    if alias not in MODEL_ALIASES or role not in P1R24_ROLES:
        raise ODEBFContractError("P1R24 result identity differs")
    suffix = {
        "P1R24_B1_RS_NEUTRAL": "b1-rs-neutral-smoke",
        "P1R24_B10_RS_PAIR": "b10-rs-neutral-soft-pair",
        "P1R24_B10_BG_PAIR": "b10-bg-neutral-soft-pair",
    }[role]
    return f"s05-p1r24-{alias}-{suffix}-tech-r6-v1"


def forecast_p1r24_panel(
    artifact_lock_path: Path, base_model_lock_path: Path, alias: str
) -> Any:
    return forecast_p1r23_panel(artifact_lock_path, base_model_lock_path, alias)


__all__ = ["P1R24_ROLES", "expected_p1r24_result_name", "forecast_p1r24_panel"]
