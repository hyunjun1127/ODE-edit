"""Closed execution identities for P1R35 full-current-residual runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import MODEL_ALIASES, ODEBFContractError
from .p1_scalable_batched_runtime_panel import forecast_p1r23_panel


P1R35_ROLES = ("P1R35_B1_RS_PAIR", "P1R35_B10_RS_PAIR")


def expected_p1r35_result_name(alias: str, role: str) -> str:
    if alias not in MODEL_ALIASES or role not in P1R35_ROLES:
        raise ODEBFContractError("P1R35 result identity differs")
    suffix = {
        "P1R35_B1_RS_PAIR": "b1-rs-full-current-residual-neutral-soft-pair",
        "P1R35_B10_RS_PAIR": "b10-rs-full-current-residual-neutral-soft-pair",
    }[role]
    return f"s05-p1r35-{alias}-{suffix}-v1"


def forecast_p1r35_panel(
    artifact_lock_path: Path, base_model_lock_path: Path, alias: str
) -> Any:
    return forecast_p1r23_panel(artifact_lock_path, base_model_lock_path, alias)


__all__ = ["P1R35_ROLES", "expected_p1r35_result_name", "forecast_p1r35_panel"]
