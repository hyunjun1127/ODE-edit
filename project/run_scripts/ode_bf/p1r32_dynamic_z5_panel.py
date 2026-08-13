"""Closed execution identities for P1R32 Dynamic-Z5/full-residual Atomic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import MODEL_ALIASES, ODEBFContractError
from .p1_scalable_batched_runtime_panel import forecast_p1r23_panel


P1R32_ROLES = ("P1R32_B1_PAIR", "P1R32_B10_PAIR")


def expected_p1r32_result_name(
    alias: str, role: str, *, attempt: str = "initial"
) -> str:
    if alias not in MODEL_ALIASES or role not in P1R32_ROLES:
        raise ODEBFContractError("P1R32 result identity differs")
    if attempt not in ("initial", "tech-r2"):
        raise ODEBFContractError("P1R32 result attempt differs")
    suffix = {
        "P1R32_B1_PAIR": "b1-pair",
        "P1R32_B10_PAIR": "b10-pair",
    }[role]
    attempt_suffix = "" if attempt == "initial" else f"-{attempt}"
    return f"s05-p1r32-{alias}-dynz5-fullres-{suffix}{attempt_suffix}-v1"


def forecast_p1r32_panel(
    artifact_lock_path: Path, base_model_lock_path: Path, alias: str
) -> Any:
    return forecast_p1r23_panel(artifact_lock_path, base_model_lock_path, alias)


__all__ = [
    "P1R32_ROLES",
    "expected_p1r32_result_name",
    "forecast_p1r32_panel",
]
