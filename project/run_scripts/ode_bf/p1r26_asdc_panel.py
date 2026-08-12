"""Closed P1R26 ASDC roles and result namespaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError
from .p1_scalable_batched_runtime_panel import forecast_p1r23_panel


P1R26_ROLES = (
    "P1R26_B1_RS_PAIR",
    "P1R26_B1_BG_PAIR",
    "P1R26_B10X10_RS_PAIR",
    "P1R26_B10X10_BG_PAIR",
)
LOCK_FILE = "numerical_lock_s05_p1r26_asdc.json"
LOCK_SCHEMA = "ode-edit-s05-p1r26-asdc-lock/v1"


def expected_p1r26_result_name(alias: str, role: str) -> str:
    if alias not in MODEL_ALIASES or role not in P1R26_ROLES:
        raise ODEBFContractError("P1R26 result identity differs")
    suffix = {
        "P1R26_B1_RS_PAIR": "b1-rs-asdc-neutral-soft-pair",
        "P1R26_B1_BG_PAIR": "b1-bg-asdc-neutral-soft-pair",
        "P1R26_B10X10_RS_PAIR": "independent-b10x10-rs-asdc-neutral-soft-pair",
        "P1R26_B10X10_BG_PAIR": "independent-b10x10-bg-asdc-neutral-soft-pair",
    }[role]
    return f"s05-p1r26-{alias}-{suffix}-v1"


def forecast_p1r26_panel(
    artifact_lock_path: Path, base_model_lock_path: Path, alias: str
) -> Any:
    return forecast_p1r23_panel(artifact_lock_path, base_model_lock_path, alias)


def validate_p1r26_lock(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != LOCK_SCHEMA
        or value.get("instruction_id")
        != "ODEEDIT-S05-P1R26-ABSOLUTE-SEMANTIC-DEFICIT-CONTROLLER-V1"
        or value.get("scientific_base")
        != "ce8c6c36348752f1407f7d713d30e6b5c727379b"
        or value.get("epsilon_z") != 0.05
        or value.get("grid") != {"K": 8, "h": 0.125, "tau_final": 1.0}
        or value.get("allocations") != ["RS", "BG"]
        or value.get("execution", {}).get("project_gpu_cap_server1") != 4
        or value.get("execution", {}).get("array_max_concurrent_gpu") != 4
        or value.get("execution", {}).get("callbacks") != 0
        or value.get("stream", {}).get("root")
        != "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
    ):
        raise ODEBFContractError("P1R26 numerical lock differs")


def load_and_validate_p1r26_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, digest = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_p1r26_lock(value)
    return value, digest


__all__ = [
    "LOCK_FILE",
    "P1R26_ROLES",
    "expected_p1r26_result_name",
    "forecast_p1r26_panel",
    "load_and_validate_p1r26_lock",
]
