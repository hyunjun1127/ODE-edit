"""Canonical one-shot Native MEMIT/AlphaEdit reference boundary for P4 Euler."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .contracts import ODEBFContractError, canonical_hash
from .p4_euler_integrator import P4_EULER_INSTRUCTION_ID


NATIVE_SOURCE_MEMBERS = {
    "native-memit": (
        "easyeditor/models/memit/memit_main.py",
        "easyeditor/models/memit/compute_z.py",
        "easyeditor.models.memit.memit_main.apply_memit_to_model",
        "easyeditor.models.memit.compute_z.compute_z",
    ),
    "native-alphaedit": (
        "easyeditor/models/alphaedit/AlphaEdit_main.py",
        "easyeditor/models/alphaedit/compute_z.py",
        "easyeditor.models.alphaedit.AlphaEdit_main.apply_AlphaEdit_to_model",
        "easyeditor.models.alphaedit.compute_z.compute_z",
    ),
}


def _regular_source(root: Path, relative: str) -> tuple[Path, str, int]:
    member = root / relative
    if member.is_symlink() or not member.is_file():
        raise ODEBFContractError("P4 Euler Native canonical source differs")
    resolved_root = root.resolve(strict=True)
    resolved = member.resolve(strict=True)
    if resolved_root not in resolved.parents:
        raise ODEBFContractError("P4 Euler Native source escaped EasyEdit root")
    payload = member.read_bytes()
    return resolved, hashlib.sha256(payload).hexdigest(), len(payload)


def build_native_source_contract(
    easyedit_root: Path, *, method: str
) -> Mapping[str, Any]:
    if method not in NATIVE_SOURCE_MEMBERS:
        raise ODEBFContractError("P4 Euler Native method differs")
    main_relative, compute_relative, entrypoint, compute_entrypoint = (
        NATIVE_SOURCE_MEMBERS[method]
    )
    main_path, main_sha, main_size = _regular_source(easyedit_root, main_relative)
    compute_path, compute_sha, compute_size = _regular_source(
        easyedit_root, compute_relative
    )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-native-source-contract/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "method": method,
        "easyedit_root": str(easyedit_root.resolve(strict=True)),
        "main_source": {
            "path": str(main_path),
            "sha256": main_sha,
            "bytes": main_size,
        },
        "compute_z_source": {
            "path": str(compute_path),
            "sha256": compute_sha,
            "bytes": compute_size,
        },
        "canonical_entrypoint": entrypoint,
        "canonical_compute_z_entrypoint": compute_entrypoint,
        "execution_mode": "CANONICAL_ONE_SHOT",
        "external_composite_reference": True,
        "causal_panel_membership": False,
        "ode_edit_optimizer_override_count": 0,
        "ode_edit_compute_z_override_count": 0,
        "k8_count": 0,
        "waypoint_count": 0,
        "provided_z_override_count": 0,
        "schedule_match_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def verify_native_execution_boundary(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    if (
        receipt.get("execution_mode") != "CANONICAL_ONE_SHOT"
        or receipt.get("causal_panel_membership") is not False
        or any(
            receipt.get(name) != 0
            for name in (
                "ode_edit_optimizer_override_count",
                "ode_edit_compute_z_override_count",
                "k8_count",
                "waypoint_count",
                "provided_z_override_count",
                "schedule_match_count",
            )
        )
    ):
        raise ODEBFContractError("P4 Euler Native execution boundary differs")
    return receipt


__all__ = [
    "NATIVE_SOURCE_MEMBERS",
    "build_native_source_contract",
    "verify_native_execution_boundary",
]
