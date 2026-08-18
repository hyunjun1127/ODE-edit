"""Closed identities for the P1R52 B100 accepted-z observation panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p1r52_accepted_z_observation import INSTRUCTION_ID
from .p1r52_sequential_runtime import (
    MEMIT_ROLE,
    NATIVE_CORRECTED_ROLE,
    R52_CONTROL_ROLE,
    R52_H_ROLE,
    RESULT_NAMES_B100X10_ACCEPTED_Z_OBS,
)


SOURCE_PARENT = "2f6bd2f371ecdd6936500bb0bb87bc52a172fc05"
SOURCE_PARENT_TREE = "cf3d6b1114068323113c861745e8a1cb3c27ac0f"
BRANCH = "codex/p1r52-b100-accepted-z-rephrase-obs-v1"
LOCK_SCHEMA = "ode-edit-s05-p1r52-b100-accepted-z-rephrase-observation-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r52_b100_accepted_z_rephrase_observation.json"
MANIFEST_SCHEMA = (
    "ode-edit-s05-p1r52-b100-accepted-z-rephrase-observation-source-manifest/v1"
)
MANIFEST_FILE = "source_manifest_s05_p1r52_b100_accepted_z_rephrase_observation.json"
MANIFEST_TECH_R1_FILE = (
    "source_manifest_s05_p1r52_b100_accepted_z_rephrase_observation_tech_r1.json"
)
MANIFEST_TECH_R1_BASELINES_FILE = (
    "source_manifest_s05_p1r52_b100_accepted_z_rephrase_observation_"
    "tech_r1_baselines.json"
)
ROLES = (MEMIT_ROLE, NATIVE_CORRECTED_ROLE, R52_H_ROLE, R52_CONTROL_ROLE)
SEALED_RESULT_ROOT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r52-llama-seq-10xb100-fourarm-r1-v1/local/odebf/results"
)
REFERENCE_NAMES = {
    MEMIT_ROLE: "s05-p1r52-official-memit-sequential-10xb100-tech-r1-v1",
    NATIVE_CORRECTED_ROLE: (
        "s05-p1r52-official-alphaedit-sequential-cache-on-10xb100-tech-r1-v1"
    ),
    R52_H_ROLE: (
        "s05-p1r52-llama-soft-sequential-structuralh-on-10xb100-"
        "tech-r3-release-r1-v1"
    ),
    R52_CONTROL_ROLE: (
        "s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb100-"
        "tech-r3-release-r1-v1"
    ),
}
REFERENCE_TERMINAL_SHA256 = {
    MEMIT_ROLE: "a6e8c27a170f00a98337fc8837f9242c4c8a1b1fce3efcfcb026159859f2a23f",
    NATIVE_CORRECTED_ROLE: (
        "f90aa3169d05bf0b5236884ca392f27990f02e78571f48ab67075713a4a800bf"
    ),
    R52_H_ROLE: "552aef6525931e6cef944017d81ea4eadb03324148ec674cf467775809de158d",
    R52_CONTROL_ROLE: (
        "ddbbcf7e3d266ba146e41c0ef50ac74f54f753b7cae503105195ea03c1f31910"
    ),
}


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "source_parent": SOURCE_PARENT,
        "source_parent_tree": SOURCE_PARENT_TREE,
        "model": "llama3-8b-inst",
        "roles": list(ROLES),
        "round_count": 10,
        "batch_size": 100,
        "request_count_per_role": 1000,
        "stream_root": (
            "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
        ),
        "stream_order": (
            "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
        ),
        "native_target_layer": 8,
        "native_lookup": "subject_last",
        "overlay": "METHOD_NATIVE_ABSOLUTE_ACCEPTED_Z_REPLACEMENT",
        "batch_entry_W_metrics_count": 0,
        "added_backward_count": 0,
        "added_generation_call_count": 0,
        "action_influence_count": 0,
        "duplicate_W_evaluator_model_forward_count": 0,
        "project_gpu_cap": 4,
        "stage_gpu_max": 4,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("accepted-z observation numerical lock differs")
    if value.get("reference_names") != REFERENCE_NAMES:
        raise ODEBFContractError("accepted-z sealed reference names differ")
    if value.get("reference_terminal_sha256") != REFERENCE_TERMINAL_SHA256:
        raise ODEBFContractError("accepted-z sealed reference terminals differ")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


def expected_result_name(role: str, *, repair_revision: str | None = None) -> str:
    if role not in RESULT_NAMES_B100X10_ACCEPTED_Z_OBS:
        raise ODEBFContractError("accepted-z observation role differs")
    name = RESULT_NAMES_B100X10_ACCEPTED_Z_OBS[role]
    if repair_revision is None:
        return name
    if repair_revision != "TECH-R1":
        raise ODEBFContractError("accepted-z observation repair revision differs")
    return name.removesuffix("-v1") + "-tech-r1-v1"


def sealed_reference_root(role: str) -> Path:
    if role not in REFERENCE_NAMES:
        raise ODEBFContractError("accepted-z sealed reference role differs")
    return SEALED_RESULT_ROOT / REFERENCE_NAMES[role]


__all__ = [
    "BRANCH",
    "INSTRUCTION_ID",
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "MANIFEST_FILE",
    "MANIFEST_TECH_R1_FILE",
    "MANIFEST_TECH_R1_BASELINES_FILE",
    "MANIFEST_SCHEMA",
    "REFERENCE_NAMES",
    "REFERENCE_TERMINAL_SHA256",
    "ROLES",
    "SOURCE_PARENT",
    "SOURCE_PARENT_TREE",
    "expected_result_name",
    "load_and_validate_lock",
    "sealed_reference_root",
    "validate_lock",
]
