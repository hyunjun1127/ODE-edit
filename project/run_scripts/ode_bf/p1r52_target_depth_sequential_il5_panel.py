"""Closed execution identities for the P1R52 IL5 sequential B1000 cell."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json, sha256_file
from .contracts import ODEBFContractError
from .p1r52_sequential_b100x10_panel import LOCK_FILE as INHERITED_LOCK_FILE
from .p1r52_sequential_b100x10_panel import load_and_validate_lock as load_inherited_lock
from .p1r52_target_depth_sequential_il5 import (
    INSTRUCTION_ID,
    METHOD_ID,
    POLICY,
    STREAM_ORDER,
    STREAM_ROOT,
)


PARENT = "db30255a39ec48dcd190fecd61cb5fd8c3ce3555"
PARENT_TREE = "f05de718e195d39b0d3c198c8e6474d200544f56"
LOCK_FILE = "numerical_lock_s05_p1r52_llama_j0_il5_sequential_10xb100_tech_r1.json"
LOCK_SCHEMA = "ode-edit-s05-p1r52-llama-j0-il5-sequential-tech-r1-lock/v1"
SOURCE_MANIFEST_FILE = (
    "source_manifest_s05_p1r52_llama_j0_il5_sequential_10xb100_tech_r1.json"
)
SOURCE_MANIFEST_SCHEMA = (
    "ode-edit-s05-p1r52-llama-j0-il5-sequential-tech-r1-source-manifest/v1"
)


def validate_lock(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "exact_parent": PARENT,
        "exact_parent_tree": PARENT_TREE,
        "alias": "llama3-8b-inst",
        "role": POLICY.role,
        "round_count": POLICY.round_count,
        "batch_size": POLICY.batch_size,
        "request_count": POLICY.request_count,
        "outer_k": POLICY.outer_count,
        "target_depth": POLICY.depth.value,
        "target_inner_count": POLICY.depth.inner_count,
        "inner_h": POLICY.inner_h,
        "expected_inner_rows_per_batch": POLICY.expected_inner_rows_per_batch,
        "expected_request_inner_rows_per_batch": (
            POLICY.expected_request_inner_rows_per_batch
        ),
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "writer": "P1R52-J0",
        "allocation": "RS-SOFT",
        "alpha_cache": "ON",
        "structural_h": "ON",
        "physical_weight_persistence": True,
        "batch_entry_evaluator_count": 0,
        "interbatch_w0_restore_count": 0,
        "terminal_w0_restore_count": 1,
        "task_gpu_cap": 3,
        "stage_gpu_max": 1,
        "technical_attempt": "TECH_R1_SERVER2_NODE_BINDING",
        "execution_node": "server2",
    }
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise ODEBFContractError("P1R52 IL5 sequential numerical lock differs")
    for key in (
        "retry_count",
        "backtracking_count",
        "imputation_count",
        "inner_writer_materialization_count",
        "inner_heldout_controller_access_count",
        "observation_action_influence_count",
        "easyedit_write_count",
    ):
        if value.get(key) != 0:
            raise ODEBFContractError("P1R52 IL5 sequential forbidden count differs")


def load_and_validate_lock(repo_root: Path) -> tuple[dict[str, Any], str, str]:
    locks = repo_root / "project/run_scripts/ode_bf/locks"
    inherited, inherited_sha = load_inherited_lock(locks / INHERITED_LOCK_FILE)
    if (
        inherited.get("stream_root") != STREAM_ROOT
        or inherited.get("stream_order") != STREAM_ORDER
    ):
        raise ODEBFContractError("P1R52 IL5 inherited stream lock differs")
    value, file_sha = load_rooted_json(
        locks / LOCK_FILE,
        expected_schema=LOCK_SCHEMA,
    )
    validate_lock(value)
    return value, file_sha, inherited_sha


def verify_source_manifest(repo_root: Path, source_head: str) -> str:
    value, file_sha = load_rooted_json(
        repo_root / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST_FILE,
        expected_schema=SOURCE_MANIFEST_SCHEMA,
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != INSTRUCTION_ID
        or value.get("parent") != PARENT
        or not isinstance(entries, list)
        or not entries
    ):
        raise ODEBFContractError("P1R52 IL5 sequential source manifest differs")
    for entry in entries:
        path = repo_root / str(entry["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(entry["size"])
            or sha256_file(path) != entry["sha256"]
        ):
            raise ODEBFContractError("P1R52 IL5 source member differs")
    return file_sha


__all__ = [
    "INHERITED_LOCK_FILE",
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "SOURCE_MANIFEST_FILE",
    "SOURCE_MANIFEST_SCHEMA",
    "load_and_validate_lock",
    "validate_lock",
    "verify_source_manifest",
]
