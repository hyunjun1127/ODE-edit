"""Closed identities for P1R52 PIR-U Alpha-cache continuity A1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json, sha256_file
from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .p1r52_piru_cache_continuity import (
    CACHE_COMPLETE_ROLE,
    CONTRACT_SHA256,
    INSTRUCTION_ID,
    LEGACY_ROLE,
)
from .p1r52_piru_sequential_panel import STREAM_ORDER, STREAM_ROOT


PARENT = "4f25efdc8ccd9e9017a7e44aa99f92dccd01c31e"
PARENT_TREE = "0b621c01a20b31af0925f1e18ab1f4932f1656b6"
LOCK_FILE = "numerical_lock_s05_p1r52_piru_cache_continuity_a1.json"
LOCK_SCHEMA = "ode-edit-s05-p1r52-piru-cache-continuity-a1-lock/v1"
SOURCE_MANIFEST_FILE = "source_manifest_s05_p1r52_piru_cache_continuity_a1.json"
SOURCE_MANIFEST_SCHEMA = "ode-edit-s05-p1r52-piru-cache-continuity-a1-source-manifest/v1"

SEALED_LEGACY_ROOT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r52-piru-seq-postenergy-warn-r1-v1/local/odebf/results/"
    "s05-p1r52-pir-u-soft-sequential-structuralh-on-10xb100-"
    "postenergy-warn-r1-tech-r1-v1"
)


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "contract_sha256": CONTRACT_SHA256,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "model": "llama3-8b-inst",
        "roles": [LEGACY_ROLE, CACHE_COMPLETE_ROLE],
        "round_count": 10,
        "batch_size": 100,
        "request_count": 1000,
        "k": 8,
        "h": 0.125,
        "writer_policy": "PIR-U",
        "gamma": 1.0,
        "legacy_prefix_history": "EMPTY_L5_L8",
        "cache_complete_prefix_history": "COMMITTED_PAST_L5_L8",
        "stage_a_cache_complete_rounds": [10],
        "stage_b_cache_complete_rounds": list(range(1, 11)),
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "batch_entry_evaluator_count": 0,
        "postsolve_energy_policy": "WARN_CONTINUE_IF_POSITIVE",
        "terminal_w0_restore_count": 1,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("PIR-U cache-continuity numerical lock differs")
    for name in (
        "dynamic_pi_count",
        "current_suffix_slope_backward_count",
        "retry_count",
        "backtracking_count",
        "line_search_count",
        "current_batch_history_inclusion_count",
        "current_prefix_history_inclusion_count",
        "easyedit_write_count",
    ):
        if value.get(name) != 0:
            raise ODEBFContractError("PIR-U cache-continuity forbidden count differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


def verify_sealed_legacy() -> dict[str, Any]:
    terminal = SEALED_LEGACY_ROOT / "terminal.json"
    manifest = SEALED_LEGACY_ROOT / "manifest.json"
    b10 = SEALED_LEGACY_ROOT / "raw/batches/b10/terminal.json"
    if any(path.is_symlink() or not path.is_file() for path in (terminal, manifest, b10)):
        raise ODEBFStateError("sealed PIR-U Legacy reference is absent")
    terminal_value = json.loads(terminal.read_text(encoding="utf-8"))
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    b10_value = json.loads(b10.read_text(encoding="utf-8"))
    terminal_sha = sha256_file(terminal)
    if (
        manifest_value.get("terminal_sha256") != terminal_sha
        or terminal_value.get("request_count") != 1000
        or terminal_value.get("terminal_W0_restore", {}).get("pointer_restored_exact") is not True
        or terminal_value.get("terminal_W0_restore", {}).get("byte_restored_exact") is not True
        or b10_value.get("history_width_at_entry") != 900
    ):
        raise ODEBFStateError("sealed PIR-U Legacy reference identity differs")
    payload = {
        "root": str(SEALED_LEGACY_ROOT),
        "terminal_sha256": terminal_sha,
        "manifest_sha256": sha256_file(manifest),
        "b10_terminal_sha256": sha256_file(b10),
        "b10_entry_weight_sha256": b10_value["entry_weight_sha256"],
        "b10_history_width": 900,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "SOURCE_MANIFEST_FILE",
    "SOURCE_MANIFEST_SCHEMA",
    "load_and_validate_lock",
    "verify_sealed_legacy",
]
