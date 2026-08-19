"""Closed identities for the P1R52 PIR-U sequential 10xB100 cell."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json, sha256_file
from .contracts import ODEBFContractError, ODEBFStateError
from .p1r52_piru_sequential_adapter import (
    P1R52_PIRU_SEQUENTIAL_INSTRUCTION_ID,
    P1R52_PIRU_SEQUENTIAL_ROLE,
)


PARENT = "1d1ba7554a50deb7a92276d32c567542771e1551"
PARENT_TREE = "196709caad3ff29c190c326626f788197df8f3cc"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
LOCK_FILE = "numerical_lock_s05_p1r52_piru_sequential_10xb100.json"
LOCK_SCHEMA = "ode-edit-s05-p1r52-piru-sequential-10xb100-lock/v1"
SOURCE_MANIFEST_FILE = "source_manifest_s05_p1r52_piru_sequential_10xb100.json"
SOURCE_MANIFEST_SCHEMA = (
    "ode-edit-s05-p1r52-piru-sequential-10xb100-source-manifest/v1"
)

REFERENCE_BASE = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r52-llama-seq-10xb100-fourarm-r1-v1/local/odebf/results"
)
REFERENCE_RESULTS = {
    "Official MEMIT": "s05-p1r52-official-memit-sequential-10xb100-tech-r1-v1",
    "Official AlphaEdit": "s05-p1r52-official-alphaedit-sequential-cache-on-10xb100-tech-r1-v1",
    "P1R52 J0 Structural-H ON": (
        "s05-p1r52-llama-soft-sequential-structuralh-on-10xb100-tech-r3-release-r1-v1"
    ),
}


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": P1R52_PIRU_SEQUENTIAL_INSTRUCTION_ID,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "model": "llama3-8b-inst",
        "role": P1R52_PIRU_SEQUENTIAL_ROLE,
        "round_count": 10,
        "batch_size": 100,
        "request_count": 1000,
        "k": 8,
        "h": 0.125,
        "writer_policy": "PIR-U",
        "gamma": 1.0,
        "alpha_zero_branch": "ZERO_FACTORS",
        "prefix_capture_per_k": 4,
        "q_solve_per_k": 4,
        "additional_current_slope_backward_per_k": 0,
        "materialization_per_k": 1,
        "alpha_cache": "ON",
        "structural_h": "ON",
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "batch_entry_evaluator_count": 0,
        "terminal_w0_restore_count": 1,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R52 PIR-U sequential numerical lock differs")
    for name in (
        "retry_count",
        "backtracking_count",
        "line_search_count",
        "layerwise_commit_count",
        "heldout_inner_access_count",
        "easyedit_write_count",
    ):
        if value.get(name) != 0:
            raise ODEBFContractError("P1R52 PIR-U forbidden count differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


def verify_reference_results() -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for label, name in REFERENCE_RESULTS.items():
        root = REFERENCE_BASE / name
        terminal = root / "terminal.json"
        manifest = root / "manifest.json"
        stage = root / "raw/stage-001-post_preflight.json"
        if any(path.is_symlink() or not path.is_file() for path in (terminal, manifest, stage)):
            raise ODEBFStateError("P1R52 PIR-U sealed reference is absent")
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
        terminal_value = json.loads(terminal.read_text(encoding="utf-8"))
        stage_value = json.loads(stage.read_text(encoding="utf-8"))["payload"]
        terminal_sha = sha256_file(terminal)
        if (
            manifest_value.get("terminal_sha256") != terminal_sha
            or manifest_value.get("request_count") != 1000
            or manifest_value.get("W0_restored") is not True
            or terminal_value.get("request_count") != 1000
            or stage_value.get("stream_root_digest") != STREAM_ROOT
            or stage_value.get("stream_request_count") != 1000
        ):
            raise ODEBFStateError("P1R52 PIR-U sealed reference identity differs")
        verified[label] = {
            "root": str(root),
            "terminal_sha256": terminal_sha,
            "manifest_sha256": sha256_file(manifest),
            "manifest_identity_sha256": manifest_value.get("identity_sha256"),
            "terminal_identity_sha256": terminal_value.get("identity_sha256"),
            "source_head": terminal_value.get("source_head"),
            "request_count": 1000,
            "W0_restored": True,
            "stream_root": STREAM_ROOT,
        }
    payload = {
        "references": verified,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
    }
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "SOURCE_MANIFEST_FILE",
    "SOURCE_MANIFEST_SCHEMA",
    "STREAM_ORDER",
    "STREAM_ROOT",
    "load_and_validate_lock",
    "validate_lock",
    "verify_reference_results",
]
