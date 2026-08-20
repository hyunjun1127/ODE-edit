"""Closed identities for P1R52 target + Official AlphaEdit writer A1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p1r52_target_official_alphaedit_writer import INSTRUCTION_ID, PHASE_A_ROLE


PARENT = "4f25efdc8ccd9e9017a7e44aa99f92dccd01c31e"
PARENT_TREE = "0b621c01a20b31af0925f1e18ab1f4932f1656b6"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
LOCK_FILE = "numerical_lock_s05_p1r52_target_official_alphaedit_writer_a1.json"
LOCK_SCHEMA = "ode-edit-s05-p1r52-target-official-alphaedit-writer-a1-lock/v1"
SOURCE_MANIFEST_FILE = "source_manifest_s05_p1r52_target_official_alphaedit_writer_a1.json"
SOURCE_MANIFEST_SCHEMA = "ode-edit-s05-p1r52-target-official-alphaedit-writer-a1-source-manifest/v1"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "model": "llama3-8b-inst",
        "role": PHASE_A_ROLE,
        "phase_a_batch_size": 100,
        "phase_a_case_count": 10,
        "k": 8,
        "h": 0.125,
        "accepted_z_source": "P1R52_K8_TERMINAL_TARGET",
        "pilot_gate": "OFFICIAL_REPHRASE_W_MINUS_Z_GAP_LT_P1R52_WRITER_GAP",
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "native_alphaedit_compute_z_call_count": 0,
        "official_alphaedit_cache_c": "ON",
        "official_alphaedit_static_p": "ON",
        "batch_entry_evaluator_count": 0,
        "retry_count": 0,
        "imputation_count": 0,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("target/Official AlphaEdit writer numerical lock differs")
    for name in (
        "r52_structural_h_influence_count",
        "r52_p_barrier_influence_count",
        "r52_energy_capacity_barrier_influence_count",
        "pir_influence_count",
        "piru_influence_count",
        "fpiq_influence_count",
        "r52_writer_historical_risk_influence_count",
        "easyedit_write_count",
    ):
        if value.get(name) != 0:
            raise ODEBFContractError("target/Official writer forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


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
]
