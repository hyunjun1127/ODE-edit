"""Typed two-cell binding for missing native sequential final-W NLL telemetry.

This module does not implement either native method.  It restricts execution to
the two released Phase-1 Official EasyEdit roles whose runtime already records
the complete final-W10 request table.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ODEBFContractError, canonical_hash
from .p1r52_c_writer_phase1_sequential import (
    expected_result_name as phase1_expected_result_name,
    label_for_role as phase1_label_for_role,
    role_for_cell as phase1_role_for_cell,
)


INSTRUCTION_ID = "ODEEDIT-S05-P1R54-NATIVE-SEQUENTIAL-W-NLL-BACKFILL-V1"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"


@dataclass(frozen=True)
class NativeSequentialWNLLCell:
    cell: int
    label: str
    role: str
    result_name: str
    scientific_call_path: str
    alpha_cache_semantics: str


def _cell(cell: int) -> NativeSequentialWNLLCell:
    if isinstance(cell, bool) or cell not in (0, 1):
        raise ODEBFContractError("native sequential W-NLL cell differs")
    role = phase1_role_for_cell(cell)
    label = phase1_label_for_role(role)
    expected_label = ("OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT")[cell]
    if label != expected_label:
        raise ODEBFContractError("native sequential Phase1 source mapping differs")
    return NativeSequentialWNLLCell(
        cell=cell,
        label=label,
        role=role,
        result_name=phase1_expected_result_name(role),
        scientific_call_path=(
            "easyeditor.models.alphaedit.AlphaEdit_main.apply_AlphaEdit_to_model"
            if cell == 0
            else "easyeditor.models.memit.memit_main.apply_memit_to_model"
        ),
        alpha_cache_semantics=(
            "SEQUENTIAL_CACHE_C_CONTINUITY_0_TO_900"
            if cell == 0
            else "NOT_APPLICABLE_STATIC_COV"
        ),
    )


CELLS = tuple(_cell(index) for index in range(2))


def cell_for_index(cell: int) -> NativeSequentialWNLLCell:
    if isinstance(cell, bool) or cell not in (0, 1):
        raise ODEBFContractError("native sequential W-NLL cell differs")
    return CELLS[cell]


def source_equivalence_receipt() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r54-native-sequential-w-nll-source-equivalence/v1",
        "instruction_id": INSTRUCTION_ID,
        "cell_count": 2,
        "cells": [
            {
                "cell": item.cell,
                "label": item.label,
                "role": item.role,
                "result_name": item.result_name,
                "scientific_call_path": item.scientific_call_path,
                "alpha_cache_semantics": item.alpha_cache_semantics,
            }
            for item in CELLS
        ],
        "phase1_original_runtime_reuse": True,
        "native_equation_change_count": 0,
        "native_writer_change_count": 0,
        "stream_or_evaluator_change_count": 0,
        "new_native_arm_count": 0,
        "rerun_reason": "MISSING_FINAL_W10_REQUEST_NLL_RAW_ON_SERVER4",
        "required_output": "immediate-post-final-w10-requests.json",
        "required_request_count_per_cell": 1000,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "CELLS",
    "INSTRUCTION_ID",
    "NativeSequentialWNLLCell",
    "STREAM_ORDER",
    "STREAM_ROOT",
    "cell_for_index",
    "source_equivalence_receipt",
]
