"""Fail-close numerical/source panel for P2R5 Stage A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p2r5_sdrt_writer import P2R5_ARMS, P2R5_INSTRUCTION_ID
from .p2r5_stage_a_runtime import P2R4_PARENT_HEAD, STAGE_A_CASES, STREAM_ORDER, STREAM_ROOT


LOCK_FILE = "numerical_lock_s05_p2r5_sdrt_stage_a.json"
PARENT = P2R4_PARENT_HEAD


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, raw_sha = load_rooted_json(
        path,
        expected_schema="ode-edit-s05-p2r5-sdrt-stage-a-numerical-lock/v1",
    )
    expected_cases = {key: list(items) for key, items in STAGE_A_CASES.items()}
    if (
        value.get("instruction_id") != P2R5_INSTRUCTION_ID
        or value.get("exact_p2r4_parent") != PARENT
        or value.get("stream_root") != STREAM_ROOT
        or value.get("all_request_order_sha256") != STREAM_ORDER
        or value.get("stage_a_cases") != expected_cases
        or value.get("arms") != list(P2R5_ARMS)
        or value.get("clamp_policy") != "ON_DECISION_ACTIVE_EVERY_TARGET_MICROSTEP"
        or value.get("target_microstep_count") != 24
        or value.get("writer_transition_count") != 8
        or value.get("request_layer_alpha_shape") != [5, 10]
        or value.get("coefficient_mass_constraint") != "NONNEGATIVE_SUM_LAYER_PER_REQUEST_LE_ONE"
        or value.get("eta_entry") != 1
        or value.get("eta_floor_count") != 0
        or value.get("candidate_materialization_count") != 0
        or value.get("stage_b_status") != "CLOSED_PENDING_GH_STAGE_A_REVIEW"
        or value.get("scientific_promotion_authorized") is not False
    ):
        raise ODEBFContractError("P2R5 numerical lock differs")
    return value, raw_sha


__all__ = ["LOCK_FILE", "PARENT", "load_and_validate_lock"]
