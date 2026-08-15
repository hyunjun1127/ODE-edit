"""Fail-close numerical/source panel for the P2R6 focused pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p2r5_stage_a_runtime import P2R4_PARENT_HEAD, STREAM_ORDER, STREAM_ROOT
from .p2r6_pilot_runtime import PHASE1_CASES, PHASE2_CASES
from .p2r6_semantic_region_controller import (
    P2R6_ARMS,
    P2R6_CAP_ARMS,
    P2R6_E1_XI_AUTHORITY,
    P2R6_HIGHS_INTERNAL_TOLERANCE,
    P2R6_INSTRUCTION_ID,
)


LOCK_FILE = "numerical_lock_s05_p2r6_semantic_region_pilot.json"
PARENT = "ff032a10eac257d23c691896458159d2f4eff23b"


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, raw_sha = load_rooted_json(
        path,
        expected_schema="ode-edit-s05-p2r6-semantic-region-pilot-numerical-lock/v1",
    )
    if (
        value.get("instruction_id") != P2R6_INSTRUCTION_ID
        or value.get("exact_p2r5_parent") != PARENT
        or value.get("exact_p2r4_parent") != P2R4_PARENT_HEAD
        or value.get("stream_root") != STREAM_ROOT
        or value.get("all_request_order_sha256") != STREAM_ORDER
        or value.get("phase1_cases") != {key: list(items) for key, items in PHASE1_CASES.items()}
        or value.get("phase2_cases") != {key: list(items) for key, items in PHASE2_CASES.items()}
        or value.get("phase1_arms") != list(P2R6_CAP_ARMS)
        or value.get("all_source_arms") != list(P2R6_ARMS)
        or value.get("clamp_policy") != "ON_DECISION_ACTIVE_EVERY_TARGET_MICROSTEP"
        or value.get("target_microstep_count") != 24
        or value.get("writer_transition_count") != 8
        or value.get("request_layer_alpha_shape") != [5, 10]
        or value.get("coefficient_mass_constraint") != "NONNEGATIVE_SUM_LAYER_PER_REQUEST_LE_ONE"
        or value.get("decision_response") != "RAW_CURRENT_STATE_S"
        or value.get("eta_decision_influence_count_aeta_ar_as") != 0
        or value.get("highs_internal_feasibility_tolerance")
        != P2R6_HIGHS_INTERNAL_TOLERANCE
        or value.get("e1_xi_authority") != P2R6_E1_XI_AUTHORITY
        or value.get("quadratic_certificate_polish")
        != "MONOTONE_ACTIVE_SET_EXPANSION_WITH_UNCHANGED_EXTERNAL_TOLERANCE"
        or value.get("shadow_added_model_forward_backward_materialization") != [0, 0, 0]
        or value.get("stage_b10x10_status") != "NOT_AUTHORIZED"
        or value.get("scientific_promotion_authorized") is not False
    ):
        raise ODEBFContractError("P2R6 numerical lock differs")
    return value, raw_sha


__all__ = ["LOCK_FILE", "PARENT", "load_and_validate_lock"]
