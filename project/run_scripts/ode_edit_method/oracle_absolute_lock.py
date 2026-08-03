"""Fail-closed lock for the oracle absolute-mean-margin V3 runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import MethodContractError, canonical_hash
from .oracle_absolute_event import ORACLE_ABSOLUTE_MEAN_MARGIN_EVENT_MODE
from .oracle_event import ORACLE_MODEL_FORWARD_CALLS, ORACLE_REALIZATION_FRACTION


ORACLE_ABSOLUTE_LOCK_PATH = Path(__file__).with_name(
    "oracle_absolute_mean_margin_v3.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_oracle_absolute_lock(
    path: str | Path = ORACLE_ABSOLUTE_LOCK_PATH,
) -> dict[str, Any]:
    candidate = Path(path).resolve(strict=True)
    try:
        raw = json.loads(
            candidate.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                MethodContractError(f"non-finite V3 lock constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MethodContractError("V3 development lock is invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise MethodContractError("V3 development lock is not an object")
    payload = dict(raw)
    event = payload.get("event")
    selection = payload.get("selection")
    runtime = payload.get("runtime")
    resources = payload.get("resources")
    boundary = payload.get("execution_boundary")
    v2 = payload.get("v2_provenance")
    legacy = payload.get("legacy_diagnostic_provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (event, selection, runtime, resources, boundary, v2, legacy)
    ):
        raise MethodContractError("V3 development lock sections are absent")
    if (
        payload.get("schema_version")
        != "ode-edit-oracle-absolute-mean-margin-development-lock/v3"
        or payload.get("status") != "TECHNICAL_V3_P0P1_EXECUTION_LOCK"
        or payload.get("instruction_id")
        != "ODEEDIT-S02-ORACLE-ABSOLUTE-MEAN-MARGIN-V3-P0P1"
        or payload.get("parent_instruction_id")
        != "ODEEDIT-S02-ORACLE-MEAN-EVENT-V2-R1-P0P1"
        or payload.get("revision_id")
        != "V3_ZERO_MEAN_MARGIN_ORACLE_ABSOLUTE_FLOOR"
        or payload.get("implementation_base")
        != "fcd5e93bf4d25d9efff7eb5912c67fdcd40146df"
        or payload.get("base_proposal_id")
        != "83eec5068acec673f7d9d0788e57ce0e9873a55785f7a6932026a4a2420c330b"
        or payload.get("base_lock_sha256")
        != "95376c554f01e00c9c5d71e99dfa3cbb5339a7534db8a62a9a05bc69c0be1bbb"
        or event.get("mode") != ORACLE_ABSOLUTE_MEAN_MARGIN_EVENT_MODE
        or event.get("rho") != ORACLE_REALIZATION_FRACTION
        or event.get("required_mean_margin") != 0.0
        or event.get("calibration_total_model_forwards") != 4
        or event.get("entry_ordinary_model_forwards") != 2
        or event.get("oracle_extra_model_forwards")
        != ORACLE_MODEL_FORWARD_CALLS
        or event.get("oracle_backward_count") != 0
        or event.get("ordinary_model_forwards_per_event") != 2
        or event.get("smooth_objective_count") != 2
        or tuple(event.get("decision_objectives", ()))
        != (
            "negative-uniform-mean-margin",
            "absolute-new-likelihood-deficit",
        )
        or event.get("oracle_margin_role") != "validity-and-diagnostic-only"
        or event.get("q_margin_decision") is not False
        or event.get("q_new_decision_floor") != ORACLE_REALIZATION_FRACTION
        or event.get("context_weights") != "uniform"
        or event.get("per_context_decision") is not False
        or event.get("legacy_event_role") != "shadow-diagnostic-only"
        or event.get("conditional_fallback") is not False
        or event.get("native_nll_equality_used") is not False
        or event.get("direct_z_before_entry_event") is not True
        or event.get("entry_hit_n_z_zero_allowed") is not False
        or tuple(selection.get("p0_case_ids", ())) != ("2022",)
        or tuple(selection.get("p1_case_ids", ()))
        != ("2022", "12498", "20964", "768")
        or tuple(selection.get("arms", ()))
        != ("native-memit", "static-synchronous", "full-ode-edit")
        or selection.get("seed") != 17
        or selection.get("execution_axis")
        != "arm-outer-sequential-edit-inner"
        or tuple(runtime.get("models", ()))
        != ("llama3-8b-inst", "qwen2.5-7b-inst")
        or runtime.get("dtype_policy") != "checkpoint-original"
        or runtime.get("model_specific_policy") is not False
        or runtime.get("evaluation") is not False
        or runtime.get("generation") is not False
        or v2.get("proposal_id")
        != "78937c0a5fa3c3e92ae20a4089bba6e6e4f6a5c910dcf31f2abb17789d804639"
        or v2.get("lock_sha256")
        != "ff75022dbfca79e11c121d64dbed4f9ad6dcf45154f17b30845e706681046715"
        or v2.get("reused_as_runtime_fallback") is not False
        or legacy.get("legacy_replay_executed") is not False
        or legacy.get("existing_legacy_diagnostic_only") is not True
        or legacy.get("llama_complete_terminal_manifest_sha256")
        != "154791d93b4af12a27d6e2b2c470e098c2035655b9cbb04899b63490f59933db"
        or legacy.get("qwen_partial_tree_sha256")
        != "7f245bb23d306a52dfbcfcee7f818deeaecbdd38e326e6c196a4615ea300b71a"
        or legacy.get("merged_into_v3_timing_or_results") is not False
        or resources.get("server1_project_gpu_cap") != 3
        or resources.get("gpu_per_job") != 1
        or resources.get("cpu_per_job") != 8
        or resources.get("host_memory_mib_per_job") != 65000
        or resources.get("p0_time") != "04:00:00"
        or resources.get("p1_time") != "08:00:00"
        or resources.get("pair_gpu") != 2
        or boundary.get("submission_authorized") is not False
        or boundary.get("retry_authorized") is not False
        or boundary.get("p0_token") != "oracle-absolute-mean-margin-v3-p0"
        or boundary.get("p1_token")
        != "oracle-absolute-mean-margin-v3-p1-after-p0-pass"
        or boundary.get("automatic_p1_condition")
        != "both-p0-terminal-technical-pass"
        or boundary.get("scientific_outcome_count") != 0
    ):
        raise MethodContractError("V3 development lock differs")
    epsilon = event.get("oracle_validity_epsilon")
    if (
        not isinstance(epsilon, (int, float))
        or isinstance(epsilon, bool)
        or epsilon <= 0
    ):
        raise MethodContractError("V3 oracle validity epsilon is invalid")
    payload["proposal_id"] = canonical_hash(payload)
    payload["lock_sha256"] = file_sha256(candidate)
    return payload


__all__ = [
    "ORACLE_ABSOLUTE_LOCK_PATH",
    "file_sha256",
    "load_oracle_absolute_lock",
]
