"""Fail-closed overlay lock for the oracle-mean V2 development runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import MethodContractError, canonical_hash
from .oracle_event import (
    ORACLE_MEAN_EVENT_MODE,
    ORACLE_MODEL_FORWARD_CALLS,
    ORACLE_REALIZATION_FRACTION,
)


ORACLE_LOCK_PATH = Path(__file__).with_name("oracle_mean_event_v2.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_oracle_lock(path: str | Path = ORACLE_LOCK_PATH) -> dict[str, Any]:
    candidate = Path(path).resolve(strict=True)
    try:
        raw = json.loads(
            candidate.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                MethodContractError(f"non-finite oracle lock constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MethodContractError("oracle development lock is invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise MethodContractError("oracle development lock is not an object")
    payload = dict(raw)
    event = payload.get("event")
    selection = payload.get("selection")
    runtime = payload.get("runtime")
    resources = payload.get("resources")
    boundary = payload.get("execution_boundary")
    if not all(
        isinstance(value, Mapping)
        for value in (event, selection, runtime, resources, boundary)
    ):
        raise MethodContractError("oracle development lock sections are absent")
    if (
        payload.get("schema_version")
        != "ode-edit-oracle-mean-event-development-lock/v2"
        or payload.get("status") != "TECHNICAL_P0P1_EXECUTION_LOCK"
        or payload.get("instruction_id")
        != "ODEEDIT-S02-ORACLE-MEAN-EVENT-V2-R1-P0P1"
        or payload.get("parent_instruction_id")
        != "ODEEDIT-S02-ORACLE-MEAN-EVENT-V2-P0P1"
        or payload.get("revision_id")
        != "R1_EXACT_ZERO_ACCOUNTING_FIREWALL"
        or event.get("mode") != ORACLE_MEAN_EVENT_MODE
        or event.get("rho") != ORACLE_REALIZATION_FRACTION
        or event.get("oracle_model_forwards_per_edit")
        != ORACLE_MODEL_FORWARD_CALLS
        or event.get("ordinary_model_forwards_per_event") != 2
        or event.get("context_weights") != "uniform"
        or event.get("per_context_decision") is not False
        or event.get("conditional_fallback") is not False
        or event.get("direct_z_before_entry_event") is not True
        or event.get("entry_hit_n_z_zero_allowed") is not False
        or tuple(selection.get("p0_case_ids", ())) != ("2022",)
        or tuple(selection.get("p1_case_ids", ()))
        != ("2022", "12498", "20964", "768")
        or tuple(selection.get("arms", ()))
        != ("native-memit", "static-synchronous", "full-ode-edit")
        or selection.get("seed") != 17
        or tuple(runtime.get("models", ()))
        != ("llama3-8b-inst", "qwen2.5-7b-inst")
        or runtime.get("model_specific_policy") is not False
        or runtime.get("evaluation") is not False
        or runtime.get("generation") is not False
        or resources.get("server1_project_gpu_cap") != 3
        or resources.get("pair_gpu") != 2
        or boundary.get("submission_authorized") is not False
        or boundary.get("retry_authorized") is not False
        or boundary.get("scientific_outcome_count") != 0
        or boundary.get("p0_token") != "oracle-mean-event-v2-r1-p0"
        or boundary.get("p1_token")
        != "oracle-mean-event-v2-r1-p1-after-p0-pass"
    ):
        raise MethodContractError("oracle development lock differs from V2")
    epsilon = event.get("oracle_validity_epsilon")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon <= 0:
        raise MethodContractError("oracle validity epsilon is invalid")
    payload["proposal_id"] = canonical_hash(payload)
    payload["lock_sha256"] = file_sha256(candidate)
    return payload


__all__ = ["ORACLE_LOCK_PATH", "file_sha256", "load_oracle_lock"]
