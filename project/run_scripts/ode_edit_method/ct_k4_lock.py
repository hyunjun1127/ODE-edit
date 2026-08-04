"""Fail-closed loader for the pre-P0 Session 03 CT-K4 lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import MethodContractError, canonical_hash
from .ct_k4 import CT_ARM_ORDER, CT_CUMULATIVE, CT_K, CT_LAMBDAS


CT_K4_LOCK_PATH = Path(__file__).with_name("ct_k4_lock.json")
CT_K4_SCHEMA = "ode-edit-session03-ct-k4-lock/v1"
CT_K4_INSTRUCTION = "ODEEDIT-S03-CT-K4-ATTRIBUTION-P0P1-V1"
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MethodContractError(f"CT-K4 lock {name} is not a mapping")
    return value


def validate_ct_k4_lock(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != CT_K4_SCHEMA
        or payload.get("instruction_id") != CT_K4_INSTRUCTION
        or payload.get("status") != "PRELOCKED_BEFORE_P0_EXECUTION"
        or payload.get("scientific_outcome_count_at_lock") != 0
        or tuple(payload.get("models", ())) != MODEL_ALIASES
        or tuple(payload.get("arms", ()))
        != tuple(arm.value for arm in CT_ARM_ORDER)
    ):
        raise MethodContractError("CT-K4 lock identity differs")
    policy = _mapping("policy", payload.get("policy"))
    if (
        policy.get("dtype") != "checkpoint-original-torch.bfloat16"
        or policy.get("seed") != 17
        or policy.get("T") != 1.0
        or policy.get("K") != CT_K
        or tuple(policy.get("lambda", ())) != CT_LAMBDAS
        or tuple(policy.get("cumulative_frozen_targets", ())) != CT_CUMULATIVE
        or policy.get("subdivision_limit_p0") != 2
        or policy.get("direct_z_compute_per_case_model") != 1
        or policy.get("direct_z_shared_across_arms") is not True
        or policy.get("zero_slope_completion") != "raw-c"
        or policy.get("post_qp_rescale") is not False
        or policy.get("primary_early_stop") is not False
        or policy.get("model_specific_policy") is not False
    ):
        raise MethodContractError("CT-K4 transport policy differs")
    selection = _mapping("selection", payload.get("selection"))
    p1_ids = tuple(selection.get("p1_case_ids", ()))
    p1_requests = tuple(selection.get("p1_request_ids", ()))
    if (
        tuple(selection.get("excluded_prior_case_ids", ()))
        != ("2022", "12498", "20964", "768")
        or tuple(selection.get("p0_case_ids", ())) != ("2022",)
        or tuple(selection.get("p0_request_ids", ()))
        != ("9f8379da67c8c754397efddcf5af0d665592632779f4cb3b125efa5502361fde",)
        or p1_ids != ("17503", "1534", "14652", "9774")
        or len(p1_requests) != 4
        or canonical_hash(list(p1_ids)) != selection.get("p1_order_sha256")
        or canonical_hash(list(p1_requests))
        != selection.get("p1_request_order_sha256")
        or set(p1_ids) & {"2022", "12498", "20964", "768"}
    ):
        raise MethodContractError("CT-K4 fresh panel identity differs")
    if selection.get("dataset_sha256") != (
        "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
    ):
        raise MethodContractError("CT-K4 CounterFact identity differs")
    evaluation = _mapping("evaluation", payload.get("evaluation"))
    if (
        evaluation.get("p0_enabled") is not False
        or evaluation.get("p1_enabled") is not True
        or evaluation.get("action_freeze_required") is not True
        or evaluation.get("generation") is not False
        or evaluation.get("controller_access") is not False
    ):
        raise MethodContractError("CT-K4 evaluation firewall differs")
    resources = _mapping("resources", payload.get("resources"))
    if (
        resources.get("server1_project_gpu_cap") != 3
        or resources.get("gpu_per_job") != 1
        or resources.get("pair_gpu") != 2
        or resources.get("submission_authorized") is not False
    ):
        raise MethodContractError("CT-K4 resource lock differs")


def load_ct_k4_lock(path: str | Path = CT_K4_LOCK_PATH) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MethodContractError("CT-K4 lock is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise MethodContractError("CT-K4 lock root is not a mapping")
    validate_ct_k4_lock(payload)
    result = dict(payload)
    result["proposal_id"] = canonical_hash(payload)
    result["lock_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result
