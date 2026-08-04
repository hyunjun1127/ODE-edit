"""Fail-closed loader for the Session 03 CT-K10 confirmation lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import MethodContractError, canonical_hash
from .ct_k10 import (
    CT_K10,
    CT_K10_ARM_ORDER,
    CT_K10_CUMULATIVE,
    CT_K10_LAMBDAS,
)
from .ct_k4 import CT_CUMULATIVE, CT_K, CT_LAMBDAS
from .ct_k4_lock import load_ct_k4_lock


CT_K10_LOCK_PATH = Path(__file__).with_name("ct_k10_lock.json")
CT_K10_SCHEMA = "ode-edit-session03-ct-k10-lock/v1"
CT_K10_INSTRUCTION = "ODEEDIT-S03-CT-K10-FIXED-HORIZON-CONFIRM-P0P1-V1"
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
P1_CASE_IDS = ("21154", "11042", "16884", "21126")
P1_REQUEST_IDS = (
    "38453fbc4a239a6bb0aa9f3c45a327d1c08ba6fad5d81efd5f49ac099495856d",
    "6e6c46fa6821a11a3a19595404398018877cd3df9909e2e93f1a7677480b0065",
    "6b591e695d5b3c34e51007610482e26714a894aa9bae0d7443437ca364603c80",
    "431c89cb031b38bd89da3ad4708c94bc4931f2237d45fecffdee119a4b11e248",
)
EXCLUDED_CASE_IDS = (
    "2022", "12498", "20964", "768", "17503", "1534", "14652", "9774",
    "21135", "6578", "5613", "13856", "16451",
)
P1_RANK_SHA256 = (
    "000130ce96f4ced2ea7946a7c3053a3701fcdbea0aea43ff8f6a5f74baf96f35",
    "0001ab9e1589b28462ffdefb29f75be09c899758331b553fb295bb20b323f671",
    "0008fd16677db8feb2abc423e93b5baba78a2821a200f2fe053b056b0f1f838e",
    "000958c1e567441c582c0e30eceda6ed91ad410ed41b803ef40c035fd5fa522b",
)
TOKENIZER_REVISIONS = {
    "llama3-8b-inst": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    "qwen2.5-7b-inst": "a09a35458c702b33eeacc393d103063234e8bc28",
}


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MethodContractError(f"CT-K10 lock {name} is not a mapping")
    return value


def validate_ct_k10_lock(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != CT_K10_SCHEMA
        or payload.get("instruction_id") != CT_K10_INSTRUCTION
        or payload.get("status") != "PRELOCKED_BEFORE_K10_P0_EXECUTION"
        or payload.get("implementation_base_commit")
        != "c99eaea45d2153cbbd18bac66bcc0fda439c098c"
        or payload.get("p0_outcome_count_at_panel_seal") != 0
        or tuple(payload.get("models", ())) != MODEL_ALIASES
        or tuple(payload.get("arms", ()))
        != tuple(arm.value for arm in CT_K10_ARM_ORDER)
    ):
        raise MethodContractError("CT-K10 lock identity differs")
    policy = _mapping("policy", payload.get("policy"))
    if (
        policy.get("dtype") != "checkpoint-original-torch.bfloat16"
        or policy.get("seed") != 17
        or policy.get("T") != 1.0
        or policy.get("K4") != CT_K
        or policy.get("K10") != CT_K10
        or tuple(policy.get("lambda_k4", ())) != CT_LAMBDAS
        or tuple(policy.get("lambda_k10", ())) != CT_K10_LAMBDAS
        or tuple(policy.get("cumulative_frozen_targets_k10", ()))
        != CT_K10_CUMULATIVE
        or policy.get("direct_z_compute_per_case_model") != 1
        or policy.get("direct_z_shared_across_arms") is not True
        or policy.get("corrector")
        != "editor-native-diagonal-cost/full-progress/nonnegative/raw-radius"
        or policy.get("zero_slope_completion") != "raw-c"
        or policy.get("post_qp_rescale") is not False
        or policy.get("first_hit_observe_only") is not True
        or policy.get("early_stop_arm") is not False
        or policy.get("model_specific_policy") is not False
    ):
        raise MethodContractError("CT-K10 transport policy differs")
    if CT_CUMULATIVE != (0.25, 0.5, 0.75, 1.0):
        raise MethodContractError("CT-K4 compatibility constants differ")
    selection = _mapping("selection", payload.get("selection"))
    if (
        selection.get("dataset_relative_path")
        != "data/counterfact/counterfact.json"
        or selection.get("dataset_size_bytes") != 45108470
        or selection.get("dataset_row_count") != 21919
        or tuple(selection.get("excluded_prior_case_ids", ()))
        != EXCLUDED_CASE_IDS
        or selection.get("salt") != "odeedit-s03-ct-k10-confirm-v1"
        or selection.get("duplicate_group_normalization")
        != (
            "unicode-casefold; collapse-whitespace; tuple(subject,prompt-template); "
            "exclude-entire-group-if-count-not-one"
        )
        or selection.get("selection_algorithm")
        != (
            "sha256(salt|canonical-request-id)-ascending; excluded-prior; "
            "unique-normalized-subject-template-group; dataset-valid; "
            "both-fixed-tokenizers-nonempty-target; first-4"
        )
        or tuple(selection.get("p0_case_ids", ())) != ("2022",)
        or tuple(selection.get("p0_request_ids", ()))
        != ("9f8379da67c8c754397efddcf5af0d665592632779f4cb3b125efa5502361fde",)
        or tuple(selection.get("p1_case_ids", ())) != P1_CASE_IDS
        or tuple(selection.get("p1_request_ids", ())) != P1_REQUEST_IDS
        or tuple(selection.get("rank_sha256", ())) != P1_RANK_SHA256
        or canonical_hash(list(P1_CASE_IDS)) != selection.get("p1_order_sha256")
        or canonical_hash(list(P1_REQUEST_IDS))
        != selection.get("p1_request_order_sha256")
        or set(P1_CASE_IDS) & set(EXCLUDED_CASE_IDS)
        or selection.get("eligible_row_count") != 21376
        or selection.get("duplicate_group_excluded_row_count") != 530
    ):
        raise MethodContractError("CT-K10 fresh panel identity differs")
    if selection.get("dataset_sha256") != (
        "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
    ):
        raise MethodContractError("CT-K10 CounterFact identity differs")
    counts = _mapping("target_token_counts", selection.get("target_token_counts"))
    if any(tuple(counts.get(alias, ())) != (1, 1, 1, 1) for alias in MODEL_ALIASES):
        raise MethodContractError("CT-K10 target token counts differ")
    if dict(_mapping("tokenizer_revisions", selection.get("tokenizer_revisions"))) != (
        TOKENIZER_REVISIONS
    ):
        raise MethodContractError("CT-K10 tokenizer revisions differ")
    evaluation = _mapping("evaluation", payload.get("evaluation"))
    if (
        evaluation.get("p0_enabled") is not False
        or evaluation.get("p1_enabled") is not True
        or evaluation.get("action_freeze_required") is not True
        or evaluation.get("generation") is not False
        or evaluation.get("controller_access") is not False
        or evaluation.get("efficacy")
        != "canonical-subject-formatted teacher-forced token accuracy"
        or evaluation.get("generalization")
        != "CounterFact paraphrase teacher-forced token accuracy"
        or evaluation.get("locality")
        != "CounterFact neighborhood pre/post token agreement and target-true accuracy"
    ):
        raise MethodContractError("CT-K10 evaluation firewall differs")
    resources = _mapping("resources", payload.get("resources"))
    if (
        resources.get("server1_project_gpu_cap") != 3
        or resources.get("gpu_per_job") != 1
        or resources.get("cpu_per_job") != 8
        or resources.get("host_memory_mib_per_job") != 65000
        or resources.get("pair_gpu") != 2
        or resources.get("p0_time") != "06:00:00"
        or resources.get("p1_time") != "12:00:00"
        or resources.get("submission_authorized") is not False
    ):
        raise MethodContractError("CT-K10 resource lock differs")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ct_k10_lock(path: str | Path = CT_K10_LOCK_PATH) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MethodContractError("CT-K10 lock is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise MethodContractError("CT-K10 lock root is not a mapping")
    validate_ct_k10_lock(payload)
    base = _mapping("base_method", payload.get("base_method"))
    directory = CT_K10_LOCK_PATH.parent
    k4 = load_ct_k4_lock(directory / "ct_k4_lock.json")
    if (
        base.get("proposal_id")
        != "83eec5068acec673f7d9d0788e57ce0e9873a55785f7a6932026a4a2420c330b"
        or base.get("lock_sha256")
        != "95376c554f01e00c9c5d71e99dfa3cbb5339a7534db8a62a9a05bc69c0be1bbb"
        or base.get("v3_event_proposal_id")
        != "bc8fc256737d4e82b83f1937633ca94709d59e51969612611e20981aa70c4c23"
        or base.get("v3_event_lock_sha256")
        != "cd4ffeb2310e1d261cea0457cd683293e53cf92b136b6d6eb01cd5c4af579882"
        or k4["proposal_id"] != base.get("ct_k4_proposal_id")
        or k4["lock_sha256"] != base.get("ct_k4_lock_sha256")
        or _sha256(directory / "ct_k4.py")
        != base.get("ct_k4_source_sha256")
        or _sha256(directory / "ct_k4_lock.json") != base.get("ct_k4_lock_sha256")
        or _sha256(directory / "ct_k4_evaluation.py")
        != base.get("corrected_evaluator_source_sha256")
    ):
        raise MethodContractError("CT-K10 pinned K4/evaluator source differs")
    result = dict(payload)
    result["proposal_id"] = canonical_hash(payload)
    result["lock_sha256"] = _sha256(source)
    return result
