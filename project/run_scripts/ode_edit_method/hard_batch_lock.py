"""Fail-closed loader for the sealed genuine-B10 attribution lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import MethodContractError, canonical_hash
from .hard_batch import HARD_BATCH_ARM_ORDER, fixed_horizon_lambdas


HARD_BATCH_LOCK_PATH = Path(__file__).with_name("hard_batch_lock.json")
HARD_BATCH_SCHEMA = "ode-edit-session03-hard-b10-k20-lock/v1"
HARD_BATCH_INSTRUCTION = "ODEEDIT-S03-HARD-B10-K20-CYCLE-ATTRIBUTION-P0P1-V1"
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
BATCH_CASE_IDS = (
    "16884", "17700", "3221", "17206", "7369",
    "2949", "7748", "19535", "11395", "14420",
)
BATCH_REQUEST_IDS = (
    "6b591e695d5b3c34e51007610482e26714a894aa9bae0d7443437ca364603c80",
    "fadb96799daa7d85df44e2afe3f64ec1987ab24f20ca9627c663afa9215a1de8",
    "59f4eb9df558bee02ca884b0167abed05ba6e767224f19b315573807bc236e73",
    "68eddfa3bf6305f1e80795c3b795b99696b34ad05389279bad0c8d06e2467d56",
    "9a21b7f45784b48c8fb8b2f04e4f24a69dc300e8ec1358e01dbffecbf4476ccc",
    "5926e8b45cca01a48309d7f7cafa83d158db3903c55189a8609fd9c24a7bf47b",
    "2d0207a924e420293b8f089095cec161e3c065025ff76c458c5ae1ac156cc31d",
    "04d94903502872b094fc15f60992b50a0b3a89efcd7c2379484cba24e4d05b74",
    "7e649dd60fc36a2db1f073367a853262c708b65f486ce28356ac7c191db4ac0e",
    "22c4272a2a9af2f64336d27144579ca121e230c0fc58601537505009e67d5650",
)


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MethodContractError(f"hard-B10 lock {name} is not a mapping")
    return value


def validate_hard_batch_lock(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != HARD_BATCH_SCHEMA
        or payload.get("status") != "PRELOCKED_BEFORE_GENUINE_B10_P0_EXECUTION"
        or payload.get("instruction_id") != HARD_BATCH_INSTRUCTION
        or payload.get("implementation_base_commit")
        != "43f0fe5fa717db9bbb56ead845f328bc0aa015f1"
        or payload.get("scientific_outcome_count_at_panel_seal") != 0
        or tuple(payload.get("models", ())) != MODEL_ALIASES
    ):
        raise MethodContractError("hard-B10 lock identity differs")
    base = _mapping("base method", payload.get("base_method"))
    if dict(base) != {
        "proposal_id": "83eec5068acec673f7d9d0788e57ce0e9873a55785f7a6932026a4a2420c330b",
        "lock_sha256": "95376c554f01e00c9c5d71e99dfa3cbb5339a7534db8a62a9a05bc69c0be1bbb",
        "v3_event_proposal_id": "bc8fc256737d4e82b83f1937633ca94709d59e51969612611e20981aa70c4c23",
        "v3_event_lock_sha256": "cd4ffeb2310e1d261cea0457cd683293e53cf92b136b6d6eb01cd5c4af579882",
        "ct_k4_source_sha256": "ec8f5bf06479173e368c5be82c197b861dff753ecaa730c7f6b9243669ae1d0a",
        "ct_k10_source_sha256": "d3f27205ea8304f64b12a2366909dac431a0282b9c238e97ffe80fb9f0573d0a",
        "corrected_token_evaluator_sha256": "d4a32b9488db8fe1bbb8303faf7306857c476b75cce19d8f75b8ed98514d7811",
    }:
        raise MethodContractError("hard-B10 base method identity differs")
    policy = _mapping("policy", payload.get("policy"))
    if (
        policy.get("dtype") != "checkpoint-original-torch.bfloat16"
        or policy.get("seed") != 17
        or policy.get("edit_batch_size") != 10
        or policy.get("one_joint_backend_transaction") is not True
        or policy.get("singleton_decomposition") is not False
        or policy.get("direct_z_receipts") != 10
        or policy.get("direct_z_compute_per_request") != 1
        or policy.get("direct_z_shared_across_arms") is not True
        or tuple(policy.get("arms", ()))
        != tuple(arm.value for arm in HARD_BATCH_ARM_ORDER)
        or policy.get("K10") != 10
        or policy.get("K20") != 20
        or policy.get("T1") != 1.0
        or policy.get("T2") != 2.0
        or tuple(policy.get("lambda_k10", ())) != fixed_horizon_lambdas(10)
        or tuple(policy.get("lambda_k20", ())) != fixed_horizon_lambdas(20)
        or policy.get("corrector")
        != "editor-native-diagonal-cost/full-progress/nonnegative/raw-radius"
        or policy.get("zero_slope_completion") != "raw-c"
        or policy.get("post_qp_rescale") is not False
        or policy.get("online_early_stop") is not False
        or policy.get("earliest_exact_hit_snapshot_selected_after_fixed_budget")
        is not True
        or policy.get("stagnation_epsilon") != 1e-8
        or policy.get("model_specific_policy") is not False
    ):
        raise MethodContractError("hard-B10 transport policy differs")
    selection = _mapping("selection", payload.get("selection"))
    if (
        selection.get("dataset_relative_path")
        != "data/counterfact/counterfact.json"
        or selection.get("dataset_sha256")
        != "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
        or selection.get("dataset_size_bytes") != 45108470
        or selection.get("dataset_row_count") != 21919
        or selection.get("salt")
        != "odeedit-s03-hard-b10-k20-cycle-attribution-v1"
        or selection.get("exclusion_digest")
        != "177f94582c8ebbbd38ce227718418a2c6c2168fd60886f4b128c88e30a0b9b2a"
        or selection.get("diagnostic_known_case_id") != "16884"
        or selection.get("diagnostic_known_case_population_evidence") is not False
        or tuple(selection.get("batch_case_ids", ())) != BATCH_CASE_IDS
        or tuple(selection.get("batch_request_ids", ())) != BATCH_REQUEST_IDS
        or canonical_hash(list(BATCH_CASE_IDS))
        != selection.get("batch_order_sha256")
        or canonical_hash(list(BATCH_REQUEST_IDS))
        != selection.get("batch_request_order_sha256")
        or len(set(BATCH_CASE_IDS)) != 10
        or len(set(BATCH_REQUEST_IDS)) != 10
    ):
        raise MethodContractError("hard-B10 sealed panel identity differs")
    benchmark = _mapping("benchmark", payload.get("benchmark"))
    if (
        benchmark.get("name")
        != "pinned-alphaedit-counterfact-length-normalized-suffix-nll"
        or benchmark.get("evaluator_source_sha256")
        != "f2de63e6cc68871cb042f614193294a433c4b76622dd410de515df8cb73416b8"
        or benchmark.get("aggregator_source_sha256")
        != "9791812222b151952c16cacf28daf3e8be9645e37321e0aa84be6ab5e24a66f8"
        or benchmark.get("zsre_future_parity_source_sha256")
        != "bf5b0c79d65b54df6ea1478798dece18cbe48f3b6bb3015e3740a9ba43877e57"
        or benchmark.get("exact_joint_hit_count") != 10
        or dict(_mapping("native floor", benchmark.get("native_floor_margin_counts")))
        != {"efficacy": 0, "generalization": 0, "locality": 0}
        or benchmark.get("generation") is not False
        or benchmark.get("controller_heldout_access") is not False
    ):
        raise MethodContractError("hard-B10 benchmark policy differs")
    memory = _mapping("memory forecast", payload.get("memory_forecast"))
    if (
        memory.get("gpu_total_bytes") != 51527024640
        or memory.get("joint_context_count") != 60
        or memory.get("max_context_tokens") != 24
        or memory.get("rank") != 10
        or memory.get("batch_split_allowed") is not False
        or float(memory.get("qwen_conservative_peak_gib", 0.0)) >= 47.99
    ):
        raise MethodContractError("hard-B10 memory forecast differs")
    resources = _mapping("resources", payload.get("resources"))
    if (
        resources.get("server1_project_gpu_cap") != 3
        or resources.get("gpu_per_job") != 1
        or resources.get("cpu_per_job") != 8
        or resources.get("host_memory_mib_per_job") != 65000
        or resources.get("pair_gpu") != 2
        or resources.get("p0_time") != "08:00:00"
        or resources.get("p1_time_max") != "24:00:00"
        or resources.get("submission_authorized") is not False
    ):
        raise MethodContractError("hard-B10 resource policy differs")


def load_hard_batch_lock(
    path: str | Path = HARD_BATCH_LOCK_PATH,
) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MethodContractError("hard-B10 lock is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise MethodContractError("hard-B10 lock root is not a mapping")
    validate_hard_batch_lock(payload)
    result = dict(payload)
    result["proposal_id"] = canonical_hash(payload)
    result["lock_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result


__all__ = [
    "BATCH_CASE_IDS",
    "BATCH_REQUEST_IDS",
    "HARD_BATCH_LOCK_PATH",
    "load_hard_batch_lock",
    "validate_hard_batch_lock",
]
