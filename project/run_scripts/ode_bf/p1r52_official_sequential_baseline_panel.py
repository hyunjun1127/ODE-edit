"""Closed identities for corrected AlphaEdit and Official MEMIT baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json, sha256_file
from .contracts import ODEBFContractError
from .p1r52_sequential_contract import HISTORY_COUNTS, INSTRUCTION_ID, ROUND_COUNT
from .p1r52_sequential_runtime import MEMIT_ROLE, NATIVE_CORRECTED_ROLE


LOCK_SCHEMA = "ode-edit-s05-p1r52-official-sequential-baselines-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r52_official_sequential_baselines.json"
SOURCE_MANIFEST = "source_manifest_s05_p1r52_official_sequential_baselines.json"
PARENT = "5f295e74304d53ee5f7f22d7635c9cf56220eb54"
PARENT_TREE = "d7f0c9b2d742c4560ed77c13d0807f718f627894"
EASYEDIT_ROOT = Path("/mnt/raid5/janghj/EasyEdit")
MEMIT_ARTIFACT_LOCK = "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
MEMIT_ARTIFACT_LOCK_SHA256 = "6c327c563e0e805eb73a09cf3a577dea3771bc44127748fe64aba70a803292d2"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "exact_parent": PARENT,
        "exact_parent_tree": PARENT_TREE,
        "model": "llama3-8b-inst",
        "roles": [NATIVE_CORRECTED_ROLE, MEMIT_ROLE],
        "round_count": ROUND_COUNT,
        "batch_size": 10,
        "request_count_per_role": 100,
        "alphaedit_reset_cache_sequence": [True] + [False] * 9,
        "alphaedit_cache_history_width_at_entry": list(HISTORY_COUNTS),
        "alphaedit_static_projector_policy": "PINNED_P_REUSED_DISTINCT_FROM_DYNAMIC_CACHE_C",
        "memit_entrypoint": "easyeditor.models.memit.memit_main.apply_memit_to_model",
        "memit_hparams_sha256": "2b81838b49b1a5e0e4ef41f229216fda473200093fcbed6bd040f07a70d43818",
        "memit_covariance_policy": "STATIC_COMPUTATION_CACHE_REUSED_NOT_REQUEST_HISTORY",
        "memit_target_z_cache_template": None,
        "physical_weight_persistence": True,
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "stream_order": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "project_gpu_cap": 4,
        "stage_gpu_max": 2,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R52 official baseline numerical lock differs")
    zero = (
        "retry_count",
        "backtracking_count",
        "imputation_count",
        "interbatch_w0_restore_count",
        "hidden_controller_state_count",
        "batch_entry_pre_backward_count",
        "batch_entry_pre_generation_count",
        "evaluator_decision_influence_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R52 official baseline forbidden count differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


def verify_memit_artifacts(repo_root: Path, *, hash_covariances: bool) -> dict[str, Any]:
    lock_path = repo_root / MEMIT_ARTIFACT_LOCK
    if sha256_file(lock_path) != MEMIT_ARTIFACT_LOCK_SHA256:
        raise ODEBFContractError("Official MEMIT artifact lock differs")
    value = json.loads(lock_path.read_text(encoding="utf-8"))
    model = value["models"]["llama3-8b-inst"]
    observed: dict[str, Any] = {
        "artifact_lock_sha256": MEMIT_ARTIFACT_LOCK_SHA256,
        "hparams_sha256": model["hparams_sha256"],
        "source_sha256": {},
        "covariance_sha256": {},
        "covariance_content_rehashed": bool(hash_covariances),
    }
    for relative, expected in sorted(value["easyedit_sources"].items()):
        path = EASYEDIT_ROOT / relative
        if sha256_file(path) != expected:
            raise ODEBFContractError("Official MEMIT source artifact differs")
        observed["source_sha256"][relative] = expected
    hparams_path = EASYEDIT_ROOT / model["hparams_path"]
    if sha256_file(hparams_path) != model["hparams_sha256"]:
        raise ODEBFContractError("Official MEMIT hparams artifact differs")
    for layer, (relative, size, expected) in sorted(model["covariance"].items()):
        path = EASYEDIT_ROOT / relative
        if path.stat().st_size != int(size):
            raise ODEBFContractError("Official MEMIT covariance size differs")
        if hash_covariances and sha256_file(path) != expected:
            raise ODEBFContractError("Official MEMIT covariance artifact differs")
        observed["covariance_sha256"][str(layer)] = expected
    return observed


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "SOURCE_MANIFEST",
    "load_and_validate_lock",
    "validate_lock",
    "verify_memit_artifacts",
]
