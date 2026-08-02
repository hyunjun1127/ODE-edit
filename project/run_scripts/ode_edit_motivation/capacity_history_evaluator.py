"""Evaluation-only replay for the capacity/history Motivation experiment.

The first evaluation field is decoded only after all four branch controllers
for the selected model are terminal, hash-valid, and policy-identical.  This
module contains no action selection and emits an NFE-free compact checkpoint
stream consumed by :mod:`capacity_history_analysis`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .capacity_geometry import compute_w0_denominators
from .capacity_qp import COMMON_FRONTIER_BARRIER_POLICY
from .capacity_history_analysis import (
    BRANCH_ALPHA_NATIVE,
    BRANCH_ALPHA_QP,
    BRANCHES,
    CHAIN_LENGTH,
    CHECKPOINT_EVENT,
    STREAM_SCHEMA,
)
from .capacity_history_controller import (
    APPLICATION_MODES,
    CONTROLLER_JOB_NAME,
    CONTROLLER_RECEIPT_SCHEMA,
    CONTROLLER_SCHEMA,
    CONTROLLER_STREAM_SCHEMA,
    MAX_ACCEPTED_ROUNDS,
    NATIVE_REFERENCE_ABS_TOL,
    POLICY_ID,
    RUN_IDS as CONTROLLER_RUN_IDS,
    RUN_SEED,
    generate_capacity_history_selection,
    is_alpha_branch,
    is_qp_branch,
    policy_parameters,
)
from .contracts import (
    ContractError,
    LowRankFactor,
    canonical_json,
    sha256_bytes,
)
from .easyedit_bridge import EasyEditBridge
from .gpu_runtime import (
    FixedModelRuntime,
    load_fixed_model,
    offline_environment,
    seed_runtime,
)
from .hooks import assert_snapshot_current, assert_tensor_sha256_device_parity
from .manifests import MODEL_SPECS, fixed_model_spec, load_counterfact_requests, preflight_fixed_artifacts
from .microseq_artifacts import apply_proposal_in_disposable_process, load_proposal_artifact
from .microseq_evaluator import (
    EvaluationFields,
    PROMPTS_PER_EDIT,
    _all_finite,
    _flatten_prompts,
    _last_token_log_probs,
    _mean_forward_kl,
    _true_requests,
    cumulative_geometry,
    load_counterfact_evaluation_fields,
)
from .mv0_fidelity import (
    DEFAULT_OUTPUT_ROOT,
    SanitizedJsonlWriter,
    _bridge_pins,
    _covariance_specs,
    _file_sha256,
    _freeze_contexts,
    _git_runtime_state,
    _load_hparams,
    _load_verified_covariances,
    _local_run_directory,
    _relative_provenance,
    _safe_payload,
    _weight_hashes,
    _write_json_exclusive,
)
from .mv1_calibration import _feature_hash, _layer_by_weight, rewrite_metrics, teacher_forced_rewrite_exact
from .quarter_step_refresh import LAYERS, _capture_source_snapshot


EVALUATOR_SCHEMA = "ode-edit-capacity-history-evaluator/v1"
EVALUATOR_RUN_IDS = {
    branch: {
        "llama3-8b-inst": CONTROLLER_RUN_IDS[branch]["llama3-8b-inst"].replace(
            "caphist_", "caphist_eval_", 1
        ),
        "qwen2.5-7b-inst": CONTROLLER_RUN_IDS[branch]["qwen2.5-7b-inst"].replace(
            "caphist_", "caphist_eval_", 1
        ),
    }
    for branch in BRANCHES
}


class CapacityHistoryEvaluationError(ContractError):
    """Controller evidence or evaluation replay differs from the lock."""


@dataclass(frozen=True, slots=True)
class ControllerActionEvidence:
    feature: Mapping[str, Any]
    action: Mapping[str, Any]
    result: Mapping[str, Any]
    proposal_manifests: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ControllerEvidence:
    branch: str
    model_alias: str
    run_id: str
    run_directory: Path
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]
    actions: tuple[ControllerActionEvidence, ...]


def evaluator_policy_parameters() -> dict[str, Any]:
    return {
        "controller": policy_parameters(),
        "metric_protocol": {
            "prompts_per_edit": PROMPTS_PER_EDIT,
            "kl_direction": "KL(W0||Wt)",
            "branch_order": list(BRANCHES),
            "evaluation_after_all_four_controllers": True,
            "raw_evaluation_text_persisted": False,
        },
    }


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapacityHistoryEvaluationError(f"{path}: mapping required")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink():
        raise CapacityHistoryEvaluationError(f"{path}: symlink JSON is forbidden")
    source = path.resolve(strict=True)
    if not source.is_file():
        raise CapacityHistoryEvaluationError(f"{path}: regular JSON file required")
    try:
        return _mapping(json.loads(source.read_text(encoding="utf-8")), str(path))
    except json.JSONDecodeError as exc:
        raise CapacityHistoryEvaluationError(f"{path}: invalid JSON") from exc


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits.lower() for character in value)
        or value != value.lower()
    ):
        raise CapacityHistoryEvaluationError(f"{path}: lowercase SHA-256 required")
    return value


def _safe_local_child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise CapacityHistoryEvaluationError("artifact name is not a basename")
    if root.is_symlink() or not root.is_dir():
        raise CapacityHistoryEvaluationError("artifact root must be a real directory")
    unresolved = root / name
    if unresolved.is_symlink():
        raise CapacityHistoryEvaluationError("artifact symlink is forbidden")
    candidate = unresolved.resolve(strict=True)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise CapacityHistoryEvaluationError("artifact escaped its run directory") from exc
    if not candidate.is_file():
        raise CapacityHistoryEvaluationError("artifact must be a regular file")
    return candidate


def _load_stream(path: Path, *, run_id: str, event: str) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    if path.is_symlink():
        raise CapacityHistoryEvaluationError(f"{path}: symlink stream is forbidden")
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                wrapper = _mapping(json.loads(line), f"{path}:{line_number}")
            except json.JSONDecodeError as exc:
                raise CapacityHistoryEvaluationError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if set(wrapper) != {
                "schema_version",
                "run_id",
                "sequence",
                "recorded_at",
                "event",
                "payload",
            }:
                raise CapacityHistoryEvaluationError(f"{path}:{line_number}: wrapper keys differ")
            if (
                wrapper["schema_version"] != CONTROLLER_STREAM_SCHEMA
                or wrapper["run_id"] != run_id
                or wrapper["sequence"] != len(records)
                or wrapper["event"] != event
            ):
                raise CapacityHistoryEvaluationError(f"{path}:{line_number}: stream identity differs")
            records.append(_mapping(wrapper["payload"], f"{path}:{line_number}.payload"))
    if len(records) != CHAIN_LENGTH:
        raise CapacityHistoryEvaluationError(f"{path}: exact four records required")
    return tuple(records)


def _verify_history_chain(branch: str, actions: Sequence[ControllerActionEvidence]) -> None:
    previous_after_id: str | None = None
    for edit_index, evidence in enumerate(actions, start=1):
        before = evidence.feature.get("history_before")
        after = evidence.feature.get("history_after")
        append = evidence.feature.get("history_append")
        if is_alpha_branch(branch):
            before_map = _mapping(before, "history_before")
            after_map = _mapping(after, "history_after")
            append_map = _mapping(append, "history_append")
            if (
                before_map.get("edit_count") != edit_index - 1
                or after_map.get("edit_count") != edit_index
                or evidence.result.get("history_edit_count") != edit_index
                or append_map.get("edit_id") != evidence.feature.get("case_id")
                or before_map.get("append_policy")
                != "post-accepted-edit-once-history-fixed-within-edit"
                or after_map.get("append_policy")
                != "post-accepted-edit-once-history-fixed-within-edit"
                or (previous_after_id is not None and before_map.get("history_id") != previous_after_id)
            ):
                raise CapacityHistoryEvaluationError("Alpha history chain differs from canonical policy")
            previous_after_id = str(after_map.get("history_id"))
        elif before is not None or after is not None or append is not None or evidence.result.get(
            "history_edit_count"
        ) != 0:
            raise CapacityHistoryEvaluationError("MEMIT branch contains Alpha history state")


def _verify_native_reference_contract(
    *, branch: str, feature: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    """Verify the c3 BF-share magnitude-control contract."""

    reference = _mapping(feature.get("native_reference"), "native_reference")
    try:
        origin = float(reference["origin_utility"])
        native = float(reference["ordered_endpoint_utility"])
        endpoint = float(reference["controller_endpoint_utility"])
        tolerance = float(reference["abs_tolerance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CapacityHistoryEvaluationError("native reference scalars are incomplete") from exc
    if (
        any(not math.isfinite(value) for value in (origin, native, endpoint, tolerance))
        or tolerance != NATIVE_REFERENCE_ABS_TOL
        or native - origin <= tolerance
        or type(reference.get("matched")) is not bool
        or type(reference.get("budget_exhausted")) is not bool
        or type(reference.get("fixed_distance_budget_completed")) is not bool
        or type(reference.get("ordered_endpoint_first_hit")) is not bool
    ):
        raise CapacityHistoryEvaluationError("native reference contract differs from c3")
    matched = bool(reference["matched"])
    budget_exhausted = bool(reference["budget_exhausted"])
    accepted_rounds = int(result.get("accepted_round_count", 0))
    if (
        result.get("native_reference_utility") != native
        or result.get("endpoint_utility") != endpoint
        or result.get("native_reference_reached") is not matched
        or matched != (endpoint >= native - tolerance)
    ):
        raise CapacityHistoryEvaluationError("native reference result differs from feature")

    diagnostics = feature.get("round_diagnostics")
    if not isinstance(diagnostics, list):
        raise CapacityHistoryEvaluationError("native reference diagnostics are absent")
    if not is_qp_branch(branch):
        if (
            diagnostics
            or accepted_rounds != 1
            or not matched
            or budget_exhausted
            or reference.get("fixed_distance_budget_completed") is not False
            or not math.isclose(endpoint, native, rel_tol=0.0, abs_tol=1e-8)
            or not math.isclose(
                float(feature.get("accepted_path_distance")),
                float(feature.get("native_c_distance")),
                rel_tol=2e-5,
                abs_tol=2e-5,
            )
        ):
            raise CapacityHistoryEvaluationError("native branch reference contract differs")
        return

    accepted = [
        _mapping(item, "round_diagnostics[]")
        for item in diagnostics
        if _mapping(item, "round_diagnostics[]").get("accepted") is True
    ]
    native_distance = float(feature.get("native_c_distance"))
    path_distance = float(feature.get("accepted_path_distance"))
    if (
        len(accepted) != MAX_ACCEPTED_ROUNDS
        or accepted_rounds != MAX_ACCEPTED_ROUNDS
        or reference.get("fixed_distance_budget_completed") is not True
        or not math.isclose(path_distance, native_distance, rel_tol=2e-5, abs_tol=2e-5)
    ):
        raise CapacityHistoryEvaluationError("QP path differs from four exact quarter hops")
    hop_distance = native_distance / MAX_ACCEPTED_ROUNDS
    for diagnostic in accepted:
        requested = float(diagnostic.get("requested_gain"))
        remaining = float(diagnostic.get("remaining_reference_gain_before"))
        remaining_rounds = int(diagnostic.get("remaining_rounds_before"))
        maximum = float(diagnostic.get("maximum_predicted_gain"))
        coefficients = diagnostic.get("coefficients")
        allocation_coefficients = diagnostic.get("allocation_coefficients")
        if (
            not isinstance(coefficients, list)
            or not isinstance(allocation_coefficients, list)
            or len(coefficients) != len(LAYERS)
            or len(allocation_coefficients) != len(LAYERS)
        ):
            raise CapacityHistoryEvaluationError("BF exact-hop coefficients are incomplete")
        coefficient_norm = math.sqrt(
            math.fsum(float(value) * float(value) for value in coefficients)
        )
        allocation_norm = math.sqrt(
            math.fsum(
                float(value) * float(value) for value in allocation_coefficients
            )
        )
        if allocation_norm <= 0.0:
            raise CapacityHistoryEvaluationError("BF allocation has no relative share")
        radial_scale = hop_distance / allocation_norm
        expected_request = (
            remaining
            if remaining > NATIVE_REFERENCE_ABS_TOL
            else maximum
        )
        if (
            remaining_rounds != MAX_ACCEPTED_ROUNDS - int(diagnostic.get("round_index")) + 1
            or not math.isclose(requested, expected_request, rel_tol=0.0, abs_tol=1e-10)
            or float(diagnostic.get("native_reference_utility")) != native
            or not math.isclose(
                float(diagnostic.get("trust_fraction")),
                1.0 / MAX_ACCEPTED_ROUNDS,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(coefficient_norm, hop_distance, rel_tol=2e-5, abs_tol=2e-5)
            or not math.isclose(
                float(diagnostic.get("coefficient_norm")),
                hop_distance,
                rel_tol=2e-5,
                abs_tol=2e-5,
            )
            or not math.isclose(
                float(diagnostic.get("share_l2_norm")), 1.0, rel_tol=2e-5, abs_tol=2e-5
            )
            or not math.isclose(
                float(diagnostic.get("allocation_coefficient_norm")),
                allocation_norm,
                rel_tol=2e-5,
                abs_tol=2e-5,
            )
            or not math.isclose(
                float(diagnostic.get("radial_scale")),
                radial_scale,
                rel_tol=2e-5,
                abs_tol=2e-5,
            )
            or any(
                not math.isclose(
                    float(applied),
                    float(allocation) * radial_scale,
                    rel_tol=2e-5,
                    abs_tol=2e-5,
                )
                for applied, allocation in zip(
                    coefficients, allocation_coefficients, strict=True
                )
            )
            or diagnostic.get("allocation_barrier_policy")
            != COMMON_FRONTIER_BARRIER_POLICY
            or diagnostic.get("applied_cap_enforced") is not False
            or diagnostic.get("native_reference_reached")
            != (
                float(diagnostic.get("utility_before"))
                + float(diagnostic.get("rewrite_gain"))
                >= native - tolerance
            )
        ):
            raise CapacityHistoryEvaluationError("BF-share exact-quarter contract differs")
    if budget_exhausted is matched:
        raise CapacityHistoryEvaluationError("QP matched/exhausted metadata is inconsistent")


def _verify_controller(
    *,
    output_root: Path,
    branch: str,
    model_alias: str,
    expected_case_ids: Sequence[str],
    selection_manifest_id: str,
    case_order_hash: str,
) -> ControllerEvidence:
    run_id = CONTROLLER_RUN_IDS[branch][model_alias]
    unresolved = output_root / run_id
    if unresolved.is_symlink():
        raise CapacityHistoryEvaluationError(f"{run_id}: symlink run directory is forbidden")
    run_directory = unresolved.resolve(strict=True)
    manifest = _read_json(run_directory / "manifest.json")
    summary = _read_json(run_directory / "summary.json")
    selection = _mapping(manifest.get("selection"), "manifest.selection")
    firewall = _mapping(manifest.get("firewall"), "manifest.firewall")
    precomputed = _mapping(manifest.get("precomputed"), "manifest.precomputed")
    policy_hash = sha256_bytes(canonical_json(policy_parameters()).encode("utf-8"))
    if (
        manifest.get("schema_version") != CONTROLLER_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("controller_branch") != branch
        or manifest.get("application_mode") != APPLICATION_MODES[branch]
        or manifest.get("policy_id") != POLICY_ID
        or manifest.get("policy_parameters") != policy_parameters()
        or manifest.get("policy_parameters_sha256") != policy_hash
        or _mapping(manifest.get("model"), "manifest.model").get("model_alias") != model_alias
        or selection.get("manifest_id") != selection_manifest_id
        or selection.get("order_hash") != case_order_hash
        or selection.get("case_ids") != list(expected_case_ids)
        or firewall
        != {
            "controller_only": True,
            "outcome_fields_loaded": False,
            "separate_evaluator_required": True,
        }
    ):
        raise CapacityHistoryEvaluationError(f"{run_id}: manifest identity differs")
    loaded = precomputed.get("covariances")
    projector = precomputed.get("projector")
    if (
        not isinstance(loaded, list)
        or len(loaded) != len(LAYERS)
        or precomputed.get("covariance_write_or_recompute") is not False
        or precomputed.get("projector_write_or_recompute") is not False
        or (is_alpha_branch(branch) and not isinstance(projector, Mapping))
        or (not is_alpha_branch(branch) and projector is not None)
        or (is_alpha_branch(branch) and precomputed.get("alpha_reference_manifest_id") is None)
        or (is_alpha_branch(branch) and precomputed.get("alpha_solver_config_id") is None)
        or (not is_alpha_branch(branch) and precomputed.get("alpha_reference_manifest_id") is not None)
        or (not is_alpha_branch(branch) and precomputed.get("alpha_solver_config_id") is not None)
    ):
        raise CapacityHistoryEvaluationError(f"{run_id}: precomputed-only contract differs")
    expected_history = CHAIN_LENGTH if is_alpha_branch(branch) else 0
    if (
        summary.get("schema_version") != CONTROLLER_SCHEMA
        or summary.get("run_id") != run_id
        or summary.get("model_alias") != model_alias
        or summary.get("controller_branch") != branch
        or summary.get("run_status") != "completed"
        or summary.get("failure_type") is not None
        or summary.get("all_pass") is not True
        or summary.get("expected_counts_exact") is not True
        or summary.get("receipt_count") != CHAIN_LENGTH
        or summary.get("direct_z_artifact_count") != CHAIN_LENGTH
        or summary.get("selection_manifest_id") != selection_manifest_id
        or summary.get("policy_parameters_sha256") != policy_hash
        or summary.get("history_edit_count") != expected_history
        or summary.get("controller_only") is not True
        or summary.get("outcome_fields_loaded") is not False
        or summary.get("disposable_process_exit_required") is not True
        or summary.get("loaded_covariances") != loaded
        or summary.get("precomputed_projector_only") is not True
    ):
        raise CapacityHistoryEvaluationError(f"{run_id}: controller is not terminal/pass-exact")
    if _mapping(summary.get("stream_sequences"), "summary.stream_sequences") != {
        "features": 4,
        "actions": 4,
        "events": 4,
    }:
        raise CapacityHistoryEvaluationError(f"{run_id}: stream counts differ")
    for filename, key in (
        ("manifest.json", "manifest_sha256"),
        ("features.jsonl", "features_sha256"),
        ("actions.jsonl", "actions_sha256"),
        ("events.jsonl", "events_sha256"),
    ):
        if _file_sha256(run_directory / filename) != _sha256(
            _mapping(summary.get("artifacts"), "summary.artifacts").get(key), key
        ):
            raise CapacityHistoryEvaluationError(f"{run_id}: {filename} hash differs")

    features = _load_stream(
        run_directory / "features.jsonl",
        run_id=run_id,
        event="capacity_history_controller_feature",
    )
    actions = _load_stream(
        run_directory / "actions.jsonl",
        run_id=run_id,
        event="capacity_history_controller_action",
    )
    results = _load_stream(
        run_directory / "events.jsonl",
        run_id=run_id,
        event="capacity_history_controller_edit",
    )
    evidence: list[ControllerActionEvidence] = []
    proposal_total = 0
    for edit_index, (case_id, feature, action, result) in enumerate(
        zip(expected_case_ids, features, actions, results, strict=True), start=1
    ):
        if (
            feature.get("case_id") != case_id
            or action.get("case_id") != case_id
            or result.get("case_id") != case_id
            or feature.get("edit_index") != edit_index
            or action.get("edit_index") != edit_index
            or result.get("edit_index") != edit_index
            or feature.get("controller_branch") != branch
            or action.get("controller_branch") != branch
            or result.get("branch") != branch
            or feature.get("application_mode") != APPLICATION_MODES[branch]
            or action.get("application_mode") != APPLICATION_MODES[branch]
            or result.get("pass") is not True
            or feature.get("controller_only") is not True
            or feature.get("outcome_fields_loaded") is not False
            or result.get("controller_only") is not True
            or result.get("outcome_fields_loaded") is not False
            or action.get("all_actions_before_outcomes") is not True
        ):
            raise CapacityHistoryEvaluationError(f"{run_id}: action {edit_index} identity differs")
        if (
            feature.get("request_id") != action.get("request_id")
            or feature.get("request_id") != result.get("request_id")
            or action.get("feature_hash") != feature.get("feature_hash")
            or result.get("feature_hash") != feature.get("feature_hash")
            or result.get("commitment_hash") != action.get("commitment_hash")
            or _feature_hash({key: value for key, value in feature.items() if key != "feature_hash"})
            != feature.get("feature_hash")
            or _feature_hash({key: value for key, value in action.items() if key != "commitment_hash"})
            != action.get("commitment_hash")
        ):
            raise CapacityHistoryEvaluationError(f"{run_id}: action {edit_index} commitment differs")
        _verify_native_reference_contract(
            branch=branch,
            feature=feature,
            result=result,
        )
        target = _mapping(feature.get("target_identity"), "feature.target_identity")
        if target.get("direct_z_compute_count") != 1 or result.get("direct_z_compute_count") != 1:
            raise CapacityHistoryEvaluationError(f"{run_id}: direct-z count differs")
        metadata_list = feature.get("proposal_artifacts")
        if not isinstance(metadata_list, list) or len(metadata_list) != result.get("proposal_count"):
            raise CapacityHistoryEvaluationError(f"{run_id}: proposal count differs")
        manifests: list[Path] = []
        action_ids: list[str] = []
        direction_hashes: list[str] = []
        for metadata_raw in metadata_list:
            metadata = _mapping(metadata_raw, "proposal_artifact")
            manifest_name = metadata.get("name")
            tensor_name = metadata.get("tensor_name")
            manifest_path = _safe_local_child(
                run_directory / "proposal_artifacts", str(manifest_name)
            )
            tensor_path = _safe_local_child(
                run_directory / "proposal_artifacts", str(tensor_name)
            )
            if (
                manifest_path.suffix != ".json"
                or tensor_path.name != manifest_path.with_suffix(".pt").name
                or _file_sha256(manifest_path) != _sha256(metadata.get("sha256"), "manifest sha")
                or _file_sha256(tensor_path)
                != _sha256(metadata.get("tensor_sha256"), "tensor sha")
            ):
                raise CapacityHistoryEvaluationError("proposal artifact identity differs")
            proposal_manifest = _read_json(manifest_path)
            manifests.append(manifest_path)
            action_ids.append(manifest_path.stem)
            direction_hashes.append(
                _sha256(
                    proposal_manifest.get("proposal_direction_sha256"),
                    "proposal_direction_sha256",
                )
            )
        if (
            action.get("branch_order") != action_ids
            or _mapping(action.get("path_action_hashes"), "path_action_hashes")
            != {branch: direction_hashes}
            or len(action.get("proposal_entry_state_ids", ())) != len(manifests)
            or len(action.get("proposal_descendant_state_ids", ())) != len(manifests)
            or result.get("terminal_state_id") != action["proposal_descendant_state_ids"][-1]
            or not isinstance(action.get("per_hop_c_energy"), list)
            or len(action["per_hop_c_energy"]) != len(manifests)
        ):
            raise CapacityHistoryEvaluationError(f"{run_id}: proposal lineage/order differs")
        native_distance = float(feature.get("native_c_distance"))
        expected_hop = (
            native_distance / MAX_ACCEPTED_ROUNDS if is_qp_branch(branch) else native_distance
        )
        if (
            not math.isclose(
                float(result.get("accepted_path_distance")),
                float(feature.get("accepted_path_distance")),
                rel_tol=2e-5,
                abs_tol=2e-5,
            )
            or any(
                not math.isclose(
                    float(energy),
                    expected_hop * expected_hop,
                    rel_tol=2e-5,
                    abs_tol=2e-5,
                )
                for energy in action["per_hop_c_energy"]
            )
        ):
            raise CapacityHistoryEvaluationError(f"{run_id}: proposal C-distance budget differs")
        receipt_path = _safe_local_child(
            run_directory / "action_receipts", str(result.get("receipt_name"))
        )
        receipt = _read_json(receipt_path)
        if (
            set(receipt)
            != {
                "schema_version",
                "case_id",
                "request_id",
                "feature_hash",
                "commitment_hash",
                "origin_lineage_id",
                "branch_order",
                "path_action_hashes",
                "per_hop_c_energy",
                "all_paths_committed_before_outcomes",
                "durability",
            }
            or receipt.get("schema_version") != CONTROLLER_RECEIPT_SCHEMA
            or receipt.get("case_id") != case_id
            or receipt.get("request_id") != feature.get("request_id")
            or receipt.get("feature_hash") != feature.get("feature_hash")
            or receipt.get("commitment_hash") != action.get("commitment_hash")
            or receipt.get("origin_lineage_id") != target.get("origin_lineage_id")
            or receipt.get("branch_order") != action_ids
            or receipt.get("path_action_hashes") != action.get("path_action_hashes")
            or receipt.get("per_hop_c_energy") != action.get("per_hop_c_energy")
            or receipt.get("all_paths_committed_before_outcomes") is not True
            or receipt.get("durability")
            != "feature+actions-write+flush+fsync-before-exclusive-receipt"
            or _file_sha256(receipt_path)
            != _sha256(result.get("receipt_sha256"), "receipt sha")
        ):
            raise CapacityHistoryEvaluationError(f"{run_id}: receipt differs")
        proposal_total += len(manifests)
        evidence.append(
            ControllerActionEvidence(
                feature=feature,
                action=action,
                result=result,
                proposal_manifests=tuple(manifests),
            )
        )
    if summary.get("proposal_artifact_count") != proposal_total:
        raise CapacityHistoryEvaluationError(f"{run_id}: proposal total differs")
    _verify_history_chain(branch, evidence)
    return ControllerEvidence(
        branch=branch,
        model_alias=model_alias,
        run_id=run_id,
        run_directory=run_directory,
        manifest=manifest,
        summary=summary,
        actions=tuple(evidence),
    )


def verify_all_controllers(
    *, output_root: str | Path, easyedit_root: str | Path, model_alias: str
) -> dict[str, ControllerEvidence]:
    """Verify all four terminal branches before any evaluation-row decode."""

    if model_alias not in MODEL_SPECS:
        raise CapacityHistoryEvaluationError("model alias is outside the fixed pair")
    root = Path(output_root).expanduser().resolve(strict=True)
    selection = generate_capacity_history_selection(easyedit_root)
    evidence = {
        branch: _verify_controller(
            output_root=root,
            branch=branch,
            model_alias=model_alias,
            expected_case_ids=selection.case_ids,
            selection_manifest_id=selection.manifest_id,
            case_order_hash=selection.order_hash,
        )
        for branch in BRANCHES
    }
    first = evidence[BRANCHES[0]].manifest
    for branch in BRANCHES[1:]:
        manifest = evidence[branch].manifest
        if (
            manifest.get("contexts") != first.get("contexts")
            or manifest.get("provenance_id") != first.get("provenance_id")
            or manifest.get("ode_edit_git") != first.get("ode_edit_git")
            or manifest.get("fixed_files") != first.get("fixed_files")
            or manifest.get("policy_parameters_sha256")
            != first.get("policy_parameters_sha256")
            or [item.feature["case_id"] for item in evidence[branch].actions]
            != [item.feature["case_id"] for item in evidence[BRANCHES[0]].actions]
        ):
            raise CapacityHistoryEvaluationError("four-controller common contract differs")
    return evidence


def _routing_diagnostic(
    *, branch: str, feature: Mapping[str, Any], accepted_round_count: int
) -> dict[str, int | float | bool]:
    """Verify the c1-compatible allocation QP, share identity, and zero support."""

    raw = feature.get("round_diagnostics")
    if not isinstance(raw, list):
        raise CapacityHistoryEvaluationError("round diagnostics must be a list")
    if not is_qp_branch(branch):
        if raw or accepted_round_count != 1:
            raise CapacityHistoryEvaluationError("native branch contains QP routing state")
        return {
            "overloaded_layer_observations": 0,
            "suppressed_overloaded_observations": 0,
            "capacity_reroute_round_count": 0,
            "capacity_barrier_max_violation": 0.0,
            "capacity_barrier_exact": True,
            "overloaded_positive_write_suppressed": True,
        }

    accepted = [
        _mapping(value, "round_diagnostics[]")
        for value in raw
        if _mapping(value, "round_diagnostics[]").get("accepted") is True
    ]
    if len(accepted) != accepted_round_count:
        raise CapacityHistoryEvaluationError("accepted routing diagnostics differ from action")
    overloaded = 0
    suppressed = 0
    reroute_rounds = 0
    max_violation = 0.0
    for diagnostic in accepted:
        coefficients = diagnostic.get("coefficients")
        allocation_coefficients = diagnostic.get("allocation_coefficients")
        terms = diagnostic.get("capacity_terms")
        if (
            not isinstance(coefficients, list)
            or not isinstance(allocation_coefficients, list)
            or not isinstance(terms, list)
            or len(coefficients) != len(LAYERS)
            or len(allocation_coefficients) != len(LAYERS)
            or len(terms) != len(LAYERS)
        ):
            raise CapacityHistoryEvaluationError("accepted routing vector differs from layers")
        round_suppressed = 0
        other_positive = False
        for coefficient_raw, allocation_raw, term_raw in zip(
            coefficients, allocation_coefficients, terms, strict=True
        ):
            coefficient = float(coefficient_raw)
            allocation = float(allocation_raw)
            term = _mapping(term_raw, "capacity_terms[]")
            psi = float(term["psi_before"])
            linear = float(term["linear"])
            quadratic = float(term["quadratic"])
            barrier = float(term["barrier"])
            cap = float(term["coefficient_cap"])
            values = (coefficient, allocation, psi, linear, quadratic, barrier, cap)
            if (
                any(not math.isfinite(value) for value in values)
                or coefficient < 0.0
                or allocation < 0.0
            ):
                raise CapacityHistoryEvaluationError("routing diagnostic contains invalid scalar")
            if allocation > cap + 1e-8:
                raise CapacityHistoryEvaluationError("allocation coefficient exceeds layer cap")
            violation = psi + linear * allocation + quadratic * allocation**2 - barrier
            max_violation = max(max_violation, violation)
            positive_overload = term.get("overloaded") is True and linear >= 0.0
            if positive_overload:
                overloaded += 1
                if allocation <= 1e-12 and coefficient <= 1e-12 and cap <= 1e-12:
                    suppressed += 1
                    round_suppressed += 1
            elif coefficient > 1e-12:
                other_positive = True
        if round_suppressed > 0 and other_positive:
            reroute_rounds += 1
    exact = max_violation <= 1e-8
    suppression_exact = overloaded == suppressed
    if not exact or not suppression_exact:
        raise CapacityHistoryEvaluationError("accepted QP action violates capacity routing")
    return {
        "overloaded_layer_observations": overloaded,
        "suppressed_overloaded_observations": suppressed,
        "capacity_reroute_round_count": reroute_rounds,
        "capacity_barrier_max_violation": max(0.0, max_violation),
        "capacity_barrier_exact": exact,
        "overloaded_positive_write_suppressed": suppression_exact,
    }


def _execution_envelope(branch: str, model_alias: str, run_id: str) -> None:
    if (
        branch not in BRANCHES
        or model_alias not in MODEL_SPECS
        or run_id != EVALUATOR_RUN_IDS[branch][model_alias]
    ):
        raise CapacityHistoryEvaluationError("capacity/history evaluator identity differs")


def _slurm_state(branch: str, model_alias: str, run_id: str) -> dict[str, Any]:
    _execution_envelope(branch, model_alias, run_id)
    values = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "role": os.environ.get("ODEEDIT_CAPACITY_PROCESS_ROLE"),
        "barrier": os.environ.get("ODEEDIT_CAPACITY_CONTROLLER_BARRIER"),
    }
    if all(value is None for value in values.values()):
        return {"under_slurm": False}
    if any(value is None for value in values.values()):
        raise CapacityHistoryEvaluationError("partial evaluator Slurm identity is forbidden")
    expected_barrier = "+".join(CONTROLLER_RUN_IDS[item][model_alias] for item in BRANCHES)
    if (
        values["job_name"] != CONTROLLER_JOB_NAME
        or values["node"] != "devbox"
        or values["role"] != "evaluator"
        or values["barrier"] != expected_barrier
        or not str(values["job_id"]).isdigit()
    ):
        raise CapacityHistoryEvaluationError("evaluator Slurm/barrier identity differs")
    return {"under_slurm": True, **values}


def run_capacity_history_evaluator(
    *,
    easyedit_root: str | Path,
    branch: str,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
    evaluation_loader: Callable[
        [str | Path, Sequence[str]], tuple[EvaluationFields, ...]
    ] = load_counterfact_evaluation_fields,
) -> dict[str, Any]:
    started = time.perf_counter()
    _execution_envelope(branch, model_alias, run_id)
    slurm = _slurm_state(branch, model_alias, run_id)
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    git_state = _git_runtime_state()
    provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_capacity_history_selection(root)

    # This full barrier must remain before the first evaluation-only row decode.
    controllers = verify_all_controllers(
        output_root=output,
        easyedit_root=root,
        model_alias=model_alias,
    )
    if _mapping(
        controllers[branch].manifest.get("ode_edit_git"), "controller.ode_edit_git"
    ) != git_state:
        raise CapacityHistoryEvaluationError("controller/evaluator Git identities differ")
    evaluation_fields = evaluation_loader(root, selection.case_ids)
    requests = load_counterfact_requests(root, selection.case_ids)
    true_requests = _true_requests(requests, evaluation_fields)
    selected = controllers[branch]

    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not set(record.path for record in bridge_provenance.files).issubset(
        set(record.path for record in provenance.files)
    ):
        raise CapacityHistoryEvaluationError("evaluator bridge provenance is outside fixed manifest")

    with offline_environment():
        assert_tensor_sha256_device_parity(torch.device("cuda", 0))
        bindings = bridge.load()
        hparams = _load_hparams(root, spec, bindings)
        seed_runtime(RUN_SEED)
        runtime = model_loader(model_alias)
        if runtime.spec != spec:
            raise CapacityHistoryEvaluationError("evaluator model loader returned different spec")
        contexts = _freeze_contexts(bridge, runtime, seed=RUN_SEED)
        if contexts.manifest_id != _mapping(
            selected.manifest.get("contexts"), "controller.contexts"
        ).get("manifest_id"):
            raise CapacityHistoryEvaluationError("evaluator/controller context identity differs")
        layer_by_weight = _layer_by_weight(hparams)
        weight_names = tuple(layer_by_weight)
        initial_hashes = _weight_hashes(runtime.model, weight_names)
        if initial_hashes != selected.actions[0].feature.get("origin_parameter_hashes"):
            raise CapacityHistoryEvaluationError("fresh W0 anchor differs from controller W0")
        covariance_specs = _covariance_specs(root, spec)
        covariance_moments, loaded_covariances = _load_verified_covariances(
            root=root, runtime=runtime, specs=covariance_specs
        )
        w0_denominators = compute_w0_denominators(
            runtime.model,
            covariance_moments,
            layer_by_weight,
            expected_layers=LAYERS,
        )

        all_neighborhood = _flatten_prompts(evaluation_fields, "neighborhood_prompts")
        all_generation = _flatten_prompts(evaluation_fields, "generation_prompts")
        w0_neighborhood = _last_token_log_probs(runtime, all_neighborhood)
        w0_generation = _last_token_log_probs(runtime, all_generation)
        w0_true_nll: list[float] = []
        for true_request in true_requests:
            teacher, target_ids = teacher_forced_rewrite_exact(runtime, true_request, contexts)
            w0_true_nll.append(rewrite_metrics(teacher, target_ids).nll)

        destination = _local_run_directory(output, run_id)
        manifest_path = destination / "manifest.json"
        checkpoints_path = destination / "checkpoints.jsonl"
        summary_path = destination / "summary.json"
        parameters = evaluator_policy_parameters()
        evaluator_sha256 = _file_sha256(Path(__file__).resolve(strict=True))
        fixed_contract = {
            "selection_manifest_id": selection.manifest_id,
            "case_order_hash": selection.order_hash,
            "chain_length": CHAIN_LENGTH,
            "layers": list(LAYERS),
            "seed": RUN_SEED,
            "policy_id": POLICY_ID,
            "policy_parameters_sha256": sha256_bytes(
                canonical_json(parameters).encode("utf-8")
            ),
            "evaluator_sha256": evaluator_sha256,
            "action_before_evaluation": True,
            "direct_z_per_branch_edit": 1,
            "history_append_policy": "post-accepted-edit-once-history-fixed-within-edit",
        }
        manifest = {
            "schema_version": EVALUATOR_SCHEMA,
            "run_id": run_id,
            "branch": branch,
            "model": runtime.metadata(),
            "ode_edit_git": git_state,
            "slurm": slurm,
            "selection": selection.to_dict(),
            "provenance_id": provenance.manifest_id,
            "fixed_files": _relative_provenance(provenance, root),
            "controller_runs": [controllers[item].run_id for item in BRANCHES],
            "all_four_controllers_verified_before_evaluation": True,
            "action_receipts_verified_before_evaluation": CHAIN_LENGTH * len(BRANCHES),
            "evaluation_fields_loaded_after_barrier": True,
            "raw_evaluation_text_persisted": False,
            "raw_logits_persisted": False,
            "raw_token_ids_persisted": False,
            "fixed_contract": fixed_contract,
            "policy_parameters": parameters,
        }
        _safe_payload(manifest)
        _write_json_exclusive(manifest_path, manifest)

        torch.cuda.reset_peak_memory_stats(0)
        cumulative_factors: list[LowRankFactor] = []
        cumulative_native_path = 0.0
        checkpoints: list[dict[str, Any]] = []
        lineage_exact = True
        setup_seconds = time.perf_counter() - started
        with SanitizedJsonlWriter(
            checkpoints_path, run_id, schema_version=STREAM_SCHEMA
        ) as writer:
            for edit_index, (request, evidence) in enumerate(
                zip(requests, selected.actions, strict=True), start=1
            ):
                checkpoint_started = time.perf_counter()
                if _weight_hashes(runtime.model, weight_names) != evidence.feature.get(
                    "origin_parameter_hashes"
                ):
                    raise CapacityHistoryEvaluationError("replay edit entry hashes differ")
                cumulative_native_path += float(evidence.feature["native_c_distance"])
                for proposal_index, proposal_manifest in enumerate(
                    evidence.proposal_manifests
                ):
                    snapshot = _capture_source_snapshot(
                        runtime=runtime,
                        request=request,
                        contexts=contexts,
                        hparams=hparams,
                        weight_names=weight_names,
                        provenance_id=bindings.provenance.manifest_id,
                    )
                    if snapshot.state_id != evidence.action["proposal_entry_state_ids"][proposal_index]:
                        raise CapacityHistoryEvaluationError("replay proposal entry state differs")
                    proposal = load_proposal_artifact(proposal_manifest, snapshot=snapshot)
                    cumulative_factors.extend(proposal.factors)
                    apply_proposal_in_disposable_process(
                        runtime.model,
                        proposal,
                        application_mode=APPLICATION_MODES[branch],
                    )
                    descendant = _capture_source_snapshot(
                        runtime=runtime,
                        request=request,
                        contexts=contexts,
                        hparams=hparams,
                        weight_names=weight_names,
                        provenance_id=bindings.provenance.manifest_id,
                    )
                    assert_snapshot_current(runtime.model, descendant)
                    if descendant.state_id != evidence.action["proposal_descendant_state_ids"][proposal_index]:
                        lineage_exact = False
                        raise CapacityHistoryEvaluationError("replay descendant state differs")

                rewrite_values = []
                for prior_request in requests[:edit_index]:
                    teacher, target_ids = teacher_forced_rewrite_exact(
                        runtime, prior_request, contexts
                    )
                    rewrite_values.append(rewrite_metrics(teacher, target_ids))
                current_utility = float(rewrite_values[-1].utility)
                all_mean = math.fsum(float(item.utility) for item in rewrite_values) / len(
                    rewrite_values
                )
                if edit_index == 1:
                    prior_mean: float | None = None
                    prior_success: float | None = None
                else:
                    prior_values = rewrite_values[:-1]
                    prior_mean = math.fsum(float(item.utility) for item in prior_values) / len(
                        prior_values
                    )
                    prior_success = sum(bool(item.exact_satisfied) for item in prior_values) / len(
                        prior_values
                    )

                prompt_count = edit_index * PROMPTS_PER_EDIT
                neighborhood_kl = _mean_forward_kl(
                    w0_neighborhood[:prompt_count],
                    _last_token_log_probs(runtime, all_neighborhood[:prompt_count]),
                )
                generation_kl = _mean_forward_kl(
                    w0_generation[:prompt_count],
                    _last_token_log_probs(runtime, all_generation[:prompt_count]),
                )
                current_true_nll: list[float] = []
                for true_request in true_requests[:edit_index]:
                    teacher, target_ids = teacher_forced_rewrite_exact(
                        runtime, true_request, contexts
                    )
                    current_true_nll.append(rewrite_metrics(teacher, target_ids).nll)
                true_delta = math.fsum(
                    current - baseline
                    for current, baseline in zip(
                        current_true_nll, w0_true_nll[:edit_index], strict=True
                    )
                ) / edit_index
                geometry = cumulative_geometry(
                    cumulative_factors,
                    covariance_by_layer=covariance_moments,
                    layer_by_weight=layer_by_weight,
                    w0_denominators=w0_denominators,
                    device=next(runtime.model.parameters()).device,
                )
                routing = _routing_diagnostic(
                    branch=branch,
                    feature=evidence.feature,
                    accepted_round_count=int(evidence.result["accepted_round_count"]),
                )
                expected_history = edit_index if is_alpha_branch(branch) else 0
                technical = {
                    "action_committed_before_evaluation": True,
                    "evaluation_firewall_pass": True,
                    "direct_z_once": bool(
                        evidence.feature["target_identity"]["direct_z_compute_count"] == 1
                        and evidence.result["direct_z_compute_count"] == 1
                    ),
                    "state_lineage_exact": lineage_exact,
                    "w0_anchor_exact": True,
                    "precomputed_covariance_only": len(loaded_covariances) == len(LAYERS),
                    "precomputed_projector_only": bool(
                        not is_alpha_branch(branch)
                        or _mapping(selected.manifest.get("precomputed"), "precomputed").get(
                            "projector_write_or_recompute"
                        )
                        is False
                    ),
                    "alpha_history_exact": evidence.result.get("history_edit_count")
                    == expected_history,
                    "capacity_barrier_exact": bool(routing["capacity_barrier_exact"]),
                    "overloaded_positive_write_suppressed": bool(
                        routing["overloaded_positive_write_suppressed"]
                    ),
                    "finite_metrics": True,
                }
                checkpoint = {
                    "model_alias": model_alias,
                    "edit_id": request.case_id,
                    "edit_index": edit_index,
                    "branch": branch,
                    "pass": True,
                    "technical": technical,
                    "fixed_contract": fixed_contract,
                    "current_utility": current_utility,
                    "all_edits_mean_utility": all_mean,
                    "prior_mean_utility": prior_mean,
                    "prior_success_rate": prior_success,
                    "neighborhood_kl": neighborhood_kl,
                    "generation_kl": generation_kl,
                    "target_true_nll_delta": true_delta,
                    **geometry,
                    "native_distance_equivalent_path_length": cumulative_native_path,
                    "accepted_path_distance": float(evidence.result["accepted_path_distance"]),
                    "wall_seconds": float(evidence.result["wall_seconds"])
                    + (time.perf_counter() - checkpoint_started)
                    + (setup_seconds if edit_index == 1 else 0.0),
                    "proposal_build_count": int(evidence.result["proposal_build_count"]),
                    "accepted_round_count": int(evidence.result["accepted_round_count"]),
                    "rejected_round_count": int(evidence.result["rejected_round_count"]),
                    "first_hit_reached": bool(evidence.result["first_hit_reached"]),
                    "history_edit_count": int(evidence.result["history_edit_count"]),
                    "overloaded_layer_observations": int(
                        routing["overloaded_layer_observations"]
                    ),
                    "suppressed_overloaded_observations": int(
                        routing["suppressed_overloaded_observations"]
                    ),
                    "capacity_reroute_round_count": int(
                        routing["capacity_reroute_round_count"]
                    ),
                    "capacity_barrier_max_violation": float(
                        routing["capacity_barrier_max_violation"]
                    ),
                }
                if not _all_finite(checkpoint):
                    technical["finite_metrics"] = False
                    checkpoint["pass"] = False
                    raise CapacityHistoryEvaluationError("checkpoint contains non-finite metrics")
                writer.write(CHECKPOINT_EVENT, checkpoint)
                writer.sync()
                checkpoints.append(checkpoint)

        summary = {
            "schema_version": EVALUATOR_SCHEMA,
            "run_id": run_id,
            "model_alias": model_alias,
            "branch": branch,
            "run_status": "completed",
            "all_pass": bool(
                len(checkpoints) == CHAIN_LENGTH
                and all(item["pass"] for item in checkpoints)
                and all(all(item["technical"].values()) for item in checkpoints)
            ),
            "checkpoint_count": len(checkpoints),
            "fixed_contract": fixed_contract,
            "controller_runs": [controllers[item].run_id for item in BRANCHES],
            "evaluation_firewall_pass": True,
            "raw_evaluation_text_persisted": False,
            "raw_logits_persisted": False,
            "raw_token_ids_persisted": False,
            "loaded_covariances": list(loaded_covariances),
            "resource": {
                "wall_seconds": time.perf_counter() - started,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "host_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
            "artifacts": {
                "manifest_sha256": _file_sha256(manifest_path),
                "checkpoints_sha256": _file_sha256(checkpoints_path),
            },
            "git_output_written": False,
        }
        _safe_payload(summary)
        _write_json_exclusive(summary_path, summary)
        return {**summary, "output_directory": str(destination)}


def merge_checkpoint_streams(
    *,
    branch_paths: Mapping[str, str | Path],
    output_path: str | Path,
    model_alias: str,
    run_id: str,
) -> Path:
    from .capacity_history_analysis import load_stream

    if set(branch_paths) != set(BRANCHES):
        raise CapacityHistoryEvaluationError("merge requires all four branch streams")
    by_branch = {branch: load_stream(branch_paths[branch]) for branch in BRANCHES}
    reference_contract = by_branch[BRANCHES[0]][0].get("fixed_contract")
    for branch in BRANCHES:
        rows = by_branch[branch]
        if (
            len(rows) != CHAIN_LENGTH
            or any(row.get("branch") != branch for row in rows)
            or any(row.get("model_alias") != model_alias for row in rows)
            or any(row.get("fixed_contract") != reference_contract for row in rows)
        ):
            raise CapacityHistoryEvaluationError("branch checkpoint streams differ from merge lock")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with SanitizedJsonlWriter(destination, run_id, schema_version=STREAM_SCHEMA) as writer:
        for branch in BRANCHES:
            for row in by_branch[branch]:
                writer.write(CHECKPOINT_EVENT, row)
        writer.sync()
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-capacity-history-evaluator", allow_abbrev=False
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", allow_abbrev=False)
    run.add_argument("--easyedit-root", required=True)
    run.add_argument("--branch", required=True, choices=BRANCHES)
    run.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    run.add_argument("--run-id", required=True)
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    merge = subparsers.add_parser("merge", allow_abbrev=False)
    merge.add_argument("--memit-native", required=True)
    merge.add_argument("--memit-qp", required=True)
    merge.add_argument("--alpha-native", required=True)
    merge.add_argument("--alpha-qp", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    merge.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        summary = run_capacity_history_evaluator(
            easyedit_root=args.easyedit_root,
            branch=args.branch,
            model_alias=args.model,
            run_id=args.run_id,
            output_root=args.output_root,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary["all_pass"] else 1
    path = merge_checkpoint_streams(
        branch_paths={
            BRANCHES[0]: args.memit_native,
            BRANCHES[1]: args.memit_qp,
            BRANCHES[2]: args.alpha_native,
            BRANCHES[3]: args.alpha_qp,
        },
        output_path=args.output,
        model_alias=args.model,
        run_id=args.run_id,
    )
    print(json.dumps({"output": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
