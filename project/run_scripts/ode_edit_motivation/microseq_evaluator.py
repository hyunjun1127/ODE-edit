"""Evaluation-only replay for the four-edit Motivation diagnostic.

The evaluator is intentionally separated from :mod:`microseq_controller`.
It verifies that both model/branch controllers are terminal and that all eight
action receipts are durable before it decodes any CounterFact evaluation
field.  It then loads a fresh W0 model, replays one committed branch, and emits
scalar-only checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import string
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .contracts import ContractError, EditRequest, LowRankFactor, canonical_json, sha256_bytes
from .easyedit_bridge import EasyEditBridge
from .gpu_runtime import FixedModelRuntime, load_fixed_model, offline_environment, seed_runtime
from .hooks import assert_snapshot_current, assert_tensor_sha256_device_parity, resolve_parameter
from .manifests import (
    COUNTERFACT_RELATIVE_PATH,
    MODEL_SPECS,
    _CASE_ID_PATTERN,
    _decode_case_id,
    _iter_top_level_json_objects,
    fixed_model_spec,
    load_counterfact_requests,
    preflight_fixed_artifacts,
)
from .microseq_analysis import (
    BRANCH_NATIVE,
    BRANCH_ODE,
    BRANCHES,
    CHAIN_LENGTH,
    CHECKPOINT_EVENT,
    STREAM_SCHEMA,
    load_stream,
)
from .microseq_artifacts import apply_proposal_in_disposable_process, load_proposal_artifact
from .microseq_controller import (
    APPLICATION_MODES,
    MICROSEQ_JOB_NAME,
    MICROSEQ_POLICY_ID,
    MICROSEQ_RECEIPT_SCHEMA,
    MICROSEQ_RUN_SEED,
    MICROSEQ_SCHEMA,
    MICROSEQ_STREAM_SCHEMA,
    RUN_IDS as CONTROLLER_RUN_IDS,
    generate_microseq_selection,
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
from .quarter_step_refresh import LAYERS, PROBE_ACTIONS, QSTEP_HOP_FRACTION, QSTEP_K, QSTEP_PROBE_FRACTION, _capture_source_snapshot


EVALUATOR_SCHEMA = "ode-edit-microseq-evaluator/v1"
EVALUATOR_RUN_IDS = {
    BRANCH_NATIVE: {
        "llama3-8b-inst": "microseq_eval_native_llama_m0_v1",
        "qwen2.5-7b-inst": "microseq_eval_native_qwen_m0_v1",
    },
    BRANCH_ODE: {
        "llama3-8b-inst": "microseq_eval_ode_llama_m0_v1",
        "qwen2.5-7b-inst": "microseq_eval_ode_qwen_m0_v1",
    },
}
PAIR_POLICY_ID = "native-vs-always-refresh-k4-d4-scoremix-v1"
PROMPTS_PER_EDIT = 4
PROMPT_BATCH_SIZE = 4
WEIGHT_ENERGY_ROW_BLOCK = 64


class MicroseqEvaluationError(RuntimeError):
    """The committed-controller or evaluation-only contract was violated."""


@dataclass(frozen=True, slots=True)
class EvaluationFields:
    case_id: str
    neighborhood_prompts: tuple[str, ...]
    generation_prompts: tuple[str, ...]
    target_true: str


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


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MicroseqEvaluationError(f"{path}: mapping required")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink():
        raise MicroseqEvaluationError(f"{path}: symlink JSON is forbidden")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise MicroseqEvaluationError(f"{path}: regular JSON file required")
    try:
        return _mapping(json.loads(resolved.read_text(encoding="utf-8")), str(path))
    except json.JSONDecodeError as exc:
        raise MicroseqEvaluationError(f"{path}: invalid JSON") from exc


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits.lower() for character in value)
        or value != value.lower()
    ):
        raise MicroseqEvaluationError(f"{path}: lowercase SHA-256 required")
    return value


def _load_controller_stream(
    path: Path, *, run_id: str, event: str
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    if path.is_symlink():
        raise MicroseqEvaluationError(f"{path}: symlink stream is forbidden")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise MicroseqEvaluationError(f"{path}: regular stream required")
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                wrapper = _mapping(json.loads(line), f"{path}:{line_number}")
            except json.JSONDecodeError as exc:
                raise MicroseqEvaluationError(f"{path}:{line_number}: invalid JSON") from exc
            if set(wrapper) != {
                "schema_version",
                "run_id",
                "sequence",
                "recorded_at",
                "event",
                "payload",
            }:
                raise MicroseqEvaluationError(f"{path}:{line_number}: wrapper keys differ")
            if (
                wrapper["schema_version"] != MICROSEQ_STREAM_SCHEMA
                or wrapper["run_id"] != run_id
                or wrapper["sequence"] != len(records)
                or wrapper["event"] != event
            ):
                raise MicroseqEvaluationError(f"{path}:{line_number}: stream identity differs")
            records.append(_mapping(wrapper["payload"], f"{path}:{line_number}.payload"))
    if len(records) != CHAIN_LENGTH:
        raise MicroseqEvaluationError(f"{path}: exact four controller records required")
    return tuple(records)


def _safe_local_child(root: Path, name: str) -> Path:
    if Path(name).name != name:
        raise MicroseqEvaluationError("controller artifact name is not a basename")
    if root.is_symlink() or not root.is_dir():
        raise MicroseqEvaluationError("controller artifact directory must be real")
    unresolved = root / name
    if unresolved.is_symlink():
        raise MicroseqEvaluationError("controller artifact symlink is forbidden")
    candidate = unresolved.resolve(strict=True)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise MicroseqEvaluationError("controller artifact escaped its run directory") from exc
    if not candidate.is_file():
        raise MicroseqEvaluationError("controller artifact must be a regular file")
    return candidate


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
    unresolved_run_directory = output_root / run_id
    if unresolved_run_directory.is_symlink():
        raise MicroseqEvaluationError(f"{run_id}: symlink run directory is forbidden")
    run_directory = unresolved_run_directory.resolve(strict=True)
    if not run_directory.is_dir():
        raise MicroseqEvaluationError(f"{run_id}: controller run directory required")
    manifest = _read_json(run_directory / "manifest.json")
    summary = _read_json(run_directory / "summary.json")
    expected_proposals = CHAIN_LENGTH if branch == BRANCH_NATIVE else CHAIN_LENGTH * QSTEP_K
    if (
        manifest.get("schema_version") != MICROSEQ_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("controller_branch") != branch
        or manifest.get("application_mode") != APPLICATION_MODES[branch]
        or _mapping(manifest.get("model"), "manifest.model").get("model_alias") != model_alias
        or _mapping(manifest.get("selection"), "manifest.selection").get("manifest_id")
        != selection_manifest_id
        or _mapping(manifest.get("selection"), "manifest.selection").get("order_hash")
        != case_order_hash
        or _mapping(manifest.get("selection"), "manifest.selection").get("case_ids")
        != list(expected_case_ids)
    ):
        raise MicroseqEvaluationError(f"{run_id}: manifest identity differs")
    constants = _mapping(manifest.get("constants"), "manifest.constants")
    firewall = _mapping(manifest.get("firewall"), "manifest.firewall")
    if (
        constants.get("layers") != list(LAYERS)
        or constants.get("chain_length") != CHAIN_LENGTH
        or constants.get("k") != QSTEP_K
        or constants.get("hop_fraction") != QSTEP_HOP_FRACTION
        or constants.get("probe_fraction") != QSTEP_PROBE_FRACTION
        or constants.get("run_seed") != MICROSEQ_RUN_SEED
        or firewall
        != {
            "controller_only": True,
            "outcome_fields_loaded": False,
            "separate_evaluator_required": True,
        }
    ):
        raise MicroseqEvaluationError(f"{run_id}: constants/firewall differ")
    if (
        summary.get("schema_version") != MICROSEQ_SCHEMA
        or summary.get("run_id") != run_id
        or summary.get("model_alias") != model_alias
        or summary.get("controller_branch") != branch
        or summary.get("run_status") != "completed"
        or summary.get("failure_type") is not None
        or summary.get("all_pass") is not True
        or summary.get("expected_counts_exact") is not True
        or summary.get("receipt_count") != CHAIN_LENGTH
        or summary.get("proposal_artifact_count") != expected_proposals
        or summary.get("direct_z_artifact_count") != CHAIN_LENGTH
        or summary.get("selection_manifest_id") != selection_manifest_id
        or summary.get("controller_only") is not True
        or summary.get("outcome_fields_loaded") is not False
        or summary.get("disposable_process_exit_required") is not True
    ):
        raise MicroseqEvaluationError(f"{run_id}: controller is not terminal/pass-exact")
    sequences = _mapping(summary.get("stream_sequences"), "summary.stream_sequences")
    if sequences != {"features": 4, "actions": 4, "events": 4}:
        raise MicroseqEvaluationError(f"{run_id}: stream counts differ")
    summary_artifacts = _mapping(summary.get("artifacts"), "summary.artifacts")
    for filename, key in (
        ("manifest.json", "manifest_sha256"),
        ("features.jsonl", "features_sha256"),
        ("actions.jsonl", "actions_sha256"),
        ("events.jsonl", "events_sha256"),
    ):
        if _file_sha256(run_directory / filename) != _sha256(
            summary_artifacts.get(key), f"summary.artifacts.{key}"
        ):
            raise MicroseqEvaluationError(f"{run_id}: {filename} hash differs")

    features = _load_controller_stream(
        run_directory / "features.jsonl",
        run_id=run_id,
        event="microseq_controller_feature",
    )
    actions = _load_controller_stream(
        run_directory / "actions.jsonl",
        run_id=run_id,
        event="microseq_controller_action",
    )
    results = _load_controller_stream(
        run_directory / "events.jsonl",
        run_id=run_id,
        event="microseq_controller_edit",
    )
    evidence: list[ControllerActionEvidence] = []
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
            or result.get("branch") != branch
            or feature.get("application_mode") != APPLICATION_MODES[branch]
            or action.get("application_mode") != APPLICATION_MODES[branch]
            or result.get("pass") is not True
            or result.get("controller_only") is not True
            or result.get("outcome_fields_loaded") is not False
            or feature.get("controller_only") is not True
            or feature.get("outcome_fields_loaded") is not False
            or action.get("all_actions_before_outcomes") is not True
        ):
            raise MicroseqEvaluationError(f"{run_id}: action {edit_index} identity differs")
        if (
            feature.get("request_id") != action.get("request_id")
            or feature.get("request_id") != result.get("request_id")
            or action.get("feature_hash") != feature.get("feature_hash")
            or _feature_hash({key: value for key, value in feature.items() if key != "feature_hash"})
            != feature.get("feature_hash")
            or _feature_hash({key: value for key, value in action.items() if key != "commitment_hash"})
            != action.get("commitment_hash")
            or result.get("feature_hash") != feature.get("feature_hash")
            or result.get("commitment_hash") != action.get("commitment_hash")
        ):
            raise MicroseqEvaluationError(f"{run_id}: action {edit_index} commitment differs")
        target_identity = _mapping(feature.get("target_identity"), "feature.target_identity")
        if (
            target_identity.get("direct_z_compute_count") != 1
            or result.get("direct_z_compute_count") != 1
        ):
            raise MicroseqEvaluationError(f"{run_id}: action {edit_index} target count differs")
        raw_metadata = feature.get("proposal_artifacts")
        if not isinstance(raw_metadata, list) or len(raw_metadata) != result.get("proposal_count"):
            raise MicroseqEvaluationError(f"{run_id}: action {edit_index} proposal count differs")
        proposal_paths: list[Path] = []
        action_ids: list[str] = []
        proposal_direction_hashes: list[str] = []
        for artifact_index, raw_metadata_item in enumerate(raw_metadata):
            metadata = _mapping(raw_metadata_item, f"proposal_artifacts[{artifact_index}]")
            manifest_name = metadata.get("name")
            tensor_name = metadata.get("tensor_name")
            if not isinstance(manifest_name, str) or not isinstance(tensor_name, str):
                raise MicroseqEvaluationError("proposal artifact names must be strings")
            manifest_path = _safe_local_child(run_directory / "proposal_artifacts", manifest_name)
            tensor_path = _safe_local_child(run_directory / "proposal_artifacts", tensor_name)
            if (
                manifest_path.suffix != ".json"
                or tensor_path.name != manifest_path.with_suffix(".pt").name
                or _file_sha256(manifest_path)
                != _sha256(metadata.get("sha256"), "proposal manifest sha256")
                or _file_sha256(tensor_path)
                != _sha256(metadata.get("tensor_sha256"), "proposal tensor sha256")
            ):
                raise MicroseqEvaluationError(f"{run_id}: proposal artifact identity differs")
            proposal_paths.append(manifest_path)
            action_ids.append(manifest_path.stem)
            proposal_manifest = _read_json(manifest_path)
            proposal_direction_hashes.append(
                _sha256(
                    proposal_manifest.get("proposal_direction_sha256"),
                    "proposal_direction_sha256",
                )
            )
        path_action_hashes = _mapping(
            action.get("path_action_hashes"), "action.path_action_hashes"
        )
        if (
            action.get("branch_order") != action_ids
            or result.get("proposal_count") != len(proposal_paths)
            or len(action.get("proposal_entry_state_ids", ())) != len(proposal_paths)
            or len(action.get("proposal_descendant_state_ids", ())) != len(proposal_paths)
            or result.get("terminal_state_id") != action["proposal_descendant_state_ids"][-1]
            or path_action_hashes != {branch: proposal_direction_hashes}
        ):
            raise MicroseqEvaluationError(f"{run_id}: proposal order/lineage differs")
        receipt_name = result.get("receipt_name")
        if not isinstance(receipt_name, str):
            raise MicroseqEvaluationError("controller receipt name must be a string")
        receipt_path = _safe_local_child(run_directory / "action_receipts", receipt_name)
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
            or receipt.get("durability")
            != "feature+actions-write+flush+fsync-before-exclusive-receipt"
            or _file_sha256(receipt_path)
            != _sha256(result.get("receipt_sha256"), "receipt sha256")
            or receipt.get("schema_version") != MICROSEQ_RECEIPT_SCHEMA
            or receipt.get("case_id") != case_id
            or receipt.get("request_id") != feature.get("request_id")
            or receipt.get("feature_hash") != feature.get("feature_hash")
            or receipt.get("commitment_hash") != action.get("commitment_hash")
            or receipt.get("origin_lineage_id") != target_identity.get("origin_lineage_id")
            or receipt.get("branch_order") != action_ids
            or receipt.get("path_action_hashes") != action.get("path_action_hashes")
            or receipt.get("per_hop_c_energy") != action.get("per_hop_c_energy")
            or receipt.get("all_paths_committed_before_outcomes") is not True
        ):
            raise MicroseqEvaluationError(f"{run_id}: receipt {edit_index} differs")
        evidence.append(
            ControllerActionEvidence(
                feature=feature,
                action=action,
                result=result,
                proposal_manifests=tuple(proposal_paths),
            )
        )
    return ControllerEvidence(
        branch=branch,
        model_alias=model_alias,
        run_id=run_id,
        run_directory=run_directory,
        manifest=manifest,
        summary=summary,
        actions=tuple(evidence),
    )


def verify_controller_pair(
    *, output_root: str | Path, easyedit_root: str | Path, model_alias: str
) -> tuple[ControllerEvidence, ControllerEvidence]:
    """Verify both terminal controllers before any evaluation-row decode."""

    if model_alias not in MODEL_SPECS:
        raise MicroseqEvaluationError("model alias is outside the fixed pair")
    root = Path(output_root).expanduser().resolve(strict=True)
    selection = generate_microseq_selection(easyedit_root)
    common = {
        "output_root": root,
        "model_alias": model_alias,
        "expected_case_ids": selection.case_ids,
        "selection_manifest_id": selection.manifest_id,
        "case_order_hash": selection.order_hash,
    }
    native = _verify_controller(branch=BRANCH_NATIVE, **common)
    ode = _verify_controller(branch=BRANCH_ODE, **common)
    native_context = _mapping(native.manifest.get("contexts"), "native.contexts")
    ode_context = _mapping(ode.manifest.get("contexts"), "ode.contexts")
    native_git = _mapping(native.manifest.get("ode_edit_git"), "native.ode_edit_git")
    ode_git = _mapping(ode.manifest.get("ode_edit_git"), "ode.ode_edit_git")
    if (
        native_context != ode_context
        or native.manifest.get("provenance_id") != ode.manifest.get("provenance_id")
        or native_git != ode_git
        or native.manifest.get("fixed_files") != ode.manifest.get("fixed_files")
        or [item.feature["case_id"] for item in native.actions]
        != [item.feature["case_id"] for item in ode.actions]
    ):
        raise MicroseqEvaluationError("controller pair common contract differs")
    return native, ode


def _validated_text(value: Any, path: str, *, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise MicroseqEvaluationError(f"{path}: non-empty bounded string required")
    if any(ord(character) < 32 and character not in "\n\t" for character in value):
        raise MicroseqEvaluationError(f"{path}: control character forbidden")
    return value


def load_counterfact_evaluation_fields(
    easyedit_root: str | Path, case_ids: Sequence[str]
) -> tuple[EvaluationFields, ...]:
    """Lazy-load only the fixed evaluation fields; never persist their text."""

    requested_order = tuple(str(case_id) for case_id in case_ids)
    if len(requested_order) != CHAIN_LENGTH or len(set(requested_order)) != CHAIN_LENGTH:
        raise MicroseqEvaluationError("evaluation requires exact four unique case IDs")
    requested = set(requested_order)
    source = Path(easyedit_root).expanduser().resolve(strict=True) / COUNTERFACT_RELATIVE_PATH
    found: dict[str, EvaluationFields] = {}
    for blob in _iter_top_level_json_objects(source):
        matches = _CASE_ID_PATTERN.findall(blob)
        if len(matches) != 1:
            raise MicroseqEvaluationError("CounterFact row has non-canonical case_id count")
        case_id = _decode_case_id(matches[0])
        if case_id not in requested:
            continue
        try:
            row = json.loads(blob)
            rewrite = row["requested_rewrite"]
            neighborhood = row["neighborhood_prompts"]
            generation = row["generation_prompts"]
            target_true = rewrite["target_true"]["str"]
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MicroseqEvaluationError(f"CounterFact case {case_id} evaluation schema differs") from exc
        if (
            not isinstance(neighborhood, list)
            or not isinstance(generation, list)
            or len(neighborhood) < PROMPTS_PER_EDIT
            or len(generation) < PROMPTS_PER_EDIT
        ):
            raise MicroseqEvaluationError(f"CounterFact case {case_id} lacks four preservation prompts")
        found[case_id] = EvaluationFields(
            case_id=case_id,
            neighborhood_prompts=tuple(
                _validated_text(item, f"{case_id}.neighborhood[{index}]", max_length=8192)
                for index, item in enumerate(neighborhood[:PROMPTS_PER_EDIT])
            ),
            generation_prompts=tuple(
                _validated_text(item, f"{case_id}.generation[{index}]", max_length=8192)
                for index, item in enumerate(generation[:PROMPTS_PER_EDIT])
            ),
            target_true=_validated_text(target_true, f"{case_id}.target_true", max_length=2048).strip(),
        )
        if len(found) == len(requested):
            break
    missing = requested - set(found)
    if missing:
        raise MicroseqEvaluationError(f"CounterFact evaluation rows missing: {sorted(missing)}")
    return tuple(found[case_id] for case_id in requested_order)


def _last_token_log_probs(runtime: FixedModelRuntime, prompts: Sequence[str]) -> torch.Tensor:
    if not prompts:
        raise MicroseqEvaluationError("next-token evaluation requires prompts")
    batches: list[torch.Tensor] = []
    device = next(runtime.model.parameters()).device
    for start in range(0, len(prompts), PROMPT_BATCH_SIZE):
        batch_prompts = list(prompts[start : start + PROMPT_BATCH_SIZE])
        encoded = runtime.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        positions = attention_mask.sum(dim=1) - 1
        if bool((positions < 0).any()):
            raise MicroseqEvaluationError("next-token prompt tokenized to empty input")
        with torch.inference_mode():
            output = runtime.model(input_ids=input_ids, attention_mask=attention_mask)
            rows = torch.arange(input_ids.shape[0], device=device)
            selected = output.logits[rows, positions, :].float()
            batches.append(torch.log_softmax(selected, dim=-1).detach())
        del output, selected, input_ids, attention_mask
    result = torch.cat(batches, dim=0)
    if result.shape[0] != len(prompts) or not bool(torch.isfinite(result).all()):
        raise MicroseqEvaluationError("next-token log-probability tensor is invalid")
    return result


def _mean_forward_kl(reference_log_probs: torch.Tensor, current_log_probs: torch.Tensor) -> float:
    if reference_log_probs.shape != current_log_probs.shape or reference_log_probs.ndim != 2:
        raise MicroseqEvaluationError("KL tensors differ in shape")
    values = torch.sum(
        reference_log_probs.exp() * (reference_log_probs - current_log_probs), dim=-1
    )
    tolerance = 1000 * torch.finfo(values.dtype).eps
    if not bool(torch.isfinite(values).all()) or bool((values < -tolerance).any()):
        raise MicroseqEvaluationError("next-token KL is non-finite or negative")
    return float(values.clamp_min(0).mean().detach().cpu())


def _true_requests(
    requests: Sequence[EditRequest], fields: Sequence[EvaluationFields]
) -> tuple[EditRequest, ...]:
    if [request.case_id for request in requests] != [field.case_id for field in fields]:
        raise MicroseqEvaluationError("rewrite/evaluation case order differs")
    return tuple(
        EditRequest.from_mapping(
            {
                "case_id": request.case_id,
                "prompt": request.prompt,
                "subject": request.subject,
                "target_new": field.target_true,
            }
        )
        for request, field in zip(requests, fields, strict=True)
    )


def _weight_c_energy(
    weight: torch.Tensor, covariance: torch.Tensor, *, row_block: int = WEIGHT_ENERGY_ROW_BLOCK
) -> float:
    """Compute ``tr(W C W.T)`` exactly in bounded GPU row blocks."""

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise MicroseqEvaluationError("covariance must be square")
    if weight.ndim != 2 or weight.shape[1] != covariance.shape[0] or row_block <= 0:
        raise MicroseqEvaluationError("weight/covariance geometry differs")
    device = weight.device
    metric = covariance.to(device=device, dtype=torch.float32)
    total = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, weight.shape[0], row_block):
            block = weight[start : start + row_block].float()
            applied = block @ metric
            total += torch.sum(block.double() * applied.double())
            del block, applied
    result = float(total.detach().cpu())
    del metric, total
    if not math.isfinite(result) or result <= 0.0:
        raise MicroseqEvaluationError("W0 C-energy must be finite and positive")
    return result


def compute_w0_denominators(
    runtime: FixedModelRuntime,
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
) -> dict[int, float]:
    denominators: dict[int, float] = {}
    for weight_name, layer in layer_by_weight.items():
        if layer in denominators or layer not in covariance_by_layer:
            raise MicroseqEvaluationError("layer/covariance mapping is not one-to-one")
        denominators[layer] = _weight_c_energy(
            resolve_parameter(runtime.model, weight_name), covariance_by_layer[layer]
        )
    if set(denominators) != set(LAYERS):
        raise MicroseqEvaluationError("W0 denominator layers differ from fixed lock")
    return denominators


def _cumulative_factor_matrices(
    factors: Sequence[LowRankFactor], weight_name: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    selected = [factor for factor in factors if factor.weight_name == weight_name]
    if not selected:
        return None
    shape = selected[0].weight_shape
    if any(factor.weight_shape != shape for factor in selected):
        raise MicroseqEvaluationError("cumulative factor shapes differ")
    return (
        torch.cat([factor.left.detach().cpu().float() for factor in selected], dim=1),
        torch.cat([factor.right.detach().cpu().float() for factor in selected], dim=1),
    )


def _factor_energy(
    left: torch.Tensor,
    right: torch.Tensor,
    covariance: torch.Tensor | None,
    *,
    device: torch.device,
) -> float:
    left_device = left.to(device=device, dtype=torch.float32)
    right_device = right.to(device=device, dtype=torch.float32)
    if covariance is None:
        applied = right_device
    else:
        metric = covariance.to(device=device, dtype=torch.float32)
        applied = metric @ right_device
    left_gram = left_device.transpose(0, 1) @ left_device
    right_gram = right_device.transpose(0, 1) @ applied
    value = float(torch.sum(left_gram.double() * right_gram.double()).detach().cpu())
    if covariance is not None:
        del metric
    del left_device, right_device, applied, left_gram, right_gram
    tolerance = 1e-6 * max(1.0, abs(value))
    if not math.isfinite(value) or value < -tolerance:
        raise MicroseqEvaluationError("cumulative factor energy is invalid")
    return max(0.0, value)


def _gini(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise MicroseqEvaluationError("Gini requires finite nonnegative values")
    total = math.fsum(values)
    if total == 0.0:
        return 0.0
    ordered = sorted(values)
    count = len(ordered)
    numerator = math.fsum((2 * index - count - 1) * value for index, value in enumerate(ordered, start=1))
    result = numerator / (count * total)
    return min(1.0, max(0.0, result))


def cumulative_geometry(
    factors: Sequence[LowRankFactor],
    *,
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
    w0_denominators: Mapping[int, float],
    device: torch.device | None = None,
) -> dict[str, float]:
    if not factors:
        raise MicroseqEvaluationError("cumulative geometry requires factors")
    capacities: list[float] = []
    frobenius_squared = 0.0
    compute_device = torch.device("cpu") if device is None else device
    for weight_name, layer in layer_by_weight.items():
        matrices = _cumulative_factor_matrices(factors, weight_name)
        if matrices is None:
            capacities.append(0.0)
            continue
        left, right = matrices
        c_energy = _factor_energy(
            left, right, covariance_by_layer[layer], device=compute_device
        )
        frobenius_squared += _factor_energy(
            left, right, None, device=compute_device
        )
        denominator = float(w0_denominators[layer])
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise MicroseqEvaluationError("W0 capacity denominator is invalid")
        capacities.append(c_energy / (denominator + torch.finfo(torch.float64).eps))
    capacity_sum = math.fsum(capacities)
    if capacity_sum <= 0.0 or not math.isfinite(frobenius_squared):
        raise MicroseqEvaluationError("cumulative geometry is zero or non-finite")
    return {
        "capacity_sum": capacity_sum,
        "max_layer_share": max(capacities) / capacity_sum,
        "layer_gini": _gini(capacities),
        "cumulative_frobenius": math.sqrt(max(0.0, frobenius_squared)),
    }


def _policy_parameters() -> dict[str, Any]:
    return {
        "branches": list(BRANCHES),
        "application_modes": dict(APPLICATION_MODES),
        "layers": list(LAYERS),
        "chain_length": CHAIN_LENGTH,
        "run_seed": MICROSEQ_RUN_SEED,
        "ode_policy_id": MICROSEQ_POLICY_ID,
        "k": QSTEP_K,
        "hop_fraction": QSTEP_HOP_FRACTION,
        "probe_fraction": QSTEP_PROBE_FRACTION,
        "prompts_per_edit": PROMPTS_PER_EDIT,
        "kl_direction": "KL(W0||Wt)",
    }


def _execution_envelope(branch: str, model_alias: str, run_id: str) -> None:
    if (
        branch not in BRANCHES
        or model_alias not in MODEL_SPECS
        or run_id != EVALUATOR_RUN_IDS[branch][model_alias]
    ):
        raise MicroseqEvaluationError("microseq evaluator identity differs from precommit")


def _slurm_state(branch: str, model_alias: str, run_id: str) -> dict[str, Any]:
    _execution_envelope(branch, model_alias, run_id)
    values = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "role": os.environ.get("ODEEDIT_MICROSEQ_PROCESS_ROLE"),
        "barrier": os.environ.get("ODEEDIT_MICROSEQ_CONTROLLER_BARRIER"),
    }
    if all(value is None for value in values.values()):
        return {"under_slurm": False}
    if any(value is None for value in values.values()):
        raise MicroseqEvaluationError("partial evaluator Slurm identity is forbidden")
    expected_barrier = "+".join(
        (CONTROLLER_RUN_IDS[BRANCH_NATIVE][model_alias], CONTROLLER_RUN_IDS[BRANCH_ODE][model_alias])
    )
    if (
        values["job_name"] != MICROSEQ_JOB_NAME
        or values["node"] != "devbox"
        or values["role"] != "evaluator"
        or values["barrier"] != expected_barrier
        or not str(values["job_id"]).isdigit()
    ):
        raise MicroseqEvaluationError("evaluator Slurm/barrier identity differs")
    return {"under_slurm": True, **values}


def _flatten_prompts(
    fields: Sequence[EvaluationFields], attribute: str
) -> tuple[str, ...]:
    return tuple(prompt for field in fields for prompt in getattr(field, attribute))


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    return False


def run_microseq_evaluator(
    *,
    easyedit_root: str | Path,
    branch: str,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
    evaluation_loader: Callable[[str | Path, Sequence[str]], tuple[EvaluationFields, ...]] = load_counterfact_evaluation_fields,
) -> dict[str, Any]:
    started = time.perf_counter()
    _execution_envelope(branch, model_alias, run_id)
    slurm = _slurm_state(branch, model_alias, run_id)
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    git_state = _git_runtime_state()
    provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_microseq_selection(root)

    # This verification must remain before the first evaluation-row decode.
    native_evidence, ode_evidence = verify_controller_pair(
        output_root=output, easyedit_root=root, model_alias=model_alias
    )
    if _mapping(
        native_evidence.manifest.get("ode_edit_git"), "controller.ode_edit_git"
    ) != git_state:
        raise MicroseqEvaluationError("controller/evaluator Git identities differ")
    selected_evidence = native_evidence if branch == BRANCH_NATIVE else ode_evidence
    evaluation_fields = evaluation_loader(root, selection.case_ids)
    requests = load_counterfact_requests(root, selection.case_ids)
    true_requests = _true_requests(requests, evaluation_fields)

    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not set(record.path for record in bridge_provenance.files).issubset(
        set(record.path for record in provenance.files)
    ):
        raise MicroseqEvaluationError("evaluator bridge provenance is outside fixed manifest")

    with offline_environment():
        assert_tensor_sha256_device_parity(torch.device("cuda", 0))
        bindings = bridge.load()
        hparams = _load_hparams(root, spec, bindings)
        seed_runtime(MICROSEQ_RUN_SEED)
        runtime = model_loader(model_alias)
        if runtime.spec != spec:
            raise MicroseqEvaluationError("evaluator model loader returned different spec")
        contexts = _freeze_contexts(bridge, runtime, seed=MICROSEQ_RUN_SEED)
        if contexts.manifest_id != _mapping(
            selected_evidence.manifest.get("contexts"), "controller.contexts"
        ).get("manifest_id"):
            raise MicroseqEvaluationError("evaluator/controller context identity differs")
        layer_by_weight = _layer_by_weight(hparams)
        weight_names = tuple(layer_by_weight)
        initial_hashes = _weight_hashes(runtime.model, weight_names)
        if initial_hashes != selected_evidence.actions[0].feature.get("origin_parameter_hashes"):
            raise MicroseqEvaluationError("fresh W0 anchor differs from controller W0")

        covariance_specs = _covariance_specs(root, spec)
        covariance_moments, loaded_covariances = _load_verified_covariances(
            root=root, runtime=runtime, specs=covariance_specs
        )
        w0_denominators = compute_w0_denominators(runtime, covariance_moments, layer_by_weight)

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
        policy_parameters = _policy_parameters()
        evaluator_sha256 = _file_sha256(Path(__file__).resolve(strict=True))
        fixed_contract = {
            "selection_manifest_id": selection.manifest_id,
            "case_order_hash": selection.order_hash,
            "chain_length": CHAIN_LENGTH,
            "layers": list(LAYERS),
            "seed": MICROSEQ_RUN_SEED,
            "policy_id": PAIR_POLICY_ID,
            "policy_parameters_sha256": sha256_bytes(
                canonical_json(policy_parameters).encode("utf-8")
            ),
            "evaluator_sha256": evaluator_sha256,
            "action_before_evaluation": True,
            "direct_z_per_branch_edit": 1,
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
            "controller_runs": [native_evidence.run_id, ode_evidence.run_id],
            "controller_summaries_verified_before_evaluation": True,
            "action_receipts_verified_before_evaluation": CHAIN_LENGTH * len(BRANCHES),
            "evaluation_fields_loaded_after_barrier": True,
            "raw_evaluation_text_persisted": False,
            "raw_logits_persisted": False,
            "raw_token_ids_persisted": False,
            "fixed_contract": fixed_contract,
            "policy_parameters": policy_parameters,
        }
        _safe_payload(manifest)
        _write_json_exclusive(manifest_path, manifest)

        torch.cuda.reset_peak_memory_stats(0)
        evaluator_setup_seconds = time.perf_counter() - started
        cumulative_factors: list[LowRankFactor] = []
        cumulative_native_path = 0.0
        checkpoints: list[dict[str, Any]] = []
        lineage_exact = True
        with SanitizedJsonlWriter(
            checkpoints_path, run_id, schema_version=STREAM_SCHEMA
        ) as writer:
            for edit_index, (request, evidence) in enumerate(
                zip(requests, selected_evidence.actions, strict=True), start=1
            ):
                checkpoint_started = time.perf_counter()
                if _weight_hashes(runtime.model, weight_names) != evidence.feature.get(
                    "origin_parameter_hashes"
                ):
                    raise MicroseqEvaluationError("controller/evaluator edit entry hashes differ")
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
                        raise MicroseqEvaluationError("replay proposal entry state differs")
                    proposal = load_proposal_artifact(proposal_manifest, snapshot=snapshot)
                    cumulative_factors.extend(proposal.factors)
                    apply_proposal_in_disposable_process(
                        runtime.model,
                        proposal,
                        application_mode=APPLICATION_MODES[branch],
                    )
                    expected_state = evidence.action["proposal_descendant_state_ids"][proposal_index]
                    descendant = _capture_source_snapshot(
                        runtime=runtime,
                        request=request,
                        contexts=contexts,
                        hparams=hparams,
                        weight_names=weight_names,
                        provenance_id=bindings.provenance.manifest_id,
                    )
                    assert_snapshot_current(runtime.model, descendant)
                    if descendant.state_id != expected_state:
                        lineage_exact = False
                        raise MicroseqEvaluationError("replay descendant state differs")

                rewrite_values = []
                for prior_request in requests[:edit_index]:
                    teacher, target_ids = teacher_forced_rewrite_exact(
                        runtime, prior_request, contexts
                    )
                    rewrite_values.append(rewrite_metrics(teacher, target_ids))
                current_utility = rewrite_values[-1].utility
                all_mean = math.fsum(item.utility for item in rewrite_values) / len(rewrite_values)
                if edit_index == 1:
                    prior_mean: float | None = None
                    prior_success: float | None = None
                else:
                    prior_values = rewrite_values[:-1]
                    prior_mean = math.fsum(item.utility for item in prior_values) / len(prior_values)
                    prior_success = sum(item.exact_satisfied for item in prior_values) / len(prior_values)

                prompt_count = edit_index * PROMPTS_PER_EDIT
                current_neighborhood = _last_token_log_probs(
                    runtime, all_neighborhood[:prompt_count]
                )
                current_generation = _last_token_log_probs(
                    runtime, all_generation[:prompt_count]
                )
                neighborhood_kl = _mean_forward_kl(
                    w0_neighborhood[:prompt_count], current_neighborhood
                )
                generation_kl = _mean_forward_kl(
                    w0_generation[:prompt_count], current_generation
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
                controlled_nfe = (
                    1
                    if branch == BRANCH_NATIVE
                    else 1 + QSTEP_K * 2 * len(PROBE_ACTIONS)
                )
                technical = {
                    "action_committed_before_evaluation": True,
                    "evaluation_firewall_pass": True,
                    "direct_z_once": bool(
                        evidence.feature["target_identity"]["direct_z_compute_count"] == 1
                        and evidence.result["direct_z_compute_count"] == 1
                    ),
                    "state_lineage_exact": lineage_exact,
                    "w0_anchor_exact": True,
                    "precomputed_cache_only": len(loaded_covariances) == len(LAYERS),
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
                    "wall_seconds": float(evidence.result["wall_seconds"])
                    + (time.perf_counter() - checkpoint_started)
                    + (evaluator_setup_seconds if edit_index == 1 else 0.0),
                    "proposal_build_count": int(evidence.result["proposal_build_count"]),
                    "controlled_nfe": controlled_nfe,
                }
                if not _all_finite(checkpoint):
                    technical["finite_metrics"] = False
                    checkpoint["pass"] = False
                    raise MicroseqEvaluationError("checkpoint contains non-finite metrics")
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
            "controller_runs": [native_evidence.run_id, ode_evidence.run_id],
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
    native_path: str | Path,
    ode_path: str | Path,
    output_path: str | Path,
    model_alias: str,
    run_id: str,
) -> Path:
    native = load_stream(native_path)
    ode = load_stream(ode_path)
    if (
        len(native) != CHAIN_LENGTH
        or len(ode) != CHAIN_LENGTH
        or any(item.get("branch") != BRANCH_NATIVE for item in native)
        or any(item.get("branch") != BRANCH_ODE for item in ode)
        or any(item.get("model_alias") != model_alias for item in (*native, *ode))
        or any(item.get("fixed_contract") != native[0].get("fixed_contract") for item in (*native, *ode))
    ):
        raise MicroseqEvaluationError("branch checkpoint streams differ from merge lock")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with SanitizedJsonlWriter(destination, run_id, schema_version=STREAM_SCHEMA) as writer:
        for item in (*native, *ode):
            writer.write(CHECKPOINT_EVENT, item)
        writer.sync()
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ode-edit-microseq-evaluator", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", allow_abbrev=False)
    run.add_argument("--easyedit-root", required=True)
    run.add_argument("--branch", required=True, choices=BRANCHES)
    run.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    run.add_argument("--run-id", required=True)
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    merge = subparsers.add_parser("merge", allow_abbrev=False)
    merge.add_argument("--native-checkpoints", required=True)
    merge.add_argument("--ode-checkpoints", required=True)
    merge.add_argument("--output-checkpoints", required=True)
    merge.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    merge.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            summary = run_microseq_evaluator(
                easyedit_root=args.easyedit_root,
                branch=args.branch,
                model_alias=args.model,
                run_id=args.run_id,
                output_root=args.output_root,
            )
            print(json.dumps(summary, sort_keys=True))
            return 0 if summary["all_pass"] else 1
        destination = merge_checkpoint_streams(
            native_path=args.native_checkpoints,
            ode_path=args.ode_checkpoints,
            output_path=args.output_checkpoints,
            model_alias=args.model,
            run_id=args.run_id,
        )
        print(json.dumps({"status": "completed", "output": str(destination)}, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "aborted",
                    "error_type": type(exc).__name__,
                    "raw_exception_persisted": False,
                },
                sort_keys=True,
            )
        )
        print(
            f"MICROSEQ_EVALUATOR_TRACEBACK_BEGIN type={type(exc).__name__}",
            file=sys.stderr,
        )
        for frame, line_number in traceback.walk_tb(exc.__traceback__):
            print(
                f'  File "{frame.f_code.co_filename}", line {line_number}, in {frame.f_code.co_name}',
                file=sys.stderr,
            )
        print("MICROSEQ_EVALUATOR_TRACEBACK_END", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
