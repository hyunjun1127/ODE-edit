#!/usr/bin/env python3
"""Run R2 absolute-LL replay followed by V2 sequential P1 development arms."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

from project.run_scripts.ode_edit_motivation.contracts import EditRequest
from project.run_scripts.ode_edit_motivation.gpu_runtime import (
    load_fixed_model_checkpoint_original,
    offline_environment,
    seed_runtime,
)
from project.run_scripts.ode_edit_method.continuation import (
    DEFAULT_R2_SOURCE_SPECS,
    PinnedDirectZEasyEditBackend,
    validate_r2_sources,
)
from project.run_scripts.ode_edit_method.contracts import Arm, canonical_hash
from project.run_scripts.ode_edit_method.controller import OmegaLedger
from project.run_scripts.ode_edit_method.easyedit_backend import (
    OracleMeanEasyEditBackend,
)
from project.run_scripts.ode_edit_method.event_strength import (
    assert_raw_free,
    build_event_strength_trace,
)
from project.run_scripts.ode_edit_method.events import EVENT_MODEL_FORWARD_CALLS
from project.run_scripts.ode_edit_method.hooks import (
    TorchCheckpoint,
    base_weight_c_energy,
)
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.legacy_rca import (
    LEGACY_RCA_ARMS,
    legacy_r2_trajectories,
)
from project.run_scripts.ode_edit_method.lock import (
    LOCK_PATH,
    controller_config,
    load_lock,
)
from project.run_scripts.ode_edit_method.mechanism import summarize_mechanism
from project.run_scripts.ode_edit_method.oracle_event import ORACLE_MEAN_EVENT_MODE
from project.run_scripts.ode_edit_method.oracle_lock import (
    ORACLE_LOCK_PATH,
    load_oracle_lock,
)
from project.run_scripts.ode_edit_method.p1_orchestration import P1SequentialArmGuard
from project.run_scripts.ode_edit_method.preflight import (
    assert_static_lock_identities,
    preflight_static_inputs,
    prepare_concrete_environment,
)
from project.run_scripts.ode_edit_method.runtime import FiveArmRunner
from project.run_scripts.session02_compute_aware_p0 import (
    EASYEDIT_ROOT,
    _file_sha256,
    _git_head,
    _json_write,
    _jsonl_append,
    _receipt_payload,
    _require_output_root,
)
from project.run_scripts.session02_compute_aware_p1 import (
    _checkpoint_original_runtime_metadata,
    _terminal_geometry,
)


P1_ARMS = (
    Arm.NATIVE_MEMIT,
    Arm.STATIC_SYNCHRONOUS,
    Arm.FULL_ODE_EDIT,
)
EXECUTION_TOKEN = "oracle-mean-event-v2-p1-after-p0-pass"
P1_OUTPUT_PREFIX = "session02-oracle-mean-event-p1-v1"
LEGACY_OUTPUT_PREFIX = "session02-oracle-mean-event-legacy-rca-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--model-alias",
        required=True,
        choices=("llama3-8b-inst", "qwen2.5-7b-inst"),
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--oracle-lock", type=Path, default=ORACLE_LOCK_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--easyedit-root", type=Path, default=EASYEDIT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def expected_output_root(model_alias: str, proposal_id: str) -> Path:
    return Path(
        "local/results/"
        f"{P1_OUTPUT_PREFIX}-{model_alias}-{proposal_id[:8]}"
    )


def expected_legacy_root(model_alias: str, proposal_id: str) -> Path:
    return Path(
        "local/results/"
        f"{LEGACY_OUTPUT_PREFIX}-{model_alias}-{proposal_id[:8]}"
    )


def _exact_output(repo: Path, raw: Path, model_alias: str, proposal_id: str) -> Path:
    expected = (repo / expected_output_root(model_alias, proposal_id)).resolve()
    candidate = raw if raw.is_absolute() else repo / raw
    if candidate.resolve() != expected:
        raise RuntimeError("oracle P1 output root differs from its dry plan")
    return candidate


def dry_plan(oracle_lock: Mapping[str, Any]) -> dict[str, Any]:
    proposal_id = str(oracle_lock["proposal_id"])
    resources = oracle_lock["resources"]
    return {
        "schema_version": "ode-edit-oracle-mean-p1-dry-plan/v1",
        "submission_authorized": False,
        "automatic_authority_condition": "both-p0-terminal-technical-pass",
        "proposal_id": proposal_id,
        "lock_sha256": oracle_lock["lock_sha256"],
        "jobs": [
            {
                "model_alias": alias,
                "output_root": str(expected_output_root(alias, proposal_id)),
                "legacy_rca_root": str(expected_legacy_root(alias, proposal_id)),
                "resources": {
                    "gpu": resources["gpu_per_job"],
                    "cpu": resources["cpu_per_job"],
                    "host_memory_mib": resources["host_memory_mib_per_job"],
                    "time": resources["p1_time"],
                },
            }
            for alias in oracle_lock["runtime"]["models"]
        ],
        "pair_gpu": resources["pair_gpu"],
        "server1_project_gpu_cap": resources["server1_project_gpu_cap"],
        "scientific_outcome_count": 0,
    }


def _request_ids(requests: Sequence[Any]) -> list[str]:
    return [
        EditRequest.from_mapping(
            {
                "case_id": request.case_id,
                "prompt": request.prompt,
                "subject": request.subject,
                "target_new": request.target_new,
            }
        ).request_id
        for request in requests
    ]


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _require_legacy_identity(
    *,
    result: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    pre_request_state: str,
    post_request_state: str,
    pre_target_state: str,
    post_target_state: str,
    ledger: OmegaLedger,
    trajectory: Any,
    source_direct_z_model_forwards: int,
) -> None:
    expected = trajectory.controller_record
    if result != expected.get("result"):
        raise RuntimeError("legacy RCA R2 action/step/status identity differs")
    if (
        pre_request_state != expected.get("pre_edit_state_id")
        or post_request_state != expected.get("post_edit_state_id")
        or pre_target_state != expected.get("pre_edit_target_weight_state_id")
        or post_target_state != expected.get("post_edit_target_weight_state_id")
    ):
        raise RuntimeError("legacy RCA R2 state identity differs")
    source_counters = dict(trajectory.compute_record.get("counters", {}))
    expected_replay_counters = dict(source_counters)
    expected_replay_counters["N_model_fwd"] = (
        int(source_counters["N_model_fwd"]) - source_direct_z_model_forwards
    )
    if snapshot.get("counters") != expected_replay_counters:
        raise RuntimeError("legacy RCA R2 reuse-adjusted counter identity differs")
    expected_omega = expected.get("Omega", {})
    if _json_normalize(ledger.state()) != expected_omega.get("after"):
        raise RuntimeError("legacy RCA R2 Omega state differs")
    if _json_normalize(_receipt_payload(ledger)) != expected_omega.get(
        "receipts_through_current_edit"
    ):
        raise RuntimeError("legacy RCA R2 receipt history differs")


def _run_legacy_rca(
    *,
    repo: Path,
    output_root: Path,
    model_alias: str,
    requests: Sequence[Any],
    runtime: Any,
    prepared: Any,
    fixed_artifacts: Any,
    config: Any,
    base_lock: Mapping[str, Any],
    denominators: Mapping[int, float],
    baseline: TorchCheckpoint,
    oracle_lock: Mapping[str, Any],
    setup_share: float,
) -> dict[str, Any]:
    selection = validate_r2_sources(repo, DEFAULT_R2_SOURCE_SPECS)
    trajectories = legacy_r2_trajectories(
        repo, model_alias, selection=selection
    )
    direct_z_forwards_by_case: dict[str, int] = {}
    for trajectory in trajectories:
        if trajectory.arm is not Arm.FULL_ODE_EDIT:
            continue
        counters = trajectory.compute_record["counters"]
        uncategorized = int(counters["N_model_fwd"]) - sum(
            int(counters[name])
            for name in (
                "N_event_fwd",
                "N_field_state_fwd",
                "N_reference_gate_fwd",
            )
        )
        if uncategorized <= 0:
            raise RuntimeError("R2 direct-z forward count is not identifiable")
        direct_z_forwards_by_case[trajectory.case_id] = uncategorized
    if set(direct_z_forwards_by_case) != {request.case_id for request in requests}:
        raise RuntimeError("R2 direct-z forward accounting panel is incomplete")
    root = _require_output_root(repo, output_root)
    for name in ("legacy_rca.jsonl", "compute.jsonl", "evaluation.jsonl"):
        (root / name).touch(exist_ok=False)
    manifest = {
        "schema_version": "ode-edit-oracle-mean-legacy-rca-manifest/v1",
        "status": "RUNNING_READ_ONLY_R2_REPLAY",
        "model": model_alias,
        "proposal_id": oracle_lock["proposal_id"],
        "source_proposal_id": oracle_lock["base_proposal_id"],
        "source_terminal_manifest_sha256": selection.source_for_alias(
            model_alias
        ).spec.terminal_manifest_sha256,
        "arms": [arm.value for arm in LEGACY_RCA_ARMS],
        "direct_z_recomputed": False,
        "excluded_from_v2_compute": True,
        "artifact_firewall": {
            "raw_prompt_target_context_persisted": False,
            "evaluation_executed": False,
            "generation_executed": False,
        },
        "scientific_outcome_count": 0,
    }
    assert_raw_free(manifest)
    _json_write(root / "manifest.json", manifest)
    request_by_case = {request.case_id: request for request in requests}
    records: list[dict[str, Any]] = []
    try:
        for arm in LEGACY_RCA_ARMS:
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)
            ledger = OmegaLedger(denominators)
            arm_rows = [item for item in trajectories if item.arm is arm]
            for trajectory in arm_rows:
                request = request_by_case[trajectory.case_id]
                metrics = EditInstrumentation(
                    f"legacy-rca:{model_alias}:{arm.value}:{trajectory.case_id}",
                    gpu_timing=True,
                    gpu_device="cuda:0",
                )
                metrics.add_wall_seconds("context_setup", setup_share)
                backend = PinnedDirectZEasyEditBackend(
                    runtime=runtime,
                    bridge=prepared.bridge,
                    hparams=prepared.hparams,
                    contexts=prepared.contexts,
                    request=request,
                    covariance_specs=prepared.covariance_specs,
                    covariance_contract=prepared.covariance_contract,
                    covariance_by_layer=prepared.covariance_by_layer,
                    direct_z_cache_root=trajectory.direct_z_path.parent,
                    direct_z_cache_path=trajectory.direct_z_path,
                    expected_direct_z_identity=trajectory.direct_z_identity,
                    expected_direct_z_tensor_sha256=trajectory.direct_z_tensor_sha256,
                    tau=config.tau,
                    unit_c_norm_epsilon=base_lock["controller"]["unit_c_norm_epsilon"],
                    unit_c_identity_atol=base_lock["controller"]["unit_c_identity_atol"],
                    unit_c_identity_rtol=base_lock["controller"]["unit_c_identity_rtol"],
                    instrumentation=metrics,
                    record_mechanism=arm is Arm.FULL_ODE_EDIT,
                )
                pre_request = backend.current_state_id()
                pre_target = backend.sequential_target_weight_state_id()
                omega_before = ledger.state()
                if _json_normalize(omega_before) != trajectory.controller_record[
                    "Omega"
                ]["before"]:
                    raise RuntimeError("legacy RCA R2 pre-edit Omega differs")
                metrics.attach_model(runtime.model)
                try:
                    result = FiveArmRunner(config, ledger).run(
                        arm,
                        edit_id=trajectory.controller_record["result"]["edit_id"],
                        request=request,
                        backend=backend,
                        instrumentation=metrics,
                    )
                finally:
                    metrics.detach_model()
                snapshot = metrics.finalize().to_dict()
                post_request = backend.current_state_id()
                post_target = backend.sequential_target_weight_state_id()
                result_payload = result.to_dict()
                _require_legacy_identity(
                    result=result_payload,
                    snapshot=snapshot,
                    pre_request_state=pre_request,
                    post_request_state=post_request,
                    pre_target_state=pre_target,
                    post_target_state=post_target,
                    ledger=ledger,
                    trajectory=trajectory,
                    source_direct_z_model_forwards=direct_z_forwards_by_case[
                        trajectory.case_id
                    ],
                )
                trace = build_event_strength_trace(
                    arm,
                    result_payload,
                    backend.event_history,
                    tau=config.tau,
                    denominator_epsilon=config.trust_denominator_epsilon,
                    event_tolerance=config.event_tolerance,
                )
                row = {
                    "schema_version": "ode-edit-oracle-mean-legacy-rca/v1",
                    "model": model_alias,
                    "arm": arm.value,
                    "case_id": trajectory.case_id,
                    "order_position": trajectory.order_position,
                    "source_identity": trajectory.raw_free_identity(),
                    "source_status": trajectory.controller_record["status"],
                    "result_identity_exact": True,
                    "logical_counter_identity_exact": True,
                    "raw_N_model_fwd_identity_expected": False,
                    "raw_N_model_fwd_difference_reason": (
                        "pinned-direct-z-reuse-removes-source-calibration-forwards"
                    ),
                    "source_direct_z_model_forwards_excluded": (
                        direct_z_forwards_by_case[trajectory.case_id]
                    ),
                    "direct_z_reuse_adjusted_counter_identity_exact": True,
                    "state_identity_exact": True,
                    "Omega_identity_exact": True,
                    "direct_z_recomputed": False,
                    "trace": trace,
                }
                compute_row = {
                    "schema_version": "ode-edit-oracle-mean-legacy-rca-compute/v1",
                    "model": model_alias,
                    "arm": arm.value,
                    "case_id": trajectory.case_id,
                    "order_position": trajectory.order_position,
                    **snapshot,
                    **snapshot["counters"],
                    "excluded_from_v2_compute": True,
                }
                assert_raw_free(row)
                assert_raw_free(compute_row)
                _jsonl_append(root / "legacy_rca.jsonl", row)
                _jsonl_append(root / "compute.jsonl", compute_row)
                records.append(row)
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)
    except BaseException:
        baseline.restore(runtime.model)
        baseline.assert_exact(runtime.model, include_rng=True)
        raise
    if len(records) != 8:
        raise RuntimeError("legacy RCA did not reproduce eight selected trajectories")
    selection = validate_r2_sources(repo, DEFAULT_R2_SOURCE_SPECS)
    summary = {
        "schema_version": "ode-edit-oracle-mean-legacy-rca-summary/v1",
        "status": "COMPLETE_EXACT_R2_REPLAY_WITH_ABSOLUTE_LL",
        "record_count": len(records),
        "all_action_status_state_omega_logical_counter_identity_exact": True,
        "raw_N_model_fwd_identity_expected": False,
        "raw_N_model_fwd_difference_reason": (
            "pinned-direct-z-reuse-removes-source-calibration-forwards"
        ),
        "source_terminal_manifest_sha256": selection.source_for_alias(
            model_alias
        ).spec.terminal_manifest_sha256,
        "scientific_outcome_count": 0,
        "evaluation_count": 0,
    }
    _json_write(root / "summary.json", summary)
    manifest["status"] = "COMPLETE_EXACT_R2_REPLAY"
    _json_write(root / "manifest.json", manifest)
    files = [
        root / "manifest.json",
        root / "legacy_rca.jsonl",
        root / "compute.jsonl",
        root / "evaluation.jsonl",
        root / "summary.json",
    ]
    _json_write(
        root / "terminal_manifest.json",
        {
            "schema_version": "ode-edit-session02-terminal-manifest/v1",
            "status": "COMPLETE",
            "files": {
                str(path.relative_to(root)): {
                    "sha256": _file_sha256(path),
                    "size": path.stat().st_size,
                }
                for path in files
            },
        },
    )
    return {
        "root": str(root.relative_to(repo)),
        "terminal_manifest_sha256": _file_sha256(root / "terminal_manifest.json"),
        "records": records,
    }


def run(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    easyedit_root = args.easyedit_root.resolve(strict=True)
    if easyedit_root != EASYEDIT_ROOT.resolve(strict=True):
        raise RuntimeError("oracle P1 EasyEdit root differs")
    base_lock_path = args.lock.resolve(strict=True)
    base_lock = load_lock(base_lock_path)
    base_proposal_id = base_lock.pop("proposal_id")
    oracle_lock = load_oracle_lock(args.oracle_lock)
    if (
        _file_sha256(base_lock_path) != oracle_lock["base_lock_sha256"]
        or base_proposal_id != oracle_lock["base_proposal_id"]
    ):
        raise RuntimeError("oracle P1 base lock identity differs")
    plan = dry_plan(oracle_lock)
    output_arg = _exact_output(
        repo, args.output_root, args.model_alias, oracle_lock["proposal_id"]
    )
    legacy_arg = repo / expected_legacy_root(
        args.model_alias, oracle_lock["proposal_id"]
    )
    if args.dry_run:
        print(json.dumps(plan, allow_nan=False, sort_keys=True))
        return 0
    if output_arg.exists() or legacy_arg.exists():
        raise RuntimeError("oracle P1 or legacy RCA output root already exists")
    if os.environ.get("ODEEDIT_ORACLE_MEAN_P1_SUBMISSION_AUTHORIZED") != EXECUTION_TOKEN:
        raise RuntimeError("conditional oracle P1 execution authority is required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("oracle P1 requires exactly one visible GPU")

    # Revalidate both complete R2 sources before creating either new root.
    validate_r2_sources(repo, DEFAULT_R2_SOURCE_SPECS)
    case_ids = tuple(oracle_lock["selection"]["p1_case_ids"])
    seed = int(oracle_lock["selection"]["seed"])
    seed_runtime(seed)
    setup_start = time.perf_counter()
    fixed_artifacts, requests = preflight_static_inputs(
        easyedit_root,
        model_alias=args.model_alias,
        case_ids=case_ids,
    )
    model_lock = base_lock["models"][args.model_alias]
    assert_static_lock_identities(
        easyedit_root,
        fixed_artifacts=fixed_artifacts,
        model_lock=model_lock,
        selection_lock=base_lock["selection"],
    )
    if _request_ids(requests) != base_lock["selection"]["request_ids"]:
        raise RuntimeError("oracle P1 request identity differs")
    model_load_start = time.perf_counter()
    with offline_environment():
        runtime = load_fixed_model_checkpoint_original(args.model_alias)
    model_load_seconds = time.perf_counter() - model_load_start
    runtime_metadata = _checkpoint_original_runtime_metadata(runtime)
    prepared = prepare_concrete_environment(
        easyedit_root,
        runtime=runtime,
        fixed_artifacts=fixed_artifacts,
        controller_requests=requests,
        expected_model_lock=model_lock,
        seed=seed,
    )
    config = controller_config(base_lock)
    if config.event_tolerance != oracle_lock["event"]["oracle_validity_epsilon"]:
        raise RuntimeError("oracle P1 validity epsilon differs")
    weights = {
        layer: f"{prepared.hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in prepared.runtime.spec.layers
    }
    denominators = base_weight_c_energy(
        runtime.model,
        prepared.covariance_by_layer,
        weights,
        denominator_epsilon=config.load_denominator_epsilon,
    )
    baseline = TorchCheckpoint.capture(
        runtime.model, tuple(weights.values()), backup_device="cpu"
    )
    setup_seconds = time.perf_counter() - setup_start - model_load_seconds
    if setup_seconds < 0.0:
        raise RuntimeError("oracle P1 setup timer became negative")
    setup_share = setup_seconds / (len(P1_ARMS) * len(requests) + 8)

    legacy = _run_legacy_rca(
        repo=repo,
        output_root=legacy_arg,
        model_alias=args.model_alias,
        requests=requests,
        runtime=runtime,
        prepared=prepared,
        fixed_artifacts=fixed_artifacts,
        config=config,
        base_lock=base_lock,
        denominators=denominators,
        baseline=baseline,
        oracle_lock=oracle_lock,
        setup_share=setup_share,
    )
    baseline.restore(runtime.model)
    baseline.assert_exact(runtime.model, include_rng=True)

    output_root = _require_output_root(repo, output_arg)
    for name in (
        "controller_steps.jsonl",
        "compute.jsonl",
        "mechanism.jsonl",
        "event_strength.jsonl",
        "evaluation.jsonl",
    ):
        (output_root / name).touch(exist_ok=False)
    git_head = _git_head(repo)
    order_sha256 = canonical_hash(list(case_ids))
    manifest = {
        "schema_version": "ode-edit-oracle-mean-p1-manifest/v1",
        "status": "RUNNING_SEQUENTIAL_DEVELOPMENT_NO_EVALUATION",
        "instruction_id": oracle_lock["instruction_id"],
        "git": {
            "commit": git_head,
            "proposal_id": oracle_lock["proposal_id"],
            "lock_sha256": oracle_lock["lock_sha256"],
            "base_proposal_id": base_proposal_id,
        },
        "model": runtime_metadata,
        "selection": {
            "case_ids": list(case_ids),
            "order_sha256": order_sha256,
            "arms": [arm.value for arm in P1_ARMS],
            "seed": seed,
            "execution_axis": "arm-outer-sequential-edit-inner",
        },
        "context_manifest_id": prepared.contexts.manifest_id,
        "fixed_artifact_manifest_id": fixed_artifacts.manifest_id,
        "easyedit_source_manifest_id": prepared.bridge.load().provenance.manifest_id,
        "policy": {
            "controller": config.to_dict(),
            "event": dict(oracle_lock["event"]),
            "event_backend": ORACLE_MEAN_EVENT_MODE,
            "trial_backend": base_lock["trial_backend"]["selected_common_backend"],
            "model_specific_policy": False,
        },
        "legacy_rca": {
            "root": legacy["root"],
            "terminal_manifest_sha256": legacy["terminal_manifest_sha256"],
            "excluded_from_v2_compute": True,
            "result_identity_exact": True,
        },
        "artifact_firewall": {
            "raw_prompt_target_context_persisted": False,
            "evaluation_executed": False,
            "generation_executed": False,
        },
        "setup_wall_seconds": setup_seconds,
        "model_load_wall_seconds_excluded": model_load_seconds,
        "expected_record_count": len(P1_ARMS) * len(requests),
        "scientific_outcome_count": 0,
    }
    assert_raw_free(manifest)
    _json_write(output_root / "manifest.json", manifest)

    compute_by_position: dict[int, dict[str, Mapping[str, Any]]] = {}
    stream_summaries: list[dict[str, Any]] = []
    compact_records: list[dict[str, Any]] = []
    records = 0
    try:
        for arm in P1_ARMS:
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)
            stream_guard = P1SequentialArmGuard(arm, case_ids)
            stream_guard.note_arm_start_restore()
            ledger = OmegaLedger(denominators)
            pending: list[dict[str, Any]] = []
            for order_position, request in enumerate(requests):
                omega_before = ledger.state()
                receipt_before = len(ledger.receipts)
                metrics = EditInstrumentation(
                    f"oracle-p1:{arm.value}:{order_position}:{request.case_id}",
                    gpu_timing=True,
                    gpu_device="cuda:0",
                )
                metrics.add_wall_seconds("context_setup", setup_share)
                cache_root = output_root / "direct_z"
                cache_path = cache_root / (
                    f"{args.model_alias}-{arm.value}-position-{order_position}-"
                    f"case-{request.case_id}.pt"
                )
                with metrics.component("context_setup"):
                    backend = OracleMeanEasyEditBackend(
                        runtime=runtime,
                        bridge=prepared.bridge,
                        hparams=prepared.hparams,
                        contexts=prepared.contexts,
                        request=request,
                        covariance_specs=prepared.covariance_specs,
                        covariance_contract=prepared.covariance_contract,
                        covariance_by_layer=prepared.covariance_by_layer,
                        direct_z_cache_root=cache_root,
                        direct_z_cache_path=cache_path,
                        tau=config.tau,
                        oracle_epsilon=config.event_tolerance,
                        unit_c_norm_epsilon=base_lock["controller"]["unit_c_norm_epsilon"],
                        unit_c_identity_atol=base_lock["controller"]["unit_c_identity_atol"],
                        unit_c_identity_rtol=base_lock["controller"]["unit_c_identity_rtol"],
                        instrumentation=metrics,
                        record_mechanism=arm is Arm.FULL_ODE_EDIT,
                    )
                pre_request = backend.current_state_id()
                pre_target = backend.sequential_target_weight_state_id()
                stream_guard.begin_edit(
                    order_position=order_position,
                    case_id=request.case_id,
                    pre_edit_state_id=pre_request,
                    pre_edit_target_weight_state_id=pre_target,
                    omega_before=omega_before,
                    history_length_before=receipt_before,
                    cache_identity=str(cache_path.relative_to(output_root)),
                )
                metrics.attach_model(runtime.model)
                try:
                    result = FiveArmRunner(config, ledger).run(
                        arm,
                        edit_id=f"oracle-p1:{arm.value}:{order_position}:{request.case_id}",
                        request=request,
                        backend=backend,
                        instrumentation=metrics,
                    )
                finally:
                    metrics.detach_model()
                post_request = backend.current_state_id()
                post_target = backend.sequential_target_weight_state_id()
                omega_after = ledger.state()
                receipt_after = len(ledger.receipts)
                failure = not result.omega_appended
                failure_request_exact = (
                    post_request == pre_request if failure else None
                )
                failure_target_exact = post_target == pre_target if failure else None
                failure_exact = (
                    failure_request_exact is True and failure_target_exact is True
                    if failure
                    else None
                )
                success_exact = (
                    post_request == result.terminal_state_id
                    if result.omega_appended
                    else None
                )
                if failure_exact is False or success_exact is False:
                    raise RuntimeError("oracle P1 edit transaction identity differs")
                stream_guard.finish_edit(
                    result=result,
                    post_edit_state_id=post_request,
                    post_edit_target_weight_state_id=post_target,
                    omega_after=omega_after,
                    history_length_after=receipt_after,
                )
                snapshot = metrics.finalize().to_dict()
                counters = snapshot["counters"]
                calibration = backend.oracle_calibration
                target = backend.oracle_target
                if calibration is None or target is None:
                    raise RuntimeError("oracle P1 calibration is absent")
                if (
                    result.direct_z_compute_count != 1
                    or counters["N_z"] != 1
                    or calibration.oracle_forward_count != 2
                    or calibration.entry_forward_count != 2
                    or counters["N_event_fwd"] % 2
                    or counters["N_eval"] != 0
                    or (
                        arm is not Arm.NATIVE_MEMIT
                        and counters["N_bw"] != counters["N_field"]
                    )
                ):
                    raise RuntimeError("oracle P1 counter contract differs")
                trace = build_event_strength_trace(
                    arm,
                    result.to_dict(),
                    backend.event_history,
                    tau=config.tau,
                    denominator_epsilon=config.trust_denominator_epsilon,
                    event_tolerance=config.event_tolerance,
                    oracle_target=target,
                )
                terminal = trace["anchors"]["terminal"]
                if result.status == "event_hit" and (
                    terminal["q_margin"] + config.event_tolerance < 0.5
                    or terminal["q_new"] + config.event_tolerance < 0.5
                ):
                    raise RuntimeError("oracle P1 hit violates realization floor")
                common = {
                    "model": args.model_alias,
                    "arm": arm.value,
                    "case_id": request.case_id,
                    "order_position": order_position,
                    "order_sha256": order_sha256,
                    "seed": seed,
                    "status": result.status,
                    "pre_edit_state_id": pre_request,
                    "post_edit_state_id": post_request,
                    "pre_edit_target_weight_state_id": pre_target,
                    "post_edit_target_weight_state_id": post_target,
                    "hashes": {
                        "proposal_id": oracle_lock["proposal_id"],
                        "context_manifest_id": prepared.contexts.manifest_id,
                        "direct_z": backend.direct_z_identity,
                    },
                }
                pending.append(
                    {
                        "common": common,
                        "result": result,
                        "snapshot": snapshot,
                        "counters": counters,
                        "event_history": backend.event_history,
                        "field_history": backend.mechanism_field_history,
                        "layers": backend.layers,
                        "target": target,
                        "calibration": calibration,
                        "trace": trace,
                        "omega_before": omega_before,
                        "omega_after": omega_after,
                        "receipt_before": receipt_before,
                        "receipt_after": receipt_after,
                        "receipts": _receipt_payload(ledger),
                        "terminal_geometry": _terminal_geometry(
                            ledger, receipt_before
                        ),
                        "failure": failure,
                        "failure_exact": failure_exact,
                        "failure_request_exact": failure_request_exact,
                        "failure_target_exact": failure_target_exact,
                        "success_exact": success_exact,
                        "native_raw_distance": backend.native_distance,
                        "d_sync_entry": (
                            backend.entry_synchronous_distance()
                            if counters["N_field"]
                            else None
                        ),
                    }
                )
                del backend

            stream_terminal_request = pending[-1]["common"]["post_edit_state_id"]
            stream_terminal_target = pending[-1]["common"][
                "post_edit_target_weight_state_id"
            ]
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)
            stream_guard.note_arm_end_restore(exact=True)
            stream_summaries.append(
                {
                    **stream_guard.summary(),
                    "order_sha256": order_sha256,
                    "completed_outer_edit_count": len(ledger.receipts),
                    "Omega_terminal": ledger.state(),
                    "stream_terminal_state_id_before_arm_isolation": stream_terminal_request,
                    "stream_terminal_target_weight_state_id_before_arm_isolation": stream_terminal_target,
                    "arm_isolation_restore_exact": True,
                }
            )

            for bundle in pending:
                common = bundle["common"]
                result = bundle["result"]
                snapshot = bundle["snapshot"]
                counters = bundle["counters"]
                sequential = {
                    "order_position": common["order_position"],
                    "pre_edit_state_id": common["pre_edit_state_id"],
                    "post_edit_state_id": common["post_edit_state_id"],
                    "pre_edit_target_weight_state_id": common[
                        "pre_edit_target_weight_state_id"
                    ],
                    "post_edit_target_weight_state_id": common[
                        "post_edit_target_weight_state_id"
                    ],
                    "Omega_before": bundle["omega_before"],
                    "Omega_after": bundle["omega_after"],
                    "history_length_before": bundle["receipt_before"],
                    "history_length_after": bundle["receipt_after"],
                    "current_edit_terminal_appended": result.omega_appended,
                }
                mechanism = summarize_mechanism(
                    arm,
                    result,
                    bundle["field_history"],
                    layers=bundle["layers"],
                    support_tolerance=config.slope_epsilon,
                    controller_gpu_seconds=float(snapshot["controller_gpu_seconds"]),
                    event_history=tuple(
                        row
                        for row in bundle["event_history"]
                        if row.get("source") == "event"
                    ),
                    progress_epsilon=config.progress_epsilon,
                    failure_rollback_required=bundle["failure"],
                    failure_rollback_exact=bundle["failure_exact"],
                    successful_terminal_endpoint_exact=bundle["success_exact"],
                    arm_isolation_restore_exact=True,
                )
                controller_row = {
                    "schema_version": "ode-edit-oracle-mean-p1-controller/v1",
                    **common,
                    "result": result.to_dict(),
                    "event_history": list(bundle["event_history"]),
                    "oracle_calibration": bundle["calibration"].to_dict(),
                    "Omega": {
                        "before": bundle["omega_before"],
                        "after": bundle["omega_after"],
                        "receipts_through_current_edit": bundle["receipts"],
                    },
                    "sequential_state": sequential,
                    "integrity": {
                        "failure_rollback_exact": bundle["failure_exact"],
                        "failure_request_state_rollback_exact": bundle[
                            "failure_request_exact"
                        ],
                        "failure_target_weight_state_rollback_exact": bundle[
                            "failure_target_exact"
                        ],
                        "successful_terminal_endpoint_exact": bundle["success_exact"],
                        "arm_isolation_restore_exact": True,
                        "functional_commit_t_c": "PASS_OR_NOT_APPLICABLE",
                        "field_backward_identity": counters["N_bw"]
                        == counters["N_field"],
                    },
                }
                compute_row = {
                    "schema_version": "ode-edit-oracle-mean-p1-compute/v1",
                    **common,
                    **snapshot,
                    **counters,
                    "oracle_forward_count": bundle[
                        "calibration"
                    ].oracle_forward_count,
                    "terminal_geometry": bundle["terminal_geometry"],
                    "D_native_raw_proposal": bundle["native_raw_distance"],
                    "D_sync_entry": bundle["d_sync_entry"],
                    "terminal_net_c_distance": (
                        math.sqrt(
                            sum(
                                bundle["terminal_geometry"][
                                    "terminal_net_energy"
                                ].values()
                            )
                        )
                        if bundle["terminal_geometry"] is not None
                        else None
                    ),
                    "sequential_state": sequential,
                }
                mechanism_row = {
                    "schema_version": "ode-edit-oracle-mean-p1-mechanism/v1",
                    **common,
                    "mechanism": mechanism,
                    "sequential_state": sequential,
                }
                strength_row = {
                    "schema_version": "ode-edit-oracle-mean-p1-strength/v1",
                    **common,
                    "oracle_target": bundle["target"].to_dict(),
                    "trace": bundle["trace"],
                }
                for row in (
                    controller_row,
                    compute_row,
                    mechanism_row,
                    strength_row,
                ):
                    assert_raw_free(row)
                _jsonl_append(output_root / "controller_steps.jsonl", controller_row)
                _jsonl_append(output_root / "compute.jsonl", compute_row)
                _jsonl_append(output_root / "mechanism.jsonl", mechanism_row)
                _jsonl_append(output_root / "event_strength.jsonl", strength_row)
                compute_by_position.setdefault(
                    int(common["order_position"]), {}
                )[arm.value] = compute_row
                terminal = bundle["trace"]["anchors"]["terminal"]
                compact_records.append(
                    {
                        "arm": arm.value,
                        "case_id": common["case_id"],
                        "order_position": common["order_position"],
                        "status": result.status,
                        "q_margin": terminal["q_margin"],
                        "q_new": terminal["q_new"],
                        "mean_margin": terminal["mean_margin"],
                        "mean_new_log_likelihood": terminal[
                            "mean_new_log_likelihood"
                        ],
                        "mean_old_log_likelihood": terminal[
                            "mean_old_log_likelihood"
                        ],
                        "controller_wall_seconds": snapshot[
                            "controller_wall_seconds"
                        ],
                        "controller_gpu_seconds": snapshot[
                            "controller_gpu_seconds"
                        ],
                        "peak_memory_allocated_bytes": snapshot[
                            "peak_memory_allocated_bytes"
                        ],
                    }
                )
                records += 1
    except BaseException:
        baseline.restore(runtime.model)
        baseline.assert_exact(runtime.model, include_rng=True)
        raise

    fixed_artifacts.assert_current()
    prepared.bridge.load().provenance.assert_current()
    ratios = []
    for position, rows in sorted(compute_by_position.items()):
        native = rows[Arm.NATIVE_MEMIT.value]
        for arm in (Arm.STATIC_SYNCHRONOUS, Arm.FULL_ODE_EDIT):
            current = rows[arm.value]
            ratios.append(
                {
                    "case_id": case_ids[position],
                    "order_position": position,
                    "arm": arm.value,
                    "wall_over_native": (
                        current["controller_wall_seconds"]
                        / native["controller_wall_seconds"]
                        if native["controller_wall_seconds"] > 0.0
                        else None
                    ),
                    "gpu_over_native": (
                        current["controller_gpu_seconds"]
                        / native["controller_gpu_seconds"]
                        if native["controller_gpu_seconds"] > 0.0
                        else None
                    ),
                    "divergent_sequential_arm_histories": True,
                }
            )
    legacy_table = [
        {
            "arm": row["arm"],
            "case_id": row["case_id"],
            "order_position": row["order_position"],
            "status": row["source_status"],
            "terminal": row["trace"]["anchors"]["terminal"],
            "last_trial": row["trace"]["anchors"]["last_trial"],
            "mean_positive_min_negative_rollback": row["trace"]["diagnostics"][
                "failed_last_trial_mean_positive_worst_context_negative"
            ],
        }
        for row in legacy["records"]
    ]
    summary = {
        "schema_version": "ode-edit-oracle-mean-p1-summary/v1",
        "status": "COMPLETE_DEVELOPMENT_NO_EVALUATION",
        "model": args.model_alias,
        "record_count": records,
        "sequential_streams": stream_summaries,
        "v2_records": compact_records,
        "legacy_under_realization_under_commit_table": legacy_table,
        "v2_compute_ratios": ratios,
        "legacy_rca_terminal_manifest_sha256": legacy[
            "terminal_manifest_sha256"
        ],
        "evaluation_count": 0,
        "scientific_outcome_count": 0,
        "scientific_superiority_claim": False,
    }
    assert_raw_free(summary)
    _json_write(output_root / "summary.json", summary)
    manifest["status"] = "COMPLETE_DEVELOPMENT_NO_EVALUATION"
    _json_write(output_root / "manifest.json", manifest)
    files = [
        output_root / "manifest.json",
        output_root / "controller_steps.jsonl",
        output_root / "compute.jsonl",
        output_root / "mechanism.jsonl",
        output_root / "event_strength.jsonl",
        output_root / "evaluation.jsonl",
        output_root / "summary.json",
        *sorted((output_root / "direct_z").glob("*.pt")),
    ]
    _json_write(
        output_root / "terminal_manifest.json",
        {
            "schema_version": "ode-edit-session02-terminal-manifest/v1",
            "status": "COMPLETE",
            "files": {
                str(path.relative_to(output_root)): {
                    "sha256": _file_sha256(path),
                    "size": path.stat().st_size,
                }
                for path in files
            },
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
