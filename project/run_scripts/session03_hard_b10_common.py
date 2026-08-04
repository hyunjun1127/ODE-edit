"""Executable genuine-B10 K10/K20/correction-cycle attribution runner."""

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
from project.run_scripts.ode_edit_method.contracts import MethodContractError
from project.run_scripts.ode_edit_method.ct_k4_evaluation import (
    load_evaluation_payloads,
    make_firewall,
)
from project.run_scripts.ode_edit_method.event_strength import assert_raw_free
from project.run_scripts.ode_edit_method.hard_batch import (
    HARD_BATCH_ARM_ORDER,
    CounterFactEndpointMetrics,
    HardBatchArm,
    endpoint_parameter_hashes,
    efficacy_specs,
    native_floor_verdict,
    run_hard_batch_arm,
    score_counterfact_endpoint,
    score_counterfact_nll_pairs,
)
from project.run_scripts.ode_edit_method.hard_batch_backend import (
    JOINT_BATCH_SIZE,
    JointHardBatchEasyEditBackend,
)
from project.run_scripts.ode_edit_method.hard_batch_lock import (
    BATCH_CASE_IDS,
    BATCH_REQUEST_IDS,
    HARD_BATCH_LOCK_PATH,
    load_hard_batch_lock,
)
from project.run_scripts.ode_edit_method.hooks import TorchCheckpoint, base_weight_c_energy
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.lock import LOCK_PATH, controller_config, load_lock
from project.run_scripts.ode_edit_method.oracle_absolute_lock import (
    ORACLE_ABSOLUTE_LOCK_PATH,
    load_oracle_absolute_lock,
)
from project.run_scripts.ode_edit_method.preflight import (
    assert_static_lock_identities,
    preflight_static_inputs,
    prepare_concrete_environment,
)
from project.run_scripts.session02_compute_aware_p0 import (
    EASYEDIT_ROOT,
    _file_sha256,
    _git_head,
    _json_write,
    _jsonl_append,
)
from project.run_scripts.session02_compute_aware_p1 import (
    _checkpoint_original_runtime_metadata,
)
from project.run_scripts.session03_ct_k4_common import (
    _PostActionForwardCounter,
    _require_session03_output_root,
)


AUTHORIZATION_TOKENS = {
    "p0": "session03-hard-b10-k20-p0-v1",
    "p1": "session03-hard-b10-k20-p1-v1",
}
OUTPUT_PREFIXES = {
    "p0": "session03-hard-b10-k20-p0",
    "p1": "session03-hard-b10-k20-p1",
}


def expected_output_root(stage: str, model_alias: str, proposal_id: str) -> Path:
    if stage not in OUTPUT_PREFIXES:
        raise MethodContractError("hard-B10 stage differs")
    return Path(
        "local/results/"
        f"{OUTPUT_PREFIXES[stage]}-{model_alias}-{proposal_id[:8]}"
    )


def dry_plan(lock: Mapping[str, Any], stage: str) -> dict[str, Any]:
    if stage not in {"p0", "p1"}:
        raise MethodContractError("hard-B10 dry stage differs")
    resources = lock["resources"]
    requested_time = (
        resources["p0_time"] if stage == "p0" else resources["p1_time_max"]
    )
    return {
        "schema_version": f"ode-edit-session03-hard-b10-{stage}-dry-plan/v1",
        "submission_authorized": False,
        "proposal_id": lock["proposal_id"],
        "lock_sha256": lock["lock_sha256"],
        "stage": stage,
        "batch_case_ids": list(BATCH_CASE_IDS),
        "edit_batch_size": JOINT_BATCH_SIZE,
        "arms": [arm.value for arm in HARD_BATCH_ARM_ORDER],
        "jobs": [
            {
                "model_alias": alias,
                "output_root": str(
                    expected_output_root(stage, alias, lock["proposal_id"])
                ),
                "resources": {
                    "gpu": resources["gpu_per_job"],
                    "cpu": resources["cpu_per_job"],
                    "host_memory_mib": resources["host_memory_mib_per_job"],
                    "time": requested_time,
                },
            }
            for alias in lock["models"]
        ],
        "pair_gpu": resources["pair_gpu"],
        "server1_project_gpu_cap": resources["server1_project_gpu_cap"],
        "p1_is_labeled_diagnostic": stage == "p1",
        "automatic_p1_condition": "both-p0-terminal-technical-pass-and-forecast<=24h",
    }


def build_parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--model-alias",
        required=True,
        choices=("llama3-8b-inst", "qwen2.5-7b-inst"),
    )
    parser.add_argument("--lock", type=Path, default=HARD_BATCH_LOCK_PATH)
    parser.add_argument("--base-lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--v3-lock", type=Path, default=ORACLE_ABSOLUTE_LOCK_PATH)
    parser.add_argument("--easyedit-root", type=Path, default=EASYEDIT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.set_defaults(stage=stage)
    return parser


def _request_ids(requests: Sequence[Any]) -> tuple[str, ...]:
    return tuple(
        EditRequest.from_mapping(
            {
                "case_id": request.case_id,
                "prompt": request.prompt,
                "subject": request.subject,
                "target_new": request.target_new,
            }
        ).request_id
        for request in requests
    )


def _validate_source_locks(
    hard_path: Path, base_path: Path, v3_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    hard = load_hard_batch_lock(hard_path)
    base = load_lock(base_path)
    v3 = load_oracle_absolute_lock(v3_path)
    pinned = hard["base_method"]
    directory = HARD_BATCH_LOCK_PATH.parent
    if (
        base["proposal_id"] != pinned["proposal_id"]
        or _file_sha256(base_path) != pinned["lock_sha256"]
        or v3["proposal_id"] != pinned["v3_event_proposal_id"]
        or _file_sha256(v3_path) != pinned["v3_event_lock_sha256"]
        or _file_sha256(directory / "ct_k4.py")
        != pinned["ct_k4_source_sha256"]
        or _file_sha256(directory / "ct_k10.py")
        != pinned["ct_k10_source_sha256"]
        or _file_sha256(directory / "ct_k4_evaluation.py")
        != pinned["corrected_token_evaluator_sha256"]
    ):
        raise MethodContractError("hard-B10 pinned method source differs")
    return hard, base


def _backend(
    *,
    runtime: Any,
    prepared: Any,
    requests: Sequence[Any],
    output_root: Path,
    config: Any,
    base_lock: Mapping[str, Any],
    metrics: EditInstrumentation,
    finite_reference_gate: bool,
) -> JointHardBatchEasyEditBackend:
    cache_root = output_root / "direct_z"
    cache_path = cache_root / f"{runtime.spec.alias}-joint-b10.pt"
    return JointHardBatchEasyEditBackend(
        runtime=runtime,
        bridge=prepared.bridge,
        hparams=prepared.hparams,
        contexts=prepared.contexts,
        requests=tuple(requests),
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
        finite_difference_gate_epsilon=(
            base_lock["derivative_backend"]["p0_finite_difference_epsilon"]
            if finite_reference_gate
            else None
        ),
        finite_difference_gate_atol=base_lock["derivative_backend"][
            "p0_scalar_gate_atol"
        ],
        finite_difference_gate_rtol=base_lock["derivative_backend"][
            "p0_scalar_gate_rtol"
        ],
        record_mechanism=True,
    )


def _benchmark_projection(value: Any) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("evaluator_wall_seconds")
    return payload


def _mechanism(result: Any) -> dict[str, Any]:
    steps = result.steps
    transitions = []
    allocation_cosines = []
    ranking_changes = []
    support_turnover = []
    for left, right in zip(steps, steps[1:]):
        transitions.append(left.direction_ids != right.direction_ids)
        numerator = math.fsum(
            a * b
            for a, b in zip(
                left.corrector_coefficients,
                right.corrector_coefficients,
                strict=True,
            )
        )
        left_norm = math.sqrt(
            math.fsum(value * value for value in left.corrector_coefficients)
        )
        right_norm = math.sqrt(
            math.fsum(value * value for value in right.corrector_coefficients)
        )
        allocation_cosines.append(
            None if left_norm == 0.0 or right_norm == 0.0 else numerator / left_norm / right_norm
        )
        left_rank = tuple(
            sorted(
                range(len(left.corrector_coefficients)),
                key=lambda index: (-left.corrector_coefficients[index], index),
            )
        )
        right_rank = tuple(
            sorted(
                range(len(right.corrector_coefficients)),
                key=lambda index: (-right.corrector_coefficients[index], index),
            )
        )
        ranking_changes.append(left_rank != right_rank)
        left_support = {
            index for index, value in enumerate(left.corrector_coefficients) if value > 0.0
        }
        right_support = {
            index for index, value in enumerate(right.corrector_coefficients) if value > 0.0
        }
        union = left_support | right_support
        support_turnover.append(
            0.0 if not union else len(left_support ^ right_support) / len(union)
        )
    return {
        "direction_transition_bits": [int(value) for value in transitions],
        "direction_transition_count": sum(transitions),
        "allocation_cosines": allocation_cosines,
        "ranking_change_bits": [int(value) for value in ranking_changes],
        "support_turnover": support_turnover,
        "exact_c_geometry_deferred": True,
        "retained_factor_tensors": False,
    }


def _assert_direct_z_identity(
    source_identity: Mapping[str, Any],
    source_receipts: Sequence[Mapping[str, Any]],
    identities: Mapping[HardBatchArm, Mapping[str, Any]],
    receipts: Mapping[HardBatchArm, Sequence[Mapping[str, Any]]],
    n_z: Mapping[HardBatchArm, int],
) -> None:
    expected_counts = {
        arm: JOINT_BATCH_SIZE if index == 0 else 0
        for index, arm in enumerate(HARD_BATCH_ARM_ORDER)
    }
    if dict(n_z) != expected_counts or sum(n_z.values()) != JOINT_BATCH_SIZE:
        raise MethodContractError("hard-B10 direct-z receipt accounting differs")
    if set(identities) != set(HARD_BATCH_ARM_ORDER):
        raise MethodContractError("hard-B10 direct-z arm set differs")
    locked_receipts = tuple(dict(value) for value in source_receipts)
    if len(locked_receipts) != JOINT_BATCH_SIZE:
        raise MethodContractError("hard-B10 direct-z receipt count differs")
    for arm in HARD_BATCH_ARM_ORDER:
        if dict(identities[arm]) != dict(source_identity):
            raise MethodContractError(f"hard-B10 direct-z identity differs for {arm.value}")
        if tuple(dict(value) for value in receipts[arm]) != locked_receipts:
            raise MethodContractError(f"hard-B10 direct-z receipts differ for {arm.value}")


def _total_energy(values: Mapping[int, float]) -> float:
    return math.fsum(float(value) for value in values.values())


def _write_terminal_manifest(output_root: Path, files: Sequence[Path]) -> None:
    _json_write(
        output_root / "terminal_manifest.json",
        {
            "schema_version": "ode-edit-session03-hard-b10-terminal/v1",
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


def run(args: argparse.Namespace) -> int:
    stage = str(args.stage)
    if stage not in {"p0", "p1"}:
        raise RuntimeError("hard-B10 stage differs")
    repo = Path(__file__).resolve().parents[2]
    easyedit_root = args.easyedit_root.resolve(strict=True)
    if easyedit_root != EASYEDIT_ROOT.resolve(strict=True):
        raise RuntimeError("hard-B10 EasyEdit root differs")
    hard, base_lock = _validate_source_locks(
        args.lock.resolve(strict=True),
        args.base_lock.resolve(strict=True),
        args.v3_lock.resolve(strict=True),
    )
    plan = dry_plan(hard, stage)
    expected_relative = expected_output_root(
        stage, args.model_alias, hard["proposal_id"]
    )
    candidate = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    expected = repo.resolve(strict=True) / expected_relative
    if Path(os.path.abspath(os.fspath(candidate))) != expected:
        raise RuntimeError("hard-B10 output root differs from dry plan")
    if args.dry_run:
        print(json.dumps(plan, allow_nan=False, sort_keys=True))
        return 0
    if os.environ.get(f"ODEEDIT_HARD_B10_{stage.upper()}_AUTHORIZED") != (
        AUTHORIZATION_TOKENS[stage]
    ):
        raise RuntimeError("separate hard-B10 execution authority is required")
    expected_head = os.environ.get("ODEEDIT_HARD_B10_EXPECTED_HEAD")
    if not expected_head or _git_head(repo) != expected_head:
        raise RuntimeError("hard-B10 source checkpoint differs")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("hard-B10 requires exactly one visible GPU")

    output_root = _require_session03_output_root(repo, candidate, expected_relative)
    output_files = tuple(
        output_root / name
        for name in (
            "controller_steps.jsonl",
            "compute.jsonl",
            "mechanism.jsonl",
            "request_events.jsonl",
            "endpoint_metrics.jsonl",
        )
    )
    for path in output_files:
        path.touch(exist_ok=False)
    seed_runtime(int(hard["policy"]["seed"]))
    setup_start = time.perf_counter()
    fixed_artifacts, requests = preflight_static_inputs(
        easyedit_root,
        model_alias=args.model_alias,
        case_ids=BATCH_CASE_IDS,
    )
    if tuple(request.case_id for request in requests) != BATCH_CASE_IDS:
        raise RuntimeError("hard-B10 case order differs")
    if _request_ids(requests) != BATCH_REQUEST_IDS:
        raise RuntimeError("hard-B10 request identities differ")
    model_lock = base_lock["models"][args.model_alias]
    assert_static_lock_identities(
        easyedit_root,
        fixed_artifacts=fixed_artifacts,
        model_lock=model_lock,
        selection_lock=hard["selection"],
    )
    model_load_start = time.perf_counter()
    with offline_environment():
        runtime = load_fixed_model_checkpoint_original(args.model_alias)
    model_load_seconds = time.perf_counter() - model_load_start
    runtime_metadata = _checkpoint_original_runtime_metadata(runtime)
    if runtime.gpu.total_memory != hard["memory_forecast"]["gpu_total_bytes"]:
        raise RuntimeError("hard-B10 GPU memory identity differs")
    prepared = prepare_concrete_environment(
        easyedit_root,
        runtime=runtime,
        fixed_artifacts=fixed_artifacts,
        controller_requests=requests,
        expected_model_lock=model_lock,
        seed=int(hard["policy"]["seed"]),
    )
    config = controller_config(base_lock)
    weight_by_layer = {
        layer: f"{prepared.hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in prepared.runtime.spec.layers
    }
    denominators = base_weight_c_energy(
        runtime.model,
        prepared.covariance_by_layer,
        weight_by_layer,
        denominator_epsilon=config.load_denominator_epsilon,
    )
    baseline = TorchCheckpoint.capture(
        runtime.model,
        tuple(weight_by_layer.values()),
        backup_device="cpu",
    )
    private_payloads = (
        load_evaluation_payloads(easyedit_root, requests) if stage == "p1" else {}
    )
    setup_seconds = time.perf_counter() - setup_start - model_load_seconds
    manifest = {
        "schema_version": f"ode-edit-session03-hard-b10-{stage}-manifest/v1",
        "status": "RUNNING",
        "instruction_id": hard["instruction_id"],
        "git": {
            "commit": _git_head(repo),
            "proposal_id": hard["proposal_id"],
            "lock_sha256": hard["lock_sha256"],
        },
        "model": runtime_metadata,
        "batch": {
            "case_ids": list(BATCH_CASE_IDS),
            "request_ids": list(BATCH_REQUEST_IDS),
            "edit_batch_size": JOINT_BATCH_SIZE,
            "single_joint_backend_transaction": True,
            "singleton_decomposition": False,
            "known_case_population_evidence": False,
        },
        "arms": [arm.value for arm in HARD_BATCH_ARM_ORDER],
        "benchmark_policy": {
            "name": hard["benchmark"]["name"],
            "evaluator_source_sha256": hard["benchmark"]["evaluator_source_sha256"],
            "aggregator_source_sha256": hard["benchmark"]["aggregator_source_sha256"],
            "native_floor_margin_counts": dict(
                hard["benchmark"]["native_floor_margin_counts"]
            ),
        },
        "endpoint_metric_stage": "enabled-after-action-freeze" if stage == "p1" else "disabled",
        "context_manifest_id": prepared.contexts.manifest_id,
        "fixed_artifact_manifest_id": fixed_artifacts.manifest_id,
        "easyedit_source_manifest_id": prepared.bridge.load().provenance.manifest_id,
        "setup_wall_seconds": setup_seconds,
        "model_load_wall_seconds_excluded": model_load_seconds,
        "server1_project_gpu_cap": 3,
    }
    assert_raw_free(manifest, "hard-B10 manifest")
    _json_write(output_root / "manifest.json", manifest)

    shared_target = None
    shared_identity: Mapping[str, Any] | None = None
    shared_receipts: tuple[Mapping[str, Any], ...] | None = None
    identities: dict[HardBatchArm, Mapping[str, Any]] = {}
    receipts: dict[HardBatchArm, Sequence[Mapping[str, Any]]] = {}
    n_z: dict[HardBatchArm, int] = {}
    rows: list[dict[str, Any]] = []
    endpoint_objects: dict[HardBatchArm, CounterFactEndpointMetrics] = {}
    try:
        for arm_index, arm in enumerate(HARD_BATCH_ARM_ORDER):
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)
            metrics = EditInstrumentation(
                f"session03:{stage}:{args.model_alias}:joint-b10:{arm.value}",
                gpu_timing=True,
                gpu_device="cuda:0",
            )
            metrics.add_wall_seconds(
                "context_setup", setup_seconds / len(HARD_BATCH_ARM_ORDER)
            )
            backend = _backend(
                runtime=runtime,
                prepared=prepared,
                requests=requests,
                output_root=output_root,
                config=config,
                base_lock=base_lock,
                metrics=metrics,
                finite_reference_gate=(
                    stage == "p0" and arm is HardBatchArm.REFRESH_K10_T1
                ),
            )
            firewalls = (
                {
                    request.case_id: make_firewall(
                        request, private_payloads[request.case_id]
                    )
                    for request in requests
                }
                if stage == "p1"
                else {}
            )
            metrics.attach_model(runtime.model)
            try:
                if shared_target is None:
                    with metrics.component("direct_z"):
                        shared_target = backend.compute_joint_direct_z(requests)
                    metrics.increment("N_z", JOINT_BATCH_SIZE)
                    shared_identity = dict(backend.direct_z_identity or {})
                    shared_receipts = tuple(backend.joint_direct_z_receipts)
                else:
                    backend.attach_shared_direct_z(shared_target)
                identity = dict(backend.direct_z_identity or {})
                arm_receipts = tuple(backend.joint_direct_z_receipts)
                identities[arm] = identity
                receipts[arm] = arm_receipts
                result = run_hard_batch_arm(
                    arm,
                    backend=backend,
                    frozen_target=shared_target,
                    denominators=denominators,
                    config=config,
                    instrumentation=metrics,
                    benchmark=lambda: score_counterfact_nll_pairs(
                        runtime.model,
                        runtime.tokenizer,
                        efficacy_specs(requests),
                    ),
                    stagnation_epsilon=float(hard["policy"]["stagnation_epsilon"]),
                )
            finally:
                if metrics.tracks_model_forwards:
                    metrics.detach_model()
            endpoint_hashes = endpoint_parameter_hashes(
                runtime.model, tuple(weight_by_layer.values())
            )
            endpoint_metrics = None
            post_action = {
                "N_post_action_model_fwd": 0,
                "post_action_model_fwd_scope_counts": {
                    "terminal_residual": 0,
                    "endpoint_metrics": 0,
                },
                "post_action_model_fwd_provenance": "none-p0",
                "post_action_wall_seconds_by_scope": {
                    "terminal_residual": 0.0,
                    "endpoint_metrics": 0.0,
                },
                "post_action_gpu_seconds_by_scope": {
                    "terminal_residual": 0.0,
                    "endpoint_metrics": 0.0,
                },
            }
            action_hashes: dict[str, str] = {}
            if stage == "p1":
                opened = {}
                for request in requests:
                    firewall = firewalls[request.case_id]
                    action_hashes[request.case_id] = firewall.freeze_action(
                        {
                            "arm": arm.value,
                            "case_id": request.case_id,
                            "selected_terminal_state_id": result.selected_terminal_state_id,
                            "endpoint_parameter_hashes": dict(endpoint_hashes),
                        }
                    )
                    opened[request.case_id] = firewall.open_evaluation()
                counter = _PostActionForwardCounter(
                    runtime.model, gpu_device="cuda:0"
                )
                with counter, counter.scope("endpoint_metrics"):
                    endpoint_metrics = score_counterfact_endpoint(
                        runtime.model,
                        runtime.tokenizer,
                        requests,
                        opened,
                    )
                post_action = counter.finalize()
                if post_action["post_action_model_fwd_scope_counts"]["endpoint_metrics"] != 3:
                    raise RuntimeError("hard-B10 endpoint metric forward count differs")
                metrics.increment("N_eval", 3)
                if _benchmark_projection(endpoint_metrics.efficacy) != (
                    _benchmark_projection(result.selected_benchmark)
                ):
                    raise RuntimeError("hard-B10 selected efficacy replay differs")
                endpoint_objects[arm] = endpoint_metrics

            snapshot = metrics.finalize().to_dict()
            counters = snapshot["counters"]
            n_z[arm] = counters["N_z"]
            expected_fields = {
                HardBatchArm.NATIVE_MEMIT_B10: 0,
                HardBatchArm.REFRESH_K10_T1: 10,
                HardBatchArm.REFRESH_K20_T1_RESOLUTION: 20,
            }.get(arm)
            if expected_fields is not None and counters["N_field"] != expected_fields:
                raise RuntimeError("hard-B10 fixed field count differs")
            if arm is HardBatchArm.REFRESH_2XK10_T2_CORRECTION and counters["N_field"] not in {10, 20}:
                raise RuntimeError("hard-B10 correction-cycle field count differs")
            if counters["N_bw"] != counters["N_field"]:
                raise RuntimeError("hard-B10 one-backward field count differs")
            if counters["N_eval"] != (3 if stage == "p1" else 0):
                raise RuntimeError("hard-B10 endpoint metric counter differs")
            geometry_rows = [step.joint_geometry for step in result.steps]
            if any(
                row.get("batch_size") != JOINT_BATCH_SIZE
                or row.get("singleton_decomposition") is not False
                or set(row.get("factor_ranks", ())) != {JOINT_BATCH_SIZE}
                for row in geometry_rows
            ):
                raise RuntimeError("hard-B10 runtime joint geometry differs")
            record = {
                "schema_version": "ode-edit-session03-hard-b10-arm/v1",
                "model": args.model_alias,
                "stage": stage,
                "arm": arm.value,
                "result": result.to_dict(),
                "direct_z_identity": identity,
                "direct_z_receipts": [dict(value) for value in arm_receipts],
                "endpoint_parameter_hashes": dict(endpoint_hashes),
                "oracle_calibrations": [
                    value.to_dict() for value in backend.joint_calibrations
                ],
                "oracle_targets": [value.to_dict() for value in backend.joint_targets],
                "hook_reference_gate": (
                    None
                    if backend.hook_reference_gate is None
                    else list(backend.hook_reference_gate)
                ),
                "action_hashes": action_hashes,
                "endpoint_metrics": (
                    None if endpoint_metrics is None else endpoint_metrics.to_dict()
                ),
                "terminal_c_energy": _total_energy(result.terminal_net_energy),
                "technical_pass": True,
            }
            compute = {
                "schema_version": "ode-edit-session03-hard-b10-compute/v1",
                "model": args.model_alias,
                "stage": stage,
                "arm": arm.value,
                **snapshot,
                **counters,
                **post_action,
            }
            mechanism = {
                "schema_version": "ode-edit-session03-hard-b10-mechanism/v1",
                "model": args.model_alias,
                "stage": stage,
                "arm": arm.value,
                **_mechanism(result),
            }
            for payload in (record, compute, mechanism):
                assert_raw_free(payload, "hard-B10 row")
            _jsonl_append(output_root / "controller_steps.jsonl", record)
            _jsonl_append(output_root / "compute.jsonl", compute)
            _jsonl_append(output_root / "mechanism.jsonl", mechanism)
            for event_row in backend.request_event_history:
                payload = {
                    "model": args.model_alias,
                    "stage": stage,
                    "arm": arm.value,
                    **dict(event_row),
                }
                assert_raw_free(payload, "hard-B10 request event")
                _jsonl_append(output_root / "request_events.jsonl", payload)
            if endpoint_metrics is not None:
                payload = {
                    "model": args.model_alias,
                    "stage": stage,
                    "arm": arm.value,
                    **endpoint_metrics.to_dict(),
                }
                assert_raw_free(payload, "hard-B10 endpoint metrics")
                _jsonl_append(output_root / "endpoint_metrics.jsonl", payload)
            rows.append(
                {
                    "arm": arm.value,
                    "status": result.status,
                    "official_success_count": result.selected_benchmark.success_count,
                    "official_success_bits": [
                        int(value) for value in result.selected_benchmark.success_bits
                    ],
                    "first_exact_hit_step": result.first_exact_hit_step,
                    "selected_earliest_exact_hit": result.selected_earliest_exact_hit,
                    "field_build_count": result.field_build_count,
                    "stagnated_cycles": list(result.stagnated_cycles),
                    "selected_direct_z_residual_fractions": [
                        0.0 if entry == 0.0 and selected == 0.0 else selected / entry
                        for selected, entry in zip(
                            result.selected_direct_z_residual_norms,
                            result.entry_direct_z_residual_norms,
                            strict=True,
                        )
                    ],
                    "terminal_c_energy": record["terminal_c_energy"],
                    "controller_wall_seconds": snapshot["controller_wall_seconds"],
                    "controller_gpu_seconds": snapshot["controller_gpu_seconds"],
                    "peak_memory_allocated_bytes": snapshot[
                        "peak_memory_allocated_bytes"
                    ],
                    "peak_memory_reserved_bytes": snapshot[
                        "peak_memory_reserved_bytes"
                    ],
                    "N_model_fwd": counters["N_model_fwd"],
                    "N_benchmark_fwd": counters["N_benchmark_fwd"],
                    "N_direct_z_residual_fwd": counters[
                        "N_direct_z_residual_fwd"
                    ],
                    "N_field": counters["N_field"],
                    "N_bw": counters["N_bw"],
                    "N_trial": counters["N_trial"],
                    "N_write": counters["N_write"],
                    "endpoint_metrics": (
                        None if endpoint_metrics is None else endpoint_metrics.to_dict()
                    ),
                }
            )
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)
    except BaseException:
        baseline.restore(runtime.model)
        baseline.assert_exact(runtime.model, include_rng=True)
        raise

    assert shared_identity is not None and shared_receipts is not None
    _assert_direct_z_identity(
        shared_identity, shared_receipts, identities, receipts, n_z
    )
    baseline.restore(runtime.model)
    baseline.assert_exact(runtime.model, include_rng=True)
    fixed_artifacts.assert_current()
    prepared.bridge.load().provenance.assert_current()
    prepared.covariance_contract.manifest.assert_current()
    native_floor = {}
    if stage == "p1":
        native = endpoint_objects[HardBatchArm.NATIVE_MEMIT_B10]
        native_floor = {
            arm.value: native_floor_verdict(native, endpoint_objects[arm])
            for arm in HARD_BATCH_ARM_ORDER
            if arm is not HardBatchArm.NATIVE_MEMIT_B10
        }
    by_arm = {row["arm"]: row for row in rows}
    native_row = by_arm[HardBatchArm.NATIVE_MEMIT_B10.value]
    for row in rows:
        row["c_energy_ratio_to_native"] = (
            None
            if native_row["terminal_c_energy"] == 0.0
            else row["terminal_c_energy"] / native_row["terminal_c_energy"]
        )
        row["controller_gpu_ratio_to_native"] = (
            None
            if native_row["controller_gpu_seconds"] == 0.0
            else row["controller_gpu_seconds"]
            / native_row["controller_gpu_seconds"]
        )
    d_row = by_arm[HardBatchArm.REFRESH_2XK10_T2_CORRECTION.value]
    summary = {
        "schema_version": f"ode-edit-session03-hard-b10-{stage}-summary/v1",
        "status": "COMPLETE_TECHNICAL_PASS",
        "model": args.model_alias,
        "stage": stage,
        "records": rows,
        "native_floor": native_floor,
        "native_completion_control": {
            "required": d_row["official_success_count"] < JOINT_BATCH_SIZE,
            "source_arm": HardBatchArm.NATIVE_MEMIT_B10.value,
            "official_success_count": native_row["official_success_count"],
            "diagnostic_only": True,
            "ode_success_relabel": False,
        },
        "technical_pass": True,
        "p1_automatic_submission_eligible": stage == "p0",
        "autoregressive_decoding_count": 0,
        "scientific_claim_promoted": False,
    }
    assert_raw_free(summary, "hard-B10 summary")
    _json_write(output_root / "summary.json", summary)
    manifest["status"] = "COMPLETE_TECHNICAL_PASS"
    _json_write(output_root / "manifest.json", manifest)
    terminal_files = [
        output_root / "manifest.json",
        *output_files,
        output_root / "summary.json",
        *sorted((output_root / "direct_z").glob("*.pt")),
    ]
    _write_terminal_manifest(output_root, terminal_files)
    return 0


__all__ = [
    "AUTHORIZATION_TOKENS",
    "OUTPUT_PREFIXES",
    "build_parser",
    "dry_plan",
    "expected_output_root",
    "run",
]
