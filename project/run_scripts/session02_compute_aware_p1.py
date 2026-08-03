"""Executable Session 02 P1 four-case technical mechanism panel.

This runner has no scheduler submission capability and performs no evaluation
or generation.  A separate GH execution envelope must invoke it.  The method
path is exactly the V7 original-BF16/two-forward/full-linear-T path; P1 adds
only outcome-free summaries of controller records and low-rank fields.
"""

from __future__ import annotations

import argparse
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
    CHECKPOINT_ORIGINAL_DTYPE_POLICY,
    load_fixed_model_checkpoint_original,
    offline_environment,
    seed_runtime,
)
from project.run_scripts.ode_edit_method.contracts import Arm, canonical_hash
from project.run_scripts.ode_edit_method.controller import OmegaLedger
from project.run_scripts.ode_edit_method.easyedit_backend import EasyEditMemitBackend
from project.run_scripts.ode_edit_method.events import (
    EVENT_BACKEND_MODE,
    EVENT_MODEL_FORWARD_CALLS,
)
from project.run_scripts.ode_edit_method.hooks import (
    TorchCheckpoint,
    base_weight_c_energy,
)
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.lock import LOCK_PATH, controller_config, load_lock
from project.run_scripts.ode_edit_method.mechanism import summarize_mechanism
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
    _require_fields,
    _require_output_root,
    _strict_ratio,
)


P1_ARMS = (
    Arm.NATIVE_MEMIT,
    Arm.STATIC_SYNCHRONOUS,
    Arm.ONE_REFRESH,
    Arm.FULL_ODE_EDIT,
)
P1_ADAPTIVE_MECHANISM_ARMS = frozenset(
    {Arm.ONE_REFRESH, Arm.FULL_ODE_EDIT}
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session02-compute-aware-p1",
        description="Execute the fixed four-case P1 technical mechanism panel",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model-alias",
        choices=("llama3-8b-inst", "qwen2.5-7b-inst"),
        required=True,
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--easyedit-root", type=Path, default=EASYEDIT_ROOT)
    parser.add_argument("--execute", action="store_true", required=True)
    return parser


def _checkpoint_original_runtime_metadata(runtime: Any) -> dict[str, Any]:
    """Expose and validate the loader's already-checked config dtype."""

    metadata = dict(runtime.metadata())
    config = getattr(runtime.model, "config", None)
    metadata["config_torch_dtype"] = str(
        getattr(config, "torch_dtype", None)
    )
    if metadata.get("dtype_policy") != CHECKPOINT_ORIGINAL_DTYPE_POLICY:
        raise RuntimeError("P1 runtime did not preserve checkpoint dtype")
    for field in (
        "checkpoint_original_dtype",
        "observed_parameter_dtype",
        "config_torch_dtype",
    ):
        if metadata.get(field) != "torch.bfloat16":
            raise RuntimeError(f"P1 runtime {field} is not torch.bfloat16")
    return metadata


def _terminal_geometry(
    ledger: OmegaLedger, receipt_count_before: int
) -> Mapping[str, Any] | None:
    if len(ledger.receipts) == receipt_count_before:
        return None
    if len(ledger.receipts) != receipt_count_before + 1:
        raise RuntimeError("one P1 outer edit appended more than one Omega receipt")
    receipt = ledger.receipts[-1]
    return {
        "terminal_net_energy": dict(receipt.terminal_net_energy),
        "increment": dict(receipt.increment),
        "receipt_id": receipt.receipt_id,
        "micro_step_energy_sum_used": False,
    }


def run(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    easyedit_root = args.easyedit_root.resolve(strict=True)
    if easyedit_root != EASYEDIT_ROOT.resolve(strict=True):
        raise ValueError("P1 EasyEdit root differs from the fixed method runtime")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")

    lock = load_lock(args.lock)
    proposal_id = lock.pop("proposal_id")
    if lock["execution_boundary"]["gpu_now"] != 0:
        raise RuntimeError("tracked P1 lock must not grant GPU authority")
    if lock["execution_boundary"]["slurm_now"] is not False:
        raise RuntimeError("tracked P1 lock unexpectedly grants Slurm authority")
    execution = lock["execution_path"]
    if execution.get("p1_runner") != "project/run_scripts/session02_compute_aware_p1.py":
        raise RuntimeError("tracked P1 execution path differs")

    model_lock = lock["models"][args.model_alias]
    case_ids = tuple(lock["selection"]["p1_case_ids"])
    if case_ids != ("2022", "12498", "20964", "768"):
        raise RuntimeError("P1 four-case canonical order differs")
    if int(lock["timing"]["p1_canonical_repetitions"]) != 1:
        raise RuntimeError("P1 repetition lock differs")

    output_root = _require_output_root(repo, args.output_root)
    for filename in (
        "controller_steps.jsonl",
        "compute.jsonl",
        "mechanism.jsonl",
        "evaluation.jsonl",
    ):
        (output_root / filename).touch(exist_ok=False)

    seed = int(lock["execution_seed"])
    seed_runtime(seed)
    setup_start = time.perf_counter()
    fixed_artifacts, requests = preflight_static_inputs(
        easyedit_root,
        model_alias=args.model_alias,
        case_ids=case_ids,
    )
    assert_static_lock_identities(
        easyedit_root,
        fixed_artifacts=fixed_artifacts,
        model_lock=model_lock,
        selection_lock=lock["selection"],
    )
    observed_request_ids = [
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
    if observed_request_ids != lock["selection"]["request_ids"]:
        raise RuntimeError("P1 rewrite request identities differ from the lock")

    model_load_start = time.perf_counter()
    with offline_environment():
        runtime = load_fixed_model_checkpoint_original(args.model_alias)
    model_load_wall_seconds = time.perf_counter() - model_load_start
    runtime_metadata = _checkpoint_original_runtime_metadata(runtime)
    _require_fields(
        runtime_metadata,
        lock["artifact_schema"]["manifest_model_required_fields"],
        label="P1 model runtime metadata",
    )

    prepared = prepare_concrete_environment(
        easyedit_root,
        runtime=runtime,
        fixed_artifacts=fixed_artifacts,
        controller_requests=requests,
        expected_model_lock=model_lock,
        seed=seed,
    )
    event_lock = lock["event_backend"]
    if (
        event_lock.get("old_new_forward") != EVENT_BACKEND_MODE
        or event_lock.get("actual_model_calls_per_event")
        != EVENT_MODEL_FORWARD_CALLS
        or event_lock.get("conditional_fallback") is not False
    ):
        raise RuntimeError("P1 event backend differs from the common two-forward lock")

    config = controller_config(lock)
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
    setup_wall_seconds = time.perf_counter() - setup_start - model_load_wall_seconds
    if setup_wall_seconds < 0.0:
        raise RuntimeError("P1 setup timer became negative")
    setup_share = setup_wall_seconds / (len(P1_ARMS) * len(requests))
    git_head = _git_head(repo)
    order_sha256 = canonical_hash(list(case_ids))
    promotion = dict(lock["p1_promotion"])

    manifest = {
        "schema_version": "ode-edit-session02-p1-manifest/v1",
        "status": "RUNNING_TECHNICAL_MECHANISM_ONLY",
        "instruction": {
            "instruction_id": lock["instruction_id"],
            "revision_id": lock["revision_id"],
            "parent_instruction_id": lock["parent_instruction_id"],
        },
        "git": {"commit": git_head, "proposal_id": proposal_id},
        "model": runtime_metadata,
        "selection": {
            "case_ids": list(case_ids),
            "order_sha256": order_sha256,
            "repetitions": 1,
        },
        "contexts": {
            "manifest_id": prepared.contexts.manifest_id,
            "tokenization_manifests": list(prepared.tokenization_manifests),
        },
        "hashes": {
            "fixed_artifact_manifest_id": fixed_artifacts.manifest_id,
            "easyedit_source_manifest_id": prepared.bridge.load().provenance.manifest_id,
            "dataset_sha256": lock["selection"]["dataset_sha256"],
            "hparams_sha256": model_lock["hparams_sha256"],
            "canonical_spec_sha256": lock["canonical_spec"]["sha256"],
        },
        "offline": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "policy": {
            "controller": config.to_dict(),
            "arms": [arm.value for arm in P1_ARMS],
            "trial_backend": lock["trial_backend"]["selected_common_backend"],
            "event_backend": EVENT_BACKEND_MODE,
            "event_nfe": EVENT_MODEL_FORWARD_CALLS,
            "dtype_policy": runtime_metadata["dtype_policy"],
            "checkpoint_original_dtype": runtime_metadata[
                "checkpoint_original_dtype"
            ],
            "observed_parameter_dtype": runtime_metadata[
                "observed_parameter_dtype"
            ],
            "model_specific_rescue": False,
            "common_formula_unchanged": True,
        },
        "p1_promotion": promotion,
        "mechanism_contract": lock["mechanism_diagnostics"],
        "artifact_firewall": {
            "raw_prompt_target_context_persisted": False,
            "evaluation_executed": False,
            "generation_executed": False,
        },
        "setup_wall_seconds": setup_wall_seconds,
        "model_load_wall_seconds_excluded": model_load_wall_seconds,
        "expected_run_count": len(P1_ARMS) * len(requests),
    }
    _require_fields(
        manifest,
        lock["artifact_schema"]["p1_manifest_required_sections"],
        label="P1 manifest",
    )
    _require_fields(
        manifest["policy"],
        lock["artifact_schema"]["manifest_policy_required_fields"],
        label="P1 manifest policy",
    )
    _json_write(output_root / "manifest.json", manifest)

    all_compute: dict[int, dict[str, Mapping[str, Any]]] = {}
    mechanism_coverage: list[dict[str, Any]] = []
    terminal_geometry_fractions: list[dict[str, Any]] = []
    stream_summaries: list[dict[str, Any]] = []
    records = 0

    for arm in P1_ARMS:
        baseline.restore(runtime.model)
        stream_guard = P1SequentialArmGuard(arm, case_ids)
        stream_guard.note_arm_start_restore()
        ledger = OmegaLedger(denominators)
        pending: list[dict[str, Any]] = []
        try:
            for order_position, request in enumerate(requests):
                omega_before = ledger.state()
                receipt_count_before = len(ledger.receipts)
                metrics = EditInstrumentation(
                    f"{arm.value}:position-{order_position}:{request.case_id}:p1",
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
                    backend = EasyEditMemitBackend(
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
                        unit_c_norm_epsilon=lock["controller"][
                            "unit_c_norm_epsilon"
                        ],
                        unit_c_identity_atol=lock["controller"][
                            "unit_c_identity_atol"
                        ],
                        unit_c_identity_rtol=lock["controller"][
                            "unit_c_identity_rtol"
                        ],
                        instrumentation=metrics,
                        record_mechanism=arm in P1_ADAPTIVE_MECHANISM_ARMS,
                    )
                pre_edit_state_id = backend.current_state_id()
                stream_guard.begin_edit(
                    order_position=order_position,
                    case_id=request.case_id,
                    pre_edit_state_id=pre_edit_state_id,
                    omega_before=omega_before,
                    history_length_before=receipt_count_before,
                    cache_identity=str(cache_path.relative_to(output_root)),
                )
                metrics.attach_model(runtime.model)
                try:
                    result = FiveArmRunner(config, ledger).run(
                        arm,
                        edit_id=(
                            f"{arm.value}:position-{order_position}:"
                            f"case-{request.case_id}"
                        ),
                        request=request,
                        backend=backend,
                        instrumentation=metrics,
                    )
                finally:
                    metrics.detach_model()
                post_edit_state_id = backend.current_state_id()
                omega_after = ledger.state()
                receipt_count_after = len(ledger.receipts)
                if result.omega_appended != (
                    receipt_count_after == receipt_count_before + 1
                ):
                    raise RuntimeError("P1 Omega append/result identity differs")
                failure_rollback_required = not result.omega_appended
                failure_rollback_exact = (
                    post_edit_state_id == pre_edit_state_id
                    if failure_rollback_required
                    else None
                )
                successful_terminal_endpoint_exact = (
                    post_edit_state_id == result.terminal_state_id
                    if result.omega_appended
                    else None
                )
                if failure_rollback_exact is False:
                    raise RuntimeError("P1 failure rollback did not preserve pre-edit state")
                if successful_terminal_endpoint_exact is False:
                    raise RuntimeError("P1 successful terminal endpoint identity differs")
                stream_guard.finish_edit(
                    result=result,
                    post_edit_state_id=post_edit_state_id,
                    omega_after=omega_after,
                    history_length_after=receipt_count_after,
                )

                snapshot = metrics.finalize().to_dict()
                counters = snapshot["counters"]
                if int(counters["N_event_fwd"]) % EVENT_MODEL_FORWARD_CALLS:
                    raise RuntimeError(
                        "P1 event forward accounting is not two-forward aligned"
                    )
                if (
                    arm is not Arm.NATIVE_MEMIT
                    and counters["N_bw"] != counters["N_field"]
                ):
                    raise RuntimeError("P1 synchronous field/backward accounting differs")
                if counters["N_eval"] != 0:
                    raise RuntimeError("P1 evaluation firewall counter is nonzero")

                exact_native_distance = None
                if arm is Arm.NATIVE_MEMIT and result.omega_appended:
                    exact_native_distance = math.sqrt(
                        sum(
                            value
                            for _layer, value in ledger.receipts[-1].terminal_net_energy
                        )
                    )
                d_sync_entry = (
                    backend.entry_synchronous_distance()
                    if counters["N_field"]
                    else None
                )
                common = {
                    "model": args.model_alias,
                    "case_id": request.case_id,
                    "arm": arm.value,
                    "order_position": order_position,
                    "order_sha256": order_sha256,
                    "seed": seed,
                    "repetition": 0,
                    "pre_edit_state_id": pre_edit_state_id,
                    "post_edit_state_id": post_edit_state_id,
                    "commit": git_head,
                    "hashes": {
                        "proposal_id": proposal_id,
                        "context_manifest_id": prepared.contexts.manifest_id,
                        "fixed_artifact_manifest_id": fixed_artifacts.manifest_id,
                        "direct_z": backend.direct_z_identity,
                    },
                    "status": result.status,
                }
                radius_provenance = {
                    "strict_radius_diagnostic_result": (
                        "P0_REFERENCE_ONLY_NO_P1_CROSS_ARM_RATIO"
                    ),
                    "strict_p0_calibration": promotion["strict_p0_observations"],
                    "strict_interval": promotion["strict_radius_interval"],
                    "gh_lenient_motivation_promotion": promotion[
                        "gh_lenient_motivation_promotion"
                    ],
                    "p1_cross_arm_case_ratio_computed": False,
                    "same_entry_cross_arm_geometry_available": False,
                    "arm_local_D_native": exact_native_distance,
                    "arm_local_D_sync_entry": d_sync_entry,
                    "common_formula_unchanged": True,
                    "model_specific_rescue": False,
                }
                pending.append(
                    {
                        "common": common,
                        "result": result,
                        "event_history": backend.event_history,
                        "field_history": backend.mechanism_field_history,
                        "layers": backend.layers,
                        "snapshot": snapshot,
                        "counters": counters,
                        "omega_before": omega_before,
                        "omega_after": omega_after,
                        "receipt_count_before": receipt_count_before,
                        "receipt_count_after": receipt_count_after,
                        "receipts": _receipt_payload(ledger),
                        "terminal_geometry": _terminal_geometry(
                            ledger, receipt_count_before
                        ),
                        "failure_rollback_required": failure_rollback_required,
                        "failure_rollback_exact": failure_rollback_exact,
                        "successful_terminal_endpoint_exact": (
                            successful_terminal_endpoint_exact
                        ),
                        "exact_native_distance": exact_native_distance,
                        "native_raw_distance": backend.native_distance,
                        "d_sync_entry": d_sync_entry,
                        "radius_provenance": radius_provenance,
                    }
                )
                # Do not carry edit-local proposal/direct-z tensors into the
                # next sequential edit.  Only raw-free Python records above
                # survive; model weights and cumulative Omega remain live.
                del backend
        except BaseException:
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)
            raise

        stream_terminal_state_id = (
            pending[-1]["common"]["post_edit_state_id"] if pending else None
        )
        baseline.restore(runtime.model)
        baseline.assert_exact(runtime.model, include_rng=True)
        arm_isolation_restore_exact = True
        stream_guard.note_arm_end_restore(exact=arm_isolation_restore_exact)
        stream_summaries.append(
            {
                **stream_guard.summary(),
                "order_sha256": order_sha256,
                "completed_outer_edit_count": len(ledger.receipts),
                "Omega_terminal": ledger.state(),
                "stream_terminal_state_id_before_arm_isolation": (
                    stream_terminal_state_id
                ),
                "arm_isolation_restore_exact": arm_isolation_restore_exact,
            }
        )

        for bundle in pending:
            common = bundle["common"]
            snapshot = bundle["snapshot"]
            counters = bundle["counters"]
            mechanism = summarize_mechanism(
                arm,
                bundle["result"],
                bundle["field_history"],
                layers=bundle["layers"],
                support_tolerance=config.slope_epsilon,
                controller_gpu_seconds=float(snapshot["controller_gpu_seconds"]),
                event_history=bundle["event_history"],
                progress_epsilon=config.progress_epsilon,
                failure_rollback_required=bundle["failure_rollback_required"],
                failure_rollback_exact=bundle["failure_rollback_exact"],
                successful_terminal_endpoint_exact=bundle[
                    "successful_terminal_endpoint_exact"
                ],
                arm_isolation_restore_exact=arm_isolation_restore_exact,
            )
            sequential_state = {
                "order_position": common["order_position"],
                "pre_edit_state_id": common["pre_edit_state_id"],
                "post_edit_state_id": common["post_edit_state_id"],
                "Omega_before": bundle["omega_before"],
                "Omega_after": bundle["omega_after"],
                "history_length_before": bundle["receipt_count_before"],
                "history_length_after": bundle["receipt_count_after"],
                "accepted_terminal_count": bundle["receipt_count_after"],
                "current_edit_terminal_appended": bundle["result"].omega_appended,
            }
            controller_record = {
                "schema_version": "ode-edit-session02-p1-controller-record/v1",
                **common,
                "result": bundle["result"].to_dict(),
                "event": list(bundle["event_history"]),
                "Omega": {
                    "before": bundle["omega_before"],
                    "after": bundle["omega_after"],
                    "receipts_through_current_edit": bundle["receipts"],
                },
                "sequential_state": sequential_state,
                "radius_provenance": bundle["radius_provenance"],
                "integrity": {
                    "failure_rollback_required": bundle[
                        "failure_rollback_required"
                    ],
                    "failure_rollback_exact": bundle["failure_rollback_exact"],
                    "successful_terminal_endpoint_exact": bundle[
                        "successful_terminal_endpoint_exact"
                    ],
                    "arm_isolation_restore_exact": arm_isolation_restore_exact,
                    "omega_used_for_rollback_inference": False,
                    "functional_commit_t_c": "PASS_OR_NOT_APPLICABLE",
                    "functional_commit_checks": int(counters["N_write"]),
                    "hook_dense_target_gradient_materialized": False,
                    "hook_reference_repeated_in_p1": False,
                    "field_backward_identity": counters["N_bw"]
                    == counters["N_field"],
                },
            }
            _require_fields(
                controller_record,
                lock["artifact_schema"]["p1_controller_required_fields"],
                label="P1 controller record",
            )
            compute_record = {
                **snapshot,
                **common,
                "schema_version": "ode-edit-session02-p1-compute-record/v1",
                "accounting_schema_version": snapshot["schema_version"],
                **counters,
                "D_native": bundle["exact_native_distance"],
                "D_native_raw_proposal": bundle["native_raw_distance"],
                "D_sync_entry": bundle["d_sync_entry"],
                "terminal_geometry": bundle["terminal_geometry"],
                "terminal_geometry_wall_fraction": (
                    snapshot["component_wall_seconds"]["terminal_geometry"]
                    / snapshot["controller_wall_seconds"]
                    if snapshot["controller_wall_seconds"] > 0.0
                    else 0.0
                ),
                "terminal_geometry_gpu_fraction": (
                    snapshot["component_gpu_seconds"]["terminal_geometry"]
                    / snapshot["controller_gpu_seconds"]
                    if snapshot["controller_gpu_seconds"] > 0.0
                    else 0.0
                ),
                "sequential_state": sequential_state,
                "radius_provenance": bundle["radius_provenance"],
            }
            _require_fields(
                compute_record,
                lock["artifact_schema"]["p1_compute_required_fields"],
                label="P1 compute record",
            )
            mechanism_record = {
                "schema_version": "ode-edit-session02-p1-mechanism-record/v1",
                **common,
                "mechanism": mechanism,
                "sequential_state": sequential_state,
                "radius_provenance": bundle["radius_provenance"],
            }
            _require_fields(
                mechanism_record,
                lock["artifact_schema"]["p1_mechanism_required_fields"],
                label="P1 mechanism record",
            )
            _jsonl_append(output_root / "controller_steps.jsonl", controller_record)
            _jsonl_append(output_root / "compute.jsonl", compute_record)
            _jsonl_append(output_root / "mechanism.jsonl", mechanism_record)
            order_position = int(common["order_position"])
            all_compute.setdefault(order_position, {})[arm.value] = compute_record
            mechanism_coverage.append(
                {
                    "case_id": common["case_id"],
                    "order_position": order_position,
                    "arm": arm.value,
                    "field_count": mechanism["field_count"],
                    "adaptive_drift_applicable": mechanism[
                        "adaptive_drift_applicable"
                    ],
                    "c_geometry_available": mechanism["c_geometry"]["available"],
                    "c_geometry_deferred_reason": mechanism["c_geometry"][
                        "deferred_reason"
                    ],
                }
            )
            terminal_geometry_fractions.append(
                {
                    "case_id": common["case_id"],
                    "order_position": order_position,
                    "arm": arm.value,
                    "wall_fraction": compute_record[
                        "terminal_geometry_wall_fraction"
                    ],
                    "gpu_fraction": compute_record[
                        "terminal_geometry_gpu_fraction"
                    ],
                }
            )
            records += 1

    terminal_integrity_start = time.perf_counter()
    fixed_artifacts.assert_current()
    prepared.bridge.load().provenance.assert_current()
    terminal_integrity_wall_seconds = time.perf_counter() - terminal_integrity_start
    full_native_ratios = [
        {
            "case_id": case_ids[order_position],
            "order_position": order_position,
            "divergent_sequential_arm_histories": True,
            "wall": _strict_ratio(
                arms[Arm.FULL_ODE_EDIT.value]["controller_wall_seconds"],
                arms[Arm.NATIVE_MEMIT.value]["controller_wall_seconds"],
                label="P1 Full/Native wall",
            ),
            "gpu": _strict_ratio(
                arms[Arm.FULL_ODE_EDIT.value]["controller_gpu_seconds"],
                arms[Arm.NATIVE_MEMIT.value]["controller_gpu_seconds"],
                label="P1 Full/Native GPU",
            ),
        }
        for order_position, arms in sorted(all_compute.items())
    ]
    summary = {
        "schema_version": "ode-edit-session02-p1-summary/v1",
        "status": "COMPLETE_TECHNICAL_MECHANISM_ONLY_NO_EVALUATION",
        "model": args.model_alias,
        "case_ids": list(case_ids),
        "record_count": records,
        "radius_provenance": {
            "strict_p0_calibration": promotion["strict_p0_observations"],
            "strict_interval": promotion["strict_radius_interval"],
            "gh_lenient_motivation_promotion": promotion[
                "gh_lenient_motivation_promotion"
            ],
            "p1_cross_arm_case_ratio_computed": False,
        },
        "p1_promotion": promotion,
        "sequential_streams": stream_summaries,
        "full_native_compute_ratios": full_native_ratios,
        "mechanism_coverage": mechanism_coverage,
        "terminal_geometry_fractions": terminal_geometry_fractions,
        "terminal_integrity_wall_seconds": terminal_integrity_wall_seconds,
        "scientific_outcome_count": 0,
        "evaluation_count": 0,
    }
    _json_write(output_root / "summary.json", summary)
    manifest["status"] = "COMPLETE_TECHNICAL_MECHANISM_ONLY"
    manifest["terminal_integrity_wall_seconds"] = terminal_integrity_wall_seconds
    _json_write(output_root / "manifest.json", manifest)
    terminal_files = [
        output_root / "manifest.json",
        output_root / "controller_steps.jsonl",
        output_root / "compute.jsonl",
        output_root / "mechanism.jsonl",
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
                for path in terminal_files
            },
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
