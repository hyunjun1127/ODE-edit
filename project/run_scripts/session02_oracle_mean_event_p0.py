#!/usr/bin/env python3
"""Run the paired-stage oracle-mean V2 P0 technical profiler.

The executable has no scheduler API.  Slurm authority is an external token;
the tracked lock remains explicitly non-authorizing.
"""

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
from project.run_scripts.ode_edit_method.lock import (
    LOCK_PATH,
    controller_config,
    load_lock,
)
from project.run_scripts.ode_edit_method.oracle_event import ORACLE_MEAN_EVENT_MODE
from project.run_scripts.ode_edit_method.oracle_lock import (
    ORACLE_LOCK_PATH,
    load_oracle_lock,
)
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
)


P0_ARMS = (
    Arm.NATIVE_MEMIT,
    Arm.STATIC_SYNCHRONOUS,
    Arm.FULL_ODE_EDIT,
)
EXECUTION_TOKEN = "oracle-mean-event-v2-r1-p0"
OUTPUT_PREFIX = "session02-oracle-mean-event-p0-r1"


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
        f"{OUTPUT_PREFIX}-{model_alias}-{proposal_id[:8]}"
    )


def _exact_output(repo: Path, raw: Path, model_alias: str, proposal_id: str) -> Path:
    expected = (repo / expected_output_root(model_alias, proposal_id)).resolve()
    candidate = raw if raw.is_absolute() else repo / raw
    if candidate.resolve() != expected:
        raise RuntimeError("oracle P0 output root differs from its dry plan")
    return candidate


def dry_plan(oracle_lock: Mapping[str, Any]) -> dict[str, Any]:
    proposal_id = str(oracle_lock["proposal_id"])
    resources = oracle_lock["resources"]
    return {
        "schema_version": "ode-edit-oracle-mean-p0-dry-plan/v1",
        "submission_authorized": False,
        "proposal_id": proposal_id,
        "lock_sha256": oracle_lock["lock_sha256"],
        "jobs": [
            {
                "model_alias": alias,
                "output_root": str(expected_output_root(alias, proposal_id)),
                "resources": {
                    "gpu": resources["gpu_per_job"],
                    "cpu": resources["cpu_per_job"],
                    "host_memory_mib": resources["host_memory_mib_per_job"],
                    "time": resources["p0_time"],
                },
            }
            for alias in oracle_lock["runtime"]["models"]
        ],
        "pair_gpu": resources["pair_gpu"],
        "server1_project_gpu_cap": resources["server1_project_gpu_cap"],
        "automatic_p1_condition": "both-p0-terminal-technical-pass",
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


def _receipt_rows(ledger: OmegaLedger) -> list[dict[str, Any]]:
    return _receipt_payload(ledger)


def run(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    easyedit_root = args.easyedit_root.resolve(strict=True)
    if easyedit_root != EASYEDIT_ROOT.resolve(strict=True):
        raise RuntimeError("oracle P0 EasyEdit root differs")
    base_lock_path = args.lock.resolve(strict=True)
    base_lock = load_lock(base_lock_path)
    base_proposal_id = base_lock.pop("proposal_id")
    oracle_lock = load_oracle_lock(args.oracle_lock)
    if (
        _file_sha256(base_lock_path) != oracle_lock["base_lock_sha256"]
        or base_proposal_id != oracle_lock["base_proposal_id"]
    ):
        raise RuntimeError("oracle P0 base lock identity differs")
    plan = dry_plan(oracle_lock)
    output_arg = _exact_output(
        repo, args.output_root, args.model_alias, oracle_lock["proposal_id"]
    )
    if args.dry_run:
        print(json.dumps(plan, allow_nan=False, sort_keys=True))
        return 0

    if os.environ.get("ODEEDIT_ORACLE_MEAN_P0_SUBMISSION_AUTHORIZED") != EXECUTION_TOKEN:
        raise RuntimeError("separate oracle P0 execution authority is required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("oracle P0 requires exactly one visible GPU")

    output_root = _require_output_root(repo, output_arg)
    for name in (
        "controller_steps.jsonl",
        "compute.jsonl",
        "event_strength.jsonl",
        "evaluation.jsonl",
    ):
        (output_root / name).touch(exist_ok=False)

    seed = int(oracle_lock["selection"]["seed"])
    case_ids = tuple(oracle_lock["selection"]["p0_case_ids"])
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
    if _request_ids(requests) != base_lock["selection"]["request_ids"][:1]:
        raise RuntimeError("oracle P0 request identity differs")

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
        raise RuntimeError("oracle validity epsilon differs from event tolerance")
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
        raise RuntimeError("oracle P0 setup timer became negative")
    git_head = _git_head(repo)
    manifest = {
        "schema_version": "ode-edit-oracle-mean-p0-manifest/v1",
        "status": "RUNNING_TECHNICAL_ONLY",
        "instruction_id": oracle_lock["instruction_id"],
        "git": {
            "commit": git_head,
            "proposal_id": oracle_lock["proposal_id"],
            "lock_sha256": oracle_lock["lock_sha256"],
            "base_proposal_id": base_proposal_id,
            "base_lock_sha256": oracle_lock["base_lock_sha256"],
        },
        "model": runtime_metadata,
        "selection": {
            "case_ids": list(case_ids),
            "arms": [arm.value for arm in P0_ARMS],
            "seed": seed,
        },
        "context_manifest_id": prepared.contexts.manifest_id,
        "fixed_artifact_manifest_id": fixed_artifacts.manifest_id,
        "easyedit_source_manifest_id": prepared.bridge.load().provenance.manifest_id,
        "policy": {
            "controller": config.to_dict(),
            "event": dict(oracle_lock["event"]),
            "event_backend": ORACLE_MEAN_EVENT_MODE,
            "teacher_forwards_per_event": EVENT_MODEL_FORWARD_CALLS,
            "trial_backend": base_lock["trial_backend"]["selected_common_backend"],
            "model_specific_policy": False,
        },
        "artifact_firewall": {
            "raw_prompt_target_context_persisted": False,
            "evaluation_executed": False,
            "generation_executed": False,
        },
        "setup_wall_seconds": setup_seconds,
        "model_load_wall_seconds_excluded": model_load_seconds,
        "expected_record_count": len(P0_ARMS),
        "scientific_outcome_count": 0,
    }
    assert_raw_free(manifest)
    _json_write(output_root / "manifest.json", manifest)

    records: list[dict[str, Any]] = []
    try:
        request = requests[0]
        for arm in P0_ARMS:
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)
            metrics = EditInstrumentation(
                f"oracle-p0:{args.model_alias}:{arm.value}:{request.case_id}",
                gpu_timing=True,
                gpu_device="cuda:0",
            )
            metrics.add_wall_seconds("context_setup", setup_seconds / len(P0_ARMS))
            cache_root = output_root / "direct_z"
            cache_path = cache_root / f"{args.model_alias}-{arm.value}-{request.case_id}.pt"
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
                    finite_difference_gate_epsilon=(
                        base_lock["derivative_backend"]["p0_finite_difference_epsilon"]
                        if arm is Arm.FULL_ODE_EDIT
                        else None
                    ),
                    finite_difference_gate_atol=base_lock["derivative_backend"]["p0_scalar_gate_atol"],
                    finite_difference_gate_rtol=base_lock["derivative_backend"]["p0_scalar_gate_rtol"],
                )
            ledger = OmegaLedger(denominators)
            metrics.attach_model(runtime.model)
            try:
                result = FiveArmRunner(config, ledger).run(
                    arm,
                    edit_id=f"oracle-p0:{arm.value}:{request.case_id}",
                    request=request,
                    backend=backend,
                    instrumentation=metrics,
                )
            finally:
                metrics.detach_model()
            snapshot = metrics.finalize().to_dict()
            calibration = backend.oracle_calibration
            target = backend.oracle_target
            if calibration is None or target is None:
                raise RuntimeError("oracle P0 calibration is absent")
            counters = snapshot["counters"]
            if (
                result.direct_z_compute_count != 1
                or counters["N_z"] != 1
                or calibration.oracle_forward_count != 2
                or calibration.entry_forward_count != 2
                or calibration.hook_call_count != 2
                or counters["N_event_fwd"] % 2
                or counters["N_eval"] != 0
            ):
                raise RuntimeError("oracle P0 counter contract differs")
            if arm is not Arm.NATIVE_MEMIT and counters["N_bw"] != counters["N_field"]:
                raise RuntimeError("oracle P0 field/backward count differs")
            if arm is Arm.FULL_ODE_EDIT and backend.hook_reference_gate is None:
                raise RuntimeError("oracle P0 Full hook/reference gate is absent")
            result_payload = result.to_dict()
            trace = build_event_strength_trace(
                arm,
                result_payload,
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
                raise RuntimeError("oracle P0 hit violates the realization floor")
            common = {
                "model": args.model_alias,
                "case_id": request.case_id,
                "arm": arm.value,
                "seed": seed,
                "status": result.status,
                "hashes": {
                    "proposal_id": oracle_lock["proposal_id"],
                    "context_manifest_id": prepared.contexts.manifest_id,
                    "direct_z": backend.direct_z_identity,
                },
            }
            controller_row = {
                "schema_version": "ode-edit-oracle-mean-p0-controller/v1",
                **common,
                "result": result_payload,
                "event_history": list(backend.event_history),
                "oracle_calibration": calibration.to_dict(),
                "Omega": {
                    "state": ledger.state(),
                    "receipts": _receipt_rows(ledger),
                },
            }
            compute_row = {
                "schema_version": "ode-edit-oracle-mean-p0-compute/v1",
                **common,
                **snapshot,
                **counters,
                "oracle_forward_count": calibration.oracle_forward_count,
                "ordinary_event_forward_count": counters["N_event_fwd"],
                "hook_reference_gate": (
                    list(backend.hook_reference_gate)
                    if backend.hook_reference_gate is not None
                    else None
                ),
            }
            trace_row = {
                "schema_version": "ode-edit-oracle-mean-p0-strength/v1",
                **common,
                "oracle_target": target.to_dict(),
                "trace": trace,
            }
            terminal_energy = (
                dict(ledger.receipts[-1].terminal_net_energy)
                if ledger.receipts
                else None
            )
            terminal_c_distance = (
                math.sqrt(sum(terminal_energy.values()))
                if terminal_energy is not None
                else None
            )
            d_sync_entry = (
                backend.entry_synchronous_distance()
                if counters["N_field"]
                else None
            )
            compute_row.update(
                {
                    "D_native_raw_proposal": backend.native_distance,
                    "D_sync_entry": d_sync_entry,
                    "terminal_net_c_energy": terminal_energy,
                    "terminal_net_c_distance": terminal_c_distance,
                }
            )
            for row in (controller_row, compute_row, trace_row):
                assert_raw_free(row)
            _jsonl_append(output_root / "controller_steps.jsonl", controller_row)
            _jsonl_append(output_root / "compute.jsonl", compute_row)
            _jsonl_append(output_root / "event_strength.jsonl", trace_row)
            records.append(
                {
                    "arm": arm.value,
                    "status": result.status,
                    "terminal": terminal,
                    "controller_wall_seconds": snapshot["controller_wall_seconds"],
                    "controller_gpu_seconds": snapshot["controller_gpu_seconds"],
                    "peak_memory_allocated_bytes": snapshot["peak_memory_allocated_bytes"],
                    "peak_memory_reserved_bytes": snapshot["peak_memory_reserved_bytes"],
                    "D_sync_entry": d_sync_entry,
                    "terminal_net_c_distance": terminal_c_distance,
                }
            )
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)
    except BaseException:
        baseline.restore(runtime.model)
        baseline.assert_exact(runtime.model, include_rng=True)
        raise

    fixed_artifacts.assert_current()
    prepared.bridge.load().provenance.assert_current()
    by_arm = {row["arm"]: row for row in records}
    native_wall = float(by_arm[Arm.NATIVE_MEMIT.value]["controller_wall_seconds"])
    native_gpu = float(by_arm[Arm.NATIVE_MEMIT.value]["controller_gpu_seconds"])
    ratios = {
        arm.value: {
            "wall_over_native": (
                float(by_arm[arm.value]["controller_wall_seconds"]) / native_wall
                if native_wall > 0.0
                else None
            ),
            "gpu_over_native": (
                float(by_arm[arm.value]["controller_gpu_seconds"]) / native_gpu
                if native_gpu > 0.0
                else None
            ),
        }
        for arm in (Arm.STATIC_SYNCHRONOUS, Arm.FULL_ODE_EDIT)
    }
    summary = {
        "schema_version": "ode-edit-oracle-mean-p0-summary/v1",
        "status": "COMPLETE_TECHNICAL_PASS_NO_EVALUATION",
        "model": args.model_alias,
        "records": records,
        "compute_ratios": ratios,
        "technical_pass": True,
        "p1_automatic_submission_eligible": True,
        "evaluation_count": 0,
        "scientific_outcome_count": 0,
    }
    assert_raw_free(summary)
    _json_write(output_root / "summary.json", summary)
    manifest["status"] = "COMPLETE_TECHNICAL_PASS"
    _json_write(output_root / "manifest.json", manifest)
    files = [
        output_root / "manifest.json",
        output_root / "controller_steps.jsonl",
        output_root / "compute.jsonl",
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
