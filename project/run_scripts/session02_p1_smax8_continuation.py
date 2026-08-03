#!/usr/bin/env python3
"""Execute the prelocked affected-case ``S_max=8`` P1 continuation.

This runner has no submission capability.  It derives eligibility from both
completed R2 technical artifacts, performs no evaluation or generation, and
does not create an output root for a zero-eligibility model.  The only method
configuration change is the predeclared controller cap from six to eight.
"""

from __future__ import annotations

import argparse
import json
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
    BASE_S_MAX,
    CONTINUATION_S_MAX,
    DEFAULT_R2_SOURCE_SPECS,
    R2_LOCK_SHA256,
    R2_PROPOSAL_ID,
    SERVER1_GPU_CAP,
    PinnedDirectZEasyEditBackend,
    assert_prefix_match,
    continuation_config,
    continuation_hit_round,
    dispatch_eligibility,
    file_sha256,
    promotion_decision,
    validate_r2_sources,
)
from project.run_scripts.ode_edit_method.contracts import Arm
from project.run_scripts.ode_edit_method.controller import OmegaLedger
from project.run_scripts.ode_edit_method.events import (
    EVENT_BACKEND_MODE,
    EVENT_MODEL_FORWARD_CALLS,
)
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
from project.run_scripts.ode_edit_method.preflight import (
    assert_static_lock_identities,
    preflight_static_inputs,
    prepare_concrete_environment,
)
from project.run_scripts.ode_edit_method.runtime import FiveArmRunner
from project.run_scripts.session02_compute_aware_p0 import (
    EASYEDIT_ROOT,
    _git_head,
    _json_write,
    _jsonl_append,
    _require_output_root,
)
from project.run_scripts.session02_compute_aware_p1 import (
    _checkpoint_original_runtime_metadata,
)


EXECUTION_TOKEN = "p1-smax8-affected-cont-v1"
OUTPUT_PREFIX = "session02-p1-smax8-affected-cont-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session02-p1-smax8-continuation",
        description="Run the prelocked affected-case P1 S_max=8 continuation",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model-alias",
        required=True,
        choices=("llama3-8b-inst", "qwen2.5-7b-inst"),
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--easyedit-root", type=Path, default=EASYEDIT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def _expected_output_root(model_alias: str) -> Path:
    return Path(
        "local/results/"
        f"{OUTPUT_PREFIX}-{model_alias}-{R2_PROPOSAL_ID[:8]}"
    )


def _dry_payload(
    *,
    model_alias: str,
    output_root: Path,
    eligible_case_ids: Sequence[str],
) -> dict[str, Any]:
    job = None
    if eligible_case_ids:
        job = {
            "model_alias": model_alias,
            "eligible_case_ids": list(eligible_case_ids),
            "output_root": str(output_root),
            "resources": {
                "gpu": 1,
                "cpu": 8,
                "host_memory_mib": 65000,
                "time": "04:00:00",
            },
        }
    return {
        "schema_version": "ode-edit-p1-smax8-dry-plan/v1",
        "submission_authorized": False,
        "server1_project_gpu_cap": SERVER1_GPU_CAP,
        "base_proposal_id": R2_PROPOSAL_ID,
        "base_lock_sha256": R2_LOCK_SHA256,
        "base_s_max": BASE_S_MAX,
        "continuation_s_max": CONTINUATION_S_MAX,
        "only_controller_difference": ["s_max"],
        "pooled_trigger_count": 3,
        "selected_job": job,
        "zero_eligibility_no_output_mutation": not bool(eligible_case_ids),
        "scientific_outcome_count": 0,
        "evaluation_count": 0,
    }


def _require_exact_output_arg(repo: Path, model_alias: str, raw: Path) -> Path:
    expected = (repo / _expected_output_root(model_alias)).resolve()
    candidate = raw if raw.is_absolute() else repo / raw
    if candidate.resolve() != expected:
        raise RuntimeError("continuation output root differs from the predeclared root")
    return candidate


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


def run(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    if args.easyedit_root.resolve(strict=True) != EASYEDIT_ROOT.resolve(strict=True):
        raise RuntimeError("continuation EasyEdit root differs from the fixed runtime")
    output_arg = _require_exact_output_arg(repo, args.model_alias, args.output_root)

    if file_sha256(args.lock.resolve(strict=True)) != R2_LOCK_SHA256:
        raise RuntimeError("base numerical lock SHA-256 differs")
    lock = load_lock(args.lock)
    proposal_id = lock.pop("proposal_id")
    if proposal_id != R2_PROPOSAL_ID:
        raise RuntimeError("base proposal identity differs")
    base_config = controller_config(lock)
    extended_config = continuation_config(base_config)
    if lock["execution_boundary"]["gpu_now"] != 0:
        raise RuntimeError("tracked base lock unexpectedly grants GPU authority")
    if lock["execution_boundary"]["slurm_now"] is not False:
        raise RuntimeError("tracked base lock unexpectedly grants Slurm authority")

    selection = validate_r2_sources(repo, DEFAULT_R2_SOURCE_SPECS)
    affected = dispatch_eligibility(
        selection,
        args.model_alias,
        output_root=output_arg,
    )
    dry = _dry_payload(
        model_alias=args.model_alias,
        output_root=_expected_output_root(args.model_alias),
        eligible_case_ids=[item.case_id for item in affected],
    )
    if args.dry_run:
        print(json.dumps(dry, allow_nan=False, sort_keys=True))
        return 0
    if not affected:
        print(
            json.dumps(
                {
                    **dry,
                    "status": "NO_ELIGIBLE_AFFECTED_CASES",
                    "model_load": False,
                    "output_mutated": False,
                },
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0

    if os.environ.get("ODEEDIT_P1_SMAX8_EXECUTION_AUTHORIZED") != EXECUTION_TOKEN:
        raise RuntimeError(
            "separate S_max=8 continuation execution authority is required"
        )
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("continuation execution requires exactly one visible GPU")

    output_root = _require_output_root(repo, output_arg)
    for filename in ("continuation.jsonl", "compute.jsonl", "evaluation.jsonl"):
        (output_root / filename).touch(exist_ok=False)

    source = selection.source_for_alias(args.model_alias)
    source_identity_before = {
        relative: {"sha256": identity.sha256, "size": identity.size}
        for relative, identity in source.file_identities
    }
    case_ids = tuple(item.case_id for item in affected)
    seed = int(lock["execution_seed"])
    seed_runtime(seed)
    setup_start = time.perf_counter()
    fixed_artifacts, requests = preflight_static_inputs(
        args.easyedit_root,
        model_alias=args.model_alias,
        case_ids=case_ids,
    )
    model_lock = lock["models"][args.model_alias]
    assert_static_lock_identities(
        args.easyedit_root,
        fixed_artifacts=fixed_artifacts,
        model_lock=model_lock,
        selection_lock=lock["selection"],
    )
    expected_by_case = dict(
        zip(
            lock["selection"]["p1_case_ids"],
            lock["selection"]["request_ids"],
            strict=True,
        )
    )
    if _request_ids(requests) != [expected_by_case[case_id] for case_id in case_ids]:
        raise RuntimeError("continuation rewrite request identities differ")

    model_load_start = time.perf_counter()
    with offline_environment():
        runtime = load_fixed_model_checkpoint_original(args.model_alias)
    model_load_wall_seconds = time.perf_counter() - model_load_start
    runtime_metadata = _checkpoint_original_runtime_metadata(runtime)
    prepared = prepare_concrete_environment(
        args.easyedit_root,
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
        raise RuntimeError("continuation event backend differs from V7")

    weight_by_layer = {
        layer: f"{prepared.hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in prepared.runtime.spec.layers
    }
    denominators = base_weight_c_energy(
        runtime.model,
        prepared.covariance_by_layer,
        weight_by_layer,
        denominator_epsilon=extended_config.load_denominator_epsilon,
    )
    baseline = TorchCheckpoint.capture(
        runtime.model,
        tuple(weight_by_layer.values()),
        backup_device="cpu",
    )
    setup_wall_seconds = time.perf_counter() - setup_start - model_load_wall_seconds
    if setup_wall_seconds < 0.0:
        raise RuntimeError("continuation setup timer became negative")

    git_head = _git_head(repo)
    manifest: dict[str, Any] = {
        "schema_version": "ode-edit-p1-smax8-continuation-manifest/v1",
        "status": "RUNNING_TECHNICAL_RESOLUTION_ONLY",
        "instruction_id": "ODEEDIT-S02-P1-SMAX8-AFFECTED-CONT-IMPL-V1",
        "git": {
            "commit": git_head,
            "base_proposal_id": R2_PROPOSAL_ID,
            "base_lock_sha256": R2_LOCK_SHA256,
        },
        "model": runtime_metadata,
        "selection": {
            "canonical_order": list(lock["selection"]["p1_case_ids"]),
            "affected_case_ids": list(case_ids),
            "eligibility_by_model": {
                candidate.spec.model_alias: [
                    item.case_id for item in candidate.affected
                ]
                for candidate in selection.sources
            },
            "excluded_trust_rejection_case": "12498",
            "pooled_resolution_cap_unresolved": selection.pooled_affected_count,
            "prelocked_trigger_minimum": 2,
        },
        "source": {
            "root": str(source.spec.root.relative_to(repo)),
            "terminal_manifest_sha256": source.spec.terminal_manifest_sha256,
            "manifest_sha256": source.manifest_sha256,
            "summary_sha256": source.summary_sha256,
            "files_read_only": True,
            "pair_terminal_manifest_sha256": {
                candidate.spec.model_alias: candidate.spec.terminal_manifest_sha256
                for candidate in selection.sources
            },
        },
        "policy": {
            "arm": Arm.FULL_ODE_EDIT.value,
            "base_controller": base_config.to_dict(),
            "continuation_controller": extended_config.to_dict(),
            "only_controller_difference": ["s_max"],
            "direct_z_reused_read_only": True,
            "direct_z_recomputed": False,
            "event_backend": EVENT_BACKEND_MODE,
            "event_nfe": EVENT_MODEL_FORWARD_CALLS,
            "trial_backend": lock["trial_backend"]["selected_common_backend"],
            "model_specific_rescue": False,
            "server1_project_gpu_cap": SERVER1_GPU_CAP,
        },
        "artifact_firewall": {
            "evaluation_executed": False,
            "generation_executed": False,
            "raw_prompt_target_context_persisted": False,
            "N_eval": 0,
        },
        "setup_wall_seconds": setup_wall_seconds,
        "model_load_wall_seconds_excluded": model_load_wall_seconds,
        "expected_record_count": len(affected),
        "scientific_outcome_count": 0,
    }
    _json_write(output_root / "manifest.json", manifest)

    records: list[Mapping[str, Any]] = []
    hit_rounds: list[int | None] = []
    try:
        for request, trajectory in zip(requests, affected, strict=True):
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)
            ledger = OmegaLedger(denominators)
            metrics = EditInstrumentation(
                f"smax8:{request.case_id}",
                gpu_timing=True,
                gpu_device="cuda:0",
            )
            metrics.add_wall_seconds(
                "context_setup", setup_wall_seconds / len(affected)
            )
            direct_z_path = source.spec.root / trajectory.direct_z_relative_path
            direct_z_before = {
                "sha256": file_sha256(direct_z_path),
                "size": direct_z_path.stat().st_size,
            }
            backend = PinnedDirectZEasyEditBackend(
                runtime=runtime,
                bridge=prepared.bridge,
                hparams=prepared.hparams,
                contexts=prepared.contexts,
                request=request,
                covariance_specs=prepared.covariance_specs,
                covariance_contract=prepared.covariance_contract,
                covariance_by_layer=prepared.covariance_by_layer,
                direct_z_cache_root=source.spec.root / "direct_z",
                direct_z_cache_path=direct_z_path,
                expected_direct_z_identity=trajectory.direct_z_identity,
                expected_direct_z_tensor_sha256=trajectory.direct_z_tensor_sha256,
                tau=extended_config.tau,
                unit_c_norm_epsilon=lock["controller"]["unit_c_norm_epsilon"],
                unit_c_identity_atol=lock["controller"]["unit_c_identity_atol"],
                unit_c_identity_rtol=lock["controller"]["unit_c_identity_rtol"],
                instrumentation=metrics,
                record_mechanism=False,
            )
            if backend.current_state_id() != trajectory.pre_edit_state_id:
                raise RuntimeError("continuation request-scoped W0 identity differs")
            if (
                backend.sequential_target_weight_state_id()
                != trajectory.pre_edit_target_weight_state_id
            ):
                raise RuntimeError("continuation target-weight W0 identity differs")

            metrics.attach_model(runtime.model)
            try:
                result = FiveArmRunner(extended_config, ledger).run(
                    Arm.FULL_ODE_EDIT,
                    edit_id=f"smax8:case-{request.case_id}",
                    request=request,
                    backend=backend,
                    instrumentation=metrics,
                )
            finally:
                metrics.detach_model()
            snapshot = metrics.finalize().to_dict()
            counters = snapshot["counters"]
            if counters["N_z"] != 1 or result.direct_z_compute_count != 1:
                raise RuntimeError("continuation direct-z once/edit accounting differs")
            if counters["N_eval"] != 0:
                raise RuntimeError(
                    "continuation evaluation firewall counter is nonzero"
                )
            if counters["N_bw"] != counters["N_field"]:
                raise RuntimeError("continuation field/backward accounting differs")
            if int(counters["N_event_fwd"]) % EVENT_MODEL_FORWARD_CALLS:
                raise RuntimeError("continuation event forward accounting differs")

            result_payload = result.to_dict()
            observed_prefix = assert_prefix_match(
                trajectory.prefix_payload,
                result_payload["steps"],
            )
            if observed_prefix != trajectory.prefix_fingerprint:
                raise RuntimeError("continuation prefix fingerprint differs")
            hit_round = continuation_hit_round(result_payload)
            hit_rounds.append(hit_round)
            d_sync_entry = (
                backend.entry_synchronous_distance()
                if counters["N_field"]
                else None
            )
            direct_z_after = {
                "sha256": file_sha256(direct_z_path),
                "size": direct_z_path.stat().st_size,
            }
            if direct_z_after != direct_z_before:
                raise RuntimeError("R2 direct-z artifact changed during continuation")

            post_state_id = backend.current_state_id()
            post_target_weight_state_id = backend.sequential_target_weight_state_id()
            failure_rollback_required = not result.omega_appended
            failure_rollback_exact: bool | None = None
            successful_terminal_endpoint_exact: bool | None = None
            if failure_rollback_required:
                failure_rollback_exact = (
                    post_state_id == trajectory.pre_edit_state_id
                    and post_target_weight_state_id
                    == trajectory.pre_edit_target_weight_state_id
                    and ledger.state() == {layer: 0.0 for layer in ledger.layers}
                    and not ledger.receipts
                )
                if not failure_rollback_exact:
                    raise RuntimeError(
                        "unresolved continuation did not roll back exactly"
                    )
            else:
                successful_terminal_endpoint_exact = (
                    post_state_id == result.terminal_state_id
                    and len(ledger.receipts) == 1
                )
                if not successful_terminal_endpoint_exact:
                    raise RuntimeError(
                        "continuation terminal endpoint/Omega identity differs"
                    )

            direct_z_identity = dict(backend.direct_z_identity or {})
            del backend
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)

            record = {
                "schema_version": "ode-edit-p1-smax8-continuation-record/v1",
                "model": args.model_alias,
                "case_id": request.case_id,
                "source_order_position": trajectory.order_position,
                "arm": Arm.FULL_ODE_EDIT.value,
                "seed": seed,
                "status": result.status,
                "result": result_payload,
                "prefix": {
                    "source_fingerprint": trajectory.prefix_fingerprint,
                    "observed_fingerprint": observed_prefix,
                    "exact_match": True,
                    "timer_fields_included": False,
                    "raw_fields_included": False,
                },
                "round_7_or_8_first_hit": hit_round,
                "pre_edit_state_id": trajectory.pre_edit_state_id,
                "post_edit_state_id": post_state_id,
                "pre_edit_target_weight_state_id": (
                    trajectory.pre_edit_target_weight_state_id
                ),
                "post_edit_target_weight_state_id": post_target_weight_state_id,
                "Omega_after": ledger.state(),
                "history_length_after": len(ledger.receipts),
                "direct_z": {
                    **direct_z_identity,
                    "source_relative_path": trajectory.direct_z_relative_path,
                    "reused_read_only": True,
                    "recomputed": False,
                    "source_bytes_unchanged": True,
                },
                "integrity": {
                    "functional_commit_t_c": (
                        "PASS" if counters["N_write"] else "NOT_APPLICABLE"
                    ),
                    "functional_commit_checks": counters["N_write"],
                    "field_backward_identity": counters["N_bw"]
                    == counters["N_field"],
                    "failure_rollback_required": failure_rollback_required,
                    "failure_rollback_exact": failure_rollback_exact,
                    "successful_terminal_endpoint_exact": (
                        successful_terminal_endpoint_exact
                    ),
                    "baseline_restore_after_case": True,
                    "N_eval": 0,
                },
                "counters": counters,
                "compute_snapshot_id": snapshot["snapshot_id"],
                "D_sync_entry": d_sync_entry,
            }
            compute_record = {
                **snapshot,
                "schema_version": "ode-edit-p1-smax8-compute-record/v1",
                "model": args.model_alias,
                "case_id": request.case_id,
                "source_order_position": trajectory.order_position,
                "status": result.status,
                "D_sync_entry": d_sync_entry,
                **counters,
            }
            _jsonl_append(output_root / "continuation.jsonl", record)
            _jsonl_append(output_root / "compute.jsonl", compute_record)
            records.append(record)
    except BaseException:
        baseline.restore(runtime.model)
        baseline.assert_exact(runtime.model, include_rng=True)
        raise

    fixed_artifacts.assert_current()
    prepared.bridge.load().provenance.assert_current()
    terminal_selection = validate_r2_sources(repo, DEFAULT_R2_SOURCE_SPECS)
    terminal_source = terminal_selection.source_for_alias(args.model_alias)
    source_identity_after = {
        relative: {"sha256": identity.sha256, "size": identity.size}
        for relative, identity in terminal_source.file_identities
    }
    if source_identity_after != source_identity_before:
        raise RuntimeError("R2 source artifacts changed during continuation")

    promotion = promotion_decision(hit_rounds)
    summary = {
        "schema_version": "ode-edit-p1-smax8-continuation-summary/v1",
        "status": "COMPLETE_TECHNICAL_RESOLUTION_ONLY_NO_EVALUATION",
        "model": args.model_alias,
        "affected_case_ids": list(case_ids),
        "record_count": len(records),
        "prefix_exact_count": sum(
            record["prefix"]["exact_match"] for record in records
        ),
        "promotion": promotion,
        "source_artifacts_unchanged": True,
        "scientific_outcome_count": 0,
        "evaluation_count": 0,
    }
    _json_write(output_root / "summary.json", summary)
    manifest["status"] = "COMPLETE_TECHNICAL_RESOLUTION_ONLY"
    manifest["promotion"] = promotion
    manifest["source"]["terminal_files_unchanged"] = True
    _json_write(output_root / "manifest.json", manifest)
    terminal_files = (
        "manifest.json",
        "continuation.jsonl",
        "compute.jsonl",
        "evaluation.jsonl",
        "summary.json",
    )
    _json_write(
        output_root / "terminal_manifest.json",
        {
            "schema_version": "ode-edit-p1-smax8-terminal-manifest/v1",
            "status": "COMPLETE",
            "files": {
                relative: {
                    "sha256": file_sha256(output_root / relative),
                    "size": (output_root / relative).stat().st_size,
                }
                for relative in terminal_files
            },
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
