"""P2R6 Phase-1/Phase-2 Atomic pilot runtime.

Only the model-free allocation policy is injected.  The target, response,
factor, materializer, W0 transaction, action freeze, and terminal evaluator are
the shared P2R5 runtime path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import BATCH_SIZE, COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .p1_backend import PinnedCovarianceRegistry
from .p1_controller import P1ControllerLock
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import _model_w0_contract
from .p1_state import ArmWeightSnapshot
from .p1r36_independent_b10x10_runtime import _hashes, _restore_exact_w0
from .p2r5_stage_a_runtime import (
    P2AtomicArmRuntimePolicy,
    P2R4_PARENT_HEAD,
    STAGE_A_CASES,
    STREAM_ORDER,
    STREAM_ROOT,
    run_p2_atomic_arm_case,
)
from .p2r6_semantic_region_controller import (
    P2R6_ARMS,
    P2R6_CAP_ARMS,
    P2R6_INSTRUCTION_ID,
    P2R6_METHOD_ID,
    p2r6_forbidden_influence_receipt,
    solve_p2r6_routing,
    solve_p2r6_shadow_panel,
)


METHOD = "P2R6-LLAMA-QP-REPAIR-PHASE1-SEMANTIC-REGION-CONTROLLER"
PHASE1_CASES = {
    "llama3-8b-inst": (5,),
    "qwen2.5-7b-inst": (1,),
}
PHASE2_CASES = STAGE_A_CASES
P2R6_RUNTIME_POLICY = P2AtomicArmRuntimePolicy(
    instruction_id=P2R6_INSTRUCTION_ID,
    method_id=P2R6_METHOD_ID,
    receipt_namespace="p2r6-red-r2-final-semantic-region",
    arm_label_prefix="P2R6",
    allowed_arms=P2R6_ARMS,
    allowed_cases=PHASE2_CASES,
    route_solver=solve_p2r6_routing,
    forbidden_receipt_builder=p2r6_forbidden_influence_receipt,
    shadow_solver=solve_p2r6_shadow_panel,
)


class P2R6PhaseTechnicalInvalid(ODEBFStateError):
    """Fail the paired phase immediately after one arm fails technically."""

    def __init__(self, arm_failure: Mapping[str, Any]) -> None:
        self.raw_free_receipt = {
            "schema": "ode-edit-s05-p2r6-phase-technical-invalid/v1",
            "instruction_id": P2R6_INSTRUCTION_ID,
            "classification": "TECHNICAL_INVALID",
            "failed_arm": arm_failure["arm"],
            "phase": arm_failure["phase"],
            "alias": arm_failure["alias"],
            "case_index": arm_failure["case_index"],
            "arm_failure_identity_sha256": arm_failure["identity_sha256"],
            "W0_restore": arm_failure["W0_restore"],
            "subsequent_arm_execution_count": 0,
        }
        self.raw_free_receipt["identity_sha256"] = canonical_hash(
            self.raw_free_receipt
        )
        super().__init__("P2R6 strict phase technical gate failed")


def _arm_compute_aggregation(
    completed: Sequence[Mapping[str, Any]], job_ledger: ComputeLedger
) -> dict[str, Any]:
    keys = (
        "target_forward_count",
        "target_backward_count",
        "kl_forward_count",
        "kl_backward_count",
        "physical_capture_forward_count",
        "physical_response_forward_count",
        "physical_response_batched_vjp_count",
        "post_write_objective_forward_count",
        "writer_materialization_count",
        "routing_qp_solve_count",
        "routing_qp_certificate_count",
        "shadow_technical_invalid_count",
        "completed_k_count",
    )
    totals = {
        key: sum(int(item.get("arm_compute", {}).get(key, 0)) for item in completed)
        for key in keys
    }
    ledger = job_ledger.raw_free_payload()
    counters = ledger["counters"]
    checks = {
        "completed_k_equal": totals["completed_k_count"]
        == ledger["completed_k_total"],
        "qp_solve_equal": totals["routing_qp_solve_count"]
        == counters["qp_solve"],
        "qp_certificate_equal": totals["routing_qp_certificate_count"]
        == counters["qp_certificate"],
        "target_backward_equal": (
            totals["target_backward_count"] + totals["kl_backward_count"]
        )
        == counters["target_backward"],
        "model_forward_nonzero_when_completed": (
            not completed or counters["model_forward"] > 0
        ),
        "backward_nonzero_when_completed": (
            not completed or counters["backward"] > 0
        ),
    }
    payload = {
        "schema": "ode-edit-s05-p2r6-arm-top-level-compute-aggregation/v1",
        "completed_arm_count": len(completed),
        "arm_totals": totals,
        "top_level_completed_k_total": ledger["completed_k_total"],
        "top_level_counters": counters,
        "top_level_component_wall_seconds": ledger["component_wall_seconds"],
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def p2r6_phase_arms(phase: str, selected_controller: str | None = None) -> tuple[str, ...]:
    if phase == "phase1":
        if selected_controller is not None:
            raise ODEBFContractError("P2R6 Phase1 selection must be absent")
        return P2R6_CAP_ARMS
    if phase == "phase2" and selected_controller in ("AR", "AS"):
        return (
            "A0-CAP",
            f"{selected_controller}-CAP",
            f"{selected_controller}-STRUCTP",
        )
    raise ODEBFContractError("P2R6 phase/controller selection differs")


def expected_p2r6_result_name(
    alias: str,
    *,
    phase: str,
    case_index: int,
    selected_controller: str | None = None,
    attempt_suffix: str | None = None,
) -> str:
    cases = PHASE1_CASES if phase == "phase1" else PHASE2_CASES
    if alias not in cases or case_index not in cases[alias]:
        raise ODEBFContractError("P2R6 result case differs")
    p2r6_phase_arms(phase, selected_controller)
    selection = f"-{selected_controller.lower()}" if selected_controller else ""
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    return (
        f"s05-p2r6-red-r2-final-{phase}{selection}-{alias}-"
        f"case-{case_index:02d}-paired{suffix}-v1"
    )


def run_p2r6_pilot_case(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    controller_lock: P1ControllerLock,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    request_microbatch_size: int,
    phase: str,
    case_index: int,
    selected_controller: str | None = None,
) -> dict[str, Any]:
    cases = PHASE1_CASES if phase == "phase1" else PHASE2_CASES
    arms = p2r6_phase_arms(phase, selected_controller)
    if (
        alias not in cases
        or case_index not in cases[alias]
        or len(stream_batches) != 10
        or any(len(batch) != BATCH_SIZE for batch in stream_batches)
    ):
        raise ODEBFContractError("P2R6 case/stream inventory differs")
    if (
        stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
    ):
        raise ODEBFContractError("P2R6 stream identity differs")
    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P2R6 entry W0 differs")
    requests = stream_batches[case_index - 1]
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for arm in arms:
        seed_all(COMMON_SEED)
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P2R6 cross-arm W0 leak detected")
        case_root = raw_root / "cases" / f"case-{case_index:02d}" / arm.lower()
        try:
            completed.append(
                run_p2_atomic_arm_case(
                    model,
                    tokenizer,
                    requests,
                    alias=alias,
                    arm=arm,
                    case_index=case_index,
                    case_root=case_root,
                    hparams=hparams,
                    projector=projector,
                    contexts=contexts,
                    covariance_registry=covariance_registry,
                    projector_sha256=projector_sha256,
                    controller_lock=controller_lock,
                    dataset_path=dataset_path,
                    mutation_lock=mutation_lock,
                    touched=touched,
                    base_values=base_values,
                    expected_w0=expected_w0,
                    request_microbatch_size=request_microbatch_size,
                    job_ledger=job_ledger,
                    runtime_policy=P2R6_RUNTIME_POLICY,
                )
            )
        except Exception as exc:
            restore = _restore_exact_w0(
                touched,
                base_values,
                mutation_lock=mutation_lock,
                expected_contract=expected_w0,
            )
            failure = {
                "schema": "ode-edit-s05-p2r6-pilot-case-failure/v1",
                "instruction_id": P2R6_INSTRUCTION_ID,
                "phase": phase,
                "alias": alias,
                "arm": arm,
                "case_index": case_index,
                "classification": "TECHNICAL_FAIL",
                "exception_class": type(exc).__name__,
                "exception_message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                "retry_count": 0,
                "W0_restore": restore,
                "paired_case_requires_full_final_head_rerun": True,
            }
            observability = getattr(exc, "raw_free_receipt", None)
            if isinstance(observability, Mapping):
                failure["technical_observability"] = dict(observability)
            failure["technical_location"] = [
                {
                    "file_identity_sha256": hashlib.sha256(frame.filename.encode()).hexdigest(),
                    "line_number": frame.lineno,
                    "symbol": frame.name,
                }
                for frame in traceback.extract_tb(exc.__traceback__)[-4:]
            ]
            failure["identity_sha256"] = canonical_hash(failure)
            _atomic_write_once(case_root / "failure.json", failure)
            failed.append(failure)
            raise P2R6PhaseTechnicalInvalid(failure) from exc
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P2R6 post-arm W0 differs")
        stages.record(
            f"post_p2r6_{phase}_case_{case_index}_{arm.lower()}",
            {
                "phase": phase,
                "case_index": case_index,
                "arm": arm,
                "complete_count": len(completed),
                "failure_count": len(failed),
                "W0_restored": True,
            },
        )
    compute_aggregation = _arm_compute_aggregation(completed, job_ledger)
    if not failed and not compute_aggregation["all_checks_pass"]:
        raise ODEBFStateError("P2R6 arm/top-level compute aggregation differs")
    terminal = {
        "schema": "ode-edit-s05-p2r6-pilot-job-terminal/v1",
        "instruction_id": P2R6_INSTRUCTION_ID,
        "method_id": P2R6_METHOD_ID,
        "p2r5_scientific_parent": "ff032a10eac257d23c691896458159d2f4eff23b",
        "p2r4_parent_head": P2R4_PARENT_HEAD,
        "source_head": source_head,
        "phase": phase,
        "selected_controller": selected_controller or "NOT_SELECTED_PHASE1",
        "alias": alias,
        "case_index": case_index,
        "arms": list(arms),
        "attempt_count": len(arms),
        "request_attempt_count": BATCH_SIZE * len(arms),
        "completed_case_arm_count": len(completed),
        "failed_case_arm_count": len(failed),
        "completed": completed,
        "failed": failed,
        "paired_case_valid": len(completed) == len(arms) and not failed,
        "W0_restored": _model_w0_contract(touched) == expected_w0,
        "job_compute": job_ledger.raw_free_payload(),
        "arm_top_level_compute_aggregation": compute_aggregation,
        "total_wall_seconds": time.perf_counter() - started,
        "stage_b10x10_status": "NOT_AUTHORIZED",
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p2r6-pilot-job-manifest/v1",
        "source_head": source_head,
        "phase": phase,
        "alias": alias,
        "case_index": case_index,
        "arms": list(arms),
        "terminal_sha256": terminal_sha,
        "completed_case_arm_count": len(completed),
        "failed_case_arm_count": len(failed),
        "paired_case_valid": terminal["paired_case_valid"],
        "W0_restored": terminal["W0_restored"],
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P2R6_PILOT_CASE_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_arm_count": len(completed),
        "failed_case_arm_count": len(failed),
        "paired_case_valid": terminal["paired_case_valid"],
        "W0_restored": terminal["W0_restored"],
    }


__all__ = [
    "METHOD",
    "PHASE1_CASES",
    "PHASE2_CASES",
    "P2R6_RUNTIME_POLICY",
    "P2R6PhaseTechnicalInvalid",
    "expected_p2r6_result_name",
    "p2r6_phase_arms",
    "run_p2r6_pilot_case",
]
