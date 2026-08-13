"""Ten independent B10 P1R38 Atomic cases for one model/router cell."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import BATCH_SIZE, COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1_runtime import _atomic_write_once
from .p1_state import ArmWeightSnapshot
from .p1_scalable_batched_experiment import _model_w0_contract
from .p1r36_independent_b10x10_runtime import (
    _case_failure,
    _hashes,
    _history_off_receipt,
    _restore_exact_w0,
    _run_ode_case,
)
from .p1r38_perrequest_target import P1R38_INSTRUCTION_ID, P1R38_METHOD_ID


INSTRUCTION_ID = P1R38_INSTRUCTION_ID
METHODS = ("PR-P1R35-NEUTRAL", "PR-P1R35-SOFT")
B1_METHODS = ("PR-P1R35-B1-NEUTRAL", "PR-P1R35-B1-SOFT")
CASE_COUNT = 10
HISTORY_MODE = "OFF"


def expected_p1r38_independent_result_name(alias: str, method: str) -> str:
    if method not in METHODS + B1_METHODS:
        raise ODEBFContractError("P1R38 independent method differs")
    arm = method.rsplit("-", 1)[-1].lower()
    if method in B1_METHODS:
        return f"s05-p1r38-pr-p1r35-b1-{alias}-{arm}-v1"
    return f"s05-p1r38-pr-p1r35-independent-b10x10-{alias}-{arm}-v1"


def run_p1r38_independent_b10x10(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    method: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    collision_by_request: Mapping[str, str],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: Any,
    theta0_cache: Any,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    artifact_guard: Any,
    artifact_receipt: Any,
    numerical_sha256: str,
    context_sha256: str,
    cuda_runtime_receipt: Mapping[str, Any],
    job_ledger: ComputeLedger,
    request_microbatch_size: int,
) -> dict[str, Any]:
    del collision_by_request, artifact_guard, artifact_receipt, numerical_sha256, context_sha256, cuda_runtime_receipt
    smoke = method in B1_METHODS
    if method not in METHODS + B1_METHODS or len(stream_batches) != CASE_COUNT:
        raise ODEBFContractError("P1R38 independent B10 matrix/count differs")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("P1R38 population is not ten B10 batches")
    if stream.get("root_digest") != "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6":
        raise ODEBFContractError("P1R38 frozen stream root differs")
    if stream.get("all_request_order_sha256") != "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c":
        raise ODEBFContractError("P1R38 frozen order differs")
    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P1R38 entry W0 differs")

    executed_method = method.replace("-B1-", "-") if smoke else method
    executed_batches: Sequence[Sequence[Mapping[str, Any]]] = (
        ((stream_batches[0][0],),) if smoke else stream_batches
    )
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for case_index, requests in enumerate(executed_batches, start=1):
        case_root = raw_root / "cases" / f"case-{case_index:02d}"
        seed_all(COMMON_SEED)
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P1R38 cross-case W0 state leak detected")
        try:
            result = _run_ode_case(
                model,
                tokenizer,
                requests,
                alias=alias,
                method=executed_method,
                case_index=case_index,
                case_root=case_root,
                hparams=hparams,
                projector=projector,
                contexts=contexts,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                controller_lock=controller_lock,
                request_by_sha256=request_by_sha256,
                population_by_sha256=population_by_sha256,
                schedule=schedule,
                theta0_cache=theta0_cache,
                dataset_path=dataset_path,
                mutation_lock=mutation_lock,
                touched=touched,
                base_receipt=base_receipt,
                base_values=base_values,
                request_microbatch_size=request_microbatch_size,
                job_ledger=job_ledger,
                p1r38=True,
                technical_smoke=smoke,
            )
            completed.append({"case_index": case_index, **result})
        except Exception as exc:
            restore = _restore_exact_w0(
                touched,
                base_values,
                mutation_lock=mutation_lock,
                expected_contract=expected_w0,
            )
            failed.append(
                _case_failure(
                    case_root,
                    exc,
                    case_index=case_index,
                    method=executed_method,
                    w0_restore=restore,
                    p1r38=True,
                )
            )
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P1R38 post-case W0 differs")
        stages.record(
            f"post_p1r38_independent_b10_case_{case_index}",
            {
                "case_index": case_index,
                "method": method,
                "complete_count": len(completed),
                "failure_count": len(failed),
                "W0_restored": True,
                "target_controller_state_reset_at_next_case": True,
                "history_mode": HISTORY_MODE,
            },
        )

    terminal = {
        "schema": "ode-edit-s05-p1r38-perrequest-independent-b10x10-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": P1R38_METHOD_ID,
        "source_head": source_head,
        "alias": alias,
        "method": method,
        "allocation_factorial": "NONE",
        "case_count": 1 if smoke else CASE_COUNT,
        "request_attempt_count": 1 if smoke else CASE_COUNT * BATCH_SIZE,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "scientific_failed_case_count": sum(
            item["classification"] == "SCIENTIFIC_FAIL" for item in failed
        ),
        "technical_failed_case_count": sum(
            item["classification"] == "TECHNICAL_FAIL" for item in failed
        ),
        "completed": completed,
        "failed_case_identity_sha256": [item["identity_sha256"] for item in failed],
        "history_mode": _history_off_receipt(),
        "cross_case_state_count": 0,
        "target_controller_state_reset_count": CASE_COUNT,
        "retry_count": 0,
        "rng_reset_count": CASE_COUNT,
        "W0_restored": _model_w0_contract(touched) == expected_w0,
        "total_wall_seconds": time.perf_counter() - started,
        "job_compute": job_ledger.raw_free_payload(),
        "scientific_promotion": False,
        "technical_smoke": smoke,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r38-perrequest-independent-b10x10-manifest/v1",
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "case_count": 1 if smoke else CASE_COUNT,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
        "history_mode": HISTORY_MODE,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R38_PR_P1R35_INDEPENDENT_B10X10_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


__all__ = [
    "CASE_COUNT",
    "B1_METHODS",
    "INSTRUCTION_ID",
    "METHODS",
    "expected_p1r38_independent_result_name",
    "run_p1r38_independent_b10x10",
]
