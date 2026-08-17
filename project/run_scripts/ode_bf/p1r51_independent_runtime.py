"""Independent P1R51 RSA-A1 B1, pilot, Neutral, and Soft runtime."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import BATCH_SIZE, COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import _model_w0_contract
from .p1_state import ArmWeightSnapshot
from .p1r36_independent_b10x10_runtime import (
    _case_failure,
    _hashes,
    _history_off_receipt,
    _restore_exact_w0,
    _run_ode_case,
)
from .p1r43_independent_b10x10_runtime import STREAM_ORDER, STREAM_ROOT
from .p1r51_requestwise_semantic_allocation import P1R51_INSTRUCTION_ID, P1R51_METHOD_ID


INSTRUCTION_ID = P1R51_INSTRUCTION_ID
PHASES = ("b1", "pilot-neutral", "neutral-full", "soft-full")
METHOD_BY_PHASE = {
    "b1": "P1R43-RSA-A1-NEUTRAL",
    "pilot-neutral": "P1R43-RSA-A1-NEUTRAL",
    "neutral-full": "P1R43-RSA-A1-NEUTRAL",
    "soft-full": "P1R43-RSA-A1-SOFT",
}
CASE_COUNT = 10
HISTORY_MODE = "OFF"


def expected_p1r51_result_name(
    alias: str,
    *,
    phase: str,
    attempt_suffix: str | None = None,
) -> str:
    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst") or phase not in PHASES:
        raise ODEBFContractError("P1R51 result identity differs")
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    if phase == "b1":
        return f"s05-p1r51-rsa-a1-b1-{alias}-neutral{suffix}-v1"
    if phase == "pilot-neutral":
        return f"s05-p1r51-rsa-a1-pilot-{alias}-neutral{suffix}-v1"
    arm = "neutral" if phase == "neutral-full" else "soft"
    return f"s05-p1r51-rsa-a1-independent-b10x10-{alias}-{arm}{suffix}-v1"


def run_p1r51_independent(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    phase: str,
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
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: Any,
    theta0_cache: Any,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    request_microbatch_size: int,
) -> dict[str, Any]:
    if phase not in PHASES or len(stream_batches) != CASE_COUNT:
        raise ODEBFContractError("P1R51 phase/matrix differs")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("P1R51 population is not ten B10 batches")
    if stream.get("root_digest") != STREAM_ROOT:
        raise ODEBFContractError("P1R51 frozen stream root differs")
    if stream.get("all_request_order_sha256") != STREAM_ORDER:
        raise ODEBFContractError("P1R51 frozen stream order differs")

    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P1R51 entry W0 differs")
    method = METHOD_BY_PHASE[phase]
    smoke = phase == "b1"
    selected_batches: Sequence[Sequence[Mapping[str, Any]]]
    if smoke:
        selected_batches = ((stream_batches[0][0],),)
    elif phase == "pilot-neutral":
        selected_batches = (stream_batches[0],)
    else:
        selected_batches = stream_batches

    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for case_index, requests in enumerate(selected_batches, start=1):
        case_root = raw_root / "cases" / f"case-{case_index:02d}"
        seed_all(COMMON_SEED)
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P1R51 cross-case W0 state leak detected")
        try:
            result = _run_ode_case(
                model,
                tokenizer,
                requests,
                alias=alias,
                method=method,
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
                p1r51=True,
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
                    method=method,
                    w0_restore=restore,
                    p1r51=True,
                )
            )
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P1R51 post-case W0 differs")
        stages.record(
            f"post_p1r51_{phase.replace('-', '_')}_case_{case_index}",
            {
                "phase": phase,
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
        "schema": "ode-edit-s05-p1r51-rsa-a1-independent-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": P1R51_METHOD_ID,
        "source_head": source_head,
        "alias": alias,
        "phase": phase,
        "method": method,
        "case_count": len(selected_batches),
        "request_attempt_count": 1 if smoke else len(selected_batches) * BATCH_SIZE,
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
        "target_controller_state_reset_count": len(selected_batches),
        "retry_count": 0,
        "rng_reset_count": len(selected_batches),
        "W0_restored": _model_w0_contract(touched) == expected_w0,
        "total_wall_seconds": time.perf_counter() - started,
        "job_compute": job_ledger.raw_free_payload(),
        "scientific_promotion": False,
        "technical_smoke": smoke,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-independent-manifest/v1",
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "phase": phase,
        "case_count": len(selected_batches),
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
        "history_mode": HISTORY_MODE,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R51_RSA_A1_PHASE_TERMINAL",
        "phase": phase,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


__all__ = [
    "CASE_COUNT",
    "INSTRUCTION_ID",
    "METHOD_BY_PHASE",
    "PHASES",
    "expected_p1r51_result_name",
    "run_p1r51_independent",
]
