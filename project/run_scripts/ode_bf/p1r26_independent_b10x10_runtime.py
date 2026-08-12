"""Ten independent whole-B10 paired ASDC experiments per model/allocation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .p1_common import COMMON_SEED, seed_all
from .p1_history import ArmWeightSnapshot
from .p1_runtime import ComputeLedger
from .p1_scalable_batched_experiment import _model_w0_contract, _run_ode_pair
from .p1r24_independent_b10x10_runtime import (
    _atomic_write_once,
    _case_failure,
    _hashes,
    _history_off_receipt,
    _restore_exact_w0,
)
from .p1r26_asdc import P1R26_INSTRUCTION_ID, P1R26_METHOD_ID


CASE_COUNT = 10
BATCH_SIZE = 10


def run_p1r26_independent_b10x10(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    allocation: str,
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
    if allocation not in ("RS", "BG") or len(stream_batches) != CASE_COUNT:
        raise ODEBFContractError("P1R26 independent matrix/count differs")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("P1R26 independent population is not ten B10 batches")
    if (
        stream.get("root_digest")
        != "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
        or stream.get("all_request_order_sha256")
        != "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
    ):
        raise ODEBFContractError("P1R26 frozen B10x10 stream differs")
    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P1R26 independent entry W0 differs")
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for case_index, requests in enumerate(stream_batches, start=1):
        case_root = raw_root / "cases" / f"case-{case_index:02d}"
        seed_all(COMMON_SEED)
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P1R26 cross-case W0 state leak detected")
        try:
            result = _run_ode_pair(
                model,
                tokenizer,
                alias=alias,
                destination=case_root,
                raw_root=case_root / "raw",
                source_head=source_head,
                requests=requests,
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
                touched=touched,
                base_receipt=base_receipt,
                base_values=base_values,
                job_ledger=job_ledger,
                write_once=_atomic_write_once,
                request_microbatch_size=request_microbatch_size,
                allocation=allocation,
                p1r26=True,
            )
            completed.append(
                {
                    "case_index": case_index,
                    "status": "CASE_COMPLETE",
                    "terminal_sha256": result["terminal_sha256"],
                    "manifest_sha256": result["manifest_sha256"],
                }
            )
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
                    method=f"{allocation}-P1R26-ASDC-PAIR",
                    w0_restore=restore,
                )
            )
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P1R26 independent post-case W0 differs")
        stages.record(
            f"post_p1r26_independent_b10_case_{case_index}",
            {
                "case_index": case_index,
                "allocation": allocation,
                "complete_count": len(completed),
                "failure_count": len(failed),
                "W0_restored": True,
                "history_mode": "OFF",
            },
        )
    terminal = {
        "schema": "ode-edit-s05-p1r26-asdc-independent-b10x10-terminal/v1",
        "instruction_id": P1R26_INSTRUCTION_ID,
        "method_id": P1R26_METHOD_ID,
        "source_head": source_head,
        "alias": alias,
        "allocation": allocation,
        "case_count": CASE_COUNT,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "completed": completed,
        "failed_case_identity_sha256": [item["identity_sha256"] for item in failed],
        "history_mode": _history_off_receipt(),
        "cross_case_state_count": 0,
        "retry_count": 0,
        "rng_reset_count": CASE_COUNT,
        "W0_restored": _model_w0_contract(touched) == expected_w0,
        "total_wall_seconds": time.perf_counter() - started,
        "job_compute": job_ledger.raw_free_payload(),
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r26-asdc-independent-b10x10-manifest/v1",
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "case_count": CASE_COUNT,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
        "history_mode": "OFF",
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R26_ASDC_INDEPENDENT_B10X10_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


__all__ = ["CASE_COUNT", "run_p1r26_independent_b10x10"]
