"""Independent P2R7 pilot and B10x10 runtime."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
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
from .p2r2_atomic_runtime import _run_arm_case
from .p2r7_shared_writer import (
    P2R7_INSTRUCTION_ID,
    P2R7_METHOD_ID,
    P2R7SharedWriterNoPositiveDirection,
)


STREAM_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
STREAM_ORDER = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
PILOT_ARMS = (
    ("P1DW", "NEUTRAL"),
    ("P1DW", "SOFT"),
    ("P1AGG", "NEUTRAL"),
    ("P1AGG", "SOFT"),
)
PRODUCTION_ARMS = (("P1DW", "NEUTRAL"), ("P1DW", "SOFT"))


def expected_p2r7_result_name(
    alias: str,
    *,
    phase: str,
    attempt_suffix: str | None = None,
) -> str:
    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst") or phase not in (
        "pilot",
        "b10x10",
    ):
        raise ODEBFContractError("P2R7 result identity differs")
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    return f"s05-p2r7-p2target-p1dw-{phase}-{alias}-paired{suffix}-v1"


def run_p2r7_atomic(
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
) -> dict[str, Any]:
    if phase not in ("pilot", "b10x10") or len(stream_batches) != 10:
        raise ODEBFContractError("P2R7 phase/case inventory differs")
    if (
        stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
        or any(len(batch) != BATCH_SIZE for batch in stream_batches)
    ):
        raise ODEBFContractError("P2R7 frozen stream identity differs")
    arms = PILOT_ARMS if phase == "pilot" else PRODUCTION_ARMS
    batches = stream_batches[:1] if phase == "pilot" else stream_batches
    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P2R7 entry W0 differs")
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case_index, requests in enumerate(batches, start=1):
        for mode, arm in arms:
            seed_all(COMMON_SEED)
            if _model_w0_contract(touched) != expected_w0:
                raise ODEBFStateError("P2R7 cross-arm/case W0 leak detected")
            case_root = raw_root / "cases" / f"case-{case_index:02d}" / f"{mode.lower()}-{arm.lower()}"
            try:
                completed.append(
                    {
                        "weighting_mode": mode,
                        **_run_arm_case(
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
                            p2r7_mode=mode,
                        ),
                    }
                )
            except Exception as exc:
                restore = _restore_exact_w0(
                    touched,
                    base_values,
                    mutation_lock=mutation_lock,
                    expected_contract=expected_w0,
                )
                scientific = isinstance(exc, P2R7SharedWriterNoPositiveDirection)
                failure = {
                    "schema": "ode-edit-s05-p2r7-case-failure/v1",
                    "alias": alias,
                    "weighting_mode": mode,
                    "arm": arm,
                    "case_index": case_index,
                    "classification": (
                        "SCIENTIFIC_SHARED_WRITER_NO_POSITIVE_DIRECTION"
                        if scientific
                        else "TECHNICAL_INVALID"
                    ),
                    "exception_class": type(exc).__name__,
                    "exception_message_sha256": hashlib.sha256(
                        str(exc).encode()
                    ).hexdigest(),
                    "retry_count": 0,
                    "W0_restore": restore,
                    "next_arm_or_case_continues": True,
                }
                failure["identity_sha256"] = canonical_hash(failure)
                _atomic_write_once(case_root / "failure.json", failure)
                failed.append(failure)
            if _model_w0_contract(touched) != expected_w0:
                raise ODEBFStateError("P2R7 post-arm W0 differs")
            stages.record(
                f"post_p2r7_{phase}_case_{case_index}_{mode.lower()}_{arm.lower()}",
                {
                    "case_index": case_index,
                    "weighting_mode": mode,
                    "arm": arm,
                    "complete_count": len(completed),
                    "failure_count": len(failed),
                    "W0_restored": True,
                },
            )
    terminal = {
        "schema": "ode-edit-s05-p2r7-job-terminal/v1",
        "instruction_id": P2R7_INSTRUCTION_ID,
        "method_id": P2R7_METHOD_ID,
        "source_head": source_head,
        "alias": alias,
        "phase": phase,
        "case_count": len(batches),
        "arm_inventory": [f"{mode}-{arm}" for mode, arm in arms],
        "attempt_count": len(batches) * len(arms),
        "completed_endpoint_count": len(completed),
        "failed_endpoint_count": len(failed),
        "completed": completed,
        "failed": failed,
        "W0_restored": _model_w0_contract(touched) == expected_w0,
        "wall_seconds": time.perf_counter() - started,
        "routing_variable_count": 5,
        "request_layer_response_matrix_count": 0,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination, terminal)
    manifest = {
        "schema": "ode-edit-s05-p2r7-job-manifest/v1",
        "terminal_sha256": terminal_sha,
        "source_head": source_head,
        "phase": phase,
        "attempt_count": terminal["attempt_count"],
        "completed_endpoint_count": len(completed),
        "failed_endpoint_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination.with_suffix(".manifest.json"), manifest)
    return {**terminal, "terminal_sha256": terminal_sha, "manifest_sha256": manifest_sha}


__all__ = [
    "PILOT_ARMS",
    "PRODUCTION_ARMS",
    "STREAM_ORDER",
    "STREAM_ROOT",
    "expected_p2r7_result_name",
    "run_p2r7_atomic",
]
