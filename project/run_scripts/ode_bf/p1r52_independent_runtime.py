"""Independent P1R52 R42-safe KDC full B10x10 cell runtime."""

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
from .p1r52_r42_safe_kdc import (
    P1R52_INSTRUCTION_ID,
    P1R52_METHOD_ID,
    P1R52_REPAIR_REASON,
    P1R52_REPAIR_REVISION,
    P1R52_SUPERSEDES_SOURCE_HEAD,
)
from .p1r52_target_depth import P1R52TargetDepth
from .p1r52_frozen_pi_quota_writer import (
    P1R52_FPIQ_INSTRUCTION_ID,
    P1R52_FPIQ_METHOD_ID,
    P1R52WriterPolicy,
)
from .p1r52_pir_writer import (
    P1R52_PIR_INSTRUCTION_ID,
    P1R52_PIR_METHOD_ID,
    P1R52PIRPolicy,
)


INSTRUCTION_ID = P1R52_INSTRUCTION_ID
ARMS = ("neutral", "soft")
FPIQ_POLICIES = tuple(item.value.lower() for item in P1R52WriterPolicy)
PIR_ARM_POLICY = {
    "pir-j0": P1R52PIRPolicy.J0,
    "pir-g": P1R52PIRPolicy.PIR_G,
    "pir-u": P1R52PIRPolicy.PIR_U,
}
PIR_POLICIES = tuple(PIR_ARM_POLICY)
CASE_COUNT = 10
HISTORY_MODE = "OFF"


def expected_p1r52_result_name(
    alias: str,
    arm: str,
    *,
    attempt_suffix: str | None = None,
    target_depth: P1R52TargetDepth | str | None = None,
) -> str:
    if (
        alias not in ("llama3-8b-inst", "qwen2.5-7b-inst")
        or arm not in (*ARMS, *FPIQ_POLICIES, *PIR_POLICIES)
        or (arm in (*FPIQ_POLICIES, *PIR_POLICIES) and alias != "llama3-8b-inst")
    ):
        raise ODEBFContractError("P1R52 result identity differs")
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    if target_depth is not None:
        depth = (
            target_depth
            if isinstance(target_depth, P1R52TargetDepth)
            else P1R52TargetDepth(target_depth)
        )
        if arm not in ARMS:
            raise ODEBFContractError("P1R52 target-depth Atomic writer differs")
        token = depth.value.lower().replace("-", "")
        return f"s05-p1r52-target-depth-atomic-b10x10-{alias}-{arm}-{token}{suffix}-v1"
    if arm in FPIQ_POLICIES:
        return f"s05-p1r52-fpiq-independent-b10x10-{alias}-{arm}{suffix}-v1"
    if arm in PIR_POLICIES:
        return f"s05-p1r52-pir-independent-b10x10-{alias}-{arm}{suffix}-v1"
    return f"s05-p1r52-rsa-r42safekdc-m1-independent-b10x10-{alias}-{arm}{suffix}-v1"


def run_p1r52_independent(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    arm: str,
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
    target_depth: P1R52TargetDepth | str | None = None,
) -> dict[str, Any]:
    if arm not in (*ARMS, *FPIQ_POLICIES, *PIR_POLICIES) or len(stream_batches) != CASE_COUNT:
        raise ODEBFContractError("P1R52 arm/matrix differs")
    depth_policy = (
        P1R52TargetDepth.IL1
        if target_depth is None
        else target_depth
        if isinstance(target_depth, P1R52TargetDepth)
        else P1R52TargetDepth(target_depth)
    )
    if target_depth is not None and arm not in ARMS:
        raise ODEBFContractError("P1R52 target-depth Atomic arm differs")
    writer_policy = (
        P1R52WriterPolicy(arm.upper()) if arm in FPIQ_POLICIES else None
    )
    pir_policy = PIR_ARM_POLICY.get(arm)
    if (writer_policy is not None or pir_policy is not None) and alias != "llama3-8b-inst":
        raise ODEBFContractError("P1R52 sequential writer is Llama-only")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("P1R52 population is not ten B10 batches")
    if stream.get("root_digest") != STREAM_ROOT or stream.get("all_request_order_sha256") != STREAM_ORDER:
        raise ODEBFContractError("P1R52 frozen stream identity differs")

    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P1R52 entry W0 differs")
    method = (
        f"P1R52-PIR-{pir_policy.value}"
        if pir_policy is not None
        else f"P1R52-FPIQ-{writer_policy.value}"
        if writer_policy is not None
        else f"P1R52-RSA-R42SAFEKDC-M1-{arm.upper()}"
    )
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for case_index, requests in enumerate(stream_batches, start=1):
        case_root = raw_root / "cases" / f"case-{case_index:02d}"
        seed_all(COMMON_SEED)
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P1R52 cross-case W0 state leak detected")
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
                p1r52=True,
                p1r52_target_depth=depth_policy.inner_count,
                p1r52_writer_policy=writer_policy,
                p1r52_pir_policy=pir_policy,
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
                    p1r52=True,
                )
            )
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P1R52 post-case W0 differs")
        stages.record(
            f"post_p1r52_{arm}_case_{case_index}",
            {
                "arm": arm,
                "case_index": case_index,
                "method": method,
                "complete_count": len(completed),
                "failure_count": len(failed),
                "W0_restored": True,
                "target_controller_state_reset_at_next_case": True,
                "history_mode": HISTORY_MODE,
                "target_depth_policy": depth_policy.value,
                "configured_inner_count": depth_policy.inner_count,
            },
        )

    terminal = {
        "schema": (
            "ode-edit-s05-p1r52-pir-independent-terminal/v1"
            if pir_policy is not None
            else "ode-edit-s05-p1r52-frozen-pi-quota-independent-terminal/v1"
            if writer_policy is not None
            else "ode-edit-s05-p1r52-rsa-r42safekdc-independent-terminal/v1"
        ),
        "instruction_id": (
            P1R52_PIR_INSTRUCTION_ID
            if pir_policy is not None
            else P1R52_FPIQ_INSTRUCTION_ID
            if writer_policy is not None
            else INSTRUCTION_ID
        ),
        "method_id": (
            P1R52_PIR_METHOD_ID
            if pir_policy is not None
            else P1R52_FPIQ_METHOD_ID
            if writer_policy is not None
            else P1R52_METHOD_ID
        ),
        "repair_revision": P1R52_REPAIR_REVISION,
        "repair_reason": P1R52_REPAIR_REASON,
        "supersedes_source_head": P1R52_SUPERSEDES_SOURCE_HEAD,
        "source_head": source_head,
        "alias": alias,
        "arm": arm,
        "writer_policy": (
            pir_policy.value
            if pir_policy is not None
            else writer_policy.value
            if writer_policy is not None
            else None
        ),
        "method": method,
        "target_depth_policy": depth_policy.value,
        "configured_inner_count": depth_policy.inner_count,
        "case_count": CASE_COUNT,
        "request_attempt_count": CASE_COUNT * BATCH_SIZE,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "scientific_failed_case_count": sum(item["classification"] == "SCIENTIFIC_FAIL" for item in failed),
        "technical_failed_case_count": sum(item["classification"] == "TECHNICAL_FAIL" for item in failed),
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
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": (
            "ode-edit-s05-p1r52-pir-independent-manifest/v1"
            if pir_policy is not None
            else "ode-edit-s05-p1r52-frozen-pi-quota-independent-manifest/v1"
            if writer_policy is not None
            else "ode-edit-s05-p1r52-rsa-r42safekdc-independent-manifest/v1"
        ),
        "source_head": source_head,
        "repair_revision": P1R52_REPAIR_REVISION,
        "supersedes_source_head": P1R52_SUPERSEDES_SOURCE_HEAD,
        "terminal_sha256": terminal_sha,
        "arm": arm,
        "target_depth_policy": depth_policy.value,
        "configured_inner_count": depth_policy.inner_count,
        "case_count": CASE_COUNT,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
        "history_mode": HISTORY_MODE,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": (
            "P1R52_PIR_CELL_TERMINAL"
            if pir_policy is not None
            else "P1R52_FPIQ_CELL_TERMINAL"
            if writer_policy is not None
            else "P1R52_RSA_R42SAFEKDC_M1_CELL_TERMINAL"
        ),
        "arm": arm,
        "target_depth_policy": depth_policy.value,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


__all__ = [
    "ARMS",
    "CASE_COUNT",
    "FPIQ_POLICIES",
    "PIR_ARM_POLICY",
    "PIR_POLICIES",
    "INSTRUCTION_ID",
    "expected_p1r52_result_name",
    "run_p1r52_independent",
]
