"""Failure-safe Llama/Qwen joint-B1 controller-validity cell runtime."""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from project.run_scripts.fixed_z_nonuniqueness.contracts import (
    MODEL_SPECS,
    Method,
    NumericalLock as PaddingLock,
)
from project.run_scripts.fixed_z_nonuniqueness.evaluation import padding_safety_gate
from project.run_scripts.fixed_z_nonuniqueness.experiment import _load_hparams
from project.run_scripts.fixed_z_nonuniqueness.padding import OfficialTokenizerHook, bind_padding

from .contracts import (
    Arm,
    FzCBEditError,
    NumericalLock,
    NumericalSensitivityFailure,
    ScientificBoundary,
    SubspaceInconclusive,
    TechnicalBoundary,
)
from .controller_validity import ArmExecutionFailure, run_validity_arm
from .data import load_sealed_rows, official_requests
from .evaluation import evaluate_rows
from .geometry import MEMITGeometryFactory, registered_batch
from .hashing import canonical_hash, tensor_sha256, write_json_once
from .journal import ArmJournal
from .runtime import _cache_identity
from .target import capture_official_memit_once, materialize_official, restore_w0
from .tolerances import UnitTolerancePolicy, synthetic_calibration_receipt
from .transaction import WeightSnapshot


ARMS = (
    Arm.OFFICIAL_MEMIT,
    Arm.TRUE_FROZEN_C_SPLIT,
    Arm.REFRESHED_EQUALITY_ONLY,
    Arm.FZCB,
    Arm.STRONG_STATIC_SAME_OBJECTIVE,
)


def _padding_gate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], hparams: Any) -> dict[str, Any]:
    selected = rows if len(rows) >= 3 else (rows * 3)[:3]
    prompts = [row["requested_rewrite"]["prompt"].format(row["requested_rewrite"]["subject"]) for row in selected]
    return padding_safety_gate(
        model,
        tokenizer,
        prompts,
        [row["requested_rewrite"]["target_new"]["str"] for row in selected],
        hparams.layer_module_tmp.format(hparams.layers[-1]),
        PaddingLock(),
        input_module=hparams.rewrite_module_tmp.format(hparams.layers[-1]),
        subject_templates=[row["requested_rewrite"]["prompt"] for row in selected],
        subjects=[row["requested_rewrite"]["subject"] for row in selected],
    )


def _failure_status(exc: BaseException) -> str:
    if isinstance(exc, SubspaceInconclusive):
        return "SUBSPACE_INCONCLUSIVE"
    if isinstance(exc, NumericalSensitivityFailure):
        return "NUMERICAL_CONTROLLER_FAILURE"
    if isinstance(exc, ScientificBoundary):
        return "SCIENTIFIC_CONTROLLER_HOLD"
    return "TECHNICAL_BLOCKED"


def _failure_payload(
    *,
    model_alias: str,
    case_ids: tuple[int, ...],
    arm: Arm,
    exc: BaseException,
    policy: UnitTolerancePolicy,
    target_receipt: dict[str, Any],
    rollback: dict[str, Any],
    cache_rollback: dict[str, Any],
) -> dict[str, Any]:
    embedded = dict(exc.receipt) if isinstance(exc, ArmExecutionFailure) else {}
    payload: dict[str, Any] = {
        "status": _failure_status(exc.__cause__ if isinstance(exc, ArmExecutionFailure) and exc.__cause__ else exc),
        "denominator": {"valid_terminal": 0, "attempted_arm": 1},
        "model": model_alias,
        "case_ids": list(case_ids),
        "arm": arm.value,
        "stage": embedded.get("stage", "arm_execution"),
        "s": embedded.get("s", "NOT_RECORDED"),
        "proposed_step": embedded.get("proposed_step", "NOT_RECORDED"),
        "A0": embedded.get("A0", "NOT_RECORDED"),
        "E": embedded.get("E", embedded.get("spent_action", "NOT_RECORDED")),
        "predicted_suffix": embedded.get("predicted_suffix", "NOT_RECORDED"),
        "h_cc": embedded.get("h_cc", "NOT_RECORDED"),
        "psi0": embedded.get("psi0", embedded.get("rectification", {}).get("psi0", "NOT_RECORDED")),
        "psi_min": embedded.get("psi_min", embedded.get("rectification", {}).get("psi_min", "NOT_RECORDED")),
        "g_eq": embedded.get("g_eq", "NOT_RECORDED"),
        "partial_s_suffix": embedded.get("partial_s_suffix", "NOT_RECORDED"),
        "g_free_norm_squared": embedded.get(
            "g_free_norm_squared",
            embedded.get("rectification", {}).get("raw_sketch_g_free_norm_squared", "NOT_RECORDED"),
        ),
        "dimensions": {
            "coefficient": embedded.get("coefficient_dimension", "NOT_RECORDED"),
            "output": embedded.get("output_dimension", "NOT_RECORDED"),
            "null": embedded.get("null_dimension", "NOT_RECORDED"),
            "effective": embedded.get("sketch", {}).get("null_dimension", "NOT_RECORDED"),
        },
        "rank_spectral_range_kkt_cg": {
            "rank": embedded.get("rank", "NOT_RECORDED"),
            "spectral": embedded.get("spectral", "NOT_RECORDED"),
            "range_residual": embedded.get("range_residual", "NOT_RECORDED"),
            "kkt_residual": embedded.get("kkt_residual", "NOT_RECORDED"),
            "cg": embedded.get("solver", "NOT_RECORDED"),
        },
        "fd": {
            "equality": embedded.get("equality_direction_fd", "NOT_RECORDED"),
            "axes": embedded.get("axis_fd_sweeps", "NOT_RECORDED"),
            "sketch": embedded.get("sketch", "NOT_RECORDED"),
        },
        "tolerance_policy": policy.receipt(),
        "tolerances": embedded.get("tolerances", "NOT_RECORDED"),
        "target_context_operator_hashes": target_receipt,
        "rollback": rollback,
        "cache_rollback": cache_rollback,
        "accepted_count": embedded.get("accepted_count", 0),
        "rejected_count": embedded.get("rejected_count", 0),
        "backtrack_count": embedded.get("backtrack_count", 0),
        "exception_type": type(exc).__name__,
        "root_exception_type": type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
        "evidence": embedded,
        "scientific_promotion": False,
    }
    return payload


def _barrier_summary(arm_payload: dict[str, Any]) -> dict[str, Any]:
    waypoints = arm_payload.get("waypoints", [])
    barriers = [row.get("barrier") for row in waypoints if row.get("barrier")]
    active = [row for row in barriers if row.get("rectification", {}).get("barrier_active")]
    hcc = [
        float(row["strict_barrier_verification"]["h_cc"])
        for row in waypoints
        if row.get("strict_barrier_verification")
    ]
    return {
        "barrier_observation_count": len(barriers),
        "barrier_active_count": len(active),
        "minimum_waypoint_h_cc": min(hcc) if hcc else "NOT_RECORDED",
        "sensitivity_authorities": sorted({str(row.get("sensitivity_method")) for row in barriers}),
        "full_projected_sensitivity_count": sum(
            row.get("sensitivity_method") == "FULL_PROJECTED_SENSITIVITY" for row in barriers
        ),
        "sketch_only_count": sum(row.get("sensitivity_method") == NumericalLock().sensitivity_method for row in barriers),
    }


def _typed_conclusion(journal_payloads: list[dict[str, Any]]) -> str:
    by_arm = {row["arm"]: row for row in journal_payloads}
    fzcb = by_arm.get(Arm.FZCB.value, {})
    if fzcb.get("status") == "NUMERICAL_CONTROLLER_FAILURE":
        return "NUMERICAL_CONTROLLER_FAILURE"
    if fzcb.get("status") == "SUBSPACE_INCONCLUSIVE":
        return "SUBSPACE_INCONCLUSIVE"
    if fzcb.get("status") not in {"TERMINAL_VALID_STRICT", "TERMINAL_VALID_STRICT_SKETCH_ONLY"}:
        return "SUBSPACE_INCONCLUSIVE" if fzcb.get("evidence", {}).get("sketch") else "NUMERICAL_CONTROLLER_FAILURE"
    summary = fzcb.get("barrier_summary", {})
    if int(summary.get("barrier_active_count", 0)) == 0:
        equality = by_arm.get(Arm.REFRESHED_EQUALITY_ONLY.value, {})
        if fzcb.get("terminal", {}).get("weight_root") == equality.get("terminal", {}).get("weight_root"):
            return "NO_BARRIER_USEFULNESS_EVIDENCE"
    if int(summary.get("sketch_only_count", 0)) > 0:
        return "SUBSPACE_INCONCLUSIVE"
    return "TERMINAL_VALID_STRICT"


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.cell_root.exists() or args.cell_root.is_symlink():
        raise TechnicalBoundary(f"refusing to reuse TECH-R1 cell root: {args.cell_root}")
    args.cell_root.mkdir(parents=True, mode=0o700)
    rows, stream_receipt = load_sealed_rows(1)
    case_ids = tuple(int(row["case_id"]) for row in rows)
    journal = ArmJournal.create(args.cell_root / "arm-journal", args.model, case_ids)
    requests = official_requests(rows)
    spec = MODEL_SPECS[args.model]
    policy = UnitTolerancePolicy.load()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        trust_remote_code=False,
    )
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TechnicalBoundary("FULL_FP32 model closure failed")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_path, local_files_only=True, use_fast=True, trust_remote_code=False,
    )
    padding_binding = bind_padding(tokenizer, model, padding_side="right")
    hparams = _load_hparams(Method.MEMIT, args.source_root / spec.memit_hparams)
    if str(hparams.model_name) != spec.statistics_model_dir:
        raise TechnicalBoundary("MEMIT statistics alias mismatch")
    padding_gate = _padding_gate(model, tokenizer, rows, hparams)
    tokenizer_hook = OfficialTokenizerHook(tokenizer, [])
    frozen = capture_official_memit_once(
        model=model, tokenizer=tokenizer_hook, requests=requests, hparams=hparams,
    )
    if frozen.direct_z_calls != 1 or any(value != 1 for value in frozen.direct_z_calls_by_case.values()):
        raise ScientificBoundary("direct-z exactly-once/edit gate failed")
    if not frozen.tokenizer_calls or not all(row["semantic_input_attention_position_equal"] for row in frozen.tokenizer_calls):
        raise TechnicalBoundary("Official/hook token-position identity receipt missing")
    restore_w0(model, frozen)
    w0 = WeightSnapshot.capture(model, tuple(frozen.originals))
    pre_edit = evaluate_rows(model, tokenizer, rows)
    batch, batch_receipt = registered_batch(tokenizer, rows, hparams, next(model.parameters()).device)
    factory = MEMITGeometryFactory(model, tokenizer, rows, requests, hparams, batch)
    entry_operator, entry_geometry = factory.build()
    cache_entry = _cache_identity()
    target = frozen.targets.to(next(model.parameters()).device, torch.float32).reshape(-1)
    phi0 = entry_operator.phi()
    delta = target - phi0
    target_receipt = {
        "target_root": frozen.target_root,
        "target_sha256": tensor_sha256(target),
        "phi0_sha256": tensor_sha256(phi0),
        "delta_sha256": tensor_sha256(delta),
        "a0_norm": float(torch.linalg.vector_norm(phi0).item()),
        "delta_star_norm": float(torch.linalg.vector_norm(delta).item()),
        "z_star_norm": float(torch.linalg.vector_norm(target).item()),
        "delta_over_a0": float(torch.linalg.vector_norm(delta).div(torch.linalg.vector_norm(phi0)).item()),
        "registered_batch_identity": canonical_hash(batch_receipt),
        "entry_geometry_identity": entry_geometry["identity"],
        "source_order_identity": stream_receipt["selected_order_root"],
    }
    arm_payloads: list[dict[str, Any]] = []

    materialize_official(model, frozen)
    official_payload = {
        "arm": Arm.OFFICIAL_MEMIT.value,
        "status": "TERMINAL_VALID",
        "denominator": {"valid_terminal": 1, "attempted_arm": 1},
        "mechanism": {
            "direct_z_compute_count": 1,
            "direct_z_recompute_count": 0,
            "official_bypass_parity": True,
            "terminal_weight_root": frozen.official_endpoint_root,
        },
        "terminal": {"weight_root": frozen.official_endpoint_root},
        "endpoint": evaluate_rows(model, tokenizer, rows),
        "scientific_promotion": False,
    }
    journal.append(0, Arm.OFFICIAL_MEMIT, official_payload)
    arm_payloads.append(official_payload)
    w0.restore(model)

    for ordinal, arm in enumerate(ARMS[1:], start=1):
        try:
            mechanism = run_validity_arm(
                arm=arm,
                model=model,
                factory=factory,
                w0=w0,
                target=target,
                namespace=f"{args.campaign_id}|{args.model}|B1|{stream_receipt['selected_order_root']}|{arm.value}",
                policy=policy,
            )
            status = str(mechanism["status"])
            if arm is Arm.FZCB:
                barrier_summary = _barrier_summary(mechanism)
                if barrier_summary["sketch_only_count"]:
                    status = "TERMINAL_VALID_STRICT_SKETCH_ONLY"
            else:
                barrier_summary = None
            endpoint = evaluate_rows(model, tokenizer, rows)
            payload = {
                **mechanism,
                "arm": arm.value,
                "status": status,
                "denominator": {"valid_terminal": 1, "attempted_arm": 1},
                "endpoint": endpoint,
                "barrier_summary": barrier_summary,
                "scientific_promotion": False,
            }
        except BaseException as exc:
            rollback = w0.restore(model)
            observed_cache = _cache_identity()
            cache_rollback = {
                "entry_identity": cache_entry["identity"],
                "observed_identity": observed_cache["identity"],
                "exact": observed_cache["identity"] == cache_entry["identity"],
            }
            payload = _failure_payload(
                model_alias=args.model,
                case_ids=case_ids,
                arm=arm,
                exc=exc,
                policy=policy,
                target_receipt=target_receipt,
                rollback=rollback,
                cache_rollback=cache_rollback,
            )
        journal.append(ordinal, arm, payload)
        arm_payloads.append(payload)
        w0.restore(model)

    cache_terminal = _cache_identity()
    cache_exact = cache_terminal["identity"] == cache_entry["identity"]
    w0_restore = w0.restore(model)
    if not cache_exact:
        raise ScientificBoundary("MEMIT covariance/context cache identity changed across joint B1 arms")
    journal_index = journal.seal()
    typed = _typed_conclusion(arm_payloads)
    valid = sum(int(row.get("denominator", {}).get("valid_terminal", 0)) for row in arm_payloads)
    return {
        "schema": "odeedit.s06.fzcb-tech-r1.joint-b1-cell.v1",
        "status": "JOINT_CELL_AUDIT_COMPLETE",
        "typed_conclusion": typed,
        "model": args.model,
        "campaign_id": args.campaign_id,
        "batch": "B1",
        "case_ids": list(case_ids),
        "valid_arm_denominator": valid,
        "attempted_arm_denominator": len(arm_payloads),
        "stream": stream_receipt,
        "padding_binding": padding_binding,
        "padding_safety_gate": padding_gate,
        "registered_batch": batch_receipt,
        "target": {
            **target_receipt,
            "direct_z_compute_count": frozen.direct_z_calls,
            "direct_z_recompute_count": 0,
            "shared_by_arm_count": len(arm_payloads),
        },
        "tolerance_policy": policy.receipt(),
        "tolerance_calibration": synthetic_calibration_receipt(policy),
        "pre_edit": pre_edit,
        "arms": arm_payloads,
        "journal": journal_index,
        "w0_restore": w0_restore,
        "cache_entry": cache_entry,
        "cache_terminal": cache_terminal,
        "cache_exact": cache_exact,
        "full_fp32": True,
        "numerical_lock": NumericalLock().payload(),
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "b10_submit_count": 0,
        "scientific_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()
    try:
        payload = run_cell(args)
    except BaseException as exc:
        payload = {
            "schema": "odeedit.s06.fzcb-tech-r1.joint-b1-cell-failure.v1",
            "status": "TECHNICAL_BLOCKED" if not isinstance(exc, ScientificBoundary) else _failure_status(exc),
            "model": args.model,
            "campaign_id": args.campaign_id,
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "traceback": traceback.format_exc(),
            "b10_submit_count": 0,
            "scientific_promotion": False,
        }
        output = args.cell_root / "result.json"
        if output.exists() or output.is_symlink():
            output = args.cell_root / "terminal-failure.json"
        write_json_once(output, payload)
        raise SystemExit(2)
    write_json_once(args.cell_root / "result.json", payload)


if __name__ == "__main__":
    main()
