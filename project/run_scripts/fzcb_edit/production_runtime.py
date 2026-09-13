"""Llama/Qwen canary and atomic-B10 production-pilot runtime."""

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

from .contracts import Arm, ScientificBoundary, TechnicalBoundary
from .controller_validity import ArmExecutionFailure, run_validity_arm
from .data import load_sealed_rows, official_requests
from .geometry import MEMITGeometryFactory, registered_batch
from .hashing import canonical_hash, tensor_sha256, write_json_once
from .production_contracts import (
    INSTRUCTION_ID,
    ProductionArm,
    ProductionMethod,
    ProductionNumericalLock,
)
from .production_controller import barrier_control_production
from .production_evaluation import capture_pre_edit, evaluate_endpoint
from .production_geometry import AlphaCacheSnapshot, AlphaEditGeometryFactory
from .production_journal import ProductionJournal
from .production_target import capture_official_alphaedit_once
from .runtime import _cache_identity
from .target import capture_official_memit_once, materialize_official, restore_w0
from .tolerances import UnitTolerancePolicy
from .transaction import WeightSnapshot


METHODS = (ProductionMethod.MEMIT, ProductionMethod.ALPHAEDIT)


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


def _arm_names(method: ProductionMethod) -> tuple[str, str, str]:
    if method is ProductionMethod.MEMIT:
        return (
            ProductionArm.OFFICIAL_MEMIT.value,
            ProductionArm.REFRESHED_EQUALITY_ONLY_MEMIT.value,
            ProductionArm.FZCB_SKETCH_K_MEMIT.value,
        )
    return (
        ProductionArm.OFFICIAL_ALPHAEDIT.value,
        ProductionArm.REFRESHED_EQUALITY_ONLY_ALPHAEDIT.value,
        ProductionArm.FZCB_SKETCH_K_ALPHAEDIT.value,
    )


def _typed_failure(
    *,
    model: str,
    method: ProductionMethod,
    arm: str,
    exc: BaseException,
    rollback: dict[str, Any],
    cache_restore: dict[str, Any],
) -> dict[str, Any]:
    evidence = dict(getattr(exc, "receipt", {}) or {})
    status = str(evidence.get("status", ""))
    if not status:
        if "NUMERICAL_INCONCLUSIVE" in str(exc):
            status = "NUMERICAL_INCONCLUSIVE"
        elif isinstance(exc, ScientificBoundary):
            status = "SCIENTIFIC_HOLD"
        else:
            status = "TECHNICAL_BLOCKED"
    return {
        "status": status,
        "model": model,
        "method": method.value,
        "arm": arm,
        "attempted_arm_denominator": 1,
        "valid_endpoint_denominator": 0,
        "FZCB_terminal_valid_denominator": 0 if "FZCB" in arm else "NOT_APPLICABLE",
        "failure_type": type(exc).__name__,
        "failure": str(exc),
        "evidence": evidence,
        "rollback": rollback,
        "cache_restore": cache_restore,
        "traceback": traceback.format_exc(),
        "scientific_promotion": False,
    }


def _method_cache_restore(method: ProductionMethod, alpha: AlphaCacheSnapshot | None) -> dict[str, Any]:
    if method is ProductionMethod.ALPHAEDIT:
        if alpha is None:
            raise TechnicalBoundary("AlphaEdit cache snapshot is missing")
        return alpha.restore()
    return {"exact": True, "identity": _cache_identity()["identity"]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.cell_root.exists() or args.cell_root.is_symlink():
        raise TechnicalBoundary(f"refusing to reuse production cell root: {args.cell_root}")
    args.cell_root.mkdir(parents=True, mode=0o700)
    count = 1 if args.stage == "canary" else 10
    rows, stream = load_sealed_rows(count)
    requests = official_requests(rows)
    case_ids = tuple(int(row["case_id"]) for row in rows)
    journal = ProductionJournal.create(args.cell_root / "arm-journal", args.model, case_ids)
    spec = MODEL_SPECS[args.model]
    policy = UnitTolerancePolicy.load()

    torch.set_grad_enabled(True)
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
    if bool(getattr(model, "is_quantized", False)):
        raise TechnicalBoundary("quantized model is forbidden")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_path,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    padding_binding = bind_padding(tokenizer, model, padding_side="right")
    evaluation_reference = capture_pre_edit(model, tokenizer, rows)
    pre_edit = evaluate_endpoint(model, tokenizer, rows, evaluation_reference)
    padding_receipts: dict[str, Any] = {}
    cohort_receipts: list[dict[str, Any]] = []
    all_arm_payloads: list[dict[str, Any]] = []

    for method in METHODS:
        official_name, equality_name, fzcb_name = _arm_names(method)
        hparams_path = spec.memit_hparams if method is ProductionMethod.MEMIT else spec.alpha_hparams
        hparams = _load_hparams(Method(method.value), args.source_root / hparams_path)
        if str(hparams.model_name) != spec.statistics_model_dir:
            raise TechnicalBoundary(f"{method.value} statistics alias mismatch")
        padding_receipts[method.value] = _padding_gate(model, tokenizer, rows, hparams)
        alpha_cache: AlphaCacheSnapshot | None = None
        if method is ProductionMethod.ALPHAEDIT:
            alpha_cache = AlphaCacheSnapshot.cold(model, tokenizer, hparams)
        tokenizer_hook = OfficialTokenizerHook(tokenizer, [])
        if method is ProductionMethod.MEMIT:
            frozen = capture_official_memit_once(
                model=model,
                tokenizer=tokenizer_hook,
                requests=requests,
                hparams=hparams,
            )
        else:
            frozen = capture_official_alphaedit_once(
                model=model,
                tokenizer=tokenizer_hook,
                requests=requests,
                hparams=hparams,
            )
        if frozen.direct_z_calls != count or any(value != 1 for value in frozen.direct_z_calls_by_case.values()):
            raise ScientificBoundary(f"{method.value} direct-z exactly-once/edit gate failed")
        if not frozen.tokenizer_calls or not all(
            row["semantic_input_attention_position_equal"] for row in frozen.tokenizer_calls
        ):
            raise TechnicalBoundary(f"{method.value} Official/hook token-position identity missing")
        names = tuple(frozen.originals)
        official_endpoint = evaluate_endpoint(model, tokenizer, rows, evaluation_reference)
        official_payload = {
            "status": "TERMINAL_VALID",
            "model": args.model,
            "method": method.value,
            "arm": official_name,
            "attempted_arm_denominator": 1,
            "valid_endpoint_denominator": 1,
            "request_denominator": count,
            "mechanism": {
                "direct_z_compute_count": count,
                "direct_z_recompute_count": 0,
                "official_bypass_parity": True,
                "terminal_weight_root": frozen.official_endpoint_root,
            },
            "endpoint": official_endpoint,
            "scientific_promotion": False,
        }
        journal.append(official_name, official_payload)
        all_arm_payloads.append(official_payload)
        restore_w0(model, frozen)
        cache_restore_after_official = _method_cache_restore(method, alpha_cache)
        w0 = WeightSnapshot.capture(model, names)
        batch, batch_receipt = registered_batch(tokenizer, rows, hparams, next(model.parameters()).device)
        if method is ProductionMethod.MEMIT:
            factory: Any = MEMITGeometryFactory(model, tokenizer, rows, requests, hparams, batch)
        else:
            if alpha_cache is None:
                raise TechnicalBoundary("AlphaEdit cache snapshot absent")
            factory = AlphaEditGeometryFactory(model, tokenizer, rows, requests, hparams, batch, alpha_cache)
        _, entry_geometry = factory.build()
        cache_entry = _cache_identity() if method is ProductionMethod.MEMIT else alpha_cache.receipt()
        target = frozen.targets.to(next(model.parameters()).device, torch.float32).reshape(-1)
        target_receipt = {
            "method": method.value,
            "target_hashes": list(frozen.target_hashes),
            "target_root": frozen.target_root,
            "target_tensor_sha256": tensor_sha256(target),
            "shared_by_arm_count": 3,
            "direct_z_compute_count": count,
            "direct_z_recompute_count": 0,
            "trajectory_target_recompute_count": 0,
            "registered_batch_identity": canonical_hash(batch_receipt),
            "entry_geometry_identity": entry_geometry["identity"],
        }
        cohort_arm_rows = [official_payload]
        for core_arm, qualified_name, barrier_fn in (
            (Arm.REFRESHED_EQUALITY_ONLY, equality_name, None),
            (Arm.FZCB, fzcb_name, barrier_control_production),
        ):
            try:
                kwargs: dict[str, Any] = {}
                if barrier_fn is not None:
                    kwargs["barrier_control_fn"] = barrier_fn
                mechanism = run_validity_arm(
                    arm=core_arm,
                    model=model,
                    factory=factory,
                    w0=w0,
                    target=target,
                    namespace=(
                        f"{INSTRUCTION_ID}|{args.campaign_id}|{args.stage}|{args.model}|"
                        f"{method.value}|{stream['selected_order_root']}|{qualified_name}"
                    ),
                    policy=policy,
                    **kwargs,
                )
                endpoint = evaluate_endpoint(model, tokenizer, rows, evaluation_reference)
                payload = {
                    **mechanism,
                    "status": (
                        "TERMINAL_VALID_STRICT_SKETCH_PROTOTYPE"
                        if core_arm is Arm.FZCB else "TERMINAL_VALID_STRICT"
                    ),
                    "model": args.model,
                    "method": method.value,
                    "arm": qualified_name,
                    "attempted_arm_denominator": 1,
                    "valid_endpoint_denominator": 1,
                    "request_denominator": count,
                    "FZCB_terminal_valid_denominator": 1 if core_arm is Arm.FZCB else "NOT_APPLICABLE",
                    "target": target_receipt,
                    "endpoint": endpoint,
                    "authority": (
                        ProductionNumericalLock().authority
                        if core_arm is Arm.FZCB else "REFRESHED_EQUALITY_ONLY"
                    ),
                    "scientific_promotion": False,
                }
            except BaseException as exc:
                rollback = w0.restore(model)
                cache_restore = _method_cache_restore(method, alpha_cache)
                payload = _typed_failure(
                    model=args.model,
                    method=method,
                    arm=qualified_name,
                    exc=exc,
                    rollback=rollback,
                    cache_restore=cache_restore,
                )
            journal.append(qualified_name, payload)
            cohort_arm_rows.append(payload)
            all_arm_payloads.append(payload)
            w0.restore(model)
            _method_cache_restore(method, alpha_cache)
        cache_terminal = _cache_identity() if method is ProductionMethod.MEMIT else alpha_cache.receipt()
        if method is ProductionMethod.MEMIT:
            cache_exact = cache_terminal["identity"] == cache_entry["identity"]
        else:
            cache_exact = cache_terminal["root"] == cache_entry["root"]
        if not cache_exact:
            raise ScientificBoundary(f"{method.value} cache identity changed across isolated arms")
        restore = w0.restore(model)
        cohort_receipts.append({
            "method": method.value,
            "target": target_receipt,
            "registered_batch": batch_receipt,
            "entry_geometry": entry_geometry,
            "cache_entry": cache_entry,
            "cache_terminal": cache_terminal,
            "cache_exact": cache_exact,
            "cache_restore_after_official": cache_restore_after_official,
            "w0_restore": restore,
            "arm_statuses": [{"arm": row["arm"], "status": row["status"]} for row in cohort_arm_rows],
        })
        torch.cuda.empty_cache()

    journal_index = journal.seal()
    fzcb_rows = [row for row in all_arm_payloads if "FZCB_SKETCH" in row["arm"]]
    terminal_fzcb = sum(row.get("FZCB_terminal_valid_denominator", 0) == 1 for row in fzcb_rows)
    valid_arms = sum(int(row.get("valid_endpoint_denominator", 0)) for row in all_arm_payloads)
    status = "TERMINAL_VALID" if valid_arms == len(all_arm_payloads) else "TERMINAL_WITH_TYPED_HOLD"
    return {
        "schema": "odeedit.s06.fzcb.atomic-b10-production-pilot.cell.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": status,
        "stage": args.stage,
        "model": args.model,
        "campaign_id": args.campaign_id,
        "case_ids": list(case_ids),
        "request_denominator": count,
        "attempted_arm_denominator": len(all_arm_payloads),
        "valid_endpoint_denominator": valid_arms,
        "FZCB_terminal_valid_denominator": terminal_fzcb,
        "FZCB_terminal_attempted_denominator": len(fzcb_rows),
        "pre_edit": pre_edit,
        "pre_edit_reference_identity": evaluation_reference.identity,
        "stream": stream,
        "padding_binding": padding_binding,
        "padding_safety_gates": padding_receipts,
        "cohorts": cohort_receipts,
        "arms": all_arm_payloads,
        "journal": journal_index,
        "production_numerical_lock": ProductionNumericalLock().payload(),
        "full_fp32": True,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "sequential_submit_count": 0,
        "scientific_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("canary", "b10"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()
    try:
        payload = run(args)
        write_json_once(args.cell_root / "result.json", payload)
    except BaseException as exc:
        failure = {
            "schema": "odeedit.s06.fzcb.atomic-b10-production-pilot.cell-failure.v1",
            "instruction_id": INSTRUCTION_ID,
            "status": "SCIENTIFIC_OR_NUMERICAL_HOLD" if isinstance(exc, ScientificBoundary) else "TECHNICAL_BLOCKED",
            "stage": args.stage,
            "model": args.model,
            "campaign_id": args.campaign_id,
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "traceback": traceback.format_exc(),
            "scientific_promotion": False,
        }
        write_json_once(args.cell_root / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
