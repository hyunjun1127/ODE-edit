"""Llama/Qwen B1/B10 FzCB-Edit initial experiment runtime."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from project.run_scripts.fixed_z_nonuniqueness.contracts import MODEL_SPECS, Method, NumericalLock as PaddingLock
from project.run_scripts.fixed_z_nonuniqueness.evaluation import padding_safety_gate
from project.run_scripts.fixed_z_nonuniqueness.experiment import _load_hparams
from project.run_scripts.fixed_z_nonuniqueness.padding import OfficialTokenizerHook, bind_padding

from .contracts import Arm, FzCBEditError, NumericalLock, ScientificBoundary, TechnicalBoundary
from .controller import run_path_arm
from .data import load_sealed_rows, official_requests
from .evaluation import evaluate_rows
from .geometry import MEMITGeometryFactory, registered_batch
from .hashing import canonical_hash, tensor_sha256, write_json_once
from .target import capture_official_memit_once, materialize_official, restore_w0
from .transaction import WeightSnapshot


def _cache_identity() -> dict[str, Any]:
    from easyeditor.models.memit import memit_main

    covariance = []
    for key, value in sorted(memit_main.COV_CACHE.items(), key=lambda item: str(item[0])):
        covariance.append({
            "key": str(key), "shape": list(value.shape), "dtype": str(value.dtype),
            "data_ptr": int(value.data_ptr()), "version": int(value._version),
        })
    payload = {"covariance": covariance, "context_templates": memit_main.CONTEXT_TEMPLATES_CACHE}
    return {"identity": canonical_hash(payload), "covariance_count": len(covariance), "payload": payload}


def _padding_gate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], hparams: Any) -> dict[str, Any]:
    selected = rows if len(rows) >= 3 else (rows * 3)[:3]
    prompts = [row["requested_rewrite"]["prompt"].format(row["requested_rewrite"]["subject"]) for row in selected]
    return padding_safety_gate(
        model, tokenizer, prompts,
        [row["requested_rewrite"]["target_new"]["str"] for row in selected],
        hparams.layer_module_tmp.format(hparams.layers[-1]), PaddingLock(),
        input_module=hparams.rewrite_module_tmp.format(hparams.layers[-1]),
        subject_templates=[row["requested_rewrite"]["prompt"] for row in selected],
        subjects=[row["requested_rewrite"]["subject"] for row in selected],
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    rows, stream_receipt = load_sealed_rows(args.batch_size)
    requests = official_requests(rows)
    spec = MODEL_SPECS[args.model]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_path, local_files_only=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, device_map={"": 0}, trust_remote_code=False,
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
    if frozen.direct_z_calls != len(rows) or any(value != 1 for value in frozen.direct_z_calls_by_case.values()):
        raise ScientificBoundary("direct-z exactly-once/edit gate failed")
    if not frozen.tokenizer_calls or not all(row["semantic_input_attention_position_equal"] for row in frozen.tokenizer_calls):
        raise TechnicalBoundary("Official/hook token-position identity receipt missing")
    restore_w0(model, frozen)
    names = tuple(frozen.originals)
    w0 = WeightSnapshot.capture(model, names)
    pre_edit = evaluate_rows(model, tokenizer, rows)
    batch, batch_receipt = registered_batch(tokenizer, rows, hparams, next(model.parameters()).device)
    factory = MEMITGeometryFactory(model, tokenizer, rows, requests, hparams, batch)
    # Warm and freeze all original-model covariance objects before arm execution.
    factory.build()
    cache_entry = _cache_identity()
    target = frozen.targets.to(next(model.parameters()).device, torch.float32).reshape(-1)
    arms = []
    materialize_official(model, frozen)
    arms.append({
        "arm": Arm.OFFICIAL_MEMIT.value,
        "status": "TERMINAL_VALID",
        "mechanism": {"direct_z_compute_count": len(rows), "direct_z_recompute_count": 0, "official_bypass_parity": True},
        "terminal_weight_root": frozen.official_endpoint_root,
        "endpoint": evaluate_rows(model, tokenizer, rows),
    })
    w0.restore(model)
    for arm in (Arm.FROZEN_SPLIT, Arm.REFRESHED_EQUALITY_ONLY, Arm.FZCB, Arm.STATIC_PATH):
        mechanism = run_path_arm(
            arm=arm, model=model, factory=factory, w0=w0, target=target,
            namespace=f"{args.model}|B{args.batch_size}|{stream_receipt['selected_order_root']}",
        )
        endpoint = evaluate_rows(model, tokenizer, rows)
        arms.append({"arm": arm.value, "status": "TERMINAL_VALID", "mechanism": mechanism, "endpoint": endpoint})
        w0.restore(model)
    cache_terminal = _cache_identity()
    if cache_terminal["identity"] != cache_entry["identity"]:
        raise ScientificBoundary("MEMIT covariance/context cache identity changed across virtual arms")
    w0_restore = w0.restore(model)
    return {
        "schema": "odeedit.s06.fzcb-edit-main-method.initial.v1",
        "status": "TERMINAL_VALID",
        "model": args.model,
        "batch_size": args.batch_size,
        "case_ids": [int(row["case_id"]) for row in rows],
        "stream": stream_receipt,
        "padding_binding": padding_binding,
        "padding_safety_gate": padding_gate,
        "registered_batch": batch_receipt,
        "target": {
            "direct_z_compute_count": frozen.direct_z_calls,
            "direct_z_recompute_count": 0,
            "target_hashes": list(frozen.target_hashes),
            "target_root": frozen.target_root,
            "shared_by_arm_count": len(arms),
        },
        "pre_edit": pre_edit,
        "arms": arms,
        "w0_restore": w0_restore,
        "cache_entry": cache_entry,
        "cache_terminal": cache_terminal,
        "full_fp32": True,
        "numerical_lock": NumericalLock().payload(),
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "scientific_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--batch-size", choices=(1, 10), type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite create-once result")
    try:
        payload = run(args)
    except FzCBEditError as exc:
        payload = {
            "schema": "odeedit.s06.fzcb-edit-main-method.failure.v1",
            "status": "SCIENTIFIC_HOLD" if isinstance(exc, ScientificBoundary) else "TECHNICAL_BLOCKED",
            "model": args.model, "batch_size": args.batch_size,
            "failure_type": type(exc).__name__, "failure": str(exc),
            "scientific_promotion": False,
        }
    except Exception as exc:
        payload = {
            "schema": "odeedit.s06.fzcb-edit-main-method.failure.v1",
            "status": "TECHNICAL_BLOCKED", "model": args.model,
            "batch_size": args.batch_size, "failure_type": type(exc).__name__,
            "failure": str(exc), "traceback": traceback.format_exc(),
            "scientific_promotion": False,
        }
    write_json_once(args.output, payload)
    if payload["status"] != "TERMINAL_VALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

