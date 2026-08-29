#!/usr/bin/env python3
"""Run staged CounterFact experiments for the EasyEdit-native barrier writer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.models.alphaedit import AlphaEditHyperParams
from easyeditor.models.alphaedit import AlphaEdit_main as official
from project.run_scripts.alphaedit_strength_neutral_barrier import (
    BarrierArm,
    BarrierWriterConfig,
    apply_strength_neutral_barrier_to_model,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.telemetry import (
    write_create_once,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.firewall import (
    verify_stock_easyedit,
)
from easyeditor.util import nethook

from project.run_scripts.alphaedit_strength_neutral_barrier.evaluator import (
    evaluate_counterfact,
    locality_preservation,
)


ARM_MAP = {
    0: (BarrierArm.OFFICIAL, 1),
    1: (BarrierArm.SPLIT, 2),
    2: (BarrierArm.SPLIT, 4),
    3: (BarrierArm.PROJECTED, 2),
    4: (BarrierArm.PROJECTED, 4),
}

STAGE_COUNTS = {
    "atomic-b1": (1, 1),
    "atomic-b10": (10, 10),
    "atomic-b100": (100, 100),
    "sequential-b10x10": (100, 10),
}


def _git_value(root: Path, value: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", value], text=True
    ).strip()


def _selected_weights(model: Any, hparams: Any) -> Dict[str, torch.Tensor]:
    return {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }


def _tensor_set_sha(weights: Dict[str, torch.Tensor]) -> str:
    digest = sha256()
    for name in sorted(weights):
        tensor = weights[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_inventory(model: Any) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    total = 0
    for parameter in model.parameters():
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + parameter.numel()
        total += parameter.numel()
    wrong = sum(value for key, value in counts.items() if key != "torch.float32")
    return {
        "total_parameters": total,
        "dtype_counts": counts,
        "non_fp32_parameter_count": wrong,
        "quantized": bool(getattr(model, "is_quantized", False)),
    }


def _request(record: Dict[str, Any]) -> Dict[str, Any]:
    rewrite = record["requested_rewrite"]
    return {
        "case_id": int(record["case_id"]),
        "prompt": rewrite["prompt"],
        "subject": rewrite["subject"],
        "target_new": rewrite["target_new"]["str"],
        "target_true": rewrite["target_true"]["str"],
    }


def _sha_file(path: Path) -> Dict[str, Any]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _stage_records(raw: Sequence[Dict[str, Any]], stage: str) -> tuple[List[Dict[str, Any]], int]:
    count, batch_size = STAGE_COUNTS[stage]
    if len(raw) < count:
        raise RuntimeError(f"dataset has {len(raw)} records, expected at least {count}")
    return list(raw[:count]), batch_size


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGE_COUNTS), required=True)
    parser.add_argument("--cell", type=int, choices=sorted(ARM_MAP), required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--easyedit-root", type=Path, required=True)
    parser.add_argument("--expected-easyedit-head", required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--eval-microbatch", type=int, default=16)
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parents[3]
    actual_head = _git_value(source_root, "HEAD")
    actual_tree = _git_value(source_root, "HEAD^{tree}")
    if actual_head != args.expected_head:
        raise RuntimeError(
            f"queued source drift: expected {args.expected_head}, observed {actual_head}"
        )
    easyedit_head = _git_value(args.easyedit_root, "HEAD")
    easyedit_tree = _git_value(args.easyedit_root, "HEAD^{tree}")
    if easyedit_head != args.expected_easyedit_head:
        raise RuntimeError("stock EasyEdit source drift")
    easyedit_dirty = subprocess.check_output(
        ["git", "-C", str(args.easyedit_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if easyedit_dirty:
        raise RuntimeError("stock EasyEdit tracked source is dirty")
    easyedit_seal = verify_stock_easyedit(
        args.easyedit_root,
        Path(__file__).resolve().parent / "official-source-lock.json",
    )
    if args.result_dir.exists() or args.result_dir.is_symlink():
        raise FileExistsError(f"create-once result directory exists: {args.result_dir}")
    args.result_dir.mkdir(parents=True, mode=0o700)
    telemetry_dir = args.result_dir / "writer-telemetry"
    telemetry_dir.mkdir(mode=0o700)

    arm, steps = ARM_MAP[args.cell]
    result_path = args.result_dir / "result.json"
    failure_path = args.result_dir / "failure-boundary.json"
    action_counts = {"model_load": 0, "writer_calls": 0, "gpu_job": 1}
    model = None
    selected = None
    global_entry = None
    cache_entry = None
    started = time.perf_counter()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        raw = json.loads(args.dataset.read_text())
        if not args.preflight_receipt.is_file() or args.preflight_receipt.is_symlink():
            raise RuntimeError("preflight receipt must be a regular non-symlink file")
        preflight = json.loads(args.preflight_receipt.read_text())
        if preflight.get("terminal_status") != "PRE_GPU_PASS":
            raise RuntimeError("preflight receipt is not terminal-valid")
        if preflight.get("source", {}).get("head") != actual_head:
            raise RuntimeError("preflight/source HEAD mismatch")
        if preflight.get("source", {}).get("tree") != actual_tree:
            raise RuntimeError("preflight/source tree mismatch")
        if preflight.get("source", {}).get("easyedit_head") != easyedit_head:
            raise RuntimeError("preflight/EasyEdit HEAD mismatch")
        if preflight.get("source", {}).get("easyedit_tree") != easyedit_tree:
            raise RuntimeError("preflight/EasyEdit tree mismatch")
        if preflight.get("tokenizer_padding_side") != "right":
            raise RuntimeError("preflight tokenizer contract mismatch")
        if not preflight.get("full_fp32") or preflight.get("quantized"):
            raise RuntimeError("preflight FP32 deployment mismatch")
        if preflight.get("gate_summary", {}).get("pass_count") != preflight.get(
            "gate_summary", {}
        ).get("expected_count"):
            raise RuntimeError("cache-aware q-KL focused G0 gate count mismatch")
        current_inputs = {
            "model_config": _sha_file(args.model_path / "config.json"),
            "tokenizer": _sha_file(args.model_path / "tokenizer.json"),
            "dataset": _sha_file(args.dataset),
            "projector": _sha_file(args.projector),
            "hparams": _sha_file(args.hparams),
            "official_source": _sha_file(Path(official.__file__).resolve(strict=True)),
        }
        sealed_inputs = preflight.get("input_identities", {})
        for name, identity in current_inputs.items():
            sealed = sealed_inputs.get(name, {})
            if (identity["sha256"], identity["bytes"]) != (
                sealed.get("sha256"),
                sealed.get("bytes"),
            ):
                raise RuntimeError(f"preflight input identity mismatch: {name}")
        records, batch_size = _stage_records(raw, args.stage)
        case_ids = [int(record["case_id"]) for record in records]

        hparams = AlphaEditHyperParams.from_hparams(str(args.hparams))
        hparams.model_name = str(args.model_path)
        hparams.P_loc = str(args.projector)
        hparams.device = 0

        _sync()
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_path),
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map={"": "cuda:0"},
        )
        tok = AutoTokenizer.from_pretrained(
            str(args.model_path), local_files_only=True, use_fast=True
        )
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "right"
        model.eval()
        _sync()
        model_load_seconds = time.perf_counter() - load_started
        action_counts["model_load"] = 1
        inventory = _parameter_inventory(model)
        if inventory["non_fp32_parameter_count"] or inventory["quantized"]:
            raise RuntimeError(f"FULL_FP32 gate failed: {inventory}")

        selected = _selected_weights(model, hparams)
        global_entry = {name: value.detach().clone() for name, value in selected.items()}
        pointer_entry = {name: int(value.data_ptr()) for name, value in selected.items()}
        global_w0_sha = _tensor_set_sha(selected)

        _sync()
        pre_started = time.perf_counter()
        global_pre = evaluate_counterfact(
            model,
            tok,
            records,
            device=torch.device("cuda:0"),
            microbatch_size=args.eval_microbatch,
        )
        _sync()
        global_pre_seconds = time.perf_counter() - pre_started

        batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
        batch_results = []
        edit_core_seconds = 0.0
        evaluation_seconds = global_pre_seconds
        for batch_index, batch_records in enumerate(batches):
            _sync()
            eval_started = time.perf_counter()
            entry_eval = evaluate_counterfact(
                model,
                tok,
                batch_records,
                device=torch.device("cuda:0"),
                microbatch_size=args.eval_microbatch,
            )
            _sync()
            evaluation_seconds += time.perf_counter() - eval_started

            config = BarrierWriterConfig(
                arm=arm,
                steps=steps,
                telemetry_path=str(telemetry_dir / f"batch-{batch_index + 1:02d}.json"),
            )
            _sync()
            edit_started = time.perf_counter()
            model, _, writer_receipt = apply_strength_neutral_barrier_to_model(
                model,
                tok,
                [_request(record) for record in batch_records],
                hparams,
                config,
                copy=False,
                return_orig_weights=False,
                cache_template=None,
                reset_cache=batch_index == 0,
            )
            _sync()
            batch_edit_seconds = time.perf_counter() - edit_started
            edit_core_seconds += batch_edit_seconds
            action_counts["writer_calls"] += 1
            if cache_entry is None:
                cache_entry = torch.zeros_like(official.cache_c)

            _sync()
            eval_started = time.perf_counter()
            immediate = evaluate_counterfact(
                model,
                tok,
                batch_records,
                device=torch.device("cuda:0"),
                microbatch_size=args.eval_microbatch,
            )
            _sync()
            batch_eval_seconds = time.perf_counter() - eval_started
            evaluation_seconds += batch_eval_seconds
            batch_results.append(
                {
                    "batch_index": batch_index + 1,
                    "case_ids": [int(record["case_id"]) for record in batch_records],
                    "entry": entry_eval,
                    "immediate": immediate,
                    "locality_preservation": locality_preservation(
                        entry_eval["locality_target_true"],
                        immediate["locality_target_true"],
                    ),
                    "writer_receipt": writer_receipt,
                    "edit_core_seconds": batch_edit_seconds,
                    "evaluation_seconds": batch_eval_seconds,
                }
            )

        _sync()
        final_started = time.perf_counter()
        final_eval = evaluate_counterfact(
            model,
            tok,
            records,
            device=torch.device("cuda:0"),
            microbatch_size=args.eval_microbatch,
        )
        _sync()
        final_eval_seconds = time.perf_counter() - final_started
        evaluation_seconds += final_eval_seconds
        final_model_sha = _tensor_set_sha(selected)
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

        with torch.no_grad():
            for name, value in selected.items():
                value.copy_(global_entry[name])
        if hasattr(official, "cache_c") and cache_entry is not None:
            official.cache_c[...] = cache_entry
        restored_sha = _tensor_set_sha(selected)
        pointer_exit = {name: int(value.data_ptr()) for name, value in selected.items()}
        restore_pass = restored_sha == global_w0_sha and pointer_exit == pointer_entry
        if not restore_pass:
            raise RuntimeError("global W0 pointer/bytes restore failed")

        payload = {
            "schema": "easyedit.alphaedit.cache-aware-qkl-projected.experiment.v1",
            "terminal_status": "TECHNICAL_PASS",
            "stage": args.stage,
            "cell": args.cell,
            "arm": arm.value,
            "steps": steps,
            "source": {
                "head": actual_head,
                "tree": actual_tree,
                "easyedit_head": easyedit_head,
                "easyedit_tree": easyedit_tree,
                "easyedit_tracked_clean": True,
                "implementation_boundary": "CACHE_AWARE_QKL_HOOK_STOCK_EASYEDIT_READ_ONLY",
                "easyedit_seal": easyedit_seal,
            },
            "model": _sha_file(args.model_path / "config.json"),
            "tokenizer": _sha_file(args.model_path / "tokenizer.json"),
            "dataset": _sha_file(args.dataset),
            "projector": _sha_file(args.projector),
            "hparams": _sha_file(args.hparams),
            "preflight_receipt": _sha_file(args.preflight_receipt),
            "case_ids": case_ids,
            "request_count": len(records),
            "batch_size": batch_size,
            "batch_count": len(batches),
            "sequential_continuity": args.stage == "sequential-b10x10",
            "parameter_inventory": inventory,
            "autocast_enabled": False,
            "tf32_enabled": False,
            "global_w0_sha256": global_w0_sha,
            "final_model_sha256_before_restore": final_model_sha,
            "restored_w0_sha256": restored_sha,
            "w0_pointer_bytes_restore_pass": restore_pass,
            "global_pre": global_pre,
            "batches": batch_results,
            "final": final_eval,
            "final_locality_preservation": locality_preservation(
                global_pre["locality_target_true"],
                final_eval["locality_target_true"],
            ),
            "timing": {
                "model_load_seconds": model_load_seconds,
                "global_pre_eval_seconds": global_pre_seconds,
                "edit_core_seconds": edit_core_seconds,
                "evaluation_seconds": evaluation_seconds,
                "final_eval_seconds": final_eval_seconds,
                "total_seconds": time.perf_counter() - started,
            },
            "memory": {
                "peak_gpu_allocated_bytes": peak_allocated,
                "peak_gpu_reserved_bytes": peak_reserved,
            },
            "action_counts": action_counts,
            "scientific_promotion": False,
        }
        write_create_once(result_path, payload)
    except Exception as error:
        if model is not None and selected is not None and global_entry is not None:
            with torch.no_grad():
                for name, value in selected.items():
                    value.copy_(global_entry[name])
        failure = {
            "schema": "easyedit.alphaedit.cache-aware-qkl-projected.failure.v1",
            "terminal_status": "FAILED_BOUNDARY",
            "stage": args.stage,
            "cell": args.cell,
            "arm": ARM_MAP[args.cell][0].value,
            "steps": ARM_MAP[args.cell][1],
            "source": {
                "head": actual_head,
                "tree": actual_tree,
                "easyedit_head": easyedit_head,
                "easyedit_tree": easyedit_tree,
                "implementation_boundary": "CACHE_AWARE_QKL_HOOK_STOCK_EASYEDIT_READ_ONLY",
                "easyedit_seal": easyedit_seal,
            },
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "action_counts": action_counts,
            "science_change_count": 0,
            "tolerance_change_count": 0,
            "retry_count": 0,
        }
        if not failure_path.exists():
            write_create_once(failure_path, failure)
        raise


if __name__ == "__main__":
    main()
