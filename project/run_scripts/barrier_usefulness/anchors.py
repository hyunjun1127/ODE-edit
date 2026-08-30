"""W0-only factual-anchor calibration and immutable ctrl/gate teacher seals."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from project.run_scripts.fixed_z_nonuniqueness.evaluation import next_token_logits, target_prefixes
from project.run_scripts.fixed_z_nonuniqueness.padding import bind_padding

from .contracts import BarrierLock, MODEL_SPECS, ScientificBoundary, TechnicalBoundary
from .hashing import canonical_hash, file_sha256, tensor_sha256, write_json_once
from .metrics import factual_margins


def _teacher_observation(model: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    texts: list[str] = []
    labels: list[int] = []
    ranges = []
    for row in rows:
        prompt = row["prompt"].format(row["subject"])
        prefixes, ids = target_prefixes(tokenizer, prompt, row["target_true"])
        begin = len(texts)
        texts.extend(prefixes)
        labels.extend(int(value) for value in ids)
        ranges.append((begin, len(texts)))
    logits = next_token_logits(model, tokenizer, texts, batch_size=16)
    label_tensor = torch.tensor(labels, device=logits.device)
    margins = factual_margins(logits, label_tensor)
    predictions = logits.argmax(dim=-1)
    return {
        "logits": logits,
        "labels": label_tensor,
        "margins": margins,
        "ranges": ranges,
        "strict": [bool(torch.equal(predictions[begin:end], label_tensor[begin:end])) for begin, end in ranges],
    }


def _write_tensor_once(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise TechnicalBoundary(f"teacher cache overwrite attempted: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    os.chmod(path, 0o600)
    return file_sha256(path)


def build(model_alias: str, source_root: Path, output_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    spec = MODEL_SPECS[model_alias]
    lock = BarrierLock()
    anchor_path = source_root / "project/run_scripts/barrier_usefulness/config/anchor-pool-manifest-v1.json"
    fresh_path = source_root / "project/run_scripts/barrier_usefulness/config/fresh-case-manifest-v1.json"
    anchor = json.loads(anchor_path.read_text())
    fresh = json.loads(fresh_path.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_path, local_files_only=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, device_map={"": 0}, trust_remote_code=False,
    )
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TechnicalBoundary("anchor calibration FULL_FP32 closure failed")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(spec.model_path, local_files_only=True, use_fast=True, trust_remote_code=False)
    padding = bind_padding(tokenizer, model, padding_side="left")
    # W0-only cold replay floor uses the first preregistered ctrl pool and is
    # fixed before any edited endpoint exists.
    calibration_rows = anchor["pools"][0]["ctrl_pool"][:32]
    first = _teacher_observation(model, tokenizer, calibration_rows)
    second = _teacher_observation(model, tokenizer, calibration_rows)
    if first["logits"].shape != second["logits"].shape:
        raise TechnicalBoundary("cold replay shape mismatch")
    margin_drift = float((first["margins"] - second["margins"]).abs().max().item())
    floor = max(lock.fp32_absolute_tolerance, 8.0 * margin_drift)
    cases = []
    for pool in anchor["pools"]:
        split_rows = {}
        for split in ("ctrl", "gate"):
            source_rows = pool[f"{split}_pool"]
            observed = _teacher_observation(model, tokenizer, source_rows)
            admitted = []
            for ordinal, (row, (begin, end), strict) in enumerate(zip(source_rows, observed["ranges"], observed["strict"], strict=True)):
                minimum = float(observed["margins"][begin:end].min().item())
                if strict and minimum > floor:
                    admitted.append((ordinal, row, begin, end, minimum))
            if len(admitted) < lock.ctrl_anchor_count:
                raise ScientificBoundary(
                    f"W0 admissible {split} anchors absent for case {pool['edit_case_id']}: {len(admitted)}/32"
                )
            selected = admitted[:32]
            token_indices = [index for _, _, begin, end, _ in selected for index in range(begin, end)]
            logits = observed["logits"][token_indices].detach().cpu()
            labels = observed["labels"][token_indices].detach().cpu()
            cache_path = output_root / f"case-{pool['edit_case_id']}" / f"{split}-w0-teacher.pt"
            cache_sha = _write_tensor_once(cache_path, {"logits": logits, "labels": labels})
            selected_rows = []
            offset = 0
            for pool_ordinal, row, begin, end, minimum in selected:
                width = end - begin
                selected_rows.append({
                    "ordinal": len(selected_rows), "pool_ordinal": pool_ordinal,
                    "case_id": row["case_id"], "row_sha256": row["row_sha256"],
                    "prompt_sha256": canonical_hash(row["prompt"].format(row["subject"])),
                    "target_true": row["target_true"], "token_ids": [int(v) for v in labels[offset : offset + width]],
                    "token_begin": offset, "token_end": offset + width,
                    "w0_minimum_margin": minimum,
                })
                offset += width
            split_rows[split] = {
                "selected": selected_rows, "selected_count": 32,
                "admissible_pool_count": len(admitted), "teacher_cache": {
                    "path": str(cache_path), "sha256": cache_sha, "logits_sha256": tensor_sha256(logits),
                    "labels_sha256": tensor_sha256(labels), "shape": list(logits.shape),
                },
                "selected_root": canonical_hash(selected_rows),
            }
        ctrl_ids = {row["case_id"] for row in split_rows["ctrl"]["selected"]}
        gate_ids = {row["case_id"] for row in split_rows["gate"]["selected"]}
        if ctrl_ids & gate_ids:
            raise ScientificBoundary("selected ctrl/gate overlap")
        cases.append({"edit_case_id": pool["edit_case_id"], **split_rows, "selected_overlap_count": 0})
    payload = {
        "schema": "odeedit.s06.barrier-usefulness.w0-anchor-seal.v1",
        "status": "W0_ANCHOR_SEAL_PASS", "model": model_alias,
        "fresh_case_root": fresh["root_digest"], "anchor_pool_root": anchor["root_digest"],
        "margin_numerical_floor": floor, "cold_replay_margin_drift_max": margin_drift,
        "padding_binding": padding, "cases": cases,
        "edited_candidate_access_count": 0, "replacement_count": 0, "final_audit_open_count": 0,
        "full_fp32": True, "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    payload["root_digest"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_once(args.output, build(args.model, args.source_root.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
