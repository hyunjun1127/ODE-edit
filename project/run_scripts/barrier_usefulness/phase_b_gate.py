"""Gate-open evaluation of exact ctrl-sealed fixed candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from project.run_scripts.fixed_z_nonuniqueness.evaluation import next_token_logits
from project.run_scripts.fixed_z_nonuniqueness.padding import bind_padding

from .contexts import build_native_contexts, compare_realized_z, public_realized_z, realized_z_by_context
from .contracts import BarrierLock, MODEL_SPECS, Method, ScientificBoundary, TechnicalBoundary
from .firewall import authorize_gate_open
from .hashing import canonical_hash, file_sha256, tensor_sha256, write_json_once
from .metrics import reference_barrier, teacher_kl
from .phase_a import _load_hparams, _request_metrics
from .phase_b_ctrl import _anchor_batch_with_tokenizer, _history_downstream, _load_inputs, _load_teacher
from .snapshots import EndpointSnapshot
from .transport import reconstruct


def _numeric_max_abs(left: Any, right: Any) -> float:
    values: list[float] = []
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ScientificBoundary("transport metric schema mismatch")
        for key in left:
            values.append(_numeric_max_abs(left[key], right[key]))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ScientificBoundary("transport metric length mismatch")
        for a, b in zip(left, right, strict=True):
            values.append(_numeric_max_abs(a, b))
    elif isinstance(left, bool) and isinstance(right, bool):
        values.append(0.0 if left == right else math.inf)
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        values.append(abs(float(left) - float(right)))
    elif left != right:
        values.append(math.inf)
    return max(values or [0.0])


def _apply_official_transport(model: Any, package: dict[str, Any]) -> tuple[EndpointSnapshot, EndpointSnapshot]:
    names = sorted(package["official_delta_factors"])
    w0 = EndpointSnapshot.capture(model, names)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, factors in package["official_delta_factors"].items():
            delta = reconstruct(factors, device=params[name].device)
            params[name].copy_(w0.tensors[name].to(params[name].device, params[name].dtype) + delta.to(params[name].dtype))
    official = EndpointSnapshot.capture(model, names)
    return w0, official


def _observe_gate(
    model: Any, tokenizer: Any, texts: list[str], labels: torch.Tensor, teacher: torch.Tensor,
    floor: float, lock: BarrierLock,
) -> dict[str, Any]:
    logits = next_token_logits(model, tokenizer, texts, batch_size=16).cpu()
    return {
        "barrier": reference_barrier(teacher, logits, labels, numerical_floor=floor),
        "teacher_kl": teacher_kl(teacher, logits, lock.cvar_alpha),
        "candidate_logits_sha256": tensor_sha256(logits),
    }


def run_case(
    *, model: Any, tokenizer: Any, hparams: Any, row: dict[str, Any], anchor: dict[str, Any],
    indexed: dict[int, dict[str, Any]], ctrl_case: dict[str, Any], selector_case: dict[str, str | None],
    history: list[dict[str, Any]], lock: BarrierLock,
) -> dict[str, Any]:
    package_path = Path(ctrl_case["candidate_bank"]["path"])
    if file_sha256(package_path) != ctrl_case["candidate_bank"]["sha256"]:
        raise TechnicalBoundary("fixed candidate bank SHA mismatch at gate")
    package = torch.load(package_path, map_location="cpu", weights_only=False)
    if package["z_sha256"] != tensor_sha256(package["z"]):
        raise ScientificBoundary("fixed-z transport identity mismatch")
    w0, official_snapshot = _apply_official_transport(model, package)
    contexts = build_native_contexts(
        tokenizer=tokenizer, request=package["normalized_request"],
        context_templates=package["context_templates"], fact_token=hparams.fact_token,
    )
    activation_module = hparams.layer_module_tmp.format(hparams.layers[-1])
    official_realized = realized_z_by_context(model, tokenizer, contexts, activation_module, package["z"])
    official_metrics = _request_metrics(model, tokenizer, row)
    official_history = _history_downstream(model, tokenizer, history)
    transport_metric_max_abs = _numeric_max_abs(official_metrics, ctrl_case["official"]["metrics"])
    ctrl_residual = ctrl_case["official"]["realized_z"]["context_residual"]
    gate_residual = official_realized["context_residual"]
    transport_realized_max_abs = max(abs(float(a) - float(b)) for a, b in zip(ctrl_residual, gate_residual, strict=True))
    if max(transport_metric_max_abs, transport_realized_max_abs) > lock.fp32_relative_tolerance:
        raise ScientificBoundary("compact Official endpoint transport parity failed")
    gate_texts, gate_labels = _anchor_batch_with_tokenizer(tokenizer, indexed, anchor["gate"])
    teacher_logits, sealed_labels = _load_teacher(Path(anchor["gate"]["teacher_cache"]["path"]), anchor["gate"]["teacher_cache"])
    if not torch.equal(gate_labels, sealed_labels):
        raise TechnicalBoundary("gate labels differ from sealed teacher")
    official_gate = _observe_gate(model, tokenizer, gate_texts, gate_labels, teacher_logits, anchor["margin_numerical_floor"], lock)
    ctrl_by_id = {candidate["candidate_id"]: candidate for candidate in ctrl_case["candidates"]}
    candidates = []
    for axis in package["axes"]:
        output = axis["output"].to(next(model.parameters()).device)
        input_factor = axis["input"].to(output.device)
        for sign in (-1, 1):
            candidate_id = f"{axis['axis_id']}-{'plus' if sign > 0 else 'minus'}"
            correction = torch.outer(output, input_factor) * (float(axis["gamma"]) * sign)
            official_snapshot.apply_correction(model, package["last_weight_name"], correction)
            gate = _observe_gate(model, tokenizer, gate_texts, gate_labels, teacher_logits, anchor["margin_numerical_floor"], lock)
            realized = realized_z_by_context(model, tokenizer, contexts, activation_module, package["z"])
            realized_comparison = compare_realized_z(realized, official_realized)
            metrics = _request_metrics(model, tokenizer, row)
            history_downstream = _history_downstream(model, tokenizer, history)
            strict_equal = {name: value["strict"] for name, value in metrics.items()} == {name: value["strict"] for name, value in official_metrics.items()}
            ctrl = ctrl_by_id[candidate_id]
            valid = bool(ctrl["valid"] and strict_equal and realized_comparison["maximum_relative"] <= lock.fp32_relative_tolerance)
            restore = official_snapshot.restore_after_candidate(model)
            candidates.append({
                "candidate_id": candidate_id, "valid": valid, "gate": gate,
                "ctrl": {"barrier": ctrl["ctrl"]["barrier"], "teacher_kl": ctrl["ctrl"]["teacher_kl"]},
                "actual_action": ctrl["actual_action"], "realized_z": realized_comparison,
                "strict_equal_official": strict_equal, "metrics": metrics,
                "history_downstream": history_downstream, "snapshot_restore": restore,
            })
    if len(candidates) != 32 or any(not row["valid"] for row in candidates):
        raise ScientificBoundary("gate candidate hard validity failed")
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    selected_gate = {}
    for selector, candidate_id in selector_case.items():
        if candidate_id == "official":
            selected_gate[selector] = official_gate
        elif candidate_id is None:
            selected_gate[selector] = None
        else:
            selected_gate[selector] = by_id[candidate_id]["gate"]
    oracle = min(candidates, key=lambda candidate: (candidate["gate"]["teacher_kl"]["cvar_0_875"], candidate["candidate_id"]))
    ranking = sorted(candidates, key=lambda candidate: (candidate["gate"]["teacher_kl"]["cvar_0_875"], candidate["candidate_id"]))
    rank = {candidate["candidate_id"]: ordinal for ordinal, candidate in enumerate(ranking)}
    safe_id = selector_case["barrier-safe"]
    safe_percentile = rank[safe_id] / 31.0
    if not w0.copy_to_model(model)["exact"]:
        raise ScientificBoundary("gate terminal W0 restore failed")
    return {
        "schema": "odeedit.s06.barrier-usefulness.gate-case.v1", "case_id": int(row["case_id"]),
        "candidate_bank_identity": ctrl_case["candidate_bank_identity"],
        "selectors": selector_case, "selected_gate": selected_gate,
        "official_gate": official_gate, "official_metrics": official_metrics,
        "official_history_downstream": official_history,
        "official_realized_z": public_realized_z(official_realized),
        "official_transport_parity": {
            "metric_max_abs": transport_metric_max_abs,
            "realized_z_residual_max_abs": transport_realized_max_abs,
            "tolerance": lock.fp32_relative_tolerance,
        },
        "candidates": candidates,
        "gate_oracle_diagnostic": {"candidate_id": oracle["candidate_id"], "gate": oracle["gate"]},
        "barrier_safe_gate_percentile": safe_percentile,
        "valid_candidate_count": 32, "candidate_denominator": 32,
        "direct_z_compute_count": 0, "candidate_regeneration_count": 0, "w0_restore": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    lock = BarrierLock()
    ctrl = json.loads(args.ctrl_result.read_text())
    selector_lock = json.loads(args.selector_lock.read_text())
    cell_id = f"{args.model}|{args.method}"
    selected_cell = next(row for row in selector_lock["cells"] if row["cell_id"] == cell_id)
    if selected_cell["result_sha256"] != file_sha256(args.ctrl_result) or selected_cell["candidate_bank_hash"] != ctrl["candidate_bank_hash"]:
        raise TechnicalBoundary("selector lock/ctrl result identity mismatch")
    gate_open = authorize_gate_open(
        ctrl_root=args.ctrl_root, gate_root=args.gate_root, selector_lock=args.selector_lock,
        expected_identity=args.selector_identity,
    )
    gate_opened_at = time.time()
    cases, roles, anchor_seal, indexed = _load_inputs(args.dataset, args.source_root, args.anchor_seal)
    if ctrl["case_ids"] != [int(row["case_id"]) for row in cases]:
        raise TechnicalBoundary("gate case order differs from ctrl")
    spec = MODEL_SPECS[args.model]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_path, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        device_map={"": 0}, trust_remote_code=False,
    )
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TechnicalBoundary("gate FULL_FP32 failed")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(spec.model_path, local_files_only=True, use_fast=True, trust_remote_code=False)
    padding = bind_padding(tokenizer, model, padding_side="right")
    hparams_rel = spec.alpha_hparams if args.method == Method.ALPHAEDIT.value else spec.memit_hparams
    hparams = _load_hparams(Method(args.method), args.source_root / hparams_rel)
    anchors = {int(row["edit_case_id"]): {**row, "margin_numerical_floor": anchor_seal["margin_numerical_floor"]} for row in anchor_seal["cases"]}
    ctrl_by_case = {int(row["case_id"]): row for row in ctrl["cases"]}
    results = []
    for row in cases:
        case_id = int(row["case_id"])
        results.append(run_case(
            model=model, tokenizer=tokenizer, hparams=hparams, row=row, anchor=anchors[case_id], indexed=indexed,
            ctrl_case=ctrl_by_case[case_id], selector_case=ctrl["selected_candidate_ids"][str(case_id)],
            history=roles["history"], lock=lock,
        ))
        torch.cuda.empty_cache()
    return {
        "schema": "odeedit.s06.barrier-usefulness.gate-cell.v1", "status": "GATE_TERMINAL_VALID",
        "model": args.model, "method": args.method, "case_ids": ctrl["case_ids"], "cases": results,
        "selector_lock": {"path": str(args.selector_lock), "sha256": file_sha256(args.selector_lock), "identity": args.selector_identity},
        "gate_open": gate_open, "gate_opened_at": gate_opened_at,
        "candidate_bank_hash": ctrl["candidate_bank_hash"], "padding_binding": padding,
        "direct_z_compute_count": 0, "candidate_regeneration_count": 0,
        "edited_gate_access_count": 8 * 32, "replacement_count": 0, "final_audit_open_count": 0,
        "full_fp32": True, "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "scientific_promotion": False,
        "source": {
            "head": os.popen(f"git -C {args.source_root} rev-parse HEAD").read().strip(),
            "tree": os.popen(f"git -C {args.source_root} rev-parse HEAD^{{tree}}").read().strip(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--method", choices=[item.value for item in Method], required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--anchor-seal", type=Path, required=True)
    parser.add_argument("--ctrl-result", type=Path, required=True)
    parser.add_argument("--ctrl-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--selector-lock", type=Path, required=True)
    parser.add_argument("--selector-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    write_json_once(args.output, payload)


if __name__ == "__main__":
    main()
