"""K0 algebra/authority smoke and K1 predictive completion-value screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from project.run_scripts.fixed_z_nonuniqueness.contracts import MODEL_SPECS, Method, NumericalLock as FixedNumericalLock
from project.run_scripts.fixed_z_nonuniqueness.evaluation import padding_safety_gate
from project.run_scripts.fixed_z_nonuniqueness.experiment import _load_hparams
from project.run_scripts.fixed_z_nonuniqueness.official import restore_originals, run_official_once
from project.run_scripts.fixed_z_nonuniqueness.padding import OfficialTokenizerHook, bind_padding

from .algebra import (
    dimension_relative_floor, frozen_geometry_identity, macro_action_receipt,
    project_equality_null, scale_null_direction, solve_minimum_action, suffix_value,
)
from .contracts import (
    FzCBError, K0_CASE_IDS, K1_CASE_IDS, NumericalLock, ScientificBoundary,
    TechnicalBoundary,
)
from .data import canonical_prompt, load_rows, official_request
from .geometry import FullModelControlOperator, MEMITGeometryFactory, canonical_batch
from .hashing import canonical_hash, tensor_sha256, write_json_once
from .snapshots import WeightSnapshot


def _seed(model: str, case_id: int, ordinal: int) -> int:
    body = f"{NumericalLock().seed_namespace}|{model}|memit|{case_id}|{ordinal}"
    return int(hashlib.sha256(body.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def _cache_identity() -> dict[str, Any]:
    from easyeditor.models.memit import memit_main

    cov = []
    for key, value in sorted(memit_main.COV_CACHE.items(), key=lambda item: str(item[0])):
        cov.append({"key": str(key), "sha256": tensor_sha256(value), "shape": list(value.shape)})
    context = memit_main.CONTEXT_TEMPLATES_CACHE
    payload = {"covariance": cov, "context_templates": context}
    return {"root": canonical_hash(payload), "covariance_count": len(cov), "context_template_count": 0 if context is None else sum(len(v) for v in context)}


def _same_basis_operator(model: Any, source: FullModelControlOperator) -> FullModelControlOperator:
    return FullModelControlOperator(
        model, source.batch, source.z_layer, source.weight_names, source.right_factors,
        source.h_sqrt_blocks, source.last_direct_gain,
    )


def _action_squared(operator: FullModelControlOperator, coefficient: torch.Tensor) -> float:
    blocks = operator.split(coefficient)
    return float(sum(
        float(scale) ** 2 * float(torch.dot(block, block).item())
        for block, scale in zip(blocks, operator.h_sqrt_blocks, strict=True)
    ))


def _layer_action(operator: FullModelControlOperator, coefficient: torch.Tensor) -> dict[str, Any]:
    blocks = operator.split(coefficient)
    values = [
        float(scale) ** 2 * float(torch.dot(block, block).item())
        for block, scale in zip(blocks, operator.h_sqrt_blocks, strict=True)
    ]
    total = float(sum(values))
    return {
        "action_squared_by_layer": dict(zip(operator.weight_names, values, strict=True)),
        "action_squared_total": total,
        "last_layer_action_fraction": values[-1] / total if total > 0 else math.nan,
    }


def _solve(operator: FullModelControlOperator, rhs: torch.Tensor, tolerance: float) -> Any:
    return solve_minimum_action(
        operator, rhs.detach().float(), operator.h_sqrt_vector(),
        relative_tolerance=tolerance, max_iterations=NumericalLock().cg_max_iterations,
        output_preconditioner=operator.output_preconditioner,
    )


def _actual_adjoint_gate(operator: FullModelControlOperator, seed: int, tolerance: float) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    coefficient = torch.randn(operator.coefficient_dimension, generator=generator, dtype=torch.float32).to(operator.primals[0].device)
    cotangent = torch.randn(operator.output_dimension, generator=generator, dtype=torch.float32).to(operator.primals[0].device)
    forward_inner = torch.dot(operator.apply(coefficient), cotangent)
    adjoint_inner = torch.dot(coefficient, operator.adjoint(cotangent))
    relative = float(
        torch.abs(forward_inner - adjoint_inner)
        / (torch.abs(forward_inner) + torch.abs(adjoint_inner) + torch.finfo(torch.float32).eps)
    )
    if not math.isfinite(relative) or relative > tolerance:
        raise TechnicalBoundary(f"full-model matrix-free adjoint implementation failed: {relative}")
    return {
        "seed": seed, "forward_inner": float(forward_inner.item()),
        "adjoint_inner": float(adjoint_inner.item()), "relative_error": relative,
        "tolerance": tolerance, "operator_counts": operator.receipt(),
    }


def _apply_step(
    model: Any, operator: FullModelControlOperator, target: torch.Tensor, delta_s: float,
    tolerance: float, initial_coefficient: torch.Tensor | None = None,
) -> dict[str, Any]:
    lock = NumericalLock()
    entry = WeightSnapshot.capture(model, operator.weight_names)
    phi_entry = operator.phi()
    rhs = (target - phi_entry) / float(delta_s)
    nominal = _solve(operator, rhs, tolerance)
    nominal_counts = operator.receipt()
    coefficient = nominal.coefficient.clone() if initial_coefficient is None else initial_coefficient.clone()
    correctors = []
    for iteration in range(lock.corrector_iterations):
        entry.apply_tangents(model, operator.raw_coefficient_tangents(coefficient), delta_s)
        current = _same_basis_operator(model, operator)
        residual = target - current.phi()
        correction = _solve(current, residual / float(delta_s), tolerance)
        coefficient = coefficient + correction.coefficient
        correctors.append({
            "iteration": iteration + 1, "residual_before": float(torch.linalg.vector_norm(residual).item()),
            "correction_action_norm": math.sqrt(correction.action_norm_squared),
            "range_residual": correction.range_residual, "cg": asdict(correction.receipt),
            "operator_counts": current.receipt(),
        })
    entry.apply_tangents(model, operator.raw_coefficient_tangents(coefficient), delta_s)
    final_operator = _same_basis_operator(model, operator)
    phi_final = final_operator.phi()
    residual = float(torch.linalg.vector_norm(phi_final - target).div(torch.linalg.vector_norm(target) + torch.finfo(torch.float32).eps).item())
    if residual > tolerance:
        raise ScientificBoundary(f"nonlinear corrector fixed-z closure failed: {residual}")
    action_squared = _action_squared(operator, coefficient)
    layer_action = _layer_action(operator, coefficient)
    action_receipt = macro_action_receipt(action_squared, float(delta_s))
    return {
        "entry_snapshot": entry, "coefficient": coefficient.detach(),
        "coefficient_sha256": tensor_sha256(coefficient),
        "target_sha256": tensor_sha256(target), "phi_entry_sha256": tensor_sha256(phi_entry),
        "phi_final_sha256": tensor_sha256(phi_final), "closure_relative": residual,
        "nominal_range_residual": nominal.range_residual, "nominal_cg": asdict(nominal.receipt),
        "correctors": correctors, "corrector_action_included": True,
        "coefficient_action_squared": action_squared, **layer_action,
        "step_action": action_receipt["step_action"],
        "displacement_metric_action": action_receipt["displacement_metric_action"],
        "equivalent_step_action": action_receipt["equivalent_step_action"],
        "corrector_action_accounting_absolute_error": action_receipt["absolute_error"],
        "nominal_operator_counts": nominal_counts,
        "final_operator_counts": final_operator.receipt(),
    }


def _serial_step(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"entry_snapshot", "coefficient"}}


def _padding_gate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], hparams: Any) -> dict[str, Any]:
    selected = rows[:3]
    return padding_safety_gate(
        model, tokenizer, [canonical_prompt(row) for row in selected],
        [row["requested_rewrite"]["target_new"]["str"] for row in selected],
        hparams.layer_module_tmp.format(hparams.layers[-1]), FixedNumericalLock(),
        input_module=hparams.rewrite_module_tmp.format(hparams.layers[-1]),
        subject_templates=[row["requested_rewrite"]["prompt"] for row in selected],
        subjects=[row["requested_rewrite"]["subject"] for row in selected],
    )


def _candidate(
    *, model: Any, factory: MEMITGeometryFactory, state25: WeightSnapshot,
    phi0: torch.Tensor, target: torch.Tensor, displacement: torch.Tensor,
    candidate_id: str, initial_coefficient: torch.Tensor, tolerance: float,
    repeat_value: bool,
) -> dict[str, Any]:
    lock = NumericalLock()
    state25.restore(model)
    operator25, geometry25 = factory.build()
    transition = _apply_step(model, operator25, phi0 + 0.5 * displacement, 0.25, tolerance, initial_coefficient)
    endpoint50 = WeightSnapshot.capture(model, operator25.weight_names)
    endpoint_hashes = []
    for _ in range(3):
        state25.restore(model)
        state25.apply_tangents(model, operator25.raw_coefficient_tangents(transition["coefficient"]), 0.25)
        endpoint_hashes.append(state25.current_root(model))
    if len(set(endpoint_hashes)) != 1 or endpoint_hashes[0] != endpoint50.root:
        raise ScientificBoundary("corrected endpoint repeat identity failed")
    endpoint50.restore(model)
    value_operator, value_geometry = factory.build()
    residual = target - value_operator.phi()
    value_runs = []
    repeat_count = 3 if repeat_value else 1
    value_failure = None
    try:
        for _ in range(repeat_count):
            solution = _solve(value_operator, residual, tolerance)
            value_runs.append({
                "value": suffix_value(solution, 0.5), "range_residual": solution.range_residual,
                "coefficient_sha256": tensor_sha256(solution.coefficient), "cg": asdict(solution.receipt),
            })
    except ScientificBoundary as exc:
        value_failure = str(exc)
        value_runs = [{"value": math.inf, "range_residual": math.inf, "failure": value_failure}]
    values = [row["value"] for row in value_runs]
    predicted = values[0]
    finite_repeats = [value for value in values if math.isfinite(value)]
    value_noise = max(finite_repeats) - min(finite_repeats) if finite_repeats else 0.0
    suffix_steps = []
    suffix_action = 0.0
    rollout_failure = None
    try:
        for next_s in (0.75, 1.0):
            current, geometry = factory.build()
            step = _apply_step(model, current, phi0 + next_s * displacement, 0.25, tolerance)
            suffix_action += step["step_action"]
            suffix_steps.append({"s": next_s, "geometry": asdict(geometry), "step": _serial_step(step)})
        terminal_operator, _ = factory.build()
        terminal_closure = float(torch.linalg.vector_norm(terminal_operator.phi() - target).div(torch.linalg.vector_norm(target) + torch.finfo(torch.float32).eps).item())
        if terminal_closure > tolerance:
            raise ScientificBoundary(f"terminal suffix target closure failed: {terminal_closure}")
    except ScientificBoundary as exc:
        suffix_action = math.inf
        terminal_closure = math.inf
        rollout_failure = str(exc)
    finally:
        state25.restore(model)
    return {
        "candidate_id": candidate_id, "predicted_suffix_value": predicted,
        "predicted_value_repeat_noise": value_noise, "predicted_value_runs": value_runs,
        "predicted_value_failure": value_failure,
        "actual_suffix_action": suffix_action, "total_realized_cost": transition["step_action"] + suffix_action,
        "transition": _serial_step(transition), "geometry_025": asdict(geometry25),
        "geometry_050": asdict(value_geometry), "suffix_steps": suffix_steps,
        "terminal_closure_relative": terminal_closure, "rollout_failure": rollout_failure,
        "corrected_endpoint_repeat_count": 3, "corrected_endpoint_root": endpoint50.root,
    }


def _run_case(model: Any, tokenizer: Any, model_alias: str, row: dict[str, Any], hparams: Any, seed_count: int) -> dict[str, Any]:
    lock = NumericalLock()
    started = time.monotonic()
    device = next(model.parameters()).device
    batch, batch_receipt = canonical_batch(tokenizer, row, hparams, device)
    tokenizer_hook = OfficialTokenizerHook(tokenizer, [])
    tokenizer.padding_side = "right"
    endpoint = run_official_once(
        method=Method.MEMIT, model=model, tokenizer=tokenizer_hook,
        request=official_request(row), hparams=hparams,
    )
    if endpoint.direct_z_compute_count != 1 or endpoint.direct_z_recompute_count != 0:
        raise ScientificBoundary("direct-z once/hash gate failed")
    if not endpoint.tokenizer_calls or not all(call["semantic_input_attention_position_equal"] for call in endpoint.tokenizer_calls):
        raise TechnicalBoundary("Official semantic input/attention/position identity failed")
    names = tuple(f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in hparams.layers)
    if not restore_originals(model, endpoint.originals):
        raise ScientificBoundary("Official W0 restore failed")
    w0 = WeightSnapshot.capture(model, names)
    factory = MEMITGeometryFactory(model, tokenizer, row, hparams, batch)
    cache_entry = _cache_identity()
    operator0, geometry0 = factory.build()
    phi0 = operator0.phi()
    target = endpoint.z.detach().to(device=device, dtype=torch.float32).reshape(-1)
    displacement = target - phi0
    tolerance = dimension_relative_floor(int(target.numel()))
    adjoint_gate = _actual_adjoint_gate(operator0, _seed(model_alias, int(row["case_id"]), 10000), tolerance)
    initial = _solve(operator0, displacement, tolerance)
    repeated = [_solve(operator0, displacement, tolerance) for _ in range(3)]
    solve_noise = max(item.action_norm_squared for item in repeated) - min(item.action_norm_squared for item in repeated)
    coefficient_roots = {tensor_sha256(item.coefficient) for item in repeated}
    if len(coefficient_roots) != 1:
        raise ScientificBoundary("repeated minimum-action solve differs")
    identity = frozen_geometry_identity(0.5 * initial.action_norm_squared, lock.progress_grid)
    if identity["maximum_absolute_error"] > tolerance:
        raise ScientificBoundary("frozen-geometry spent+suffix identity failed")
    step25 = _apply_step(model, operator0, phi0 + 0.25 * displacement, 0.25, tolerance)
    state25 = WeightSnapshot.capture(model, names)
    operator25, geometry25 = factory.build()
    rhs25 = (phi0 + 0.5 * displacement - operator25.phi()) / 0.25
    eq25 = _solve(operator25, rhs25, tolerance)
    candidates = [
        _candidate(
            model=model, factory=factory, state25=state25, phi0=phi0,
            target=target, displacement=displacement, candidate_id="equality-only",
            initial_coefficient=eq25.coefficient, tolerance=tolerance, repeat_value=True,
        )
    ]
    null_receipts = []
    for ordinal in range(seed_count):
        generator = torch.Generator().manual_seed(_seed(model_alias, int(row["case_id"]), ordinal))
        seed = torch.randn(operator25.coefficient_dimension, generator=generator, dtype=torch.float32).to(device)
        projected, null_receipt = project_equality_null(
            operator25, seed, relative_tolerance=tolerance,
            max_iterations=lock.cg_max_iterations,
            refinement_iterations=lock.null_projection_refinement_iterations,
            output_preconditioner=operator25.output_preconditioner,
        )
        candidate_id = f"seed-{ordinal:02d}"
        receipt = {
            "candidate_id": candidate_id, "seed": _seed(model_alias, int(row["case_id"]), ordinal),
            **null_receipt,
        }
        if not bool(null_receipt["cg_converged"]):
            receipt.update({"status": "PROJECTION_SOLVER_FAILURE", "valid_weight_authority": False})
            null_receipts.append(receipt)
            continue
        if float(null_receipt["norm"]) <= tolerance:
            receipt.update({"status": "NO_WEIGHT_AUTHORITY", "valid_weight_authority": False})
            null_receipts.append(receipt)
            continue
        scaled = scale_null_direction(projected, math.sqrt(eq25.action_norm_squared), lock.candidate_rho)
        scaled_norm = float(torch.linalg.vector_norm(scaled).item())
        null_residual = float(torch.linalg.vector_norm(operator25.apply(scaled)).div(torch.linalg.vector_norm(scaled) + torch.finfo(torch.float32).eps).item())
        receipt.update({
            "scaled_null_residual": null_residual, "scaled_action_norm": scaled_norm,
            "status": "VALID" if null_residual <= tolerance and scaled_norm > tolerance else "PROJECTION_SOLVER_FAILURE",
            "valid_weight_authority": bool(null_residual <= tolerance and scaled_norm > tolerance),
        })
        null_receipts.append(receipt)
        if not receipt["valid_weight_authority"]:
            continue
        coefficient = eq25.coefficient + scaled / operator25.h_sqrt_vector()
        candidates.append(_candidate(
            model=model, factory=factory, state25=state25, phi0=phi0,
            target=target, displacement=displacement, candidate_id=candidate_id,
            initial_coefficient=coefficient, tolerance=tolerance, repeat_value=False,
        ))
    state25.restore(model)
    w0_restore = w0.restore(model)
    cache_terminal = _cache_identity()
    if cache_terminal["root"] != cache_entry["root"]:
        raise ScientificBoundary("MEMIT cache rollback identity failed")
    finite_values = [row["predicted_suffix_value"] for row in candidates if math.isfinite(row["predicted_suffix_value"])]
    spread = max(finite_values) - min(finite_values) if finite_values else 0.0
    value_noise = max(row["predicted_value_repeat_noise"] for row in candidates)
    return {
        "schema": "odeedit.s06.fzcb-completion-value.case.v1", "status": "TERMINAL_VALID",
        "model": model_alias, "method": "memit", "case_id": int(row["case_id"]),
        "direct_z_compute_count": 1, "direct_z_recompute_count": 0,
        "target_sha256": tensor_sha256(target), "phi0_sha256": tensor_sha256(phi0),
        "canonical_batch": batch_receipt, "official_tokenizer_calls": endpoint.tokenizer_calls,
        "numerical_tolerance": tolerance, "initial_geometry": asdict(geometry0),
        "actual_full_model_adjoint_gate": adjoint_gate,
        "state025_geometry": asdict(geometry25), "initial_range_residual": initial.range_residual,
        "repeated_solve_count": 3, "repeated_action_noise": solve_noise,
        "initial_operator_counts": operator0.receipt(),
        "frozen_geometry_identity": identity, "step_000_025": _serial_step(step25),
        "null_directions": null_receipts,
        "valid_null_direction_count": sum(bool(item["valid_weight_authority"]) for item in null_receipts),
        "projection_solver_failure_count": sum(item["status"] == "PROJECTION_SOLVER_FAILURE" for item in null_receipts),
        "candidates": candidates, "candidate_value_spread": spread,
        "candidate_value_repeat_noise": value_noise,
        "spread_gt_3x_noise": spread > 3.0 * value_noise,
        "w0_restore": w0_restore, "cache_entry": cache_entry, "cache_terminal": cache_terminal,
        "wall_seconds": time.monotonic() - started,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    lock = NumericalLock()
    stage = args.stage
    model_alias = args.model
    if stage == "k0" and model_alias != "llama3-8b-inst":
        raise TechnicalBoundary("K0 is Llama-only")
    rows = load_rows(args.source_root.resolve(), stage)
    spec = MODEL_SPECS[model_alias]
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
    padding = _padding_gate(model, tokenizer, rows, hparams)
    cases = []
    seed_count = lock.k0_random_seed_count if stage == "k0" else lock.k1_random_seed_count
    terminal_failure = None
    terminal_status = "SCIENTIFIC_HOLD"
    for row in rows:
        try:
            case = _run_case(model, tokenizer, model_alias, row, hparams, seed_count)
            cases.append(case)
            torch.cuda.empty_cache()
            if stage == "k0" and case["projection_solver_failure_count"]:
                terminal_failure = (
                    f"K0 projection solver implementation failed case={case['case_id']} "
                    f"count={case['projection_solver_failure_count']}"
                )
                terminal_status = "TECHNICAL_BLOCKED"
                break
            if stage == "k0" and (
                case["valid_null_direction_count"] < 3 or not case["spread_gt_3x_noise"]
            ):
                terminal_failure = (
                    f"K0 kill gate failed case={case['case_id']} "
                    f"null={case['valid_null_direction_count']} "
                    f"spread={case['candidate_value_spread']} "
                    f"noise={case['candidate_value_repeat_noise']}"
                )
                break
        except FzCBError as exc:
            terminal_failure = f"case={row['case_id']} {type(exc).__name__}: {exc}"
            terminal_status = "SCIENTIFIC_HOLD" if isinstance(exc, ScientificBoundary) else "TECHNICAL_BLOCKED"
            break
    if terminal_failure is not None:
        return {
            "schema": f"odeedit.s06.fzcb-completion-value.{stage}.failure.v1",
            "status": terminal_status, "stage": stage, "model": model_alias,
            "failure": terminal_failure, "completed_case_count": len(cases),
            "cases": cases, "padding_binding": padding_binding,
            "padding_safety_gate": padding, "full_fp32": True,
            "numerical_lock": lock.payload(), "wall_seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "scientific_promotion": False,
        }
    return {
        "schema": f"odeedit.s06.fzcb-completion-value.{stage}.v1", "status": "TERMINAL_VALID",
        "stage": stage, "model": model_alias, "method": "memit",
        "case_ids": [int(row["case_id"]) for row in rows], "cases": cases,
        "padding_binding": padding_binding, "padding_safety_gate": padding,
        "full_fp32": True, "numerical_lock": lock.payload(),
        "dense_inverse_count": 0, "explicit_kronecker_count": 0,
        "value_gradient_count": 0, "cbf_count": 0, "full_qcqp_count": 0,
        "alphaedit_count": 0, "b10_count": 0, "b100_count": 0,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "scientific_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("k0", "k1"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite create-once result")
    try:
        payload = run(args)
    except FzCBError as exc:
        payload = {
            "schema": "odeedit.s06.fzcb-completion-value.failure.v1",
            "status": "SCIENTIFIC_HOLD" if isinstance(exc, ScientificBoundary) else "TECHNICAL_BLOCKED",
            "stage": args.stage, "model": args.model, "failure_type": type(exc).__name__,
            "failure": str(exc), "scientific_promotion": False,
        }
    except Exception as exc:  # preserve unexpected technical evidence for TECH-Rn
        payload = {
            "schema": "odeedit.s06.fzcb-completion-value.failure.v1",
            "status": "TECHNICAL_BLOCKED", "stage": args.stage, "model": args.model,
            "failure_type": type(exc).__name__, "failure": str(exc),
            "traceback": traceback.format_exc(), "scientific_promotion": False,
        }
    write_json_once(args.output, payload)
    if payload["status"] != "TERMINAL_VALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
