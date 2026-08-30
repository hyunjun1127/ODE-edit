"""Corrected F1b causal-closure runtime for Phase A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from project.run_scripts.fixed_z_nonuniqueness.evaluation import (
    next_token_logits,
    padding_safety_gate,
    sequence_metrics,
)
from project.run_scripts.fixed_z_nonuniqueness.padding import OfficialTokenizerHook, bind_padding

from .contexts import (
    build_constraint_bank,
    build_native_contexts,
    compare_realized_z,
    public_realized_z,
    realized_z_by_context,
)
from .contracts import BarrierLock, MODEL_SPECS, Method, ScientificBoundary, TechnicalBoundary
from .geometry import build_axes, projector_probe_audit
from .hashing import canonical_hash, file_sha256, tensor_sha256, write_json_once
from .metrics import teacher_kl
from .official import restore_originals, run_official_once
from .snapshots import EndpointSnapshot


def _official_request(row: dict[str, Any]) -> dict[str, Any]:
    request = row["requested_rewrite"]
    return {
        "case_id": str(row["case_id"]),
        "prompt": request["prompt"],
        "subject": request["subject"],
        "target_new": request["target_new"]["str"],
        "target_true": request["target_true"]["str"],
    }


def _prompt(row: dict[str, Any]) -> str:
    req = row["requested_rewrite"]
    return req["prompt"].format(req["subject"])


def _load_rows(dataset: Path, manifest: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    data = json.loads(dataset.read_text())
    indexed = {int(row["case_id"]): row for row in data}
    seal = json.loads(manifest.read_text())
    cases = [indexed[int(row["case_id"])] for row in seal["screen_cases"]]
    roles = {
        role: [indexed[int(row["case_id"])] for row in selected]
        for role, selected in seal["roles"].items() if role != "final_audit_sealed"
    }
    return cases, roles


def _load_hparams(method: Method, path: Path) -> Any:
    if method is Method.ALPHAEDIT:
        from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams
        return AlphaEditHyperParams.from_hparams(str(path))
    from easyeditor.models.memit.memit_hparams import MEMITHyperParams
    return MEMITHyperParams.from_hparams(str(path))


class SecondMoment:
    """GPU registered uncentered second moment with exact dense-D action."""

    def __init__(self, path: Path, device: torch.device, *, scale: float, additive: float) -> None:
        archive = np.load(path)
        raw = archive["mom2.mom2"]
        self.count = int(archive["mom2.count"])
        self.sample_size = int(archive["sample_size"])
        self.matrix = torch.from_numpy(np.asarray(raw)).to(device=device, dtype=torch.float32) / self.count
        self.scale = float(scale)
        self.additive = float(additive)
        self.path = path

    def matvec(self, value: torch.Tensor) -> torch.Tensor:
        return self.scale * (self.matrix @ value) + self.additive * value

    def matrix_action(self, value: torch.Tensor, chunk: int = 32) -> float:
        total = torch.zeros((), device=value.device, dtype=torch.float64)
        for begin in range(0, value.shape[0], chunk):
            block = value[begin : begin + chunk]
            transformed = self.scale * (block @ self.matrix) + self.additive * block
            total += (block.double() * transformed.double()).sum()
        result = float(total.item())
        if not math.isfinite(result) or result <= 0:
            raise ScientificBoundary("dense registered base action invalid")
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "sha256": file_sha256(self.path), "count": self.count,
            "sample_size": self.sample_size, "shape": list(self.matrix.shape),
            "dtype": "float32", "scale": self.scale, "additive": self.additive,
            "name": "registered_w0_uncentered_second_moment_action",
        }


def _request_metrics(model: Any, tok: Any, row: dict[str, Any]) -> dict[str, Any]:
    req = row["requested_rewrite"]
    rewrite = [req["prompt"].format(req["subject"])]
    rephrase = list(row.get("paraphrase_prompts", []))
    return {
        "rewrite_target_new": sequence_metrics(model, tok, rewrite, req["target_new"]["str"]),
        "rewrite_target_true": sequence_metrics(model, tok, rewrite, req["target_true"]["str"]),
        "rephrase_target_new": sequence_metrics(model, tok, rephrase, req["target_new"]["str"]),
        "rephrase_target_true": sequence_metrics(model, tok, rephrase, req["target_true"]["str"]),
    }


def _strict_signature(metrics: dict[str, Any]) -> dict[str, list[bool]]:
    return {name: list(row["strict"]) for name, row in metrics.items()}


def _observe(
    model: Any, tok: Any, row: dict[str, Any], contexts: list[Any], activation_module: str,
    z: torch.Tensor, teacher_logits: torch.Tensor, screen_prompts: list[str], lock: BarrierLock,
) -> dict[str, Any]:
    functional = teacher_kl(teacher_logits, next_token_logits(model, tok, screen_prompts), lock.cvar_alpha)
    realized = realized_z_by_context(model, tok, contexts, activation_module, z)
    return {"functional": functional, "realized_z": realized, "metrics": _request_metrics(model, tok, row)}


def _public_observation(row: dict[str, Any]) -> dict[str, Any]:
    return {"functional": row["functional"], "realized_z": public_realized_z(row["realized_z"]), "metrics": row["metrics"]}


def run_case(
    *, model: Any, tok: Any, method: Method, hparams: Any, row: dict[str, Any],
    roles: dict[str, list[dict[str, Any]]], moment: SecondMoment, lock: BarrierLock,
    model_alias: str, reverse_audit: bool,
) -> dict[str, Any]:
    request = _official_request(row)
    hook = OfficialTokenizerHook(tok, [])
    tok.padding_side = "right"
    captured = run_official_once(method=method, model=model, tokenizer=hook, request=request, hparams=hparams)
    endpoint = captured.endpoint
    if endpoint.direct_z_compute_count != 1 or endpoint.direct_z_recompute_count != 0:
        raise ScientificBoundary("direct-z count failed")
    if not endpoint.tokenizer_calls or not all(call["semantic_input_attention_position_equal"] for call in endpoint.tokenizer_calls):
        raise TechnicalBoundary("Official semantic input/attention/position receipt failed")
    contexts = build_native_contexts(
        tokenizer=tok, request=captured.normalized_request,
        context_templates=captured.compute_z_context_templates, fact_token=hparams.fact_token,
    )
    names = sorted(endpoint.originals)
    snapshot = EndpointSnapshot.capture(model, names)
    screen_prompts = [_prompt(item) for item in roles["screen"]]
    if not restore_originals(model, endpoint.originals):
        raise ScientificBoundary("W0 restore before teacher failed")
    teacher_logits = next_token_logits(model, tok, screen_prompts)
    snapshot.copy_to_model(model)
    input_module = hparams.rewrite_module_tmp.format(hparams.layers[-1])
    activation_module = hparams.layer_module_tmp.format(hparams.layers[-1])
    history_templates = [item["requested_rewrite"]["prompt"] for item in roles["history"]]
    history_subjects = [item["requested_rewrite"]["subject"] for item in roles["history"]]
    constraints, constraint_receipt = build_constraint_bank(
        model=model, tokenizer=tok, request=captured.normalized_request, contexts=contexts,
        history_templates=history_templates, history_subjects=history_subjects, input_module=input_module,
    )
    official_first = _observe(model, tok, row, contexts, activation_module, endpoint.z, teacher_logits, screen_prompts, lock)
    snapshot.copy_to_model(model)
    official_second = _observe(model, tok, row, contexts, activation_module, endpoint.z, teacher_logits, screen_prompts, lock)
    replay_difference = compare_realized_z(official_second["realized_z"], official_first["realized_z"])
    realized_tolerance = max(lock.fp32_relative_tolerance, 8.0 * replay_difference["maximum_relative"])
    duplicate_noise = abs(official_second["functional"]["cvar_0_875"] - official_first["functional"]["cvar_0_875"])
    # Nuisance is calibrated in the same teacher-KL/CVaR units as the signal.
    # Algebraic relative tolerance is dimensionless and must not be inserted as
    # an absolute CVaR floor.
    official_rebatch = teacher_kl(
        teacher_logits, next_token_logits(model, tok, screen_prompts, batch_size=8), lock.cvar_alpha
    )["cvar_0_875"]
    order = list(reversed(range(len(screen_prompts))))
    teacher_reordered = teacher_logits[order]
    candidate_reordered = next_token_logits(model, tok, [screen_prompts[index] for index in order], batch_size=16)
    official_reorder = teacher_kl(teacher_reordered, candidate_reordered, lock.cvar_alpha)["cvar_0_875"]
    official_primary = official_first["functional"]["cvar_0_875"]
    nuisance_components = {
        "cold_replay": duplicate_noise,
        "rebatch": abs(official_rebatch - official_primary),
        "batch_permutation": abs(official_reorder - official_primary),
        "fp32_identity": teacher_kl(teacher_logits, teacher_logits.clone(), lock.cvar_alpha)["cvar_0_875"],
    }
    base_delta = endpoint.deltas[endpoint.last_weight_name].to(next(model.parameters()).device)
    base_action = moment.matrix_action(base_delta)
    projector = endpoint.projector.to(base_delta.device) if endpoint.projector is not None else None
    projector_audit = None
    if projector is not None:
        seed = int(hashlib.sha256(f"projector|{model_alias}|{row['case_id']}".encode()).hexdigest()[:16], 16)
        projector_audit = projector_probe_audit(projector, constraints.to(projector.device), seed=seed)
    axes = build_axes(
        base_delta=base_delta, constraints=constraints.to(base_delta.device), matvec=moment.matvec,
        lock=lock, model=model_alias, method=method.value, case_id=str(row["case_id"]),
        count=lock.phase_a_axis_count, projector=projector, base_action=base_action,
    )
    strict_official = _strict_signature(official_first["metrics"])
    candidates = []
    order = [(axis, sign) for axis in axes for sign in (-1, 1)]

    def evaluate(axis: Any, sign: int, pass_name: str) -> dict[str, Any]:
        correction = axis.correction(sign)
        applied_hash = snapshot.apply_correction(model, endpoint.last_weight_name, correction)
        observed = _observe(model, tok, row, contexts, activation_module, endpoint.z, teacher_logits, screen_prompts, lock)
        equality = float((correction @ constraints.to(correction.device)).norm().div(correction.norm() * constraints.to(correction.device).norm()).item())
        realized = compare_realized_z(observed["realized_z"], official_first["realized_z"])
        strict_equal = _strict_signature(observed["metrics"]) == strict_official
        correction_action = axis.gamma * axis.gamma * axis.unit_action
        cross = float(sign * axis.gamma * axis.cross)
        total_action = axis.base_action + 2.0 * cross + correction_action
        action_ratio = total_action / axis.base_action
        valid = (
            equality <= lock.fp32_relative_tolerance
            and realized["maximum_relative"] <= realized_tolerance
            and abs(action_ratio - 1.05) <= lock.fp32_relative_tolerance
            and abs(cross) / math.sqrt(axis.base_action * correction_action) <= lock.fp32_relative_tolerance
            and strict_equal
        )
        restore = snapshot.restore_after_candidate(model)
        return {
            "candidate_id": f"{axis.axis_id}-{'plus' if sign > 0 else 'minus'}",
            "pass": pass_name, "axis": axis.axis_id, "sign": sign, "seed": axis.seed,
            "candidate_weight_sha256": applied_hash, "snapshot_restore": restore,
            "equality_NA_star_relative": equality,
            "actual_action": {
                "Q_delta_off": axis.base_action, "Q_N": correction_action,
                "cross_delta_N": cross, "Q_total": total_action, "total_ratio": action_ratio,
            },
            "realized_z": realized,
            "strict_equal_official": strict_equal,
            "valid": valid,
            "observation": _public_observation(observed),
        }

    for ordinal, (axis, sign) in enumerate(order):
        candidates.append(evaluate(axis, sign, "forward"))
        if ordinal in {0, len(order) - 1}:
            snapshot.copy_to_model(model)
            replay = _observe(model, tok, row, contexts, activation_module, endpoint.z, teacher_logits, screen_prompts, lock)
            duplicate_noise = max(duplicate_noise, abs(replay["functional"]["cvar_0_875"] - official_first["functional"]["cvar_0_875"]))
    reverse = []
    if reverse_audit:
        for axis, sign in reversed(order):
            reverse.append(evaluate(axis, sign, "reverse"))
    forward_map = {candidate["candidate_id"]: candidate for candidate in candidates}
    reverse_map = {candidate["candidate_id"]: candidate for candidate in reverse}
    reverse_delta = max(
        [abs(forward_map[key]["observation"]["functional"]["cvar_0_875"] - reverse_map[key]["observation"]["functional"]["cvar_0_875"]) for key in reverse_map]
        or [0.0]
    )
    cvars = [candidate["observation"]["functional"]["cvar_0_875"] for candidate in candidates if candidate["valid"]]
    spread = max(cvars) - min(cvars) if cvars else 0.0
    nuisance_components["candidate_reverse_order"] = reverse_delta
    nuisance = max(nuisance_components.values())
    terminal_restore = restore_originals(model, endpoint.originals)
    if not terminal_restore:
        raise ScientificBoundary("terminal W0 restore failed")
    return {
        "schema": "odeedit.s06.barrier-usefulness.phase-a-case.v1",
        "model": model_alias, "method": method.value, "case_id": int(row["case_id"]),
        "direct_z_compute_count": 1, "direct_z_recompute_count": 0,
        "z_sha256": tensor_sha256(endpoint.z), "native_context_count": len(contexts),
        "constraints": constraint_receipt, "official_snapshot_root": snapshot.root,
        "official": _public_observation(official_first),
        "cold_replay": {"realized_z": replay_difference, "functional_duplicate_noise": duplicate_noise},
        "realized_z_tolerance": realized_tolerance, "projector_audit": projector_audit,
        "registered_action": moment.receipt(), "base_action": base_action,
        "candidates": candidates, "reverse_order_audit": reverse,
        "reverse_order_cvar_max_abs": reverse_delta,
        "valid_candidate_count": sum(int(candidate["valid"]) for candidate in candidates),
        "candidate_denominator": len(candidates), "functional_spread": spread,
        "nuisance_components_cvar_units": nuisance_components,
        "epsilon_nuisance": nuisance, "spread_gt_3x_nuisance": spread > 3.0 * nuisance,
        "w0_pointer_bytes_restore": True, "full_fp32": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    lock = BarrierLock()
    spec = MODEL_SPECS[args.model]
    cases, roles = _load_rows(args.dataset, args.case_manifest)
    selected = cases[:1] if args.stage == "sentinel" else cases
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_path, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        device_map={"": 0}, trust_remote_code=False,
    )
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TechnicalBoundary("FULL_FP32 model closure failed")
    model.eval()
    tok = AutoTokenizer.from_pretrained(spec.model_path, local_files_only=True, use_fast=True, trust_remote_code=False)
    padding_binding = bind_padding(tok, model, padding_side="right")
    hparams_rel = spec.alpha_hparams if args.method == Method.ALPHAEDIT.value else spec.memit_hparams
    hparams = _load_hparams(Method(args.method), args.source_root / hparams_rel)
    calibration = roles["calibration"][:3]
    padding_gate = padding_safety_gate(
        model, tok, [_prompt(row) for row in calibration], [row["requested_rewrite"]["target_new"]["str"] for row in calibration],
        hparams.layer_module_tmp.format(hparams.layers[-1]),
        # The reused left-padding gate only consumes these common fields.
        type("PaddingLock", (), {"fp32_relative_tolerance": lock.fp32_relative_tolerance})(),
        input_module=hparams.rewrite_module_tmp.format(hparams.layers[-1]),
        subject_templates=[row["requested_rewrite"]["prompt"] for row in calibration],
        subjects=[row["requested_rewrite"]["subject"] for row in calibration],
    )
    stats_path = spec.statistics_root / spec.statistics_model_dir / "wikipedia_stats" / "model.layers.8.mlp.down_proj_float32_mom2_100000.npz"
    moment = SecondMoment(stats_path, next(model.parameters()).device, scale=float(hparams.mom2_update_weight), additive=float(getattr(hparams, "L2", 0.0)))
    outputs = []
    for ordinal, row in enumerate(selected):
        outputs.append(run_case(
            model=model, tok=tok, method=Method(args.method), hparams=hparams, row=row, roles=roles,
            moment=moment, lock=lock, model_alias=args.model, reverse_audit=ordinal == 0,
        ))
        torch.cuda.empty_cache()
    valid = sum(case["valid_candidate_count"] == case["candidate_denominator"] for case in outputs)
    spread = sum(case["spread_gt_3x_nuisance"] for case in outputs)
    spreads = sorted(case["functional_spread"] for case in outputs)
    nuisance = sorted(case["epsilon_nuisance"] for case in outputs)
    gate = None if args.stage == "sentinel" else {
        "valid_cases": valid, "valid_denominator": 8, "valid_gate": valid >= 7,
        "spread_gt_3x_nuisance_cases": spread, "spread_denominator": 8, "spread_gate": spread >= 6,
        "median_spread": float(np.median(spreads)), "median_nuisance": float(np.median(nuisance)),
        "median_gate": float(np.median(spreads)) > 5.0 * float(np.median(nuisance)),
    }
    if gate is not None:
        gate["phase_a_pass"] = bool(gate["valid_gate"] and gate["spread_gate"] and gate["median_gate"])
    return {
        "schema": "odeedit.s06.barrier-usefulness.phase-a-run.v1", "status": "TERMINAL_VALID",
        "stage": args.stage, "model": args.model, "method": args.method, "case_ids": [int(row["case_id"]) for row in selected],
        "padding_binding": padding_binding, "padding_safety_gate": padding_gate,
        "cases": outputs, "engineering_gate": gate, "full_fp32": True,
        "q_gate_edited_access_count": 0, "final_audit_open_count": 0,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "source": {"head": os.popen(f"git -C {args.source_root} rev-parse HEAD").read().strip()},
        "case_manifest_sha256": file_sha256(args.case_manifest),
        "scientific_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["sentinel", "screen"], required=True)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--method", choices=[item.value for item in Method], required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_once(args.output, run(args))


if __name__ == "__main__":
    main()
