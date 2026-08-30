"""One-model/method smoke and exact eight-case engineering screen runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .algebra import candidate_action, generate_axes
from .contracts import (
    MODEL_ENDPOINT_TOLERANCE,
    MODEL_SPECS,
    Method,
    NumericalLock,
    ScientificBoundary,
    TechnicalBoundary,
)
from .evaluation import (
    capture_subject_keys,
    next_token_logits,
    padding_safety_gate,
    relative_error,
    sequence_metrics,
    target_path_observation,
    teacher_kl,
)
from .manifest import canonical_hash
from .official import restore_originals, run_official_once, set_official_endpoint
from .padding import OfficialTokenizerHook, bind_padding


def tensor_sha(value: torch.Tensor) -> str:
    raw = value.detach().to("cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_requests(dataset: Path, manifest: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = json.loads(dataset.read_text())
    by_id = {int(row["case_id"]): row for row in rows}
    seal = json.loads(manifest.read_text())
    screen = [by_id[int(row["case_id"])] for row in seal["screen_cases"]]
    roles = {role: [by_id[int(row["case_id"])] for row in selected] for role, selected in seal["roles"].items() if role != "final_audit_sealed"}
    return screen, roles


def official_request(row: dict[str, Any]) -> dict[str, Any]:
    request = row["requested_rewrite"]
    return {
        "case_id": str(row["case_id"]),
        "prompt": request["prompt"],
        "subject": request["subject"],
        "target_new": request["target_new"]["str"],
        "target_true": request["target_true"]["str"],
    }


def prompt(row: dict[str, Any]) -> str:
    req = row["requested_rewrite"]
    return req["prompt"].format(req["subject"])


def _load_hparams(method: Method, path: Path) -> Any:
    if method is Method.ALPHAEDIT:
        from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams
        return AlphaEditHyperParams.from_hparams(str(path))
    from easyeditor.models.memit.memit_hparams import MEMITHyperParams
    return MEMITHyperParams.from_hparams(str(path))


def _covariance(path: Path, device: torch.device, *, scale: float, additive: float) -> tuple[Any, dict[str, Any]]:
    archive = np.load(path, mmap_mode="r")
    raw = archive["mom2.mom2"]
    count = int(archive["mom2.count"])

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        cpu = vector.detach().to("cpu", torch.float32).numpy()
        result = (raw @ cpu) / count
        result = scale * result + additive * cpu
        return torch.from_numpy(np.asarray(result)).to(device=device, dtype=torch.float32)

    return matvec, {"count": count, "sample_size": int(archive["sample_size"]), "shape": list(raw.shape), "dtype": str(raw.dtype), "scale": scale, "additive": additive}


def _request_eval(model: Any, tok: Any, row: dict[str, Any]) -> dict[str, Any]:
    req = row["requested_rewrite"]
    rewrite = [req["prompt"].format(req["subject"])]
    rephrase = list(row.get("paraphrase_prompts", []))
    locality = list(row.get("neighborhood_prompts", []))
    return {
        "rewrite_target_new": sequence_metrics(model, tok, rewrite, req["target_new"]["str"]),
        "rewrite_target_true": sequence_metrics(model, tok, rewrite, req["target_true"]["str"]),
        "rephrase_target_new": sequence_metrics(model, tok, rephrase, req["target_new"]["str"]),
        "rephrase_target_true": sequence_metrics(model, tok, rephrase, req["target_true"]["str"]),
        "locality_target_true": sequence_metrics(model, tok, locality, req["target_true"]["str"]),
    }


def _candidate_observation(
    model: Any,
    tok: Any,
    row: dict[str, Any],
    hparams: Any,
    screen_prompts: list[str],
    teacher_logits: torch.Tensor,
) -> dict[str, Any]:
    req = row["requested_rewrite"]
    canonical = req["prompt"].format(req["subject"])
    target_path = target_path_observation(
        model,
        tok,
        canonical,
        req["target_new"]["str"],
        input_module=hparams.rewrite_module_tmp.format(hparams.layers[-1]),
        activation_module=hparams.layer_module_tmp.format(hparams.layers[-1]),
    )
    functional = teacher_kl(teacher_logits, next_token_logits(model, tok, screen_prompts), NumericalLock().cvar_alpha)
    return {"target_path": target_path, "metrics": _request_eval(model, tok, row), "functional": functional}


def _serializable_observation(value: dict[str, Any]) -> dict[str, Any]:
    path = value["target_path"]
    return {
        "target_path": {
            "key_sha256": tensor_sha(path["keys"]),
            "activation_sha256": tensor_sha(path["activations"]),
            "logits_sha256": tensor_sha(path["logits"]),
            "target_ids": [int(v) for v in path["target_ids"]],
            "nll": path["nll"],
            "margin": path["margin"],
            "strict": path["strict"],
            "token_count": path["token_count"],
        },
        "metrics": value["metrics"],
        "functional": value["functional"],
    }


def run_case(
    *,
    model: Any,
    tok: Any,
    method: Method,
    hparams: Any,
    row: dict[str, Any],
    role_rows: dict[str, list[dict[str, Any]]],
    stats_path: Path,
    lock: NumericalLock,
    model_alias: str,
) -> dict[str, Any]:
    request = official_request(row)
    tokenizer_hook = OfficialTokenizerHook(tok, [])
    tok.padding_side = "right"
    endpoint = run_official_once(method=method, model=model, tokenizer=tokenizer_hook, request=request, hparams=hparams)
    if endpoint.direct_z_compute_count != 1:
        raise ScientificBoundary("direct-z count differs")
    # Official endpoint is currently materialized.
    history_templates = [item["requested_rewrite"]["prompt"] for item in role_rows["history"]]
    history_subjects = [item["requested_rewrite"]["subject"] for item in role_rows["history"]]
    history_keys = capture_subject_keys(
        model, tok, history_templates, history_subjects, hparams.rewrite_module_tmp.format(hparams.layers[-1])
    )
    screen_prompts = [prompt(item) for item in role_rows["screen"]]
    # Teacher is W0, so capture it after restoring; official/reference risk is
    # still evaluated relative to exactly the same W0 logits.
    official_target = target_path_observation(
        model, tok, prompt(row), request["target_new"],
        input_module=hparams.rewrite_module_tmp.format(hparams.layers[-1]),
        activation_module=hparams.layer_module_tmp.format(hparams.layers[-1]),
    )
    official_metrics = _request_eval(model, tok, row)
    official_weight_snapshot = {name: dict(model.named_parameters())[name].detach().clone() for name in endpoint.originals}
    if not restore_originals(model, endpoint.originals):
        raise ScientificBoundary("W0 bytes restore failed after Official endpoint")
    teacher_logits = next_token_logits(model, tok, screen_prompts)
    set_official_endpoint(model, endpoint)
    reference = _candidate_observation(model, tok, row, hparams, screen_prompts, teacher_logits)
    gamma0_endpoint_match = all(torch.equal(dict(model.named_parameters())[name], weight) for name, weight in official_weight_snapshot.items())
    gamma0_target_logit_relative = relative_error(reference["target_path"]["logits"], official_target["logits"])
    if not gamma0_endpoint_match or gamma0_target_logit_relative > lock.fp32_relative_tolerance or reference["metrics"] != official_metrics:
        raise ScientificBoundary("gamma=0 Official parity failed")
    edit_key = endpoint.edit_key.to(next(model.parameters()).device).reshape(-1, 1)
    history_keys = history_keys.to(edit_key.device)
    # target_path keys are already [d_in, target_token_count].
    target_keys = reference["target_path"]["keys"].to(edit_key.device)
    constraints = torch.cat([edit_key, history_keys, target_keys], dim=1)
    base_delta = endpoint.deltas[endpoint.last_weight_name].to(edit_key.device)
    projector = None if endpoint.projector is None else endpoint.projector.to(edit_key.device)
    scale = float(hparams.mom2_update_weight)
    additive = float(getattr(hparams, "L2", 0.0))
    matvec, cov_meta = _covariance(stats_path, edit_key.device, scale=scale, additive=additive)
    axes, factor = generate_axes(
        base_delta=base_delta,
        constraints=constraints,
        covariance_matvec=matvec,
        lock=lock,
        model_alias=model_alias,
        method=method.value,
        case_id=request["case_id"],
        projector=projector,
    )
    del projector
    params = dict(model.named_parameters())
    last_weight = params[endpoint.last_weight_name]
    records = []
    reference_serial = _serializable_observation(reference)
    for duplicate in range(lock.duplicate_reference_evaluations):
        observed = _candidate_observation(model, tok, row, hparams, screen_prompts, teacher_logits)
        records.append({
            "candidate_id": f"reference-duplicate-{duplicate}", "baseline": "official", "axis": None, "sign": 0,
            "seed": None, "rho": 0.0, "gamma": 0.0, "z_sha256": tensor_sha(endpoint.z),
            "valid": True, "failure": None, "observation": _serializable_observation(observed),
        })
    axis_actions: dict[int, dict[str, float]] = {}
    for axis in axes:
        axis_actions[axis.axis] = {}
        for sign in (-1, 1):
            correction = axis.correction(sign)
            with torch.no_grad():
                last_weight.add_(correction.to(last_weight.device, last_weight.dtype))
            observed = _candidate_observation(model, tok, row, hparams, screen_prompts, teacher_logits)
            correction_key_residual = float((correction @ edit_key).norm().div(correction.norm() * edit_key.norm()).item())
            correction_history_residual = float((correction @ history_keys).norm().div(correction.norm() * history_keys.norm()).item())
            correction_target_residual = float((correction @ target_keys).norm().div(correction.norm() * target_keys.norm()).item())
            activation_error = relative_error(observed["target_path"]["activations"], reference["target_path"]["activations"])
            logit_error = relative_error(observed["target_path"]["logits"], reference["target_path"]["logits"])
            action = candidate_action(axis, sign)
            axis_actions[axis.axis][str(sign)] = action
            algebra_valid = max(
                correction_key_residual,
                correction_history_residual,
                correction_target_residual,
                axis.cross_action_relative,
            ) <= lock.fp32_relative_tolerance
            full_model_valid = max(activation_error, logit_error) <= MODEL_ENDPOINT_TOLERANCE[model_alias]
            valid = algebra_valid and full_model_valid and observed["target_path"]["strict"] == reference["target_path"]["strict"]
            records.append({
                "candidate_id": f"axis-{axis.axis}-{'plus' if sign > 0 else 'minus'}", "baseline": "official",
                "axis": axis.axis, "sign": sign, "seed": axis.seed, "rho": lock.rho_tangent, "gamma": axis.gamma,
                "z_sha256": tensor_sha(endpoint.z), "key_sha256": tensor_sha(constraints),
                "equality": {"NK_E": correction_key_residual, "NK_H": correction_history_residual, "NK_T": correction_target_residual, "target_activation": activation_error, "target_logits": logit_error, "algebra_tolerance": lock.fp32_relative_tolerance, "full_model_tolerance": MODEL_ENDPOINT_TOLERANCE[model_alias]},
                "action": action, "cross_action_relative": axis.cross_action_relative,
                "rank": axis.rank, "valid": valid, "failure": None if valid else "FIXED_Z_OR_ACTION_IDENTITY",
                "observation": _serializable_observation(observed),
            })
            with torch.no_grad():
                last_weight.sub_(correction.to(last_weight.device, last_weight.dtype))
    pair_mismatch = {str(axis): abs(values["1"] - values["-1"]) / max(values["1"], values["-1"]) for axis, values in axis_actions.items()}
    if max(pair_mismatch.values()) > lock.fp32_relative_tolerance:
        raise ScientificBoundary("paired covariance action mismatch")
    restored = restore_originals(model, endpoint.originals)
    if not restored:
        raise ScientificBoundary("terminal W0 bytes restore failed")
    cvars = [row["observation"]["functional"]["cvar_0_875"] for row in records if row["axis"] is not None and row["valid"]]
    duplicates = [row["observation"]["functional"]["cvar_0_875"] for row in records if row["axis"] is None]
    spread = max(cvars) - min(cvars) if cvars else None
    noise = max(duplicates) - min(duplicates) if duplicates else None
    return {
        "schema": "odeedit.s06.fixed-z-nonuniqueness.case-result.v1",
        "model": model_alias, "method": method.value, "case_id": request["case_id"], "layer": int(hparams.layers[-1]),
        "direct_z_compute_count": 1, "direct_z_recompute_count": 0, "z_sha256": tensor_sha(endpoint.z),
        "official": {"gamma0_endpoint_exact": gamma0_endpoint_match, "gamma0_target_logit_relative": gamma0_target_logit_relative, "metrics": official_metrics, "target_path": _serializable_observation({"target_path": official_target, "metrics": {}, "functional": {}})["target_path"]},
        "constraints": {"edit": 1, "history": history_keys.shape[1], "target": target_keys.shape[1], "sha256": tensor_sha(constraints)},
        "covariance": cov_meta, "factor": factor, "pair_action_relative_mismatch": pair_mismatch,
        "reference": reference_serial, "candidates": records,
        "valid_candidate_count": sum(int(row["valid"]) for row in records if row["axis"] is not None),
        "candidate_denominator": 8, "functional_spread": spread, "duplicate_noise": noise,
        "spread_gt_3x_noise": bool(spread is not None and noise is not None and spread > 3.0 * noise),
        "w0_pointer_bytes_restore": True, "full_fp32": True, "final_audit_open_count": 0,
        "tokenizer_official_calls": endpoint.tokenizer_calls,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    source_root = args.source_root.resolve()
    spec = MODEL_SPECS[args.model]
    lock = NumericalLock()
    screen, roles = load_requests(args.dataset, args.case_manifest)
    selected = screen[:1] if args.stage == "smoke" else screen
    torch.set_grad_enabled(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_path, local_files_only=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, device_map={"": 0}, trust_remote_code=False,
    )
    if any(param.dtype != torch.float32 for param in model.parameters()):
        raise TechnicalBoundary("FULL_FP32 model closure failed")
    model.eval()
    tok = AutoTokenizer.from_pretrained(spec.model_path, local_files_only=True, use_fast=True, trust_remote_code=False)
    padding_binding = bind_padding(tok, model, padding_side="right")
    calibration = roles["calibration"][:3]
    calibration_prompts = [prompt(row) for row in calibration]
    calibration_targets = [row["requested_rewrite"]["target_new"]["str"] for row in calibration]
    padding_gate = padding_safety_gate(
        model, tok, calibration_prompts, calibration_targets,
        f"model.layers.{31 if args.model.startswith('llama') else 27}", lock,
    )
    special = {
        "tokenizer_class": type(tok).__name__, "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id,
        "padding_binding": padding_binding, "chat_template_present": bool(getattr(tok, "chat_template", None)),
        "add_special_tokens_ids": tok.encode(calibration_prompts[0], add_special_tokens=True)[:8],
        "without_special_tokens_ids": tok.encode(calibration_prompts[0], add_special_tokens=False)[:8],
    }
    hparams_path = source_root / (spec.alpha_hparams if args.method == "alphaedit" else spec.memit_hparams)
    hparams = _load_hparams(Method(args.method), hparams_path)
    if str(hparams.model_name) != spec.statistics_model_dir:
        raise TechnicalBoundary("Official statistics model alias mismatch")
    stats_path = spec.statistics_root / spec.statistics_model_dir / "wikipedia_stats" / "model.layers.8.mlp.down_proj_float32_mom2_100000.npz"
    cases = []
    for row in selected:
        cases.append(run_case(model=model, tok=tok, method=Method(args.method), hparams=hparams, row=row, role_rows=roles, stats_path=stats_path, lock=lock, model_alias=args.model))
        torch.cuda.empty_cache()
    valid_cases = sum(case["valid_candidate_count"] == 8 for case in cases)
    spread_cases = sum(case["spread_gt_3x_noise"] for case in cases)
    spreads = sorted(case["functional_spread"] for case in cases if case["functional_spread"] is not None)
    noises = sorted(case["duplicate_noise"] for case in cases if case["duplicate_noise"] is not None)
    median_spread = spreads[len(spreads) // 2] if spreads else None
    median_noise = noises[len(noises) // 2] if noises else None
    gate = None if args.stage == "smoke" else {
        "valid_cases": valid_cases, "valid_denominator": 8, "valid_gate": valid_cases >= 7,
        "spread_gt_3x_noise_cases": spread_cases, "spread_denominator": 8, "spread_gate": spread_cases >= 6,
        "median_spread": median_spread, "median_noise": median_noise,
        "median_gate": bool(median_spread is not None and median_noise is not None and median_spread > 5.0 * median_noise),
    }
    return {
        "schema": "odeedit.s06.fixed-z-nonuniqueness.run-result.v1", "status": "TERMINAL_VALID",
        "stage": args.stage, "model": args.model, "method": args.method, "full_fp32": True,
        "padding_safety_gate": padding_gate, "tokenizer_special_boundary": special,
        "official_statistics_model_name_binding": str(hparams.model_name),
        "case_manifest_sha256": hashlib.sha256(args.case_manifest.read_bytes()).hexdigest(),
        "case_ids": [int(row["case_id"]) for row in selected], "cases": cases, "engineering_gate": gate,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "scientific_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "screen"], required=True)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--method", choices=[item.value for item in Method], required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite result")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    os.chmod(args.output, 0o600)


if __name__ == "__main__":
    main()
