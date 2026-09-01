"""Llama/Qwen one-request Official dense Alpha replay (HA1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch

from project.run_scripts.fixed_z_nonuniqueness.contracts import NumericalLock as PaddingNumericalLock
from project.run_scripts.fixed_z_nonuniqueness.evaluation import padding_safety_gate

from .contracts import EngineeringBoundary, NumericalLock
from .dense_backend import ContextIncidence, DenseEntryFactor, match_update_to_weight
from .model_support import (
    MODEL_SPECS,
    load_case_rows,
    load_hparams,
    load_model_and_tokenizer,
    official_model_name,
    official_requests,
    restore_selected,
    selected_weight_identity,
    selected_weights,
    tensor_sha,
    tokenizer_hook,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def relative(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(left.float() - right.float())
        .div(torch.linalg.vector_norm(right.float()).clamp_min(torch.finfo(torch.float32).tiny))
        .item()
    )


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left.reshape(1, -1).float(), right.reshape(1, -1).float()).item())


def direct_official_update(
    projector: torch.Tensor,
    history: torch.Tensor,
    keys: torch.Tensor,
    residual: torch.Tensor,
    l2: float,
) -> tuple[torch.Tensor, float]:
    identity = torch.eye(keys.shape[0], dtype=keys.dtype, device=keys.device)
    matrix = projector @ (keys @ keys.T + history) + l2 * identity
    rhs = projector @ keys @ residual.T
    solution = torch.linalg.solve(matrix, rhs)
    equation_residual = torch.linalg.vector_norm(matrix @ solution - rhs) / torch.linalg.vector_norm(rhs).clamp_min(
        torch.finfo(rhs.dtype).tiny
    )
    return solution.T, float(equation_residual.item())


def backend_update(
    projector: torch.Tensor,
    history: torch.Tensor,
    keys: torch.Tensor,
    residual: torch.Tensor,
    l2: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    factor = DenseEntryFactor.build(projector, history, l2)
    solve = factor.woodbury(keys)
    direct_d = factor.direct(keys)
    update = solve.update(residual)
    return update, {
        "woodbury_direct_d_relative": float(solve.direct_relative(direct_d).item()),
        "solve_relative": float(solve.solve_relative.item()),
        "projector_leakage": float(solve.coefficient_leakage(projector).item()),
        "small_condition": float(torch.linalg.cond(solve.small.double()).item()),
        "d_sha256": tensor_sha(solve.d),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    source_root = args.source_root.resolve()
    if git(source_root, "rev-parse", "HEAD") != args.expected_head or git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise EngineeringBoundary("source HEAD/clean gate failed")
    if sha256(args.preflight) != args.preflight_sha256:
        raise EngineeringBoundary("preflight receipt SHA mismatch")
    spec = MODEL_SPECS[args.model]
    rows = load_case_rows(args.dataset, args.case_manifest, 3)
    request = official_requests(rows[:1])
    model, tokenizer, model_receipt = load_model_and_tokenizer(spec)
    hparams = load_hparams(spec, args.official_root)
    calibration_prompts = [row["requested_rewrite"]["prompt"].format(row["requested_rewrite"]["subject"]) for row in rows]
    calibration_targets = [row["requested_rewrite"]["target_new"]["str"] for row in rows]
    padding = padding_safety_gate(
        model,
        tokenizer,
        calibration_prompts,
        calibration_targets,
        hparams.layer_module_tmp.format(hparams.layers[-1]),
        PaddingNumericalLock(),
        input_module=hparams.rewrite_module_tmp.format(hparams.layers[-1]),
        subject_templates=[row["requested_rewrite"]["prompt"] for row in rows],
        subjects=[row["requested_rewrite"]["subject"] for row in rows],
    )
    official_tokenizer = tokenizer_hook(tokenizer)
    from easyeditor.models.alphaedit import AlphaEdit_main as official
    from easyeditor.models.alphaedit.compute_z import get_module_input_output_at_words

    weights = selected_weights(model, hparams)
    pointers = {name: int(value.data_ptr()) for name, value in weights.items()}
    entry = {name: value.detach().clone() for name, value in weights.items()}
    w0_sha = selected_weight_identity(weights)
    projector_all = torch.load(spec.projector, map_location="cpu", weights_only=True, mmap=True)
    context_templates = official.get_context_templates(model, official_tokenizer)
    z_layer = int(hparams.layers[-1])
    with official_model_name(model, spec.statistics_alias):
        z = official.compute_z(model, official_tokenizer, request[0], hparams, z_layer, context_templates)
    zs = z.reshape(-1, 1)
    layer_receipts = []
    for layer_index, layer in enumerate(hparams.layers):
        keys = official.compute_ks(model, official_tokenizer, request, hparams, layer, context_templates).T.float()
        current = get_module_input_output_at_words(
            model,
            official_tokenizer,
            z_layer,
            context_templates=[request[0]["prompt"]],
            words=[request[0]["subject"]],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T.float()
        target_residual = zs.float() - current
        repeat_factor = keys.shape[1] // target_residual.shape[1]
        if repeat_factor * target_residual.shape[1] != keys.shape[1]:
            raise EngineeringBoundary("nonintegral Official key repeat factor")
        residual = target_residual.repeat_interleave(repeat_factor, dim=1) / (len(hparams.layers) - layer_index)
        incidence = ContextIncidence.from_context_request_ids(
            (int(request[0]["case_id"]),),
            tuple(int(request[0]["case_id"]) for _ in range(keys.shape[1])),
            dtype=keys.dtype,
            device=keys.device,
        )
        if incidence.repeat_factor != repeat_factor:
            raise EngineeringBoundary("Official repeat/incidence mismatch")
        projector = projector_all[layer_index].to(keys.device, torch.float32)
        histories = {
            "zero": torch.zeros_like(projector),
            "nonzero_real_key": (0.125 * (keys @ keys.T)).float(),
        }
        parity = {}
        authoritative_update = None
        for history_name, history in histories.items():
            official_update, official_residual = direct_official_update(
                projector, history, keys, residual, float(hparams.L2)
            )
            new_update, backend = backend_update(projector, history, keys, residual, float(hparams.L2))
            matched_official, official_transposed = match_update_to_weight(official_update, weights[f"{hparams.rewrite_module_tmp.format(layer)}.weight"].shape)
            matched_new, new_transposed = match_update_to_weight(new_update, weights[f"{hparams.rewrite_module_tmp.format(layer)}.weight"].shape)
            difference = relative(matched_new, matched_official)
            if difference > NumericalLock().fp32_backend_relative_tolerance:
                raise EngineeringBoundary(f"Official/backend parity failure layer={layer} history={history_name}")
            parity[history_name] = {
                "official_equation_residual": official_residual,
                "relative_update_difference": difference,
                "update_cosine": cosine(matched_new, matched_official),
                "official_update_sha256": tensor_sha(matched_official),
                "backend_update_sha256": tensor_sha(matched_new),
                "official_transposed": official_transposed,
                "backend_transposed": new_transposed,
                **backend,
            }
            if history_name == "zero":
                authoritative_update = matched_new
            del history, official_update, new_update, matched_official, matched_new
            torch.cuda.empty_cache()
        if authoritative_update is None:
            raise EngineeringBoundary("zero-history authoritative update missing")
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        with torch.no_grad():
            weights[weight_name].add_(authoritative_update)
        layer_receipts.append(
            {
                "layer": int(layer),
                "keys_shape": list(keys.shape),
                "keys_sha256": tensor_sha(keys),
                "target_residual_sha256": tensor_sha(target_residual),
                "residual_sha256": tensor_sha(residual),
                "repeat_factor": repeat_factor,
                "incidence_shape": list(incidence.matrix.shape),
                "incidence_sha256": tensor_sha(incidence.matrix),
                "parity": parity,
            }
        )
        del projector, histories, keys, residual, target_residual, current, authoritative_update
        torch.cuda.empty_cache()
    post_keys = {}
    for layer in hparams.layers:
        key = official.compute_ks(model, official_tokenizer, request, hparams, layer, context_templates).T.float()
        post_keys[int(layer)] = {"shape": list(key.shape), "sha256": tensor_sha(key)}
    terminal_weight_sha = selected_weight_identity(weights)
    restored = restore_selected(weights, entry)
    pointer_restored = all(int(weights[name].data_ptr()) == pointers[name] for name in weights)
    bytes_restored = restored and selected_weight_identity(weights) == w0_sha
    if not pointer_restored or not bytes_restored:
        raise EngineeringBoundary("W0 pointer/bytes restore failure")
    if not official_tokenizer.call_identities or not all(item["semantic_input_attention_position_equal"] for item in official_tokenizer.call_identities):
        raise EngineeringBoundary("Official tokenizer semantic identity receipt missing")
    return {
        "schema": "odeedit.s06.fzcb-hard-alpha-densec.ha1-official-replay.v1",
        "status": "TERMINAL_VALID",
        "stage": "HA1_B1",
        "model": args.model,
        "case_id": int(request[0]["case_id"]),
        "source": {"head": args.expected_head, "tree": git(source_root, "rev-parse", "HEAD^{tree}")},
        "preflight": {"path": str(args.preflight), "sha256": args.preflight_sha256},
        "model_receipt": model_receipt,
        "padding_gate": padding,
        "official_source_root": str(args.official_root),
        "official_target_compute_count": 1,
        "official_target_recompute_count": 0,
        "z_sha256": tensor_sha(z),
        "w0_selected_sha256": w0_sha,
        "terminal_selected_sha256": terminal_weight_sha,
        "layers": layer_receipts,
        "post_commit_key_candidates": post_keys,
        "cache_append_count": 0,
        "cache_mutation_count": 0,
        "w0_pointer_restore": pointer_restored,
        "w0_bytes_restore": bytes_restored,
        "tokenizer_calls": official_tokenizer.call_identities,
        "full_fp32": True,
        "bf16_fp16_conversion_count": 0,
        "autocast_count": 0,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "scientific_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite HA1 result")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    os.chmod(args.output, 0o600)


if __name__ == "__main__":
    main()
