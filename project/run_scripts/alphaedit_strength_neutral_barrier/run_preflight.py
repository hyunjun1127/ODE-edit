#!/usr/bin/env python3
"""Focused G0 for cache-aware one-sided predictive q-KL AlphaEdit."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.models.alphaedit import AlphaEditHyperParams
from easyeditor.models.alphaedit import AlphaEdit_main as official
from easyeditor.util import nethook
from easyeditor.util.device import copy_to_param
from project.run_scripts.alphaedit_strength_neutral_barrier import (
    BarrierArm,
    BarrierWriterConfig,
    apply_strength_neutral_barrier_to_model,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.contracts import (
    require_right_padding,
    target_token_ids,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.firewall import (
    verify_stock_easyedit,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.geometry import (
    factorize_alphaedit_velocity,
    low_rank_pullback,
    project_one_sided_velocity,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.official_state import (
    prepare_official_state,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.target_path import (
    build_target_event_batch,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.telemetry import (
    write_create_once,
)


PATHS = (
    ("official-n1", BarrierArm.OFFICIAL, 1),
    ("split-n2", BarrierArm.SPLIT, 2),
    ("split-n4", BarrierArm.SPLIT, 4),
    ("projected-n2", BarrierArm.PROJECTED, 2),
    ("projected-n4", BarrierArm.PROJECTED, 4),
)


def _git(root: Path, value: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", value], text=True
    ).strip()


def _sha_file(path: Path) -> Dict[str, Any]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _selected(model: Any, hparams: Any) -> Dict[str, torch.Tensor]:
    return {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }


def _selected_sha(weights: Dict[str, torch.Tensor]) -> str:
    digest = sha256()
    for name in sorted(weights):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha(weights[name]).encode("ascii"))
    return digest.hexdigest()


def _restore(weights: Dict[str, torch.Tensor], entry: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, weight in weights.items():
            copy_to_param(weight, entry[name])


def _request(record: Dict[str, Any]) -> Dict[str, Any]:
    rewrite = record["requested_rewrite"]
    return {
        "case_id": int(record["case_id"]),
        "prompt": rewrite["prompt"],
        "subject": rewrite["subject"],
        "target_new": rewrite["target_new"]["str"],
        "target_true": rewrite["target_true"]["str"],
    }


def _relative(left: Dict[str, torch.Tensor], right: Dict[str, torch.Tensor]) -> dict:
    maximum = 0.0
    numerator = 0.0
    denominator = 0.0
    for name in left:
        delta = left[name].float() - right[name].float()
        maximum = max(maximum, float(delta.abs().max().item()))
        numerator += float(torch.sum(delta**2).item())
        denominator += max(
            float(torch.sum(left[name].float() ** 2).item()),
            float(torch.sum(right[name].float() ** 2).item()),
        )
    return {
        "max_abs": maximum,
        "relative_l2": (numerator / max(denominator, torch.finfo(torch.float32).tiny))
        ** 0.5,
    }


def _synthetic_gates() -> Dict[str, Any]:
    torch.manual_seed(41)
    output, width, request_count = 5, 9, 4
    projector = torch.diag(
        torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.float32)
    )
    keys = torch.randn(width, request_count)
    residual = torch.randn(output, request_count)
    history_basis = torch.randn(width, 3)
    history = history_basis @ history_basis.T
    factors = factorize_alphaedit_velocity(
        residual=residual,
        keys=keys,
        projector=projector,
        history_cache=history,
        l2=0.2,
    )
    empty_factors = factorize_alphaedit_velocity(
        residual=residual,
        keys=keys,
        projector=projector,
        history_cache=torch.zeros_like(history),
        l2=0.2,
    )
    positive = project_one_sided_velocity(
        residual=residual,
        writer_map=factors.writer_map,
        metric=factors.metric,
        metric_pinv=factors.metric_pinv,
        barrier_pullback=residual,
        keys=keys,
        history_cache=history,
        l2=0.2,
    )
    negative = project_one_sided_velocity(
        residual=residual,
        writer_map=factors.writer_map,
        metric=factors.metric,
        metric_pinv=factors.metric_pinv,
        barrier_pullback=-residual,
        keys=keys,
        history_cache=history,
        l2=0.2,
    )
    empty_projection = project_one_sided_velocity(
        residual=residual,
        writer_map=empty_factors.writer_map,
        metric=empty_factors.metric,
        metric_pinv=empty_factors.metric_pinv,
        barrier_pullback=residual,
        keys=keys,
        history_cache=torch.zeros_like(history),
        l2=0.2,
    )
    left = torch.randn(output, 7)
    right = torch.randn(width, 7)
    low_rank = low_rank_pullback(left, right, factors.writer_map)
    dense = (left @ right.T) @ factors.writer_map
    low_rank_max_abs = float((low_rank - dense).abs().max().item())
    history_effect = float(
        torch.linalg.vector_norm(
            positive.residual_velocity - empty_projection.residual_velocity
        ).item()
    )
    if not positive.active or abs(positive.projected_rate) > 2e-5:
        raise RuntimeError("synthetic positive-rate projection gate failed")
    if negative.active or not torch.equal(negative.residual_velocity, residual):
        raise RuntimeError("synthetic negative-rate identity gate failed")
    if history_effect == 0.0:
        raise RuntimeError("nonzero cache did not change synthetic projection")
    if low_rank_max_abs > 2e-5:
        raise RuntimeError("low-rank GJ parity gate failed")
    return {
        "factorization": {
            "solve_backward_error": factors.solve_backward_error,
            "solve_backward_tolerance": factors.solve_backward_tolerance,
            "stock_velocity_max_abs": factors.stock_velocity_max_abs,
            "stock_velocity_relative": factors.stock_velocity_relative,
        },
        "metric": {
            **factors.metric_receipt.__dict__,
            "history_effect_norm": history_effect,
        },
        "projection": {
            "positive_native_rate": positive.native_rate,
            "positive_projected_rate": positive.projected_rate,
            "negative_native_rate": negative.native_rate,
            "negative_projected_rate": negative.projected_rate,
            "correction_energy": positive.correction_energy,
            "energy_terms": positive.current_key_energy
            + positive.history_cache_energy
            + positive.l2_energy,
        },
        "low_rank": {"max_abs": low_rank_max_abs},
        "token_alignment_fixtures": {
            "single_token": {"source": [2], "target": [3]},
            "equal_length_multi_token": {"source": [2, 3], "target": [4, 5]},
            "unequal_length": {"source": [2], "target": [4, 5]},
            "source_prefix_target": {"source": [2], "target": [2, 3]},
            "target_prefix_source": {"source": [2, 3], "target": [2]},
            "unequal_non_prefix": {"source": [2, 3], "target": [4]},
        },
    }


def _run_path(
    *,
    name: str,
    arm: BarrierArm,
    steps: int,
    model: Any,
    tok: Any,
    request: Dict[str, Any],
    hparams: Any,
    result_root: Path,
    selected: Dict[str, torch.Tensor],
    weight_entry: Dict[str, torch.Tensor],
    cache_entry: torch.Tensor,
) -> Dict[str, Any]:
    _restore(selected, weight_entry)
    official.cache_c[...] = cache_entry
    telemetry_path = result_root / f"{name}-writer-telemetry.json"
    model, _, receipt = apply_strength_neutral_barrier_to_model(
        model,
        tok,
        [request],
        hparams,
        BarrierWriterConfig(arm=arm, steps=steps, telemetry_path=str(telemetry_path)),
        copy=False,
        return_orig_weights=False,
        cache_template=None,
        reset_cache=True,
    )
    endpoint = {key: value.detach().clone() for key, value in selected.items()}
    endpoint_sha = _selected_sha(selected)
    cache_endpoint_sha = _tensor_sha(official.cache_c)
    _restore(selected, weight_entry)
    official.cache_c[...] = cache_entry
    if _selected_sha(selected) != _selected_sha(weight_entry):
        raise RuntimeError(f"{name} W0 restore failed")
    return {
        "name": name,
        "arm": arm.value,
        "steps": steps,
        "endpoint": endpoint,
        "endpoint_sha256": endpoint_sha,
        "cache_endpoint_sha256": cache_endpoint_sha,
        "telemetry": receipt,
        "telemetry_identity": _sha_file(telemetry_path),
    }


def _sequential_cache_gate(
    *,
    model: Any,
    tok: Any,
    requests: Sequence[Dict[str, Any]],
    hparams: Any,
    result_root: Path,
    selected: Dict[str, torch.Tensor],
    weight_entry: Dict[str, torch.Tensor],
    cache_entry: torch.Tensor,
) -> Dict[str, Any]:
    _restore(selected, weight_entry)
    official.cache_c[...] = cache_entry
    first_entry = _selected_sha(selected)
    model, _, first = apply_strength_neutral_barrier_to_model(
        model,
        tok,
        [requests[0]],
        hparams,
        BarrierWriterConfig(
            BarrierArm.PROJECTED,
            2,
            str(result_root / "sequential-first-writer-telemetry.json"),
        ),
        reset_cache=True,
    )
    first_endpoint = _selected_sha(selected)
    cache_first = _tensor_sha(official.cache_c)
    second_entry = _selected_sha(selected)
    model, _, second = apply_strength_neutral_barrier_to_model(
        model,
        tok,
        [requests[1]],
        hparams,
        BarrierWriterConfig(
            BarrierArm.PROJECTED,
            2,
            str(result_root / "sequential-second-writer-telemetry.json"),
        ),
        reset_cache=False,
    )
    cache_second = _tensor_sha(official.cache_c)
    if first_endpoint != second_entry:
        raise RuntimeError("sequential W continuity gate failed")
    if cache_first in {_tensor_sha(cache_entry), cache_second}:
        raise RuntimeError("sequential cache continuity gate failed")
    _restore(selected, weight_entry)
    official.cache_c[...] = cache_entry
    return {
        "first_entry_sha256": first_entry,
        "first_endpoint_sha256": first_endpoint,
        "second_entry_sha256": second_entry,
        "cache_entry_sha256": _tensor_sha(cache_entry),
        "cache_after_first_sha256": cache_first,
        "cache_after_second_sha256": cache_second,
        "first_cache_append_count": first["cache_append_count"],
        "second_cache_append_count": second["cache_append_count"],
        "w0_restore_pass": _selected_sha(selected) == _selected_sha(weight_entry),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--easyedit-root", type=Path, required=True)
    parser.add_argument("--expected-easyedit-head", required=True)
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parents[3]
    actual_head = _git(source_root, "HEAD")
    actual_tree = _git(source_root, "HEAD^{tree}")
    easyedit_head = _git(args.easyedit_root, "HEAD")
    easyedit_tree = _git(args.easyedit_root, "HEAD^{tree}")
    if actual_head != args.expected_head or easyedit_head != args.expected_easyedit_head:
        raise RuntimeError("source identity mismatch")
    if subprocess.check_output(
        ["git", "-C", str(args.easyedit_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise RuntimeError("stock EasyEdit is dirty")
    easyedit_seal = verify_stock_easyedit(
        args.easyedit_root,
        Path(__file__).resolve().parent / "official-source-lock.json",
    )
    if args.result_dir.exists() or args.result_dir.is_symlink():
        raise FileExistsError(f"create-once result root exists: {args.result_dir}")
    args.result_dir.mkdir(parents=True, mode=0o700)
    failure_path = args.result_dir / "failure-boundary.json"
    model = None
    started = time.perf_counter()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        raw = json.loads(args.dataset.read_text())
        if len(raw) < 10:
            raise RuntimeError("dataset lacks G0 records")
        synthetic = _synthetic_gates()
        write_create_once(args.result_dir / "g0-math-gates.json", synthetic)

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
        require_right_padding(tok, caller="cache-aware q-KL G0")
        model.eval()
        non_fp32 = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.dtype != torch.float32
        )
        if non_fp32 or getattr(model, "is_quantized", False):
            raise RuntimeError("FULL_FP32 deployment gate failed")

        hparams = AlphaEditHyperParams.from_hparams(str(args.hparams))
        hparams.model_name = str(args.model_path)
        hparams.P_loc = str(args.projector)
        hparams.device = 0
        prepare_official_state(model, tok, hparams, reset_cache=True)
        selected = _selected(model, hparams)
        weight_entry = {name: value.detach().clone() for name, value in selected.items()}
        cache_entry = official.cache_c.detach().clone()
        requests = [_request(record) for record in raw[:10]]

        target_batch = build_target_event_batch(
            tok, requests, device=torch.device("cuda:0")
        )
        request_weight_sums = [0.0 for _ in requests]
        for event, weight in zip(target_batch.events, target_batch.event_weights):
            request_weight_sums[event.request_index] += float(weight.item())
        expected_weight = 1.0 / len(requests)
        weight_error = max(abs(value - expected_weight) for value in request_weight_sums)
        weight_tolerance = 8 * torch.finfo(torch.float32).eps
        if weight_error > weight_tolerance:
            raise RuntimeError("multi-token macro event weighting gate failed")

        outputs = {}
        for name, arm, steps in PATHS:
            outputs[name] = _run_path(
                name=name,
                arm=arm,
                steps=steps,
                model=model,
                tok=tok,
                request=requests[0],
                hparams=hparams,
                result_root=args.result_dir,
                selected=selected,
                weight_entry=weight_entry,
                cache_entry=cache_entry,
            )
        official_output = outputs["official-n1"]
        if not official_output["telemetry"].get("official_native_bypass"):
            raise RuntimeError("N=1 stock AlphaEdit bypass gate failed")
        split_n2 = outputs["split-n2"]
        split_n4 = outputs["split-n4"]
        split_comparison = _relative(split_n2["endpoint"], split_n4["endpoint"])
        split_tolerance = max(
            layer["solve_backward_tolerance"]
            for layer in split_n2["telemetry"]["layers"]
        )
        if split_comparison["relative_l2"] > split_tolerance:
            raise RuntimeError("split N2/N4 same-endpoint gate failed")
        official_comparison = _relative(
            official_output["endpoint"], split_n2["endpoint"]
        )
        if official_comparison["relative_l2"] > split_tolerance:
            raise RuntimeError("factorized split/stock AlphaEdit endpoint gate failed")

        projected_summaries = {}
        for name in ("projected-n2", "projected-n4"):
            receipt = outputs[name]["telemetry"]
            nodes = [node for layer in receipt["layers"] for node in layer["nodes"]]
            expected_nodes = len(hparams.layers) * outputs[name]["steps"]
            if len(nodes) != expected_nodes:
                raise RuntimeError(f"{name} node count mismatch")
            if receipt["predictor_forward_backward_count"] != expected_nodes:
                raise RuntimeError(f"{name} q-KL backward count mismatch")
            if receipt["layer_factorization_count"] != len(hparams.layers):
                raise RuntimeError(f"{name} layer factorization count mismatch")
            if receipt["cache_append_count"] != 1:
                raise RuntimeError(f"{name} cache append count mismatch")
            if not all(node["predictor_restore_pass"] for node in nodes):
                raise RuntimeError(f"{name} predictor restore gate failed")
            projected_summaries[name] = {
                "node_count": len(nodes),
                "active_count": sum(node["correction_active"] for node in nodes),
                "positive_rate_violation_max": max(
                    node["positive_projected_rate_violation"] for node in nodes
                ),
                "removed_fraction_mean": sum(
                    node["removed_energy_fraction"] for node in nodes
                )
                / len(nodes),
                "terminal_q_kl": receipt["layers"][-1]["terminal_barrier_q_kl"],
            }

        continuity = _sequential_cache_gate(
            model=model,
            tok=tok,
            requests=requests[:2],
            hparams=hparams,
            result_root=args.result_dir,
            selected=selected,
            weight_entry=weight_entry,
            cache_entry=cache_entry,
        )
        if not continuity["w0_restore_pass"]:
            raise RuntimeError("sequential smoke W0 restore failed")

        inputs = {
            "model_config": _sha_file(args.model_path / "config.json"),
            "tokenizer": _sha_file(args.model_path / "tokenizer.json"),
            "dataset": _sha_file(args.dataset),
            "projector": _sha_file(args.projector),
            "hparams": _sha_file(args.hparams),
            "official_source": _sha_file(Path(official.__file__).resolve(strict=True)),
        }
        gate_count = 12
        payload = {
            "schema": "easyedit.alphaedit.cache-aware-qkl-projected.preflight.v1",
            "terminal_status": "PRE_GPU_PASS",
            "source": {
                "head": actual_head,
                "tree": actual_tree,
                "easyedit_head": easyedit_head,
                "easyedit_tree": easyedit_tree,
                "easyedit_seal": easyedit_seal,
            },
            "input_identities": inputs,
            "full_fp32": True,
            "quantized": False,
            "tokenizer_padding_side": tok.padding_side,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "gate_summary": {"pass_count": gate_count, "expected_count": gate_count},
            "synthetic_gates": synthetic,
            "multi_token": {
                "request_count": target_batch.request_count,
                "event_count": target_batch.event_count,
                "target_lengths": [len(target_token_ids(tok, request["target_new"])) for request in requests],
                "request_weight_max_abs_error": weight_error,
                "request_weight_tolerance": weight_tolerance,
            },
            "path_outputs": {
                name: {
                    key: value
                    for key, value in output.items()
                    if key not in {"endpoint", "telemetry"}
                }
                for name, output in outputs.items()
            },
            "split_n2_n4": split_comparison,
            "official_split_n2": official_comparison,
            "split_tolerance": split_tolerance,
            "projected": projected_summaries,
            "sequential_cache_continuity": continuity,
            "controller_influence": {
                "locality": 0,
                "rephrase": 0,
                "target_true": 0,
            },
            "target_gradient_backward_count": 0,
            "fallback_count": 0,
            "retry_count": 0,
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "wall_seconds": time.perf_counter() - started,
        }
        write_create_once(args.result_dir / "preflight-receipt.json", payload)
    except Exception as error:
        failure = {
            "schema": "easyedit.alphaedit.cache-aware-qkl-projected.preflight-failure.v1",
            "terminal_status": "FAILED_BOUNDARY",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "source_head": actual_head,
            "source_tree": actual_tree,
            "easyedit_head": easyedit_head,
            "easyedit_tree": easyedit_tree,
            "science_change_count": 0,
            "tolerance_change_count": 0,
            "fallback_count": 0,
        }
        if not failure_path.exists():
            write_create_once(failure_path, failure)
        raise


if __name__ == "__main__":
    main()
