#!/usr/bin/env python3
"""One-GPU native-parity and multi-token alignment gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.editors.editor import BaseEditor
from easyeditor.models.alphaedit import AlphaEditHyperParams
from easyeditor.models.alphaedit import AlphaEdit_main as official
from easyeditor.models.alphaedit.compute_ks import compute_ks
from easyeditor.models.alphaedit.compute_z import get_module_input_output_at_words
from easyeditor.models.alphaedit.compute_z import find_fact_lookup_idx
from project.run_scripts.alphaedit_strength_neutral_barrier import (
    BarrierArm,
    BarrierWriterConfig,
    apply_strength_neutral_barrier_to_model,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.contracts import (
    require_right_padding,
    select_multitoken_alignment_fixtures,
    target_token_ids,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.target_path import (
    build_target_event_batch,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.telemetry import (
    write_create_once,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.endpoint_adoption import (
    fp32_comparison,
)
from project.run_scripts.alphaedit_strength_neutral_barrier import writer as guided_writer
from project.run_scripts.alphaedit_strength_neutral_barrier.firewall import (
    verify_stock_easyedit,
)
from easyeditor.util import nethook

from project.run_scripts.alphaedit_strength_neutral_barrier.evaluator import (
    evaluate_counterfact,
)


def _git_value(root: Path, value: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", value], text=True
    ).strip()


def _file_identity(
    path: Path, *, allow_pinned_snapshot_symlink: bool = False
) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"expected regular input: {path}")
    resolved = path.resolve(strict=True)
    if path.is_symlink() and not allow_pinned_snapshot_symlink:
        raise RuntimeError(f"expected regular non-symlink input: {path}")
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"resolved input is not a regular file: {resolved}")
    data = resolved.read_bytes()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "logical_path_is_symlink": path.is_symlink(),
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
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


def _weights(model: Any, hparams: Any) -> Dict[str, torch.Tensor]:
    return {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }


def _restore(weights: Dict[str, torch.Tensor], entry: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, weight in weights.items():
            weight.copy_(entry[name])


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _max_ulp_distance(left: torch.Tensor, right: torch.Tensor) -> int:
    if left.shape != right.shape or left.dtype != torch.float32 or right.dtype != torch.float32:
        raise RuntimeError("ULP comparison requires shape-matched FP32 tensors")
    left = left.detach().cpu().contiguous().view(torch.int32).flatten()
    right = right.detach().cpu().contiguous().view(torch.int32).flatten()
    maximum = 0
    chunk_size = 4 * 1024 * 1024
    offset = 0
    while offset < left.numel():
        left_chunk = left[offset : offset + chunk_size].to(torch.int64)
        right_chunk = right[offset : offset + chunk_size].to(torch.int64)
        left_ordered = torch.where(
            left_chunk < 0, -left_chunk - 1, left_chunk + 2**31
        )
        right_ordered = torch.where(
            right_chunk < 0, -right_chunk - 1, right_chunk + 2**31
        )
        maximum = max(
            maximum, int((left_ordered - right_ordered).abs().max().item())
        )
        offset += chunk_size
    return maximum


def _path_endpoint(
    *,
    path: str,
    model: Any,
    tok: Any,
    record: Dict[str, Any] | Sequence[Dict[str, Any]],
    hparams: Any,
) -> Dict[str, Any]:
    records = [record] if isinstance(record, dict) else list(record)
    if not records:
        raise RuntimeError("writer preflight record set is empty")
    weights = _weights(model, hparams)
    entry = {name: weight.detach().clone() for name, weight in weights.items()}
    pointers = {name: int(weight.data_ptr()) for name, weight in weights.items()}
    official.cache_c_new = False
    from project.run_scripts.alphaedit_strength_neutral_barrier.official_state import prepare_official_state

    prepare_official_state(model, tok, hparams, reset_cache=True)
    cache_entry_sha = _tensor_sha(official.cache_c)
    writer_receipt: Dict[str, Any] = {}
    captured_native_deltas: Dict[str, torch.Tensor] = {}
    original_execute = official.execute_AlphaEdit
    original_native_layer_delta = guided_writer._native_layer_delta

    def capture_execute(*execute_args, **execute_kwargs):
        result = original_execute(*execute_args, **execute_kwargs)
        captured_native_deltas.update(
            {name: value.detach().cpu().clone() for name, value in result.items()}
        )
        return result

    def capture_native_layer_delta(*execute_args, **execute_kwargs):
        result = original_native_layer_delta(*execute_args, **execute_kwargs)
        layer = int(execute_kwargs["layer"])
        weight_name = hparams.rewrite_module_tmp.format(layer) + ".weight"
        captured_native_deltas[weight_name] = result[0].detach().cpu().clone()
        return result

    if path in {"base_editor", "wrapper", "direct"}:
        official.execute_AlphaEdit = capture_execute
    elif path.startswith("split-") or path.startswith("barrier-"):
        guided_writer._native_layer_delta = capture_native_layer_delta
    try:
        if path == "base_editor":
            if len(records) != 1:
                raise RuntimeError("BaseEditor parity path requires one request")
            record = records[0]
            editor = BaseEditor.__new__(BaseEditor)
            editor.model = model
            editor.tok = tok
            editor.hparams = hparams
            editor.model_name = hparams.model_name
            editor.alg_name = "AlphaEdit"
            editor.apply_algo = official.apply_AlphaEdit_to_model
            rewrite = record["requested_rewrite"]
            editor.edit(
                prompts=[rewrite["prompt"].format(rewrite["subject"])],
                target_new=[rewrite["target_new"]["str"]],
                ground_truth=[rewrite["target_true"]["str"]],
                subject=[rewrite["subject"]],
                sequential_edit=True,
                verbose=False,
                test_generation=False,
            )
        elif path == "wrapper":
            _, _, writer_receipt = apply_strength_neutral_barrier_to_model(
                model,
                tok,
                [_request(item) for item in records],
                hparams,
                BarrierWriterConfig(BarrierArm.OFFICIAL, 1),
                copy=False,
                return_orig_weights=False,
                cache_template=None,
                reset_cache=True,
            )
        elif path == "direct":
            official.apply_AlphaEdit_to_model(
                model,
                tok,
                [_request(item) for item in records],
                hparams,
                copy=False,
                return_orig_weights=False,
                cache_template=None,
                reset_cache=True,
            )
        elif path.startswith("split-") or path.startswith("barrier-"):
            steps = int(path.rsplit("-", 1)[1])
            arm = BarrierArm.SPLIT if path.startswith("split-") else BarrierArm.BARRIER
            _, _, writer_receipt = apply_strength_neutral_barrier_to_model(
                model,
                tok,
                [_request(item) for item in records],
                hparams,
                BarrierWriterConfig(arm, steps),
                copy=False,
                return_orig_weights=False,
                cache_template=None,
                reset_cache=True,
            )
        else:
            raise ValueError(path)
    finally:
        official.execute_AlphaEdit = original_execute
        guided_writer._native_layer_delta = original_native_layer_delta
    endpoint_weights = {name: weight.detach().cpu().clone() for name, weight in weights.items()}
    endpoint_deltas = {
        name: endpoint_weights[name] - entry[name].cpu() for name in weights
    }
    split_fractional_replay = None
    if path.startswith("split-"):
        steps = int(path.rsplit("-", 1)[1])
        split_fractional_replay = {}
        for name in weights:
            replayed = entry[name].detach().cpu().clone()
            for _ in range(steps):
                replayed.add_(captured_native_deltas[name].to(replayed), alpha=1.0 / steps)
            split_fractional_replay[name] = fp32_comparison(
                replayed.float(), endpoint_weights[name].float()
            )
            if not split_fractional_replay[name]["bitwise_equal"]:
                raise RuntimeError(
                    f"G0-B split fractional replay mismatch: N={steps} {name}"
                )
    deltas = captured_native_deltas or endpoint_deltas
    endpoint = evaluate_counterfact(
        model,
        tok,
        records,
        device=next(model.parameters()).device,
        microbatch_size=max(1, len(records)),
    )
    _restore(weights, entry)
    if any(int(weights[name].data_ptr()) != pointers[name] for name in weights):
        raise RuntimeError(f"{path} W0 pointer restore failed")
    if any(not torch.equal(weights[name], entry[name]) for name in weights):
        raise RuntimeError(f"{path} W0 byte restore failed")
    return {
        "deltas": deltas,
        "endpoint_deltas": endpoint_deltas,
        "endpoint_weights": endpoint_weights,
        "endpoint": endpoint,
        "writer_receipt": writer_receipt,
        "native_deltas": captured_native_deltas,
        "split_fractional_replay": split_fractional_replay,
        "cache_entry_sha256": cache_entry_sha,
        "cache_endpoint_sha256": _tensor_sha(official.cache_c),
        "cache_changed": cache_entry_sha != _tensor_sha(official.cache_c),
    }


def _nll_rows(endpoint: Dict[str, Any]) -> Dict[str, list[float]]:
    return {
        kind: [float(row["nll"]) for row in rows]
        for kind, rows in endpoint.items()
    }


def _pred_rows(endpoint: Dict[str, Any]) -> Dict[str, list[list[int]]]:
    return {
        kind: [list(row["token_predictions"]) for row in rows]
        for kind, rows in endpoint.items()
    }


def _replay_envelope(
    model: Any,
    tok: Any,
    record: Dict[str, Any],
    hparams: Any,
    direct_deltas: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    weights = _weights(model, hparams)
    entry = {name: weight.detach().clone() for name, weight in weights.items()}
    layer_max_abs: Dict[str, float] = {name: 0.0 for name in weights}
    layer_max_ulp: Dict[str, int] = {name: 0 for name in weights}
    endpoint_nll_max_abs: Dict[str, float] = {}
    by_steps: Dict[str, Any] = {}
    direct_endpoint = None
    for steps in (1, 2, 4, 8):
        _restore(weights, entry)
        with torch.no_grad():
            for name, weight in weights.items():
                for _ in range(steps):
                    weight.add_(direct_deltas[name].to(weight), alpha=1.0 / steps)
        if steps == 1:
            direct_weights = {name: weight.detach().clone() for name, weight in weights.items()}
        else:
            step_layer_abs = {}
            step_layer_ulp = {}
            for name, weight in weights.items():
                observed_abs = float(
                    (weight - direct_weights[name]).abs().max().item()
                )
                observed_ulp = _max_ulp_distance(
                    weight.float(), direct_weights[name].float()
                )
                step_layer_abs[name] = observed_abs
                step_layer_ulp[name] = observed_ulp
                layer_max_abs[name] = max(
                    layer_max_abs[name],
                    observed_abs,
                )
                layer_max_ulp[name] = max(
                    layer_max_ulp[name],
                    observed_ulp,
                )
        endpoint = evaluate_counterfact(
            model,
            tok,
            [record],
            device=next(model.parameters()).device,
            microbatch_size=1,
        )
        if direct_endpoint is None:
            direct_endpoint = endpoint
        else:
            step_nll_abs = {}
            for kind, rows in endpoint.items():
                for observed, expected in zip(rows, direct_endpoint[kind]):
                    error = abs(float(observed["nll"]) - float(expected["nll"]))
                    step_nll_abs[kind] = max(
                        step_nll_abs.get(kind, 0.0), error
                    )
                    endpoint_nll_max_abs[kind] = max(
                        endpoint_nll_max_abs.get(kind, 0.0),
                        error,
                    )
            by_steps[str(steps)] = {
                "layer_endpoint_max_abs": step_layer_abs,
                "layer_endpoint_max_ulp": step_layer_ulp,
                "endpoint_nll_max_abs": step_nll_abs,
            }
    _restore(weights, entry)
    return {
        "definition": "actual FP32 one-add versus N={2,4,8} fractional-add replay",
        "layer_delta_max_abs": layer_max_abs,
        "layer_endpoint_max_ulp": layer_max_ulp,
        "endpoint_nll_max_abs": endpoint_nll_max_abs,
        "by_steps": by_steps,
        "direct_endpoint": direct_endpoint,
    }


def _assert_official_parity(
    outputs: Dict[str, Dict[str, Any]],
    envelope: Dict[str, Any],
) -> Dict[str, Any]:
    reference_delta = outputs["direct"]["deltas"]
    reference_weights = outputs["direct"]["endpoint_weights"]
    reference_endpoint = outputs["direct"]["endpoint"]
    comparisons = {}
    for path, output in outputs.items():
        deltas = output["deltas"]
        endpoint = output["endpoint"]
        layer_errors = {
            name: float(
                (output["endpoint_weights"][name] - reference_weights[name])
                .abs()
                .max()
                .item()
            )
            for name in reference_delta
        }
        layer_ulp = {
            name: _max_ulp_distance(
                output["endpoint_weights"][name].float(), reference_weights[name].float()
            )
            for name in reference_delta
        }
        for name, error in layer_errors.items():
            if error > envelope["layer_delta_max_abs"][name]:
                raise RuntimeError(
                    f"Official parity layer boundary: path={path} name={name} "
                    f"error={error} envelope={envelope['layer_delta_max_abs'][name]}"
                )
            if layer_ulp[name] > envelope["layer_endpoint_max_ulp"][name]:
                raise RuntimeError(
                    f"Official parity ULP boundary: path={path} name={name} "
                    f"error={layer_ulp[name]} "
                    f"envelope={envelope['layer_endpoint_max_ulp'][name]}"
                )
        nll_errors = {}
        for kind, rows in endpoint.items():
            for observed, expected in zip(rows, reference_endpoint[kind]):
                error = abs(float(observed["nll"]) - float(expected["nll"]))
                nll_errors[kind] = max(nll_errors.get(kind, 0.0), error)
                if error > envelope["endpoint_nll_max_abs"].get(kind, 0.0):
                    raise RuntimeError(
                        f"Official parity endpoint boundary: path={path} kind={kind} "
                        f"error={error} envelope={envelope['endpoint_nll_max_abs'].get(kind, 0.0)}"
                    )
        if _pred_rows(endpoint) != _pred_rows(reference_endpoint):
            raise RuntimeError(f"Official parity prediction mismatch: {path}")
        comparisons[path] = {"layer_max_abs": layer_errors, "nll_max_abs": nll_errors}
        comparisons[path]["layer_max_ulp"] = layer_ulp
        comparisons[path]["layer_delta_sha256"] = {
            name: _tensor_sha(deltas[name].float()) for name in deltas
        }
        comparisons[path]["endpoint_nll"] = _nll_rows(endpoint)
        comparisons[path]["predictions"] = _pred_rows(endpoint)
    return comparisons


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return bool(torch.isfinite(torch.tensor(value)))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _guided_integrity(
    output: Dict[str, Any], *, arm: BarrierArm, steps: int, request_count: int
) -> Dict[str, Any]:
    receipt = output["writer_receipt"]
    if receipt.get("arm") != arm.value or int(receipt.get("steps", 0)) != steps:
        raise RuntimeError("guided writer arm/step receipt mismatch")
    if int(receipt.get("request_count", 0)) != request_count:
        raise RuntimeError("guided writer request denominator mismatch")
    if int(receipt.get("fixed_z_compute_count", 0)) != request_count:
        raise RuntimeError("fixed-z compute count mismatch")
    if int(receipt.get("fixed_z_recompute_count", -1)) != 0:
        raise RuntimeError("fixed-z recompute is forbidden")
    if int(receipt.get("cache_append_count", 0)) != 1 or not output["cache_changed"]:
        raise RuntimeError("AlphaEdit cache append/continuity gate failed")
    if not receipt.get("w0_restore_pass"):
        raise RuntimeError("internal temporary W0 restore failed")
    if not receipt.get("authoritative_endpoint_adoption_pass"):
        raise RuntimeError("G0-C exact temporary/authoritative endpoint adoption failed")
    if (
        receipt.get("temporary_endpoint_selected_sha256")
        == receipt.get("w0_selected_sha256")
    ):
        raise RuntimeError("selected weights did not change at temporary endpoint")
    if (
        receipt.get("restored_w0_selected_sha256")
        != receipt.get("w0_selected_sha256")
    ):
        raise RuntimeError("restored-W0 SHA mismatch")
    if not _all_finite(receipt):
        raise RuntimeError("writer receipt contains nonfinite telemetry")
    nodes = [node for layer in receipt.get("layers", []) for node in layer.get("nodes", [])]
    expected_nodes = len(receipt.get("layers", [])) * steps
    if len(nodes) != expected_nodes:
        raise RuntimeError(f"writer node count mismatch: {len(nodes)} != {expected_nodes}")
    event_count = int(receipt.get("target_event_count", 0))
    for node in nodes:
        if node.get("native_predictor"):
            raise RuntimeError("legacy first-node native bypass remains active")
        if not node.get("predictor_restore_pass"):
            raise RuntimeError("predictor W_n restore failed")
        if not node.get("selected_weight_endpoint_sha256"):
            raise RuntimeError("node endpoint SHA is absent")
        if arm is BarrierArm.BARRIER:
            residuals = node.get(
                "target_strength_constraint_residual_per_event_abs", []
            )
            if len(residuals) != event_count:
                raise RuntimeError(
                    "a target token is missing its strength-gradient residual"
                )
    return {
        "node_count": len(nodes),
        "target_event_count": event_count,
        "cache_append_count": receipt["cache_append_count"],
        "cache_entry_sha256": output["cache_entry_sha256"],
        "cache_endpoint_sha256": output["cache_endpoint_sha256"],
        "temporary_endpoint_selected_sha256": receipt[
            "temporary_endpoint_selected_sha256"
        ],
        "authoritative_endpoint_selected_sha256": receipt[
            "authoritative_endpoint_selected_sha256"
        ],
        "restored_w0_selected_sha256": receipt["restored_w0_selected_sha256"],
        "temporary_endpoint_component_sha256": receipt[
            "temporary_endpoint_component_sha256"
        ],
        "authoritative_endpoint_component_sha256": receipt[
            "authoritative_endpoint_component_sha256"
        ],
        "authoritative_endpoint_adoption_pass": receipt[
            "authoritative_endpoint_adoption_pass"
        ],
    }


def _assert_split_parity(
    split_outputs: Dict[int, Dict[str, Any]],
    official_output: Dict[str, Any],
    envelope: Dict[str, Any],
) -> Dict[str, Any]:
    reference_weights = official_output["endpoint_weights"]
    reference_endpoint = official_output["endpoint"]
    results = {}
    for steps, output in split_outputs.items():
        locked = envelope["by_steps"][str(steps)]
        layer_max_abs = {}
        layer_max_ulp = {}
        for name, expected in reference_weights.items():
            observed = output["endpoint_weights"][name]
            layer_max_abs[name] = float((observed - expected).abs().max().item())
            layer_max_ulp[name] = _max_ulp_distance(
                observed.float(), expected.float()
            )
            if layer_max_abs[name] > locked["layer_endpoint_max_abs"][name]:
                raise RuntimeError(
                    f"Split N={steps} weight parity boundary: {name} "
                    f"{layer_max_abs[name]} > {locked['layer_endpoint_max_abs'][name]}"
                )
            if layer_max_ulp[name] > locked["layer_endpoint_max_ulp"][name]:
                raise RuntimeError(
                    f"Split N={steps} ULP parity boundary: {name} "
                    f"{layer_max_ulp[name]} > {locked['layer_endpoint_max_ulp'][name]}"
                )
        nll_max_abs = {}
        for kind, rows in output["endpoint"].items():
            for observed, expected in zip(rows, reference_endpoint[kind]):
                error = abs(float(observed["nll"]) - float(expected["nll"]))
                nll_max_abs[kind] = max(nll_max_abs.get(kind, 0.0), error)
                if error > locked["endpoint_nll_max_abs"].get(kind, 0.0):
                    raise RuntimeError(
                        f"Split N={steps} endpoint parity boundary: {kind}"
                    )
        if _pred_rows(output["endpoint"]) != _pred_rows(reference_endpoint):
            raise RuntimeError(f"Split N={steps} prediction parity mismatch")
        results[str(steps)] = {
            "layer_endpoint_max_abs": layer_max_abs,
            "layer_endpoint_max_ulp": layer_max_ulp,
            "layer_delta_sha256": {
                name: _tensor_sha(value.float())
                for name, value in output["deltas"].items()
            },
            "endpoint_nll_max_abs": nll_max_abs,
            "g0_b_fractional_replay": output["split_fractional_replay"],
            "integrity": _guided_integrity(
                output, arm=BarrierArm.SPLIT, steps=steps, request_count=1
            ),
        }
    return results


def _assert_native_field_parity(
    split_outputs: Dict[int, Dict[str, Any]], official_output: Dict[str, Any]
) -> Dict[str, Any]:
    """G0-A: compare Official native field with the hook field before splitting."""

    official_deltas = official_output["native_deltas"]
    results: Dict[str, Any] = {}
    for steps, output in split_outputs.items():
        per_weight = {}
        for name, expected in official_deltas.items():
            observed = output["native_deltas"][name]
            comparison = fp32_comparison(expected.float(), observed.float())
            # The first rewrite layer is evaluated at exact W0 in both paths.
            # Later layers are reported as observed because earlier-layer state
            # follows their respective one-add/fractional trajectories.
            comparison["same_state_required"] = name == sorted(official_deltas)[0]
            if comparison["same_state_required"] and not comparison["bitwise_equal"]:
                raise RuntimeError(f"G0-A native field parity mismatch: N={steps} {name}")
            per_weight[name] = comparison
        results[str(steps)] = per_weight
    return results


def _tensor_gate(left: torch.Tensor, right: torch.Tensor, *, label: str) -> Dict[str, Any]:
    if left.shape != right.shape:
        raise RuntimeError(f"{label} shape mismatch: {left.shape} != {right.shape}")
    scale = max(float(left.abs().max().item()), float(right.abs().max().item()), 1.0)
    tolerance = max(left.shape) * torch.finfo(torch.float32).eps * scale
    max_abs = float((left.float() - right.float()).abs().max().item())
    if max_abs > tolerance:
        raise RuntimeError(f"{label} batch/single mismatch: {max_abs} > {tolerance}")
    return {"max_abs": max_abs, "backward_error_tolerance": tolerance}


def _alignment_gate(
    model: Any,
    tok: Any,
    raw: Sequence[Dict[str, Any]],
    hparams: Any,
) -> Dict[str, Any]:
    fixtures = select_multitoken_alignment_fixtures(raw, tok)
    records = [raw[item.ordinal] for item in fixtures]
    requests = [_request(record) for record in records]
    target_batch = build_target_event_batch(
        tok, requests, device=next(model.parameters()).device
    )
    if target_batch.event_count <= target_batch.request_count:
        raise RuntimeError("multi-token target_event_count must exceed request_count")
    grouped_ids: Dict[int, list[int]] = {index: [] for index in range(len(requests))}
    for event in target_batch.events:
        grouped_ids[event.request_index].append(event.target_token_id)
    for index, record in enumerate(records):
        expected = target_token_ids(tok, record["requested_rewrite"]["target_new"]["str"])
        if grouped_ids[index] != expected:
            raise RuntimeError("teacher-forced/evaluator target ID mismatch")

    context_templates = official.get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    prompts = [request["prompt"] for request in requests]
    subjects = [request["subject"] for request in requests]
    current_batch = get_module_input_output_at_words(
        model,
        tok,
        z_layer,
        context_templates=prompts,
        words=subjects,
        module_template=hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[1]
    current_single = torch.cat(
        [
            get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[prompt],
                words=[subject],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1]
            for prompt, subject in zip(prompts, subjects)
        ],
        dim=0,
    )
    current_gate = _tensor_gate(current_batch, current_single, label="current-z")

    key_layer = hparams.layers[0]
    ks_batch = compute_ks(model, tok, requests, hparams, key_layer, context_templates)
    ks_single = torch.cat(
        [
            compute_ks(model, tok, [request], hparams, key_layer, context_templates)
            for request in requests
        ],
        dim=0,
    )
    ks_gate = _tensor_gate(ks_batch, ks_single, label="compute_ks")

    compute_z_prompts = []
    for request in requests:
        target_ids = target_token_ids(tok, request["target_new"])
        compute_z_prompts.append(
            context_templates[0][0].format(request["prompt"])
            + tok.decode(target_ids[:-1])
        )

    label_rows = []
    rewriting_rows = []
    for request_index, request in enumerate(requests):
        target_ids = target_token_ids(tok, request["target_new"])
        for context_type in context_templates:
            for context in context_type:
                template = context.format(request["prompt"])
                rewriting_rows.append(
                    {
                        "request_index": request_index,
                        "template": template,
                        "subject": request["subject"],
                        "target_ids": target_ids,
                        "text": template.format(request["subject"])
                        + tok.decode(target_ids[:-1]),
                    }
                )
    encoded_rows = tok(
        [row["text"] for row in rewriting_rows]
        + ["{} is a".format(requests[0]["subject"])],
        return_tensors="pt",
        padding=True,
    )
    for row_index, row in enumerate(rewriting_rows):
        ex_len = int(encoded_rows["attention_mask"][row_index].sum().item())
        batch_ids = [
            int(value)
            for value in encoded_rows["input_ids"][row_index, :ex_len].tolist()
        ]
        single_ids = [
            int(value)
            for value in tok(row["text"], add_special_tokens=True)["input_ids"]
        ]
        if batch_ids != single_ids:
            raise RuntimeError("compute_z right-padding batch/single token mismatch")
        target_ids = list(row["target_ids"])
        label_start = ex_len - len(target_ids)
        label_positions = list(range(label_start, ex_len))
        if label_start < 0:
            raise RuntimeError("compute_z target labels exceed prompt sequence")
        if len(target_ids) > 1:
            observed_prefix = batch_ids[label_start + 1 : ex_len]
            if observed_prefix != target_ids[:-1]:
                raise RuntimeError(
                    "compute_z teacher-forced prefix/target ID alignment failed"
                )
        lookup_index = find_fact_lookup_idx(
            row["template"],
            row["subject"],
            tok,
            hparams.fact_token,
            verbose=False,
        )
        label_rows.append(
            {
                "request_index": row["request_index"],
                "target_ids": target_ids,
                "label_positions": label_positions,
                "subject_lookup_index": int(lookup_index),
                "batch_single_input_ids_match": True,
                "teacher_forced_prefix_ids_match": True,
            }
        )
    compute_z_batch = get_module_input_output_at_words(
        model,
        tok,
        z_layer,
        context_templates=compute_z_prompts,
        words=subjects,
        module_template=hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[1]
    compute_z_single = torch.cat(
        [
            get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[prompt],
                words=[subject],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1]
            for prompt, subject in zip(compute_z_prompts, subjects)
        ],
        dim=0,
    )
    compute_z_gate = _tensor_gate(
        compute_z_batch, compute_z_single, label="compute_z subject representation"
    )
    return {
        "fixtures": [item.__dict__ for item in fixtures],
        "request_count": target_batch.request_count,
        "target_event_count": target_batch.event_count,
        "teacher_forced_evaluator_target_ids_match": True,
        "compute_z_target_label_alignment": {
            "row_count": len(label_rows),
            "rows": label_rows,
            "omission_count": 0,
        },
        "current_z": current_gate,
        "compute_ks": ks_gate,
        "compute_z_subject_representation": compute_z_gate,
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
    actual_head = _git_value(source_root, "HEAD")
    actual_tree = _git_value(source_root, "HEAD^{tree}")
    if actual_head != args.expected_head:
        raise RuntimeError("queued source drift")
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
        raise FileExistsError(args.result_dir)
    args.result_dir.mkdir(parents=True, mode=0o700)
    receipt_path = args.result_dir / "preflight-receipt.json"
    failure_path = args.result_dir / "failure-boundary.json"
    partial_dir = args.result_dir / "partial-receipts"
    partial_inventory: list[Dict[str, Any]] = []

    def publish_gate(ordinal: int, gate: str, evidence: Dict[str, Any]) -> None:
        identity = write_create_once(
            partial_dir / f"{ordinal:02d}-{gate.lower().replace('_', '-')}.json",
            {
                "schema": "easyedit.alphaedit.strength-neutral-barrier.partial-gate.v1",
                "gate": gate,
                "terminal_status": "PASS",
                "source": {
                    "head": actual_head,
                    "tree": actual_tree,
                    "easyedit_head": easyedit_head,
                    "easyedit_tree": easyedit_tree,
                },
                "evidence": evidence,
                "science_change_count": 0,
                "tolerance_change_count": 0,
            },
        )
        partial_inventory.append(identity)

    input_identities = {
        "model_config": _file_identity(
            args.model_path / "config.json", allow_pinned_snapshot_symlink=True
        ),
        "tokenizer": _file_identity(
            args.model_path / "tokenizer.json", allow_pinned_snapshot_symlink=True
        ),
        "dataset": _file_identity(args.dataset),
        "projector": _file_identity(args.projector),
        "hparams": _file_identity(args.hparams),
        "official_source": _file_identity(Path(official.__file__).resolve(strict=True)),
    }
    publish_gate(
        1,
        "INPUT_SOURCE_IDENTITY",
        {
            "input_identities": input_identities,
            "easyedit_seal": easyedit_seal,
        },
    )
    model = None
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        raw = json.loads(args.dataset.read_text())
        hparams = AlphaEditHyperParams.from_hparams(str(args.hparams))
        hparams.model_name = str(args.model_path)
        hparams.P_loc = str(args.projector)
        hparams.device = 0
        hparams.batch_size = 1
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
        require_right_padding(tok, caller="preflight")
        model.eval()
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("FULL_FP32 preflight gate failed")
        publish_gate(
            2,
            "FULL_FP32_OFFLINE_LOAD",
            {
                "full_fp32": True,
                "tf32_enabled": False,
                "autocast_enabled": False,
                "padding_side": tok.padding_side,
            },
        )

        alignment = _alignment_gate(model, tok, raw, hparams)
        publish_gate(3, "ALIGNMENT", alignment)
        parity_record = raw[0]
        outputs = {
            path: _path_endpoint(
                path=path,
                model=model,
                tok=tok,
                record=parity_record,
                hparams=hparams,
            )
            for path in ("base_editor", "wrapper", "direct")
        }
        envelope = _replay_envelope(
            model, tok, parity_record, hparams, outputs["direct"]["deltas"]
        )
        comparisons = _assert_official_parity(outputs, envelope)
        if not all(output["cache_changed"] for output in outputs.values()):
            raise RuntimeError("Official path cache append gate failed")
        publish_gate(
            4,
            "OFFICIAL_N1",
            {"comparisons": comparisons, "cache_append_all_paths": True},
        )

        split_outputs = {
            steps: _path_endpoint(
                path=f"split-{steps}",
                model=model,
                tok=tok,
                record=parity_record,
                hparams=hparams,
            )
            for steps in (2, 4, 8)
        }
        split_parity = _assert_split_parity(
            split_outputs, outputs["direct"], envelope
        )
        native_field_parity = _assert_native_field_parity(
            split_outputs, outputs["direct"]
        )
        publish_gate(5, "G0_A_NATIVE_FIELD_PARITY", native_field_parity)
        publish_gate(
            6,
            "G0_B_FRACTIONAL_REPLAY",
            {
                str(steps): output["split_fractional_replay"]
                for steps, output in split_outputs.items()
            },
        )
        publish_gate(
            7,
            "G0_C_EXACT_ENDPOINT_ADOPTION",
            {
                str(steps): {
                    "temporary": output["writer_receipt"][
                        "temporary_endpoint_selected_sha256"
                    ],
                    "authoritative": output["writer_receipt"][
                        "authoritative_endpoint_selected_sha256"
                    ],
                    "pass": output["writer_receipt"][
                        "authoritative_endpoint_adoption_pass"
                    ],
                }
                for steps, output in split_outputs.items()
            },
        )
        barrier_outputs = {
            steps: _path_endpoint(
                path=f"barrier-{steps}",
                model=model,
                tok=tok,
                record=parity_record,
                hparams=hparams,
            )
            for steps in (2, 4, 8)
        }
        single_token_barrier = {
            str(steps): {
                "integrity": _guided_integrity(
                    output,
                    arm=BarrierArm.BARRIER,
                    steps=steps,
                    request_count=1,
                ),
                "endpoint_nll": _nll_rows(output["endpoint"]),
                "predictions": _pred_rows(output["endpoint"]),
            }
            for steps, output in barrier_outputs.items()
        }
        publish_gate(8, "BARRIER_EXECUTED", single_token_barrier)

        multi_ordinals = []
        for fixture in alignment["fixtures"]:
            ordinal = int(fixture["ordinal"])
            if ordinal not in multi_ordinals:
                multi_ordinals.append(ordinal)
        multi_records = [raw[ordinal] for ordinal in multi_ordinals]
        multi_output = _path_endpoint(
            path="barrier-4",
            model=model,
            tok=tok,
            record=multi_records,
            hparams=hparams,
        )
        multi_integrity = _guided_integrity(
            multi_output,
            arm=BarrierArm.BARRIER,
            steps=4,
            request_count=len(multi_records),
        )
        expected_events = sum(
            len(target_token_ids(tok, record["requested_rewrite"]["target_new"]["str"]))
            for record in multi_records
        )
        if multi_integrity["target_event_count"] != expected_events:
            raise RuntimeError("multi-token writer event denominator mismatch")
        if expected_events <= len(multi_records):
            raise RuntimeError("multi-token writer event count must exceed requests")
        publish_gate(
            9,
            "MULTITOKEN_BARRIER",
            {
                "request_count": len(multi_records),
                "target_event_count": expected_events,
                "integrity": multi_integrity,
            },
        )

        official_source = Path(official.__file__).resolve(strict=True)
        payload = {
            "schema": "easyedit.alphaedit.strength-neutral-barrier.preflight.v3",
            "terminal_status": "PRE_GPU_PASS",
            "source": {
                "head": actual_head,
                "tree": actual_tree,
                "easyedit_head": easyedit_head,
                "easyedit_tree": easyedit_tree,
                "easyedit_tracked_clean": True,
                "implementation_boundary": "ODE_EDIT_HOOK_STOCK_EASYEDIT_READ_ONLY",
                "easyedit_seal": easyedit_seal,
            },
            "input_identities": input_identities,
            "tokenizer_padding_side": tok.padding_side,
            "full_fp32": True,
            "non_fp32_parameter_count": 0,
            "quantized": bool(getattr(model, "is_quantized", False)),
            "tf32_enabled": False,
            "autocast_enabled": False,
            "alignment": alignment,
            "official_parity": {
                "case_id": int(parity_record["case_id"]),
                "paths": ["BaseEditor", "wrapper OFFICIAL_ALPHAEDIT", "direct apply"],
                "fp32_replay_envelope": {
                    key: value for key, value in envelope.items() if key != "direct_endpoint"
                },
                "comparisons": comparisons,
                "prediction_identity": True,
                "cache_append_all_paths": True,
            },
            "split_weight_parity": split_parity,
            "g0_a_native_field_parity": native_field_parity,
            "g0_b_fractional_replay": {
                str(steps): output["split_fractional_replay"]
                for steps, output in split_outputs.items()
            },
            "g0_c_exact_endpoint_adoption": {
                str(steps): output["writer_receipt"][
                    "authoritative_endpoint_adoption_pass"
                ]
                for steps, output in split_outputs.items()
            },
            "partial_receipts": partial_inventory,
            "single_token_b1": {
                "case_id": int(parity_record["case_id"]),
                "barrier": single_token_barrier,
            },
            "multi_token_writer": {
                "ordinals": multi_ordinals,
                "case_ids": [int(record["case_id"]) for record in multi_records],
                "request_count": len(multi_records),
                "target_event_count": expected_events,
                "integrity": multi_integrity,
                "endpoint_nll": _nll_rows(multi_output["endpoint"]),
                "predictions": _pred_rows(multi_output["endpoint"]),
                "omission_count": 0,
                "imputation_count": 0,
            },
            "science_change_count": 0,
            "tolerance_change_count": 0,
            "model_load_count": 1,
            "gpu_job_count": 1,
        }
        write_create_once(receipt_path, payload)
    except Exception as error:
        if not failure_path.exists():
            write_create_once(
                failure_path,
                {
                    "schema": "easyedit.alphaedit.strength-neutral-barrier.preflight-failure.v1",
                    "terminal_status": "FAILED_BOUNDARY",
                    "source": {
                        "head": actual_head,
                        "tree": actual_tree,
                        "easyedit_head": easyedit_head,
                        "easyedit_tree": easyedit_tree,
                        "implementation_boundary": "ODE_EDIT_HOOK_STOCK_EASYEDIT_READ_ONLY",
                        "easyedit_seal": easyedit_seal,
                    },
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "partial_receipts": partial_inventory,
                    "completed_gate_count": len(partial_inventory),
                    "science_change_count": 0,
                    "tolerance_change_count": 0,
                },
            )
        raise


if __name__ == "__main__":
    main()
