"""GPU runtime for B1 parity gates and independent B10x10 observation."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from project.run_scripts.fixed_z_nonuniqueness.contracts import MODEL_SPECS, NumericalLock
from project.run_scripts.fixed_z_nonuniqueness.evaluation import padding_safety_gate, sequence_metrics
from project.run_scripts.fixed_z_nonuniqueness.official import official_model_name_binding
from project.run_scripts.fixed_z_nonuniqueness.padding import OfficialTokenizerHook, bind_padding
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r24_independent_b10x10_selection import load_historical_h0_batches
from project.run_scripts.ode_bf.p4_sealed_stream import preflight_transferred_stream_v2
from project.run_scripts.ode_bf.p1r52_official_sequential_baselines import run_official_memit_apply
from project.run_scripts.ode_bf.scalable_batched_native import run_official_native_apply

from .contracts import (
    DATASET,
    DATASET_SHA256,
    EVALUATOR_IDENTITY,
    INSTRUCTION_ID,
    LAYERS,
    Method,
    NONCE,
    ORDER_IDENTITY,
    ObservationBoundary,
    ObservationLock,
    STREAM_ARCHIVE,
    STREAM_IDENTITY,
    STREAM_ROOT,
)
from .observer import OfficialCallAudit, OfficialLayerObserver


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_hparams(method: Method, path: Path) -> Any:
    if method is Method.ALPHAEDIT:
        from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams
        return AlphaEditHyperParams.from_hparams(str(path))
    from easyeditor.models.memit.memit_hparams import MEMITHyperParams
    return MEMITHyperParams.from_hparams(str(path))


def _method_module(method: Method) -> Any:
    if method is Method.ALPHAEDIT:
        from easyeditor.models.alphaedit import AlphaEdit_main
        return AlphaEdit_main
    from easyeditor.models.memit import memit_main
    return memit_main


def _touched(model: Any, hparams: Any) -> dict[str, torch.nn.Parameter]:
    parameters = dict(model.named_parameters())
    names = [f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in LAYERS]
    if tuple(int(value) for value in hparams.layers) != LAYERS or any(name not in parameters for name in names):
        raise ObservationBoundary("Official editable layer inventory differs")
    return {name: parameters[name] for name in names}


def _w0_identity(touched: Mapping[str, torch.nn.Parameter]) -> dict[str, Any]:
    return {
        "sha256": {name: tensor_sha256(value) for name, value in touched.items()},
        "pointers": {name: int(value.data_ptr()) for name, value in touched.items()},
        "dtypes": {name: str(value.dtype) for name, value in touched.items()},
    }


def _restore_w0(
    touched: Mapping[str, torch.nn.Parameter],
    originals: Mapping[str, torch.Tensor],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    from easyeditor.util.device import copy_to_param
    if set(touched) != set(originals):
        raise ObservationBoundary("Official original weight inventory differs")
    with torch.no_grad():
        for name, parameter in touched.items():
            copy_to_param(parameter, originals[name])
    observed = _w0_identity(touched)
    exact = observed["sha256"] == expected["sha256"] and observed["pointers"] == expected["pointers"]
    if not exact:
        raise ObservationBoundary("W0 pointer/bytes restore failed")
    return {"exact": True, **observed}


def _official_requests(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    required = (
        "case_id",
        "prompt",
        "subject",
        "target_new",
        "target_true",
        "request_sha256",
    )
    answer = []
    for row in rows:
        if not set(required).issubset(row):
            raise ObservationBoundary("sealed request schema differs")
        answer.append({key: row[key] for key in required})
    return answer


def _endpoint_metrics(model: Any, tokenizer: Any, requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    for request in requests:
        prompt = str(request["prompt"]).format(str(request["subject"]))
        new = sequence_metrics(model, tokenizer, [prompt], str(request["target_new"]))
        true = sequence_metrics(model, tokenizer, [prompt], str(request["target_true"]))
        records.append({
            "request_sha256": str(request["request_sha256"]),
            "target_new_nll": float(new["nll"][0]),
            "target_true_nll": float(true["nll"][0]),
            "target_new_strict": bool(new["strict"][0]),
            "target_true_strict": bool(true["strict"][0]),
            "target_new_margin": float(new["margin"][0]),
            "target_true_margin": float(true["margin"][0]),
        })
    if any(not all(torch.isfinite(torch.tensor(value)) for key, value in row.items() if isinstance(value, float)) for row in records):
        raise ObservationBoundary("endpoint metric is nonfinite")
    return {"request_count": len(records), "records": records, "raw_prompt_or_logit_count": 0}


def _warm_official_state(model: Any, tokenizer: Any, method: Method, hparams: Any) -> dict[str, Any]:
    module = _method_module(method)
    with official_model_name_binding(model, str(hparams.model_name)):
        contexts = module.get_context_templates(model, tokenizer)
        if method is Method.MEMIT:
            module.COV_CACHE.clear()
            for layer in LAYERS:
                cov = module.get_cov(
                    model,
                    tokenizer,
                    hparams.rewrite_module_tmp.format(layer),
                    hparams.mom2_dataset,
                    hparams.mom2_n_samples,
                    hparams.mom2_dtype,
                    hparams=hparams,
                )
                del cov
                torch.cuda.empty_cache()
            cache = {"kind": "STATIC_MEMIT_COVARIANCE_COMPUTATION_CACHE", "entry_count": len(module.COV_CACHE)}
        else:
            if not bool(getattr(module, "P_loaded", False)):
                module.P = torch.load(hparams.P_loc, map_location="cpu", weights_only=True)
                module.P_loaded = True
                module.P_loaded_from = str(hparams.P_loc)
            module.cache_c_new = False
            cache = {
                "kind": "ALPHAEDIT_STATIC_P_PRELOADED_DYNAMIC_CACHE_COLD",
                "projector_sha256": tensor_sha256(module.P),
                "cache_c_new": False,
            }
    return {"context_identity": canonical_hash(contexts), "cache": cache}


def _reset_effective_entry(method: Method) -> None:
    module = _method_module(method)
    if method is Method.ALPHAEDIT:
        module.cache_c_new = False


def _run_apply(
    *,
    model: Any,
    tokenizer: Any,
    method: Method,
    hparams: Any,
    requests: Sequence[Mapping[str, Any]],
    touched: Mapping[str, torch.nn.Parameter],
    capture_layers: bool,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    request_hashes = [str(value["request_sha256"]) for value in requests]
    observer = OfficialLayerObserver(method=method, request_sha256=request_hashes, capture_layers=capture_layers)
    call_audit = OfficialCallAudit()
    hook = OfficialTokenizerHook(tokenizer, [])
    _reset_effective_entry(method)
    with official_model_name_binding(model, str(hparams.model_name)):
        if method is Method.MEMIT:
            payload, originals = run_official_memit_apply(
                model,
                hook,
                requests,
                hparams,
                touched=touched,
                observer=observer,
                call_audit=call_audit,
            )
        else:
            payload, originals = run_official_native_apply(
                model,
                hook,
                requests,
                hparams,
                touched=touched,
                reset_cache=True,
                cache_history_width=0,
                cache_template=None,
                expected_native_compute_z_call_count=len(requests),
                accepted_z_source="STOCK_COMPUTE_Z_ENTRY_STATE_ONCE_PER_REQUEST",
                observer=observer,
                call_audit=call_audit,
            )
    if not hook.call_identities or not all(row["semantic_input_attention_position_equal"] for row in hook.call_identities):
        raise ObservationBoundary("Official tokenizer semantic identity receipt differs")
    payload["official_tokenizer_hook"] = {
        "call_count": len(hook.call_identities),
        "semantic_identity_root": canonical_hash(hook.call_identities),
        "all_semantic_equal": True,
        "raw_prompt_publish_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "identity_sha256"}
    )
    return payload, originals


def _cache_exit_identity(method: Method, payload: Mapping[str, Any]) -> str:
    if method is Method.MEMIT:
        return str(payload["covariance_cache"]["exit"]["identity_sha256"])
    return str(payload["alphaedit_dynamic_cache_contract"]["exit"]["sha256"])


def _counter_tuple(payload: Mapping[str, Any]) -> tuple[int, int, int]:
    audit = payload["official_call_audit"]
    return (
        int(payload["target_backward_count"]),
        int(audit["compute_ks_call_count"]),
        int(audit["torch_linalg_solve_call_count"]),
    )


def _b1_gate(
    *,
    model: Any,
    tokenizer: Any,
    method: Method,
    hparams: Any,
    request: Mapping[str, Any],
    touched: Mapping[str, torch.nn.Parameter],
    w0: Mapping[str, Any],
) -> dict[str, Any]:
    requests = [request]
    torch.manual_seed(541_001)
    torch.cuda.manual_seed_all(541_001)
    on_payload, on_originals = _run_apply(
        model=model, tokenizer=tokenizer, method=method, hparams=hparams,
        requests=requests, touched=touched, capture_layers=True,
    )
    on_endpoint = _endpoint_metrics(model, tokenizer, requests)
    on_weights = dict(on_payload["edited_sha256"])
    on_restore = _restore_w0(touched, on_originals, w0)

    torch.manual_seed(541_001)
    torch.cuda.manual_seed_all(541_001)
    off_payload, off_originals = _run_apply(
        model=model, tokenizer=tokenizer, method=method, hparams=hparams,
        requests=requests, touched=touched, capture_layers=False,
    )
    off_endpoint = _endpoint_metrics(model, tokenizer, requests)
    off_weights = dict(off_payload["edited_sha256"])
    off_restore = _restore_w0(touched, off_originals, w0)

    on_observer = on_payload["layer_realization_observer"]
    off_observer = off_payload["layer_realization_observer"]
    parity = {
        "final_weight_sha_byte_exact": on_weights == off_weights,
        "official_endpoint_metric_exact": canonical_hash(on_endpoint) == canonical_hash(off_endpoint),
        "z_hash_exact": on_observer["z_sha256"] == off_observer["z_sha256"],
        "request_order_exact": on_payload["request_order_sha256"] == off_payload["request_order_sha256"],
        "cache_exit_exact": _cache_exit_identity(method, on_payload) == _cache_exit_identity(method, off_payload),
        "backward_key_solver_counts_exact": _counter_tuple(on_payload) == _counter_tuple(off_payload),
        "observer_layer_calls_5_of_5": on_observer["layer_loop_observation_copy_count"] == 5,
        "observer_terminal_call_1": on_observer["terminal_post_L8_forward_count"] == 1,
        "observer_off_layer_copy_count_0": off_observer["layer_loop_observation_copy_count"] == 0,
        "nonfinite_count_0": on_observer["residual_debt"]["nonfinite_count"] == 0,
        "recurrence_closure_pass": (
            on_observer["residual_debt"]["maximum_recurrence_closure_relative_error"]
            < ObservationLock().recurrence_relative_tolerance
        ),
        "w0_restore_on": bool(on_restore["exact"]),
        "w0_restore_off": bool(off_restore["exact"]),
    }
    if not all(parity.values()):
        raise ObservationBoundary(f"B1 observer on/off parity failed: {parity}")
    return {
        "status": "B1_OBSERVER_ON_OFF_GATE_PASS",
        "method": method.value,
        "request_sha256": str(request["request_sha256"]),
        "parity": parity,
        "on": {"apply": on_payload, "endpoint": on_endpoint, "restore": on_restore},
        "off": {"apply": off_payload, "endpoint": off_endpoint, "restore": off_restore},
        "scientific_promotion": False,
    }


def _load_model(source_root: Path, alias: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    spec = MODEL_SPECS[alias]
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        trust_remote_code=False,
    )
    if any(parameter.dtype is not torch.float32 for parameter in model.parameters()) or bool(getattr(model, "is_quantized", False)):
        raise ObservationBoundary("FULL_FP32 model closure failed")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_path,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    padding = bind_padding(tokenizer, model, padding_side="right")
    return model, tokenizer, spec, padding


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.cell_root.exists() or args.cell_root.is_symlink():
        raise ObservationBoundary(f"refusing to reuse cell root: {args.cell_root}")
    args.cell_root.mkdir(parents=True, mode=0o700)
    source_head = _git(args.source_root, "rev-parse", "HEAD")
    source_tree = _git(args.source_root, "rev-parse", "HEAD^{tree}")
    if source_head != args.expected_head or _git(args.source_root, "status", "--porcelain", "--untracked-files=no"):
        raise ObservationBoundary("source HEAD/clean identity differs")

    stream_receipt = preflight_transferred_stream_v2(
        STREAM_ROOT,
        archive=STREAM_ARCHIVE,
        dataset_path=DATASET,
    )
    if hashlib.sha256(DATASET.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ObservationBoundary("dataset identity differs")
    if stream_receipt["stream_root"] != STREAM_IDENTITY or stream_receipt["order_sha256"] != ORDER_IDENTITY or stream_receipt["evaluator_identity"] != EVALUATOR_IDENTITY:
        raise ObservationBoundary("stream/evaluator identity differs")
    seal = json.loads((STREAM_ROOT / "canonical/p1r24_independent_b10x10_stream_seal.json").read_text())
    batches = load_historical_h0_batches(DATASET, seal)
    if len(batches) != 10 or any(len(batch) != 10 for batch in batches):
        raise ObservationBoundary("B10x10 stream geometry differs")

    model, tokenizer, spec, padding_binding = _load_model(args.source_root, args.model)
    method = Method(args.method)
    hparams_path = args.source_root / (spec.memit_hparams if method is Method.MEMIT else spec.alpha_hparams)
    hparams = _load_hparams(method, hparams_path)
    hparams.device = 0
    if tuple(int(value) for value in hparams.layers) != LAYERS or str(hparams.model_name) != spec.statistics_model_dir:
        raise ObservationBoundary("Official hparams/layer/model binding differs")
    touched = _touched(model, hparams)
    w0 = _w0_identity(touched)
    if set(w0["dtypes"].values()) != {"torch.float32"}:
        raise ObservationBoundary("editable weights are not FULL_FP32")

    first_three = _official_requests(batches[0][:3])
    prompts = [str(row["prompt"]).format(str(row["subject"])) for row in first_three]
    padding_gate = padding_safety_gate(
        model,
        tokenizer,
        prompts,
        [str(row["target_new"]) for row in first_three],
        hparams.layer_module_tmp.format(LAYERS[-1]),
        NumericalLock(),
        input_module=hparams.rewrite_module_tmp.format(LAYERS[-1]),
        subject_templates=[str(row["prompt"]) for row in first_three],
        subjects=[str(row["subject"]) for row in first_three],
    )
    warm = _warm_official_state(model, tokenizer, method, hparams)

    common = {
        "instruction_id": INSTRUCTION_ID,
        "nonce": NONCE,
        "stage": args.stage,
        "campaign_id": args.campaign_id,
        "model": args.model,
        "method": method.value,
        "source": {"head": source_head, "tree": source_tree, "tracked_clean": True},
        "stream": {
            "root": STREAM_IDENTITY,
            "order": ORDER_IDENTITY,
            "evaluator": EVALUATOR_IDENTITY,
            "sample_duplication_count": 0,
        },
        "model_binding": {
            "revision": spec.model_revision,
            "snapshot": str(spec.model_path),
            "full_fp32": True,
            "quantized": False,
            "padding": padding_binding,
            "padding_gate": padding_gate,
        },
        "warm_state": warm,
        "w0": w0,
        "observation_lock": ObservationLock().payload(),
        "controller_or_update_influence_count": 0,
        "scientific_promotion": False,
    }
    if args.stage == "b1":
        result = {
            **common,
            "schema": "odeedit.s06.official-layer-realization-debt.b1-gate.v1",
            **_b1_gate(
                model=model,
                tokenizer=tokenizer,
                method=method,
                hparams=hparams,
                request=_official_requests(batches[0][:1])[0],
                touched=touched,
                w0=w0,
            ),
            "wall_seconds": time.perf_counter() - started,
        }
    else:
        b1 = json.loads(args.b1_result.read_text())
        if b1.get("status") != "B1_OBSERVER_ON_OFF_GATE_PASS" or b1.get("model") != args.model or b1.get("method") != method.value or b1.get("source", {}).get("head") != source_head:
            raise ObservationBoundary("sealed B1 release receipt differs")
        journals = []
        for index, rows in enumerate(batches, start=1):
            requests = _official_requests(rows)
            torch.manual_seed(541_100 + index)
            torch.cuda.manual_seed_all(541_100 + index)
            apply_payload, originals = _run_apply(
                model=model,
                tokenizer=tokenizer,
                method=method,
                hparams=hparams,
                requests=requests,
                touched=touched,
                capture_layers=True,
            )
            endpoint = _endpoint_metrics(model, tokenizer, requests)
            restore = _restore_w0(touched, originals, w0)
            batch_payload = {
                "schema": "odeedit.s06.official-layer-realization-debt.b10-slice.v1",
                "status": "TERMINAL_VALID",
                "batch_index": index,
                "model": args.model,
                "method": method.value,
                "request_count": len(requests),
                "request_order_sha256": apply_payload["request_order_sha256"],
                "apply": apply_payload,
                "endpoint": endpoint,
                "w0_restore": restore,
                "cross_batch_weight_continuity": 0,
                "cross_batch_cache_history_continuity": 0,
                "scientific_promotion": False,
            }
            path = args.cell_root / f"batch-{index:02d}.json"
            digest = _atomic_json(path, batch_payload)
            journals.append({"batch_index": index, "path": str(path), "sha256": digest, "request_count": len(requests)})
        result = {
            **common,
            "schema": "odeedit.s06.official-layer-realization-debt.b10x10-cell.v1",
            "status": "B10X10_TERMINAL_VALID",
            "batch_denominator": len(journals),
            "request_denominator": sum(row["request_count"] for row in journals),
            "valid_batch_denominator": len(journals),
            "valid_request_denominator": sum(row["request_count"] for row in journals),
            "journals": journals,
            "b1_release": {"path": str(args.b1_result), "sha256": hashlib.sha256(args.b1_result.read_bytes()).hexdigest()},
            "wall_seconds": time.perf_counter() - started,
        }
    result["identity_sha256"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("b1", "b10x10"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--method", choices=tuple(value.value for value in Method), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--b1-result", type=Path)
    args = parser.parse_args()
    if args.stage == "b10x10" and args.b1_result is None:
        parser.error("--b1-result is required for b10x10")
    cell_root_preexisted = args.cell_root.exists() or args.cell_root.is_symlink()
    try:
        payload = run(args)
        _atomic_json(args.cell_root / "result.json", payload)
    except BaseException as exc:
        if cell_root_preexisted:
            raise
        args.cell_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        failure = {
            "schema": "odeedit.s06.official-layer-realization-debt.failure.v1",
            "instruction_id": INSTRUCTION_ID,
            "stage": args.stage,
            "model": args.model,
            "method": args.method,
            "status": "INSTRUMENTATION_OR_TECHNICAL_HOLD",
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
            "scientific_denominator": 0,
            "imputation_count": 0,
            "scientific_promotion": False,
        }
        failure["identity_sha256"] = canonical_hash(failure)
        path = args.cell_root / "failure.json"
        if not path.exists() and not path.is_symlink():
            _atomic_json(path, failure)
        raise


if __name__ == "__main__":
    main()
