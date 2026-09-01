"""Cumulative B1-to-B10 Official MEMIT/AlphaEdit observation runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping

import torch

from project.run_scripts.fixed_z_nonuniqueness.contracts import MODEL_SPECS, NumericalLock
from project.run_scripts.fixed_z_nonuniqueness.evaluation import padding_safety_gate
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r24_independent_b10x10_selection import load_historical_h0_batches
from project.run_scripts.ode_bf.p4_sealed_stream import preflight_transferred_stream_v2

from .continuity import (
    verify_cache_continuity,
    verify_cell_totals,
    verify_observed_batch,
    verify_weight_continuity,
)
from .contracts import (
    DATASET,
    DATASET_SHA256,
    EVALUATOR_IDENTITY,
    LAYERS,
    Method,
    ORDER_IDENTITY,
    ObservationBoundary,
    STREAM_ARCHIVE,
    STREAM_IDENTITY,
    STREAM_ROOT,
)
from .runtime import (
    _atomic_json,
    _endpoint_metrics,
    _git,
    _load_hparams,
    _load_model,
    _method_module,
    _official_requests,
    _restore_w0,
    _run_apply,
    _touched,
    _warm_official_state,
    _w0_identity,
)
from .sequential_contracts import INSTRUCTION_ID, NONCE, SequentialObservationLock


def _alpha_cache_entry_snapshot(method: Method) -> dict[str, Any]:
    if method is not Method.ALPHAEDIT:
        return {"applicable": False}
    module = _method_module(method)
    present = hasattr(module, "cache_c")
    value = getattr(module, "cache_c", None)
    clone = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else None
    return {
        "applicable": True,
        "cache_c_new": bool(getattr(module, "cache_c_new", False)),
        "present": present,
        "tensor": clone,
        "sha256": None if clone is None else tensor_sha256(clone),
    }


def _restore_alpha_cache(method: Method, entry: Mapping[str, Any]) -> dict[str, Any]:
    if method is not Method.ALPHAEDIT:
        return {"applicable": False, "exact": True}
    module = _method_module(method)
    module.cache_c_new = bool(entry["cache_c_new"])
    if bool(entry["present"]):
        tensor = entry["tensor"]
        if isinstance(tensor, torch.Tensor):
            module.cache_c = tensor.detach().cpu().clone()
        else:
            module.cache_c = tensor
    elif hasattr(module, "cache_c"):
        delattr(module, "cache_c")
    observed_present = hasattr(module, "cache_c")
    observed = getattr(module, "cache_c", None)
    observed_sha = tensor_sha256(observed) if isinstance(observed, torch.Tensor) else None
    exact = (
        bool(getattr(module, "cache_c_new", False)) == bool(entry["cache_c_new"])
        and observed_present == bool(entry["present"])
        and observed_sha == entry["sha256"]
    )
    if not exact:
        raise ObservationBoundary("AlphaEdit terminal cache restore differs")
    return {
        "applicable": True,
        "exact": True,
        "cache_c_new": bool(entry["cache_c_new"]),
        "present": bool(entry["present"]),
        "sha256": entry["sha256"],
    }


def _load_stream() -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    stream = preflight_transferred_stream_v2(
        STREAM_ROOT,
        archive=STREAM_ARCHIVE,
        dataset_path=DATASET,
    )
    if hashlib.sha256(DATASET.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ObservationBoundary("dataset identity differs")
    if (
        stream["stream_root"] != STREAM_IDENTITY
        or stream["order_sha256"] != ORDER_IDENTITY
        or stream["evaluator_identity"] != EVALUATOR_IDENTITY
    ):
        raise ObservationBoundary("stream/evaluator identity differs")
    seal = json.loads(
        (STREAM_ROOT / "canonical/p1r24_independent_b10x10_stream_seal.json").read_text()
    )
    batches = load_historical_h0_batches(DATASET, seal)
    if len(batches) != 10 or any(len(batch) != 10 for batch in batches):
        raise ObservationBoundary("sequential B1-to-B10 geometry differs")
    return batches, stream


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.cell_root.exists() or args.cell_root.is_symlink():
        raise ObservationBoundary(f"refusing to reuse cell root: {args.cell_root}")
    args.cell_root.mkdir(parents=True, mode=0o700)
    source_head = _git(args.source_root, "rev-parse", "HEAD")
    source_tree = _git(args.source_root, "rev-parse", "HEAD^{tree}")
    if source_head != args.expected_head or _git(
        args.source_root, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ObservationBoundary("source HEAD/clean identity differs")

    batches, stream = _load_stream()
    model, tokenizer, spec, padding_binding = _load_model(args.source_root, args.model)
    method = Method(args.method)
    hparams_path = args.source_root / (
        spec.memit_hparams if method is Method.MEMIT else spec.alpha_hparams
    )
    hparams = _load_hparams(method, hparams_path)
    hparams.device = 0
    if tuple(int(value) for value in hparams.layers) != LAYERS or str(
        hparams.model_name
    ) != spec.statistics_model_dir:
        raise ObservationBoundary("Official hparams/layer/model binding differs")
    touched = _touched(model, hparams)
    w0 = _w0_identity(touched)
    if set(w0["dtypes"].values()) != {"torch.float32"}:
        raise ObservationBoundary("editable weights are not FULL_FP32")
    w0_values = {
        name: value.detach().to(device="cpu", dtype=value.dtype).clone()
        for name, value in touched.items()
    }

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
    alpha_cache_entry = _alpha_cache_entry_snapshot(method)

    journals: list[dict[str, Any]] = []
    previous_commit = dict(w0["sha256"])
    prior_cache_exit: str | None = None
    terminal_w0_restore: dict[str, Any] | None = None
    terminal_cache_restore: dict[str, Any] | None = None
    terminal_commit_sha: dict[str, str] | None = None
    try:
        for batch_index, rows in enumerate(batches, start=1):
            requests = _official_requests(rows)
            torch.manual_seed(641_100 + batch_index)
            torch.cuda.manual_seed_all(641_100 + batch_index)
            apply_payload, _ = _run_apply(
                model=model,
                tokenizer=tokenizer,
                method=method,
                hparams=hparams,
                requests=requests,
                touched=touched,
                capture_layers=True,
                reset_alpha_cache=batch_index == 1,
                alpha_cache_history_width=(batch_index - 1) * len(requests),
            )
            observer_checks = verify_observed_batch(
                apply_payload, method=method, request_count=len(requests)
            )
            weight_continuity = verify_weight_continuity(
                apply_payload, expected_entry=previous_commit
            )
            cache_continuity = verify_cache_continuity(
                method,
                apply_payload,
                batch_index=batch_index,
                request_count=len(requests),
                prior_exit_sha256=prior_cache_exit,
            )
            endpoint = _endpoint_metrics(model, tokenizer, requests)
            observer = apply_payload["layer_realization_observer"]
            first_valid = None
            if batch_index == 1:
                first_valid = {
                    "status": "FIRST_B1_TRANSACTION_GATE_PASS",
                    "observer_checks": observer_checks,
                    "weight_entry_is_pristine_w0": apply_payload["entry_sha256"]
                    == w0["sha256"],
                    "cache_entry_width": cache_continuity["entry_width"],
                    "cache_exit_width": cache_continuity["exit_width"],
                    "counter_influence_count": 0,
                }
                if not first_valid["weight_entry_is_pristine_w0"]:
                    raise ObservationBoundary("first sequential B1 did not start at W0")
            batch_payload = {
                "schema": "odeedit.s06.official-layer-realization-debt.sequential-batch.v1",
                "instruction_id": INSTRUCTION_ID,
                "status": "SEQUENTIAL_BATCH_TERMINAL_VALID",
                "model": args.model,
                "method": method.value,
                "batch_index": batch_index,
                "request_count": len(requests),
                "request_order_sha256": apply_payload["request_order_sha256"],
                "apply": apply_payload,
                "endpoint": endpoint,
                "weight_continuity": weight_continuity,
                "cache_continuity": cache_continuity,
                "weight_link_from_previous": int(batch_index > 1),
                "first_valid_gate": first_valid,
                "cross_batch_weight_continuity": 1,
                "cross_batch_cache_history_continuity": int(
                    method is Method.ALPHAEDIT
                ),
                "scientific_promotion": False,
            }
            journal_path = args.cell_root / f"batch-{batch_index:02d}.json"
            digest = _atomic_json(journal_path, batch_payload)
            journals.append(
                {
                    "batch_index": batch_index,
                    "path": str(journal_path),
                    "sha256": digest,
                    "request_count": len(requests),
                    "direct_z_compute_count": int(observer["direct_z_compute_count"]),
                    "direct_z_recompute_count": int(
                        observer["direct_z_recompute_count"]
                    ),
                    "layer_observation_count": int(
                        observer["layer_loop_observation_copy_count"]
                    ),
                    "terminal_forward_count": int(
                        observer["terminal_post_L8_forward_count"]
                    ),
                    "weight_link_from_previous": int(batch_index > 1),
                }
            )
            previous_commit = dict(apply_payload["edited_sha256"])
            prior_cache_exit = str(cache_continuity["exit_sha256"])

        terminal_commit_sha = dict(previous_commit)
        totals = verify_cell_totals(journals)
        terminal_w0_restore = _restore_w0(touched, w0_values, w0)
        terminal_cache_restore = _restore_alpha_cache(method, alpha_cache_entry)
    except BaseException:
        # No model state may escape the task-owned process, including on a
        # typed instrumentation boundary.  Partial journals remain immutable.
        _restore_w0(touched, w0_values, w0)
        _restore_alpha_cache(method, alpha_cache_entry)
        raise

    result = {
        "schema": "odeedit.s06.official-layer-realization-debt.sequential-cell.v1",
        "instruction_id": INSTRUCTION_ID,
        "nonce": NONCE,
        "status": "SEQUENTIAL_B1_TO_B10_TERMINAL_VALID",
        "campaign_id": args.campaign_id,
        "model": args.model,
        "method": method.value,
        "source": {"head": source_head, "tree": source_tree, "tracked_clean": True},
        "stream": {
            "root": STREAM_IDENTITY,
            "order": ORDER_IDENTITY,
            "evaluator": EVALUATOR_IDENTITY,
            "preflight_identity": stream["identity_sha256"],
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
        "initial_w0": w0,
        "terminal_committed_weight_sha256": terminal_commit_sha,
        "terminal_w0_restore": terminal_w0_restore,
        "terminal_cache_restore": terminal_cache_restore,
        "observation_lock": SequentialObservationLock().payload(),
        "batch_denominator": 10,
        "request_denominator": 100,
        "valid_batch_denominator": 10,
        "valid_request_denominator": 100,
        "totals": totals,
        "journals": journals,
        "wall_seconds": time.perf_counter() - started,
        "controller_or_update_influence_count": 0,
        "raw_prompt_logit_generation_publish_count": 0,
        "scientific_promotion": False,
    }
    result["identity_sha256"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--method", choices=tuple(value.value for value in Method), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()
    preexisted = args.cell_root.exists() or args.cell_root.is_symlink()
    try:
        payload = run(args)
        _atomic_json(args.cell_root / "result.json", payload)
    except BaseException as exc:
        if preexisted:
            raise
        args.cell_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        failure = {
            "schema": "odeedit.s06.official-layer-realization-debt.sequential-failure.v1",
            "instruction_id": INSTRUCTION_ID,
            "model": args.model,
            "method": args.method,
            "status": "INSTRUMENTATION_OR_TECHNICAL_HOLD",
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
            "partial_journal_count": len(list(args.cell_root.glob("batch-*.json"))),
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
