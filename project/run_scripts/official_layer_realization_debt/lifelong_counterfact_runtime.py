"""Evaluation-only locality-target-new backfill for frozen lifelong states."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.fixed_z_nonuniqueness.contracts import MODEL_SPECS, NumericalLock
from project.run_scripts.fixed_z_nonuniqueness.evaluation import padding_safety_gate, sequence_metrics
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.official_layer_realization_debt.contracts import Method
from project.run_scripts.official_layer_realization_debt.runtime import (
    _load_hparams,
    _load_model,
    _restore_w0,
    _touched,
    _w0_identity,
)

from .lifelong_counterfact_contracts import (
    CounterFactMetricBoundary,
    CounterFactMetricLock,
    FINALW_INPUT_LOCK,
    FINALW_INPUT_LOCK_SHA256,
    FINALW_RESULT_ROOT,
    INSTRUCTION_ID,
    SCHEMA,
)
from .lifelong_finalw_contracts import (
    AMENDED_CHECKPOINTS,
    EvaluationLock,
    LAYERS,
    ORDER_ROOT,
    STREAM_ROOT,
    STREAM_SEAL,
    STREAM_SEAL_SHA256,
    cell_mapping,
)
from .lifelong_finalw_evaluation import age_stratum, load_stream_and_rows, sha256_file
from .lifelong_finalw_runtime import (
    _atomic_json,
    _copy_checkpoint_weights,
    _git,
    _gzip_jsonl_create,
    _load_lock,
    _parameter_identity,
)


def _load_audit(path: Path, expected_sha: str, expected_head: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha:
        raise CounterFactMetricBoundary("availability audit SHA differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    identity = payload.pop("identity_sha256", None)
    if identity != canonical_hash(payload):
        raise CounterFactMetricBoundary("availability audit identity differs")
    if (
        value.get("status") != "MISSING_REQUIRES_EVALUATION_ONLY_BACKFILL"
        or value.get("source", {}).get("head") != expected_head
        or value.get("sealed_evaluation", {}).get("locality_target_new_prompt_denominator") != 0
        or value.get("sealed_evaluation", {}).get("request_state_rows") != 120_000
    ):
        raise CounterFactMetricBoundary("availability audit contract differs")
    return value


def _original_records_path(cell_index: int, count: int) -> Path:
    return (
        FINALW_RESULT_ROOT
        / f"cell-{cell_index}"
        / f"checkpoint-{count:05d}"
        / "request-metrics.jsonl.gz"
    )


def _first_original_records(cell_index: int, count: int, request_count: int = 3) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with gzip.open(_original_records_path(cell_index, count), "rt", encoding="utf-8") as handle:
        for line in handle:
            values.append(json.loads(line))
            if len(values) == request_count:
                break
    if len(values) != request_count:
        raise CounterFactMetricBoundary("sealed parity rows are incomplete")
    return values


def locality_true_parity_gate(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    original: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    if len(rows) != len(original) or len(rows) != 3:
        raise CounterFactMetricBoundary("locality parity requires exact three requests")
    max_abs = 0.0
    strict_equal = True
    prompt_count = 0
    forward_batches = 0
    with torch.inference_mode():
        for row, sealed in zip(rows, original, strict=True):
            if row["request_sha256"] != sealed["request_sha256"]:
                raise CounterFactMetricBoundary("locality parity request order differs")
            observed = sequence_metrics(
                model,
                tokenizer,
                list(row["locality_prompts"]),
                str(row["target_true"]),
            )
            expected = sealed["metrics"]["locality_target_true"]["prompts"]
            if int(observed["prompt_count"]) != len(expected) or len(expected) != 10:
                raise CounterFactMetricBoundary("locality parity prompt cardinality differs")
            for index, value in enumerate(expected):
                max_abs = max(
                    max_abs,
                    abs(float(observed["nll"][index]) - float(value["nll"])),
                    abs(float(observed["margin"][index]) - float(value["margin"])),
                )
                strict_equal = strict_equal and bool(observed["strict"][index]) == bool(value["strict"])
            prompt_count += len(expected)
            token_examples = len(expected) * int(observed["target_token_count"])
            forward_batches += math.ceil(token_examples / EvaluationLock().evaluator_batch_size)
    if max_abs > tolerance or not strict_equal:
        raise CounterFactMetricBoundary(
            f"sealed locality-target-true parity failed: max_abs={max_abs}, strict={strict_equal}"
        )
    return {
        "status": "PASS",
        "request_count": len(rows),
        "prompt_count": prompt_count,
        "maximum_nll_or_margin_absolute_error": max_abs,
        "strict_identity": strict_equal,
        "tolerance": tolerance,
        "forward_batch_count": forward_batches,
    }


def _evaluate_locality_target_new(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    seen_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    forward_batches = token_examples = prompt_count = 0
    with torch.inference_mode():
        for row in rows:
            value = sequence_metrics(
                model,
                tokenizer,
                list(row["locality_prompts"]),
                str(row["target_new"]),
            )
            prompts = [
                {
                    "nll": float(value["nll"][index]),
                    "margin": float(value["margin"][index]),
                    "strict": bool(value["strict"][index]),
                    "target_token_count": int(value["target_token_count"]),
                    "token_accuracy": "NOT_RECORDED_EVALUATOR_SCHEMA",
                    "token_correct_count": "NOT_RECORDED_EVALUATOR_SCHEMA",
                }
                for index in range(int(value["prompt_count"]))
            ]
            if len(prompts) != 10 or not all(
                math.isfinite(float(prompt[key]))
                for prompt in prompts
                for key in ("nll", "margin")
            ):
                raise CounterFactMetricBoundary("invalid locality-target-new evaluation")
            metric = {
                "prompt_count": len(prompts),
                "strict_count": sum(int(prompt["strict"]) for prompt in prompts),
                "token_correct_count": "NOT_RECORDED_EVALUATOR_SCHEMA",
                "target_token_count": len(prompts) * int(value["target_token_count"]),
                "prompts": prompts,
            }
            record: dict[str, Any] = {
                "ordinal": int(row["ordinal"]),
                "batch_index": int(row["batch_index"]),
                "case_identity_sha256": str(row["case_identity_sha256"]),
                "request_sha256": str(row["request_sha256"]),
                "age_stratum": age_stratum(int(row["ordinal"]), seen_count),
                "metrics": {"locality_target_new": metric},
                "raw_prompt_logit_generation_publish_count": 0,
            }
            record["identity_sha256"] = canonical_hash(record)
            records.append(record)
            prompt_count += len(prompts)
            examples = len(prompts) * int(value["target_token_count"])
            token_examples += examples
            forward_batches += math.ceil(examples / EvaluationLock().evaluator_batch_size)
    return records, {
        "request_count": len(records),
        "locality_prompt_count": prompt_count,
        "token_example_count": token_examples,
        "forward_batch_count": forward_batches,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.cell_root.exists() or args.cell_root.is_symlink():
        raise CounterFactMetricBoundary(f"refusing to reuse result root: {args.cell_root}")
    args.cell_root.mkdir(parents=True, mode=0o700)
    source_head = _git(args.source_root, "rev-parse", "HEAD")
    source_tree = _git(args.source_root, "rev-parse", "HEAD^{tree}")
    if source_head != args.expected_head or _git(
        args.source_root, "status", "--porcelain", "--untracked-files=no"
    ):
        raise CounterFactMetricBoundary("runtime source identity differs")
    audit = _load_audit(args.availability_audit, args.expected_audit_sha, source_head)
    lock = _load_lock(FINALW_INPUT_LOCK, FINALW_INPUT_LOCK_SHA256)
    model_alias, method_name, raw_root = cell_mapping(args.cell_index)
    arm = next(
        value for value in lock["arms"]
        if value["model"] == model_alias and value["method"] == method_name
    )
    if Path(arm["raw_root"]).resolve() != raw_root.resolve():
        raise CounterFactMetricBoundary("cell/raw-root mapping differs")
    checkpoints = [
        value for value in lock["checkpoints"]
        if value["model"] == model_alias and value["method"] == method_name
    ]
    if [value["accepted_edit_count"] for value in checkpoints] != list(AMENDED_CHECKPOINTS):
        raise CounterFactMetricBoundary("checkpoint schedule differs")
    if sha256_file(STREAM_SEAL) != STREAM_SEAL_SHA256:
        raise CounterFactMetricBoundary("stream seal SHA differs")
    seal, rows = load_stream_and_rows(STREAM_SEAL, Path(lock["stream"]["dataset_path"]))
    if seal["root_digest"] != STREAM_ROOT or seal["training_order_sha256"] != ORDER_ROOT:
        raise CounterFactMetricBoundary("stream/order identity differs")

    model, tokenizer, spec, padding_binding = _load_model(args.source_root, model_alias)
    method = Method(method_name)
    hparams_path = args.source_root / (
        spec.memit_hparams if method is Method.MEMIT else spec.alpha_hparams
    )
    hparams = _load_hparams(method, hparams_path)
    hparams.device = 0
    touched = _touched(model, hparams)
    if any(parameter.dtype is not torch.float32 for parameter in model.parameters()):
        raise CounterFactMetricBoundary("model parameter inventory is not FULL_FP32")
    w0 = _w0_identity(touched)
    originals = {name: value.detach().to("cpu").clone() for name, value in touched.items()}
    first = rows[:3]
    padding_gate = padding_safety_gate(
        model,
        tokenizer,
        [str(row["prompt"]) for row in first],
        [str(row["target_new"]) for row in first],
        hparams.layer_module_tmp.format(LAYERS[-1]),
        NumericalLock(),
        input_module=hparams.rewrite_module_tmp.format(LAYERS[-1]),
        subject_templates=[str(row["prompt_template"]) for row in first],
        subjects=[str(row["subject"]) for row in first],
    )
    torch.cuda.reset_peak_memory_stats()
    checkpoint_receipts: list[dict[str, Any]] = []
    total_counts = {
        "request_count": 0,
        "locality_prompt_count": 0,
        "token_example_count": 0,
        "model_forward_batch_count": 0,
    }
    for checkpoint in checkpoints:
        checkpoint_started = time.perf_counter()
        count = int(checkpoint["accepted_edit_count"])
        state_path = Path(checkpoint["state"]["path"])
        if sha256_file(state_path) != checkpoint["state"]["sha256"]:
            raise CounterFactMetricBoundary("checkpoint state SHA differs")
        state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=False)
        _copy_checkpoint_weights(touched, state)
        del state
        before = _parameter_identity(touched)
        expected_weights = {value["name"]: value["sha256"] for value in checkpoint["edited_weights"]}
        if before["sha256"] != expected_weights or set(before["dtypes"].values()) != {"torch.float32"}:
            raise CounterFactMetricBoundary("loaded checkpoint weight identity differs")
        selected = rows[:count]
        parity = locality_true_parity_gate(
            model,
            tokenizer,
            selected[:3],
            _first_original_records(args.cell_index, count),
            tolerance=NumericalLock().fp32_relative_tolerance,
        )
        output_records, counts = _evaluate_locality_target_new(
            model, tokenizer, selected, seen_count=count
        )
        if counts["request_count"] != count or counts["locality_prompt_count"] != 10 * count:
            raise CounterFactMetricBoundary("backfill denominator differs")
        after = _parameter_identity(touched)
        if before != after or any(parameter.grad is not None for parameter in model.parameters()):
            raise CounterFactMetricBoundary("evaluation mutated weights/pointers/versions or gradients")
        checkpoint_root = args.cell_root / f"checkpoint-{count:05d}"
        checkpoint_root.mkdir(mode=0o700)
        records_path = checkpoint_root / "locality-target-new.jsonl.gz"
        records_sha = _gzip_jsonl_create(records_path, output_records)
        receipt: dict[str, Any] = {
            "schema": f"{SCHEMA}.checkpoint-evaluation",
            "instruction_id": INSTRUCTION_ID,
            "model": model_alias,
            "method": method_name,
            "accepted_edit_count": count,
            "evaluation_type": "CHECKPOINT_FINAL_W_LOCALITY_TARGET_NEW_ONLY",
            "source_checkpoint": checkpoint,
            "state_loaded_once": True,
            "cache_loaded_into_runtime": False,
            "before_evaluation_weight_identity": before,
            "after_evaluation_weight_identity": after,
            "weight_pointer_version_bytes_exact": True,
            "locality_target_true_parity_gate": parity,
            "denominators": {
                "request": count,
                "locality_prompt": 10 * count,
            },
            "records": {
                "path": str(records_path),
                "bytes": records_path.stat().st_size,
                "sha256": records_sha,
                "rows": len(output_records),
            },
            "compute": {
                **counts,
                "model_forward_batch_count": counts["forward_batch_count"] + parity["forward_batch_count"],
                "parity_forward_batch_count": parity["forward_batch_count"],
                "checkpoint_load_count": 1,
                "compute_z_count": 0,
                "writer_count": 0,
                "key_count": 0,
                "solve_count": 0,
                "cache_append_count": 0,
                "cache_consume_count": 0,
                "history_append_count": 0,
                "optimizer_count": 0,
                "backward_count": 0,
                "parameter_gradient_count": 0,
                "model_update_during_evaluation_count": 0,
                "duplicate_evaluation_count": 0,
                "imputation_count": 0,
                "wall_seconds": time.perf_counter() - checkpoint_started,
            },
            "nonfinite_count": 0,
            "raw_prompt_logit_generation_publish_count": 0,
            "scientific_promotion": False,
        }
        receipt["identity_sha256"] = canonical_hash(receipt)
        receipt_path = checkpoint_root / "evaluation-receipt.json"
        receipt_sha = _atomic_json(receipt_path, receipt)
        checkpoint_receipts.append(
            {
                "accepted_edit_count": count,
                "path": str(receipt_path),
                "bytes": receipt_path.stat().st_size,
                "sha256": receipt_sha,
                "identity_sha256": receipt["identity_sha256"],
                "records_sha256": records_sha,
            }
        )
        total_counts["request_count"] += counts["request_count"]
        total_counts["locality_prompt_count"] += counts["locality_prompt_count"]
        total_counts["token_example_count"] += counts["token_example_count"]
        total_counts["model_forward_batch_count"] += receipt["compute"]["model_forward_batch_count"]
        del output_records
        torch.cuda.empty_cache()
    restore = _restore_w0(touched, originals, w0)
    terminal: dict[str, Any] = {
        "schema": f"{SCHEMA}.cell-terminal",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_PASS",
        "cell_index": args.cell_index,
        "model": model_alias,
        "method": method_name,
        "source": {
            "root": str(args.source_root),
            "head": source_head,
            "tree": source_tree,
            "tracked_clean": True,
        },
        "availability_audit": {
            "path": str(args.availability_audit),
            "sha256": args.expected_audit_sha,
            "identity_sha256": audit["identity_sha256"],
        },
        "metric_lock": CounterFactMetricLock().payload(),
        "stream": {"root": STREAM_ROOT, "order": ORDER_ROOT},
        "model_binding": {
            "snapshot": str(MODEL_SPECS[model_alias].model_path),
            "revision": MODEL_SPECS[model_alias].model_revision,
            "full_fp32": True,
            "autocast": False,
            "quantized": False,
            "padding": padding_binding,
            "padding_gate": padding_gate,
        },
        "checkpoint_denominator": len(checkpoint_receipts),
        "request_state_denominator": sum(AMENDED_CHECKPOINTS),
        "locality_prompt_denominator": 10 * sum(AMENDED_CHECKPOINTS),
        "checkpoint_receipts": checkpoint_receipts,
        "counts": total_counts,
        "terminal_w0_restore": restore,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.perf_counter() - started,
        "edit_replay_count": 0,
        "nonfinite_count": 0,
        "failure_count": 0,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    _atomic_json(args.cell_root / "terminal.json", terminal)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--availability-audit", type=Path, required=True)
    parser.add_argument("--expected-audit-sha", required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "model": result["model"],
                    "method": result["method"],
                    "cell_root": str(args.cell_root),
                },
                sort_keys=True,
            )
        )
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema": f"{SCHEMA}.cell-failure",
            "instruction_id": INSTRUCTION_ID,
            "status": "FAIL_CLOSE",
            "cell_index": args.cell_index,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "scientific_denominator": 0,
            "scientific_promotion": False,
        }
        if args.cell_root.exists() and not (args.cell_root / "failure.json").exists():
            failure["identity_sha256"] = canonical_hash(failure)
            _atomic_json(args.cell_root / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
