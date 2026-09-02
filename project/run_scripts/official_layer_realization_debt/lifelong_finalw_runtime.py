"""Evaluation-only loader for exact lifelong checkpoints and seen prefixes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.fixed_z_nonuniqueness.contracts import MODEL_SPECS, NumericalLock
from project.run_scripts.fixed_z_nonuniqueness.evaluation import padding_safety_gate
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.official_layer_realization_debt.contracts import Method
from project.run_scripts.official_layer_realization_debt.runtime import (
    _load_hparams,
    _load_model,
    _restore_w0,
    _touched,
    _w0_identity,
)

from .lifelong_finalw_contracts import (
    AMENDED_CHECKPOINTS,
    EVALUATOR_IDENTITY,
    EvaluationLock,
    FinalWeightBoundary,
    INSTRUCTION_ID,
    LAYERS,
    ORDER_ROOT,
    SCHEMA,
    STREAM_ROOT,
    STREAM_SEAL,
    STREAM_SEAL_SHA256,
    cell_mapping,
)
from .lifelong_finalw_evaluation import (
    CATEGORIES,
    age_stratum,
    evaluate_requests,
    evaluator_parity_gate,
    load_stream_and_rows,
    sha256_file,
)
from .lifelong_finalw_preflight import tensor_sha256_chunked


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_lock(path: Path, expected_sha: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha:
        raise FinalWeightBoundary("availability amendment receipt SHA differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.pop("identity_sha256", None)
    if identity != canonical_hash(value):
        raise FinalWeightBoundary("availability amendment canonical identity differs")
    value["identity_sha256"] = identity
    if (
        value.get("status") != "EXACT_STORED_CHECKPOINT_SCHEDULE_PASS"
        or tuple(value.get("amendment", {}).get("exact_stored_schedule", ())) != AMENDED_CHECKPOINTS
        or value.get("checkpoint_count") != 28
    ):
        raise FinalWeightBoundary("availability amendment gate differs")
    return value


def _parameter_identity(touched: Mapping[str, torch.nn.Parameter]) -> dict[str, Any]:
    return {
        "sha256": {name: tensor_sha256_chunked(value) for name, value in touched.items()},
        "pointers": {name: int(value.data_ptr()) for name, value in touched.items()},
        "versions": {name: int(value._version) for name, value in touched.items()},
        "dtypes": {name: str(value.dtype) for name, value in touched.items()},
    }


def _copy_checkpoint_weights(
    touched: Mapping[str, torch.nn.Parameter], state: Mapping[str, Any]
) -> None:
    from easyeditor.util.device import copy_to_param

    weights = state.get("weights")
    if not isinstance(weights, Mapping) or set(weights) != set(touched):
        raise FinalWeightBoundary("checkpoint/runtime edited-weight inventory differs")
    with torch.no_grad():
        for name, parameter in touched.items():
            source = weights[name]
            if source.dtype is not torch.float32 or source.shape != parameter.shape:
                raise FinalWeightBoundary("checkpoint/runtime edited tensor binding differs")
            copy_to_param(parameter, source)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    if not all(math.isfinite(float(value)) for value in values):
        raise FinalWeightBoundary("nonfinite checkpoint aggregate")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _quantile(values, 0.9),
        "max": max(values),
    }


class SummaryAccumulator:
    def __init__(self, seen_count: int) -> None:
        self.seen_count = seen_count
        self.request_count = 0
        self.prompt_nll: dict[str, list[float]] = {key: [] for key in CATEGORIES}
        self.prompt_margin: dict[str, list[float]] = {key: [] for key in CATEGORIES}
        self.request_nll: dict[str, list[float]] = {key: [] for key in CATEGORIES}
        self.request_margin: dict[str, list[float]] = {key: [] for key in CATEGORIES}
        self.strict_count = {key: 0 for key in CATEGORIES}
        self.prompt_count = {key: 0 for key in CATEGORIES}
        self.token_correct = {key: 0 for key in CATEGORIES}
        self.token_count = {key: 0 for key in CATEGORIES}
        self.gen_request_strict = 0
        self.rewrite_preference = 0
        self.rephrase_preference = 0
        self.age: dict[str, dict[str, Any]] = {}

    def add(self, record: Mapping[str, Any]) -> str:
        self.request_count += 1
        metrics = record["metrics"]
        for category in CATEGORIES:
            value = metrics[category]
            prompts = value["prompts"]
            nll = [float(item["nll"]) for item in prompts]
            margin = [float(item["margin"]) for item in prompts]
            self.prompt_nll[category].extend(nll)
            self.prompt_margin[category].extend(margin)
            self.request_nll[category].append(statistics.fmean(nll))
            self.request_margin[category].append(min(margin))
            self.strict_count[category] += int(value["strict_count"])
            self.prompt_count[category] += int(value["prompt_count"])
            self.token_correct[category] += int(value["token_correct_count"])
            self.token_count[category] += int(value["target_token_count"])
        if metrics["rephrase_target_new"]["strict_count"] == metrics["rephrase_target_new"]["prompt_count"]:
            self.gen_request_strict += 1
        rewrite_new = self.request_nll["rewrite_target_new"][-1]
        rewrite_true = self.request_nll["rewrite_target_true"][-1]
        rephrase_new = self.request_nll["rephrase_target_new"][-1]
        rephrase_true = self.request_nll["rephrase_target_true"][-1]
        self.rewrite_preference += int(rewrite_new < rewrite_true)
        self.rephrase_preference += int(rephrase_new < rephrase_true)
        stratum = age_stratum(int(record["ordinal"]), self.seen_count)
        group = self.age.setdefault(
            stratum,
            {
                "request_count": 0,
                "rewrite_strict_count": 0,
                "rephrase_prompt_count": 0,
                "rephrase_strict_count": 0,
                "rephrase_request_strict_count": 0,
                "locality_prompt_count": 0,
                "locality_strict_count": 0,
                "rewrite_target_new_nll": [],
                "rephrase_target_new_nll": [],
                "locality_target_true_nll": [],
            },
        )
        group["request_count"] += 1
        group["rewrite_strict_count"] += int(metrics["rewrite_target_new"]["strict_count"])
        group["rephrase_prompt_count"] += int(metrics["rephrase_target_new"]["prompt_count"])
        group["rephrase_strict_count"] += int(metrics["rephrase_target_new"]["strict_count"])
        group["rephrase_request_strict_count"] += int(
            metrics["rephrase_target_new"]["strict_count"] == metrics["rephrase_target_new"]["prompt_count"]
        )
        group["locality_prompt_count"] += int(metrics["locality_target_true"]["prompt_count"])
        group["locality_strict_count"] += int(metrics["locality_target_true"]["strict_count"])
        group["rewrite_target_new_nll"].append(rewrite_new)
        group["rephrase_target_new_nll"].append(rephrase_new)
        group["locality_target_true_nll"].append(self.request_nll["locality_target_true"][-1])
        return stratum

    def payload(self) -> dict[str, Any]:
        if self.request_count != self.seen_count:
            raise FinalWeightBoundary("checkpoint evaluated request denominator differs")
        categories = {}
        for category in CATEGORIES:
            categories[category] = {
                "prompt_denominator": self.prompt_count[category],
                "strict_numerator": self.strict_count[category],
                "strict_rate": self.strict_count[category] / self.prompt_count[category],
                "token_denominator": self.token_count[category],
                "token_correct_numerator": self.token_correct[category],
                "token_accuracy": self.token_correct[category] / self.token_count[category],
                "prompt_nll": _distribution(self.prompt_nll[category]),
                "request_cluster_nll": _distribution(self.request_nll[category]),
                "prompt_margin": _distribution(self.prompt_margin[category]),
                "request_cluster_min_margin": _distribution(self.request_margin[category]),
            }
        age = {}
        for name, group in sorted(self.age.items()):
            requests = int(group["request_count"])
            rephrase_prompts = int(group["rephrase_prompt_count"])
            locality_prompts = int(group["locality_prompt_count"])
            age[name] = {
                "request_denominator": requests,
                "rewrite_strict_numerator": group["rewrite_strict_count"],
                "eff": group["rewrite_strict_count"] / requests,
                "rephrase_prompt_denominator": rephrase_prompts,
                "rephrase_strict_numerator": group["rephrase_strict_count"],
                "gen_prompt": group["rephrase_strict_count"] / rephrase_prompts,
                "gen_strict_numerator": group["rephrase_request_strict_count"],
                "gen_strict": group["rephrase_request_strict_count"] / requests,
                "locality_prompt_denominator": locality_prompts,
                "locality_strict_numerator": group["locality_strict_count"],
                "loc": group["locality_strict_count"] / locality_prompts,
                "rewrite_target_new_request_nll": _distribution(group["rewrite_target_new_nll"]),
                "rephrase_target_new_request_nll": _distribution(group["rephrase_target_new_nll"]),
                "locality_target_true_request_nll": _distribution(group["locality_target_true_nll"]),
            }
        return {
            "request_denominator": self.request_count,
            "rewrite_prompt_denominator": self.prompt_count["rewrite_target_new"],
            "eff_strict_numerator": self.strict_count["rewrite_target_new"],
            "eff": self.strict_count["rewrite_target_new"] / self.request_count,
            "rephrase_prompt_denominator": self.prompt_count["rephrase_target_new"],
            "gen_prompt_strict_numerator": self.strict_count["rephrase_target_new"],
            "gen_prompt": self.strict_count["rephrase_target_new"] / self.prompt_count["rephrase_target_new"],
            "gen_strict_request_denominator": self.request_count,
            "gen_strict_numerator": self.gen_request_strict,
            "gen_strict": self.gen_request_strict / self.request_count,
            "locality_prompt_denominator": self.prompt_count["locality_target_true"],
            "loc_strict_numerator": self.strict_count["locality_target_true"],
            "loc": self.strict_count["locality_target_true"] / self.prompt_count["locality_target_true"],
            "rewrite_new_preferred_numerator": self.rewrite_preference,
            "rewrite_new_preferred_rate": self.rewrite_preference / self.request_count,
            "rephrase_new_preferred_numerator": self.rephrase_preference,
            "rephrase_new_preferred_rate": self.rephrase_preference / self.request_count,
            "categories": categories,
            "age_strata": age,
        }


def _gzip_jsonl_create(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for record in records:
                    text.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
                    text.write("\n")
    os.chmod(path, 0o600)
    return sha256_file(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.cell_root.exists() or args.cell_root.is_symlink():
        raise FinalWeightBoundary(f"refusing to reuse result root: {args.cell_root}")
    args.cell_root.mkdir(parents=True, mode=0o700)
    source_head = _git(args.source_root, "rev-parse", "HEAD")
    source_tree = _git(args.source_root, "rev-parse", "HEAD^{tree}")
    if source_head != args.expected_head or _git(args.source_root, "status", "--porcelain", "--untracked-files=no"):
        raise FinalWeightBoundary("runtime source identity differs")
    lock = _load_lock(args.input_lock, args.expected_input_lock_sha)
    model_alias, method_name, raw_root = cell_mapping(args.cell_index)
    arm = next(
        value for value in lock["arms"]
        if value["model"] == model_alias and value["method"] == method_name
    )
    if Path(arm["raw_root"]).resolve() != raw_root.resolve():
        raise FinalWeightBoundary("cell/raw-root mapping differs")
    checkpoints = [
        value for value in lock["checkpoints"]
        if value["model"] == model_alias and value["method"] == method_name
    ]
    if [value["accepted_edit_count"] for value in checkpoints] != list(AMENDED_CHECKPOINTS):
        raise FinalWeightBoundary("cell checkpoint schedule differs")
    if sha256_file(STREAM_SEAL) != STREAM_SEAL_SHA256:
        raise FinalWeightBoundary("runtime stream seal SHA differs")
    seal, rows = load_stream_and_rows(STREAM_SEAL, Path(lock["stream"]["dataset_path"]))
    if seal["root_digest"] != STREAM_ROOT or seal["training_order_sha256"] != ORDER_ROOT:
        raise FinalWeightBoundary("runtime stream/order identity differs")

    model, tokenizer, spec, padding_binding = _load_model(args.source_root, model_alias)
    method = Method(method_name)
    hparams_path = args.source_root / (spec.memit_hparams if method is Method.MEMIT else spec.alpha_hparams)
    hparams = _load_hparams(method, hparams_path)
    hparams.device = 0
    touched = _touched(model, hparams)
    if any(parameter.dtype is not torch.float32 for parameter in model.parameters()):
        raise FinalWeightBoundary("model parameter inventory is not FULL_FP32")
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
    evaluator_gate = evaluator_parity_gate(
        model,
        tokenizer,
        first,
        tolerance=NumericalLock().fp32_relative_tolerance,
        batch_size=EvaluationLock().evaluator_batch_size,
    )
    torch.cuda.reset_peak_memory_stats()
    checkpoint_receipts: list[dict[str, Any]] = []
    total_counts = {"request_count": 0, "forward_batch_count": 0, "token_example_count": 0}
    for checkpoint in checkpoints:
        checkpoint_started = time.perf_counter()
        count = int(checkpoint["accepted_edit_count"])
        state_path = Path(checkpoint["state"]["path"])
        if sha256_file(state_path) != checkpoint["state"]["sha256"]:
            raise FinalWeightBoundary("runtime checkpoint state SHA differs")
        state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=False)
        _copy_checkpoint_weights(touched, state)
        del state
        before = _parameter_identity(touched)
        expected_weights = {
            value["name"]: value["sha256"] for value in checkpoint["edited_weights"]
        }
        if before["sha256"] != expected_weights or set(before["dtypes"].values()) != {"torch.float32"}:
            raise FinalWeightBoundary("loaded checkpoint edited-weight identity differs")
        selected = rows[:count]
        accumulator = SummaryAccumulator(count)
        output_records: list[dict[str, Any]] = []
        local_counts = {"request_count": 0, "forward_batch_count": 0, "token_example_count": 0}
        for record, counts in evaluate_requests(
            model,
            tokenizer,
            selected,
            batch_size=EvaluationLock().evaluator_batch_size,
            request_chunk_size=EvaluationLock().request_chunk_size,
        ):
            record["age_stratum"] = accumulator.add(record)
            output_records.append(record)
            for key, value in counts.items():
                local_counts[key] += int(value)
        summary = accumulator.payload()
        expected_denominators = {
            "request": count,
            "rewrite_prompt": count,
            "rephrase_prompt": 2 * count,
            "locality_prompt": 10 * count,
        }
        if (
            summary["request_denominator"] != count
            or summary["rewrite_prompt_denominator"] != count
            or summary["rephrase_prompt_denominator"] != 2 * count
            or summary["locality_prompt_denominator"] != 10 * count
        ):
            raise FinalWeightBoundary("checkpoint evaluator denominator differs")
        after = _parameter_identity(touched)
        if before != after or any(value.grad is not None for value in model.parameters()):
            raise FinalWeightBoundary("frozen evaluation mutated weights/pointers/versions or gradients")
        checkpoint_root = args.cell_root / f"checkpoint-{count:05d}"
        checkpoint_root.mkdir(mode=0o700)
        records_path = checkpoint_root / "request-metrics.jsonl.gz"
        records_sha = _gzip_jsonl_create(records_path, output_records)
        receipt = {
            "schema": f"{SCHEMA}.checkpoint-evaluation",
            "instruction_id": INSTRUCTION_ID,
            "model": model_alias,
            "method": method_name,
            "accepted_edit_count": count,
            "evaluation_type": "CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS",
            "is_final_w_full10k": count == 10_000,
            "source_checkpoint": checkpoint,
            "state_loaded_once": True,
            "cache_loaded_into_runtime": False,
            "before_evaluation_weight_identity": before,
            "after_evaluation_weight_identity": after,
            "weight_pointer_version_bytes_exact": True,
            "summary": summary,
            "expected_denominators": expected_denominators,
            "records": {
                "path": str(records_path),
                "bytes": records_path.stat().st_size,
                "sha256": records_sha,
                "rows": len(output_records),
            },
            "compute": {
                **local_counts,
                "wall_seconds": time.perf_counter() - checkpoint_started,
                "checkpoint_load_count": 1,
                "model_forward_batch_count": local_counts["forward_batch_count"],
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
        for key in total_counts:
            total_counts[key] += local_counts[key]
        del output_records
        torch.cuda.empty_cache()
    restore = _restore_w0(touched, originals, w0)
    terminal = {
        "schema": f"{SCHEMA}.cell-terminal",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_PASS",
        "cell_index": args.cell_index,
        "model": model_alias,
        "method": method_name,
        "source": {"root": str(args.source_root), "head": source_head, "tree": source_tree, "tracked_clean": True},
        "input_lock": {"path": str(args.input_lock), "sha256": args.expected_input_lock_sha, "identity_sha256": lock["identity_sha256"]},
        "stream": {"root": STREAM_ROOT, "order": ORDER_ROOT, "evaluator": EVALUATOR_IDENTITY},
        "model_binding": {
            "snapshot": str(MODEL_SPECS[model_alias].model_path),
            "revision": MODEL_SPECS[model_alias].model_revision,
            "full_fp32": True,
            "autocast": False,
            "quantized": False,
            "padding": padding_binding,
            "padding_gate": padding_gate,
            "evaluator_parity_gate": evaluator_gate,
        },
        "evaluation_lock": EvaluationLock().payload(),
        "checkpoint_denominator": len(checkpoint_receipts),
        "request_state_denominator": sum(AMENDED_CHECKPOINTS),
        "checkpoint_receipts": checkpoint_receipts,
        "counts": total_counts,
        "terminal_w0_restore": restore,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.perf_counter() - started,
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
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--expected-input-lock-sha", required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps({"status": result["status"], "model": result["model"], "method": result["method"], "cell_root": str(args.cell_root)}, sort_keys=True))
    except Exception as exc:
        failure = {
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
