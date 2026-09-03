"""Read-only v6 aggregation for canonical CounterFact NLL-pair metrics."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from project.run_scripts.ode_bf.contracts import canonical_hash

from .lifelong_counterfact_contracts import (
    ARM_KEY,
    ARM_LABEL,
    ARM_ORDER,
    CANONICAL_COUNTERFACT_EVALUATOR,
    CANONICAL_COUNTERFACT_EVALUATOR_SHA256,
    CANONICAL_COUNTERFACT_SUMMARIZER,
    CANONICAL_COUNTERFACT_SUMMARIZER_SHA256,
    CounterFactMetricBoundary,
    FINALW_RESULT_ROOT,
    INSTRUCTION_ID,
    SCHEMA,
    V4_MANIFEST_SHA256,
    V4_RECEIPT_SHA256,
    V4_RELATIVE,
    V4_REPORT_SHA256,
    V5_MANIFEST_SHA256,
    V5_RECEIPT_SHA256,
    V5_RELATIVE,
    V5_REPORT_SHA256,
)
from .lifelong_counterfact_figures import generate as generate_figures
from .lifelong_counterfact_metrics import distribution, strict_nll_pair_success
from .lifelong_finalw_contracts import AMENDED_CHECKPOINTS, ORDER_ROOT, STREAM_ROOT, cell_mapping
from .lifelong_finalw_evaluation import age_stratum, sha256_file
from .lifelong_v5_analysis import _verify_package


REPORT_NAME = "official-layer-debt-lifelong-counterfact-metrics-v6-factual-ko.md"
BACKFILL_JOB_ID = "33539"
CAMPAIGN_STATE_ROOT = Path(
    "/data/janghj/ODE-edit/local/state/"
    "official-layer-realization-debt-lifelong-counterfact-metrics-v6/"
    "campaign-20260903-v1"
)
SOURCE_MANIFEST = CAMPAIGN_STATE_ROOT / "source-manifest.json"
SOURCE_MANIFEST_SHA256 = "300978510c849bf22fe0506a37a4416c2625d1d6b965dc735e82b057924ad557"
PREGPU_RECEIPT = CAMPAIGN_STATE_ROOT / "pregpu-receipt.json"
PREGPU_RECEIPT_SHA256 = "5c07467e00cf1143cd339116f69810c287b7c55de5190cfea054a393aae16296"
AVAILABILITY_AUDIT = Path(
    "/data/janghj/ODE-edit/local/state/"
    "official-layer-realization-debt-lifelong-counterfact-metrics-v6/"
    "campaign-20260903-v1/availability-audit.json"
)
AVAILABILITY_AUDIT_SHA256 = "c0bbd1cdbd82b6b325c09158cd06b12498033ef2a39a9f9f4ebda8c28de7d265"
STATS = ("mean", "median", "p90", "max")
PRIMARY = ("rs", "ps", "ns")
PRIMARY_LABEL = {"rs": "RS", "ps": "PS", "ns": "NS"}
AGE_ORDER = ("EARLY_FIRST_20PCT", "MIDDLE_60PCT", "RECENT_LAST_20PCT")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _identity(value: Mapping[str, Any], label: str, *, derived_age: bool = False) -> None:
    payload = dict(value)
    expected = payload.pop("identity_sha256", None)
    if derived_age:
        payload.pop("age_stratum", None)
    if expected != canonical_hash(payload):
        raise CounterFactMetricBoundary(f"canonical identity differs: {label}")


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CounterFactMetricBoundary(f"JSON object expected: {path}")
    return value


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise CounterFactMetricBoundary(f"refusing empty table: {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"path": path.name, "rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(raw)


def _fmt(value: Any, digits: int = 5) -> str:
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(result)


def _open_pair_writer(path: Path) -> tuple[Any, Any, Any]:
    raw = path.open("xb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    return raw, compressed, text


def _record_iter(path: Path, expected_sha: str, count: int, *, derived_age: bool) -> Iterable[dict[str, Any]]:
    if sha256_file(path) != expected_sha:
        raise CounterFactMetricBoundary(f"record member SHA differs: {path}")
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            _identity(value, f"{path}:{rows + 1}", derived_age=derived_age)
            if value.get("age_stratum") != age_stratum(int(value["ordinal"]), count):
                raise CounterFactMetricBoundary("record edit-age derivation differs")
            rows += 1
            yield value
    if rows != count:
        raise CounterFactMetricBoundary(f"record row denominator differs: {path}")


def _terminal_and_receipts(
    backfill_root: Path,
) -> tuple[dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    backfill_members: list[dict[str, Any]] = []
    compute: list[dict[str, Any]] = []
    for cell_index in range(4):
        model, method, _ = cell_mapping(cell_index)
        arm = ARM_KEY[(model, method)]
        cell = backfill_root / f"cell-{cell_index}"
        terminal_path = cell / "terminal.json"
        terminal = _object(terminal_path)
        _identity(terminal, str(terminal_path))
        if (
            terminal.get("status") != "TERMINAL_PASS"
            or terminal.get("model") != model
            or terminal.get("method") != method
            or terminal.get("source", {}).get("head")
            != "c6edbf62671ab66cdfda8a5af59a7043c2ef81b2"
            or terminal.get("source", {}).get("tree")
            != "16f5d75cb3e4bc6707688a0ff6cb2fd9e59186ed"
            or terminal.get("source", {}).get("tracked_clean") is not True
            or terminal.get("stream") != {"root": STREAM_ROOT, "order": ORDER_ROOT}
            or terminal.get("checkpoint_denominator") != 7
            or terminal.get("request_state_denominator") != sum(AMENDED_CHECKPOINTS)
            or terminal.get("locality_prompt_denominator") != 10 * sum(AMENDED_CHECKPOINTS)
            or terminal.get("counts", {}).get("request_count") != sum(AMENDED_CHECKPOINTS)
            or terminal.get("counts", {}).get("locality_prompt_count")
            != 10 * sum(AMENDED_CHECKPOINTS)
            or terminal.get("edit_replay_count") != 0
            or terminal.get("nonfinite_count") != 0
            or terminal.get("failure_count") != 0
            or terminal.get("imputation_count") != 0
        ):
            raise CounterFactMetricBoundary(f"backfill terminal contract differs: {arm}")
        model_binding = terminal.get("model_binding", {})
        if (
            model_binding.get("full_fp32") is not True
            or model_binding.get("autocast") is not False
            or model_binding.get("quantized") is not False
            or model_binding.get("padding_gate", {}).get("status") != "PASS"
        ):
            raise CounterFactMetricBoundary(f"backfill model/padding binding differs: {arm}")
        backfill_members.append(
            {
                "kind": "CELL_TERMINAL",
                "arm": arm,
                "path": str(terminal_path),
                "bytes": terminal_path.stat().st_size,
                "sha256": sha256_file(terminal_path),
            }
        )
        if terminal.get("terminal_w0_restore", {}).get("exact") is not True:
            raise CounterFactMetricBoundary(f"backfill W0 restore differs: {arm}")
        receipt_index = {
            int(value["accepted_edit_count"]): value for value in terminal["checkpoint_receipts"]
        }
        if tuple(sorted(receipt_index)) != AMENDED_CHECKPOINTS:
            raise CounterFactMetricBoundary(f"backfill checkpoint schedule differs: {arm}")
        for count in AMENDED_CHECKPOINTS:
            spec = receipt_index[count]
            receipt_path = Path(spec["path"])
            if sha256_file(receipt_path) != spec["sha256"]:
                raise CounterFactMetricBoundary("backfill receipt member SHA differs")
            receipt = _object(receipt_path)
            _identity(receipt, str(receipt_path))
            records_path = Path(receipt["records"]["path"])
            if (
                receipt.get("evaluation_type") != "CHECKPOINT_FINAL_W_LOCALITY_TARGET_NEW_ONLY"
                or receipt.get("accepted_edit_count") != count
                or receipt.get("weight_pointer_version_bytes_exact") is not True
                or receipt.get("locality_target_true_parity_gate", {}).get("status") != "PASS"
                or receipt.get("denominators") != {"request": count, "locality_prompt": 10 * count}
                or receipt.get("nonfinite_count") != 0
            ):
                raise CounterFactMetricBoundary("backfill checkpoint invariant differs")
            checkpoint = receipt.get("source_checkpoint", {})
            edited_weights = checkpoint.get("edited_weights", [])
            expected_weight_sha = {
                value.get("name"): value.get("sha256") for value in edited_weights
            }
            if (
                checkpoint.get("model") != model
                or checkpoint.get("method") != method
                or checkpoint.get("accepted_edit_count") != count
                or len(edited_weights) != 5
                or any(value.get("dtype") != "torch.float32" for value in edited_weights)
                or checkpoint.get("edited_weight_identity_sha256") is None
                or receipt.get("before_evaluation_weight_identity", {}).get("sha256")
                != expected_weight_sha
            ):
                raise CounterFactMetricBoundary("backfill checkpoint/state binding differs")
            forbidden = (
                "compute_z_count",
                "writer_count",
                "key_count",
                "solve_count",
                "cache_append_count",
                "cache_consume_count",
                "history_append_count",
                "optimizer_count",
                "backward_count",
                "parameter_gradient_count",
                "model_update_during_evaluation_count",
                "duplicate_evaluation_count",
                "imputation_count",
            )
            if any(int(receipt["compute"][field]) != 0 for field in forbidden):
                raise CounterFactMetricBoundary("backfill forbidden compute/mutation count differs")
            if receipt["before_evaluation_weight_identity"] != receipt["after_evaluation_weight_identity"]:
                raise CounterFactMetricBoundary("backfill before/after weight identity differs")
            if (
                sha256_file(records_path) != receipt["records"]["sha256"]
                or receipt["records"]["rows"] != count
            ):
                raise CounterFactMetricBoundary("backfill records binding differs")
            original_path = (
                FINALW_RESULT_ROOT
                / f"cell-{cell_index}"
                / f"checkpoint-{count:05d}"
                / "request-metrics.jsonl.gz"
            )
            original_receipt = _object(original_path.with_name("evaluation-receipt.json"))
            _identity(original_receipt, str(original_path.with_name("evaluation-receipt.json")))
            if (
                original_receipt.get("model") != model
                or original_receipt.get("method") != method
                or original_receipt.get("accepted_edit_count") != count
                or original_receipt.get("evaluation_type") != "CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS"
                or original_receipt.get("weight_pointer_version_bytes_exact") is not True
                or original_receipt.get("before_evaluation_weight_identity")
                != original_receipt.get("after_evaluation_weight_identity")
                or original_receipt.get("nonfinite_count") != 0
                or original_receipt.get("records", {}).get("rows") != count
            ):
                raise CounterFactMetricBoundary("existing final-W checkpoint invariant differs")
            if any(int(original_receipt["compute"][field]) != 0 for field in forbidden):
                raise CounterFactMetricBoundary("existing final-W forbidden compute/mutation count differs")
            if sha256_file(original_path) != original_receipt["records"]["sha256"]:
                raise CounterFactMetricBoundary("existing final-W records binding differs")
            indexed[(arm, count)] = (original_receipt, receipt)
            backfill_members.extend(
                [
                    {
                        "kind": "CHECKPOINT_RECEIPT",
                        "arm": arm,
                        "accepted_edit_count": count,
                        "path": str(receipt_path),
                        "bytes": receipt_path.stat().st_size,
                        "sha256": spec["sha256"],
                    },
                    {
                        "kind": "LOCALITY_TARGET_NEW_RECORDS",
                        "arm": arm,
                        "accepted_edit_count": count,
                        "path": str(records_path),
                        "bytes": records_path.stat().st_size,
                        "sha256": receipt["records"]["sha256"],
                        "rows": count,
                    },
                ]
            )
            compute.append(
                {
                    "arm": arm,
                    "accepted_edit_count": count,
                    "wall_seconds": receipt["compute"]["wall_seconds"],
                    "model_forward_batch_count": receipt["compute"]["model_forward_batch_count"],
                    "parity_forward_batch_count": receipt["compute"]["parity_forward_batch_count"],
                    "request_count": count,
                    "locality_prompt_count": 10 * count,
                    "peak_gpu_memory_bytes": terminal["peak_gpu_memory_bytes"],
                    "full_fp32": True,
                    "weight_pointer_version_bytes_exact": True,
                    "forbidden_mutation_count": 0,
                    "nonfinite_count": 0,
                }
            )
    if len(indexed) != 28 or len(backfill_members) != 60:
        raise CounterFactMetricBoundary("backfill terminal/member denominator differs")
    return indexed, backfill_members, compute


def _metric_store() -> dict[str, dict[str, list[Any]]]:
    return {
        metric: {"bits": [], "new": [], "true": [], "advantage": [], "order": [], "pairs": []}
        for metric in PRIMARY
    }


def _append_pair(
    store: dict[str, dict[str, list[Any]]],
    *,
    metric: str,
    arm: str,
    count: int,
    ordinal: int,
    request_sha: str,
    prompt_index: int,
    new: float,
    true: float,
    writer: Any,
) -> None:
    locality = metric == "ns"
    bit = strict_nll_pair_success(new, true, locality=locality)
    advantage = (new - true) if locality else (true - new)
    value = store[metric]
    value["bits"].append(bit)
    value["new"].append(float(new))
    value["true"].append(float(true))
    value["advantage"].append(float(advantage))
    value["order"].append([request_sha, prompt_index])
    value["pairs"].append([float(new), float(true)])
    local = {
        "arm": arm,
        "accepted_edit_count": count,
        "ordinal": ordinal,
        "request_sha256": request_sha,
        "metric": metric.upper(),
        "prompt_index": prompt_index,
        "target_new_nll": float(new),
        "target_true_nll": float(true),
        "desired_nll_advantage": float(advantage),
        "success": bit,
    }
    local["identity_sha256"] = canonical_hash(local)
    writer.write(json.dumps(local, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
    writer.write("\n")


def _store_summary(store: Mapping[str, Mapping[str, Sequence[Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    core: dict[str, Any] = {}
    distributions: list[dict[str, Any]] = []
    for metric in PRIMARY:
        value = store[metric]
        bits = [int(bit) for bit in value["bits"]]
        denominator = len(bits)
        if denominator == 0:
            raise CounterFactMetricBoundary("empty primary metric")
        core[f"{metric}_numerator"] = sum(bits)
        core[f"{metric}_denominator"] = denominator
        core[metric] = sum(bits) / denominator
        row: dict[str, Any] = {
            "metric": metric.upper(),
            "numerator": sum(bits),
            "denominator": denominator,
            "rate": sum(bits) / denominator,
            "tie_count": sum(int(float(left) == float(right)) for left, right in value["pairs"]),
            "bit_vector_sha256": canonical_hash(bits),
            "prompt_order_sha256": canonical_hash(value["order"]),
            "ordered_nll_pair_sha256": canonical_hash(value["pairs"]),
        }
        for prefix, values in (
            ("target_new_nll", value["new"]),
            ("target_true_nll", value["true"]),
            ("nll_advantage", value["advantage"]),
        ):
            for statistic, result in distribution(values).items():
                row[f"{prefix}_{statistic}"] = result
        distributions.append(row)
    return core, distributions


def _aggregate(
    indexed: Mapping[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]],
    local_pairs_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    core_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    age_rows: list[dict[str, Any]] = []
    age_distribution_rows: list[dict[str, Any]] = []
    raw, compressed, writer = _open_pair_writer(local_pairs_path)
    pair_rows = 0
    try:
        for arm in ARM_ORDER:
            for count in AMENDED_CHECKPOINTS:
                original_receipt, backfill_receipt = indexed[(arm, count)]
                original_path = Path(original_receipt["records"]["path"])
                backfill_path = Path(backfill_receipt["records"]["path"])
                original_iter = _record_iter(
                    original_path, original_receipt["records"]["sha256"], count, derived_age=True
                )
                backfill_iter = _record_iter(
                    backfill_path, backfill_receipt["records"]["sha256"], count, derived_age=False
                )
                store = _metric_store()
                age_store = {age: _metric_store() for age in AGE_ORDER}
                rewrite_acc = rephrase_acc = rephrase_all = locality_true_acc = 0
                age_acc = {
                    age: {"requests": 0, "rewrite": 0, "rephrase": 0, "rephrase_all": 0, "locality_true": 0}
                    for age in AGE_ORDER
                }
                request_hashes: list[str] = []
                rows_seen = 0
                for original, backfill in zip(original_iter, backfill_iter, strict=True):
                    if (
                        original["request_sha256"] != backfill["request_sha256"]
                        or original["ordinal"] != backfill["ordinal"]
                        or original["batch_index"] != backfill["batch_index"]
                        or original["case_identity_sha256"] != backfill["case_identity_sha256"]
                        or original["age_stratum"] != backfill["age_stratum"]
                    ):
                        raise CounterFactMetricBoundary("old/new backfill row pairing differs")
                    ordinal = int(original["ordinal"])
                    request_sha = str(original["request_sha256"])
                    age = str(original["age_stratum"])
                    request_hashes.append(request_sha)
                    old = original["metrics"]
                    new_locality = backfill["metrics"]["locality_target_new"]
                    rewrite_new = old["rewrite_target_new"]["prompts"]
                    rewrite_true = old["rewrite_target_true"]["prompts"]
                    rephrase_new = old["rephrase_target_new"]["prompts"]
                    rephrase_true = old["rephrase_target_true"]["prompts"]
                    locality_new = new_locality["prompts"]
                    locality_true = old["locality_target_true"]["prompts"]
                    if not (
                        len(rewrite_new) == len(rewrite_true) == 1
                        and len(rephrase_new) == len(rephrase_true) == 2
                        and len(locality_new) == len(locality_true) == 10
                    ):
                        raise CounterFactMetricBoundary("paired prompt cardinality differs")
                    pairs = (
                        ("rs", rewrite_new, rewrite_true),
                        ("ps", rephrase_new, rephrase_true),
                        ("ns", locality_new, locality_true),
                    )
                    for metric, new_prompts, true_prompts in pairs:
                        for prompt_index, (new_prompt, true_prompt) in enumerate(
                            zip(new_prompts, true_prompts, strict=True)
                        ):
                            for target_store in (store, age_store[age]):
                                _append_pair(
                                    target_store,
                                    metric=metric,
                                    arm=arm,
                                    count=count,
                                    ordinal=ordinal,
                                    request_sha=request_sha,
                                    prompt_index=prompt_index,
                                    new=float(new_prompt["nll"]),
                                    true=float(true_prompt["nll"]),
                                    writer=writer if target_store is store else _NullWriter(),
                                )
                            pair_rows += 1
                    rewrite_acc += int(rewrite_new[0]["strict"])
                    rephrase_acc += sum(int(value["strict"]) for value in rephrase_new)
                    rephrase_all += int(all(bool(value["strict"]) for value in rephrase_new))
                    locality_true_acc += sum(int(value["strict"]) for value in locality_true)
                    age_acc[age]["requests"] += 1
                    age_acc[age]["rewrite"] += int(rewrite_new[0]["strict"])
                    age_acc[age]["rephrase"] += sum(int(value["strict"]) for value in rephrase_new)
                    age_acc[age]["rephrase_all"] += int(all(bool(value["strict"]) for value in rephrase_new))
                    age_acc[age]["locality_true"] += sum(int(value["strict"]) for value in locality_true)
                    rows_seen += 1
                if rows_seen != count or len(set(request_hashes)) != count:
                    raise CounterFactMetricBoundary("paired checkpoint request denominator differs")
                core, distributions = _store_summary(store)
                model, method = next(key for key, value in ARM_KEY.items() if value == arm)
                core_row = {
                    "arm": arm,
                    "model": model,
                    "method": method,
                    "accepted_edit_count": count,
                    "evaluation_type": "CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS_COUNTERFACT_NLL_PAIR",
                    "request_denominator": count,
                    **core,
                    "rewrite_acc_numerator": rewrite_acc,
                    "rewrite_acc_denominator": count,
                    "rewrite_acc": rewrite_acc / count,
                    "rephrase_acc_numerator": rephrase_acc,
                    "rephrase_acc_denominator": 2 * count,
                    "rephrase_acc": rephrase_acc / (2 * count),
                    "rephrase_acc_strict_all_2_numerator": rephrase_all,
                    "rephrase_acc_strict_all_2_denominator": count,
                    "rephrase_acc_strict_all_2": rephrase_all / count,
                    "neighborhood_target_true_teacher_forced_acc_numerator": locality_true_acc,
                    "neighborhood_target_true_teacher_forced_acc_denominator": 10 * count,
                    "neighborhood_target_true_teacher_forced_acc": locality_true_acc / (10 * count),
                    "request_order_sha256": canonical_hash(request_hashes),
                }
                core_rows.append(core_row)
                for row in distributions:
                    distribution_rows.append({"arm": arm, "model": model, "method": method, "accepted_edit_count": count, **row})
                for age in AGE_ORDER:
                    age_core, age_distributions = _store_summary(age_store[age])
                    request_count = age_acc[age]["requests"]
                    age_rows.append(
                        {
                            "arm": arm,
                            "model": model,
                            "method": method,
                            "accepted_edit_count": count,
                            "age_stratum": age,
                            "request_denominator": request_count,
                            **age_core,
                            "rewrite_acc_numerator": age_acc[age]["rewrite"],
                            "rewrite_acc_denominator": request_count,
                            "rewrite_acc": age_acc[age]["rewrite"] / request_count,
                            "rephrase_acc_numerator": age_acc[age]["rephrase"],
                            "rephrase_acc_denominator": 2 * request_count,
                            "rephrase_acc": age_acc[age]["rephrase"] / (2 * request_count),
                            "rephrase_acc_strict_all_2_numerator": age_acc[age]["rephrase_all"],
                            "rephrase_acc_strict_all_2_denominator": request_count,
                            "rephrase_acc_strict_all_2": age_acc[age]["rephrase_all"] / request_count,
                            "neighborhood_target_true_teacher_forced_acc_numerator": age_acc[age]["locality_true"],
                            "neighborhood_target_true_teacher_forced_acc_denominator": 10 * request_count,
                            "neighborhood_target_true_teacher_forced_acc": age_acc[age]["locality_true"] / (10 * request_count),
                        }
                    )
                    for row in age_distributions:
                        age_distribution_rows.append(
                            {
                                "arm": arm,
                                "model": model,
                                "method": method,
                                "accepted_edit_count": count,
                                "age_stratum": age,
                                "request_denominator": request_count,
                                **row,
                            }
                        )
    finally:
        writer.flush()
        writer.close()
        compressed.close()
        raw.close()
    if pair_rows != 1_560_000:
        raise CounterFactMetricBoundary("per-prompt primary pair denominator differs")
    return core_rows, distribution_rows, age_rows, age_distribution_rows, {
        "path": str(local_pairs_path),
        "bytes": local_pairs_path.stat().st_size,
        "sha256": sha256_file(local_pairs_path),
        "rows": pair_rows,
    }


class _NullWriter:
    def write(self, _: str) -> None:
        return None


def _spans(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any], str]]:
    ordered = sorted(rows, key=lambda row: int(row["accepted_edit_count"]))
    values = [(ordered[index - 1], ordered[index], "ADJACENT") for index in range(1, len(ordered))]
    values.append((ordered[0], ordered[-1], "TOTAL_1K_TO_10K"))
    return values


def _transition_rows(core: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    answer: list[dict[str, Any]] = []
    fields = (*PRIMARY, "rewrite_acc", "rephrase_acc", "rephrase_acc_strict_all_2", "neighborhood_target_true_teacher_forced_acc")
    for arm in ARM_ORDER:
        for before, after, span in _spans([row for row in core if row["arm"] == arm]):
            for field in fields:
                answer.append(
                    {
                        "arm": arm,
                        "from_checkpoint": before["accepted_edit_count"],
                        "to_checkpoint": after["accepted_edit_count"],
                        "span": span,
                        "metric": field,
                        "from_value": before[field],
                        "to_value": after[field],
                        "delta": float(after[field]) - float(before[field]),
                    }
                )
    return answer


def _paired_rows(core: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    comparisons = (
        ("LA", "LM", "ALPHAEDIT_MINUS_MEMIT_LLAMA"),
        ("QA", "QM", "ALPHAEDIT_MINUS_MEMIT_QWEN"),
        ("LM", "QM", "LLAMA_MINUS_QWEN_MEMIT"),
        ("LA", "QA", "LLAMA_MINUS_QWEN_ALPHAEDIT"),
    )
    indexed = {(row["arm"], int(row["accepted_edit_count"])): row for row in core}
    answer: list[dict[str, Any]] = []
    for left_arm, right_arm, label in comparisons:
        for count in AMENDED_CHECKPOINTS:
            left, right = indexed[(left_arm, count)], indexed[(right_arm, count)]
            for metric in (*PRIMARY, "rewrite_acc", "rephrase_acc", "neighborhood_target_true_teacher_forced_acc"):
                answer.append(
                    {
                        "comparison": label,
                        "left_arm": left_arm,
                        "right_arm": right_arm,
                        "accepted_edit_count": count,
                        "metric": metric,
                        "left_value": left[metric],
                        "right_value": right[metric],
                        "delta_left_minus_right": float(left[metric]) - float(right[metric]),
                    }
                )
    return answer


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda value: value[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in range(start, end):
            result[indexed[index][0]] = rank
        start = end
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right))
    return 0.0 if denominator == 0.0 else numerator / denominator


def _mechanism_associations(source_root: Path, core: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    mechanism = _rows(source_root / V5_RELATIVE / "mechanism-cumulative-checkpoints.csv")
    indexed = {(row["arm"], int(row["accepted_edit_count"])): row for row in core}
    excluded = {"arm", "accepted_edit_count", "eff", "gen_prompt", "gen_strict", "loc", "rewrite_new_nll_median", "rephrase_new_nll_median"}
    fields = [field for field in mechanism[0] if field not in excluded]
    answer: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        selected = sorted(
            (row for row in mechanism if row["arm"] == arm),
            key=lambda row: int(row["accepted_edit_count"]),
        )
        if len(selected) != 7:
            raise CounterFactMetricBoundary("mechanism checkpoint denominator differs")
        for mechanism_field in fields:
            x = [float(row[mechanism_field]) for row in selected]
            for outcome in PRIMARY:
                y = [float(indexed[(arm, int(row["accepted_edit_count"]))][outcome]) for row in selected]
                answer.append(
                    {
                        "arm": arm,
                        "mechanism_metric": mechanism_field,
                        "outcome_metric": outcome,
                        "checkpoint_n": 7,
                        "spearman": _pearson(_average_ranks(x), _average_ranks(y)),
                        "pearson": _pearson(x, y),
                        "causal_claim": 0,
                    }
                )
    return answer


def _relabel_v5_body(source_root: Path) -> str:
    path = source_root / V5_RELATIVE / "official-layer-realization-debt-lifelong-v5-exhaustive-cumulative-factual-ko.md"
    text = path.read_text(encoding="utf-8")
    start = text.find("## 3. Provenance")
    if start < 0:
        raise CounterFactMetricBoundary("v5 detailed-body anchor missing")
    text = text[start:]
    # Replace complete legacy metric labels only. Substring replacement would
    # corrupt ordinary words such as Locality or Effective.
    replacements = (
        (r"(?<![A-Za-z0-9_])Gen prompt(?![A-Za-z0-9_])", "rephrase_acc (prompt)"),
        (r"(?<![A-Za-z0-9_])Gen strict(?![A-Za-z0-9_])", "rephrase_acc strict-all-2"),
        (r"(?<![A-Za-z0-9_])Eff(?![A-Za-z0-9_])", "rewrite_acc"),
        (r"(?<![A-Za-z0-9_])Loc(?![A-Za-z0-9_])", "neighborhood_target_true_teacher_forced_acc"),
        (r"`eff`", "`rewrite_acc`"),
        (r"`gen_prompt`", "`rephrase_acc`"),
        (r"`gen_strict`", "`rephrase_acc_strict_all_2`"),
        (r"`loc`", "`neighborhood_target_true_teacher_forced_acc`"),
        (r"(?<![A-Za-z0-9_])rephrase pref(?![A-Za-z0-9_])", "rephrase request-cluster mean preference (not PS)"),
    )
    for pattern, new in replacements:
        text = re.sub(pattern, new, text)
    text = re.sub(
        r"\]\((?!\.\./)([^/)]+\.png)\)",
        rf"](../{V5_RELATIVE.name}/\1)",
        text,
    )
    text = re.sub(
        r"^(##+ )([0-9]+)(\.\d+)?(\.)",
        lambda match: f"{match.group(1)}{int(match.group(2)) + 6}{match.group(3) or ''}{match.group(4)}",
        text,
        flags=re.MULTILINE,
    )
    return text


def _report(
    source_root: Path,
    head: str,
    tree: str,
    core: Sequence[Mapping[str, Any]],
    distributions: Sequence[Mapping[str, Any]],
    age: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    compute: Sequence[Mapping[str, Any]],
    prompt_pairs: Mapping[str, Any],
    backfill_members: Sequence[Mapping[str, Any]],
) -> str:
    final = {row["arm"]: row for row in core if int(row["accepted_edit_count"]) == 10_000}
    first = {row["arm"]: row for row in core if int(row["accepted_edit_count"]) == 1_000}
    headline = [
        (
            ARM_LABEL[arm],
            f"{row['rs_numerator']}/{row['rs_denominator']} ({_fmt(row['rs'])})",
            f"{row['ps_numerator']}/{row['ps_denominator']} ({_fmt(row['ps'])})",
            f"{row['ns_numerator']}/{row['ns_denominator']} ({_fmt(row['ns'])})",
            f"{row['rewrite_acc_numerator']}/{row['rewrite_acc_denominator']} ({_fmt(row['rewrite_acc'])})",
            f"{row['rephrase_acc_numerator']}/{row['rephrase_acc_denominator']} ({_fmt(row['rephrase_acc'])})",
        )
        for arm, row in ((arm, final[arm]) for arm in ARM_ORDER)
    ]
    core_table = [
        (
            row["arm"], row["accepted_edit_count"], row["request_denominator"],
            f"{row['rs_numerator']}/{row['rs_denominator']} ({_fmt(row['rs'], 4)})",
            f"{row['ps_numerator']}/{row['ps_denominator']} ({_fmt(row['ps'], 4)})",
            f"{row['ns_numerator']}/{row['ns_denominator']} ({_fmt(row['ns'], 4)})",
            _fmt(row["rewrite_acc"], 4), _fmt(row["rephrase_acc"], 4),
            _fmt(row["neighborhood_target_true_teacher_forced_acc"], 4),
        )
        for row in core
    ]
    dist_table = [
        (
            row["arm"], row["accepted_edit_count"], row["metric"],
            f"{row['numerator']}/{row['denominator']} ({_fmt(row['rate'], 4)})",
            "/".join(_fmt(row[f"target_new_nll_{stat}"], 3) for stat in STATS),
            "/".join(_fmt(row[f"target_true_nll_{stat}"], 3) for stat in STATS),
            "/".join(_fmt(row[f"nll_advantage_{stat}"], 3) for stat in STATS),
            row["tie_count"],
        )
        for row in distributions
    ]
    age_table = [
        (
            row["arm"], row["accepted_edit_count"], row["age_stratum"], row["request_denominator"],
            f"{row['rs_numerator']}/{row['rs_denominator']} ({_fmt(row['rs'], 4)})",
            f"{row['ps_numerator']}/{row['ps_denominator']} ({_fmt(row['ps'], 4)})",
            f"{row['ns_numerator']}/{row['ns_denominator']} ({_fmt(row['ns'], 4)})",
            _fmt(row["rewrite_acc"], 4), _fmt(row["rephrase_acc"], 4),
        )
        for row in age
    ]
    transition_table = [
        (
            row["arm"], f"{row['from_checkpoint']}→{row['to_checkpoint']}", row["metric"],
            _fmt(row["from_value"], 4), _fmt(row["to_value"], 4), _fmt(row["delta"], 4),
        )
        for row in transitions
        if row["metric"] in PRIMARY
    ]
    paired_table = [
        (
            row["comparison"], row["accepted_edit_count"], row["metric"],
            _fmt(row["left_value"], 4), _fmt(row["right_value"], 4), _fmt(row["delta_left_minus_right"], 4),
        )
        for row in paired
        if row["metric"] in PRIMARY
    ]
    compute_table = [
        (
            row["arm"], row["accepted_edit_count"], row["request_count"],
            row["locality_prompt_count"], _fmt(row["wall_seconds"], 1),
            row["model_forward_batch_count"], row["parity_forward_batch_count"],
            _fmt(float(row["peak_gpu_memory_bytes"]) / (1024 ** 3), 2),
        )
        for row in compute
    ]
    sections = [
        "# Official layer-write realization debt — CounterFact primary metrics v6 사실 보고서",
        "",
        "이 문서는 v3/v4-r2/v5를 수정하지 않고, canonical CounterFact NLL-pair metric을 추가한 successor다. `scientific_promotion=false`.",
        "",
        "## 1. Final W₁₀₀₀₀ — canonical CounterFact headline",
        "",
        "아래 표는 frozen final W₁₀₀₀₀에서 sealed 10,000 requests 전체를 평가한 결과다. RS/PS/NS가 primary이며 두 accuracy 열은 secondary teacher-forced diagnostic이다.",
        "",
        _markdown_table(("model/method", "RS", "PS", "NS", "rewrite_acc", "rephrase_acc"), headline),
        "",
        "### 1.1 핵심 metric 교정 결과",
        "",
        "- 1k의 낮은 종전 `Loc`는 실험의 locality 실패율이 아니라 target-true teacher-forced strict accuracy였다. 동일 frozen W₁₀₀₀에서 canonical NS는 "
        + ", ".join(
            f"{arm} {row['ns_numerator']}/{row['ns_denominator']} ({_fmt(row['ns'], 4)})"
            for arm, row in ((arm, first[arm]) for arm in ARM_ORDER)
        )
        + "이다.",
        "- NS 보충 평가의 기존 locality-target-true 3-request parity는 모든 checkpoint/model/method에서 오차 0이며, 평가 전후 edited-weight pointer/version/bytes가 동일하다. 따라서 현재 증거는 edit trajectory 오류보다 metric-label/schema 불일치를 지지한다.",
        "- RS/PS와 rewrite_acc/rephrase_acc는 성공 조건이 다르므로 값의 크기를 서로 치환하거나 같은 efficacy 정의로 pooling하지 않는다.",
        "",
        "![Final CounterFact primary metrics](counterfact-finalw-full10k-primary.png)",
        "",
        "## 2. Metric glossary / 읽는 법",
        "",
        "- **RS (Rewrite Success, higher):** request당 rewrite prompt 1개에서 length-normalized `target_new NLL < target_true NLL`. tie는 실패. 분모=t prompts.",
        "- **PS (Paraphrase Success, higher):** request당 rephrase prompt 2개 각각에서 `target_new NLL < target_true NLL`. prompt 단위이며 request-cluster 평균 비교가 아니다. 분모=2t prompts.",
        "- **NS (Neighborhood Success, higher):** request당 neighborhood prompt 10개 각각에서 `target_true NLL < target_new NLL`. 분모=10t prompts. 이것이 canonical locality primary metric이다.",
        "- **rewrite_acc / rephrase_acc (secondary, higher):** target-new teacher-forced continuation의 모든 target token이 top-1인 prompt 비율. generic token-mean accuracy와 다르며 RS/PS가 아니다.",
        "- **rephrase_acc strict-all-2 (secondary):** 한 request의 rephrase prompts 2개가 모두 teacher-forced strict일 때 성공 1건.",
        "- **neighborhood_target_true_teacher_forced_acc (secondary):** target-true continuation이 top-1인 prompt 비율. v5의 `Loc` 숫자는 이 값이었으며 NS로 재명명하지 않았다.",
        "- **NLL (lower):** target continuation token들의 length-normalized mean negative log likelihood. new/true는 같은 request·category·prompt index에서 pair한다. prompt-level과 request 내부 prompts를 먼저 묶은 request-cluster 통계를 분리한다. **desired NLL advantage (higher):** RS/PS는 `true-new`, NS는 `new-true`; 양수일 때만 success다.",
        "- **margin (higher):** 각 target token logit에서 최대 비-target logit을 뺀 뒤 continuation 최소값. 평균 margin과 all-token strict는 같은 통계가 아니다.",
        "- **mean/median/IQR/p90/max:** 산술평균/중앙값/(p75−p25)/90백분위/최댓값. 본 v6 primary 표는 prompt-level NLL pair 분포이며 prompt와 request denominator를 혼합하지 않는다.",
        "- **R=z*−Φ(W):** layer write 직전 activation residual. **A=R/n_remaining:** Official layer allocation. **Y=R_pre−R_post:** 실제 write가 줄인 residual. **E=A−Y:** allocation-realization gap. activation-space vector이며 NLL 단위가 아니다.",
        "- **q=||R||/||R_entry|| (lower):** entry-normalized remaining residual. **rho=<Y,A>/||A||²:** target-aligned realization ratio(1 exact, 0–1 under, >1 overshoot, <0 opposite). **tau=||Y−rho A||/||A||:** orthogonal distortion.",
        "- **d_parallel/d_perp:** L8 entry residual과 ideal `(1/5)R_entry` 차이의 entry-target 평행/직교 정규화 성분. **recurrence closure (lower):** exact residual recurrence의 FP64 relative error.",
        "- **D_TV (lower for profile agreement):** 다섯 layer update-share와 positive target-progress profile의 total variation distance(0 identical, 1 maximally separated). negative progress는 normalization에서 숨기지 않고 별도 센다.",
        "- **Layer-wise Update Magnitude:** `||ΔW_l||_F`; **share:** 다섯 layer magnitude 합 중 해당 layer 비중. activation progress와 별개 축이며 squared Frobenius telemetry와 혼용하지 않는다. 모든 mechanism metric은 관찰 telemetry이고 RS/PS/NS controller input이 아니다.",
        "",
        "## 3. Metric/source audit와 실행 provenance",
        "",
        f"- v6 source HEAD/tree: `{head}` / `{tree}`.",
        f"- Evaluation-only backfill Slurm job: `{BACKFILL_JOB_ID}` array `0-3%2`; LM/LA/QM/QA one GPU each, explicit `--mem=60416M`.",
        f"- Common stream/order: `{STREAM_ROOT}` / `{ORDER_ROOT}`.",
        f"- Canonical CounterFact context/pair construction: `{CANONICAL_COUNTERFACT_EVALUATOR}` lines 43–76, SHA `{CANONICAL_COUNTERFACT_EVALUATOR_SHA256}`; strict success directions: `{CANONICAL_COUNTERFACT_SUMMARIZER}` lines 63–108, SHA `{CANONICAL_COUNTERFACT_SUMMARIZER_SHA256}`.",
        "- Secondary accuracy source: `project/run_scripts/fixed_z_nonuniqueness/evaluation.py` lines 140–170. `nll`은 target-token mean이고 `strict`는 해당 prompt의 모든 target token top-1 exact다.",
        "- Existing final-W raw: 4 terminals, 28 receipts, 120,000 request-state rows full rehash PASS. Rewrite NLL pairs=120,000 prompts; rephrase NLL pairs=240,000 prompts.",
        "- Missing field audit: locality target-new=0/1,200,000 prompts. Evaluation-only backfill produced exactly 1,200,000 prompts; edit replay/compute_z/writer/key/solve/cache-history mutation/backward/gradient/model update/imputation=0.",
        f"- Backfill sealed members={len(backfill_members)}; local prompt-pair table=`{prompt_pairs['path']}` ({prompt_pairs['rows']} rows, `{prompt_pairs['sha256']}`). Raw prompt/logit/generation publication=0.",
        "- 기존 v5 `Loc` 저값은 `neighborhood_target_true_teacher_forced_acc`였고 NS가 아니었다. 따라서 그 수치를 canonical locality failure rate로 사용하지 않는다.",
        "",
        "## 4. Seven-checkpoint cumulative RS/PS/NS",
        "",
        "각 W_t는 first t seen requests 전체를 평가한다. current B100나 online-at-write가 아니다.",
        "",
        _markdown_table(("arm", "t", "request n", "RS", "PS", "NS", "rewrite_acc", "rephrase_acc", "neighborhood true TF acc"), core_table),
        "",
        "### 4.1 1k→10k primary 변화",
        "",
        *(
            f"- **{arm}:** RS {_fmt(first[arm]['rs'], 4)}→{_fmt(final[arm]['rs'], 4)} "
            f"(Δ{_fmt(final[arm]['rs'] - first[arm]['rs'], 4)}), PS "
            f"{_fmt(first[arm]['ps'], 4)}→{_fmt(final[arm]['ps'], 4)} "
            f"(Δ{_fmt(final[arm]['ps'] - first[arm]['ps'], 4)}), NS "
            f"{_fmt(first[arm]['ns'], 4)}→{_fmt(final[arm]['ns'], 4)} "
            f"(Δ{_fmt(final[arm]['ns'] - first[arm]['ns'], 4)})."
            for arm in ARM_ORDER
        ),
        "",
        "모든 변화는 frozen checkpoint W_t에서 seen-prefix 전체를 다시 평가한 descriptive 값이다. method 효과나 인과로 해석하지 않는다.",
        "",
        "![Cumulative CounterFact primary metrics](counterfact-cumulative-primary.png)",
        "",
        "![Secondary teacher-forced accuracy](counterfact-secondary-accuracy.png)",
        "",
        "## 5. Prompt-level NLL-pair distributions",
        "",
        "각 행은 prompt-level pair다. 열 순서는 mean/median/p90/max다.",
        "",
        _markdown_table(("arm", "t", "metric", "success n/d", "new NLL", "true NLL", "desired advantage", "ties"), dist_table),
        "",
        "![Cumulative NLL advantage](counterfact-cumulative-nll-advantage.png)",
        "",
        "## 6. Edit-age strata",
        "",
        "각 checkpoint seen-prefix 안에서 early first20% / middle60% / recent last20%로 고정했다.",
        "",
        _markdown_table(("arm", "t", "age", "request n", "RS", "PS", "NS", "rewrite_acc", "rephrase_acc"), age_table),
        "",
        "![CounterFact primary metrics by edit age](counterfact-age-strata-primary.png)",
        "",
        "## 7. Checkpoint 변화와 paired arithmetic",
        "",
        _markdown_table(("arm", "span", "metric", "from", "to", "delta"), transition_table),
        "",
        _markdown_table(("comparison", "t", "metric", "left", "right", "left-right"), paired_table),
        "",
        "## 8. Mechanism ↔ canonical cumulative metric descriptive association",
        "",
        "동일 7 checkpoints/arm에서 계산한 기술 통계다. n=7이고 causal claim=0이다.",
        "",
        "![Mechanism association](counterfact-mechanism-associations.png)",
        "",
        "### 8.1 Evaluation-only backfill compute",
        "",
        _markdown_table(
            ("arm", "t", "requests", "locality prompts", "wall s", "forward batches", "parity batches", "peak GiB"),
            compute_table,
        ),
        "",
        "각 checkpoint는 frozen W_t를 한 번 로드했다. 표의 wall time은 checkpoint 평가이며 edit replay와 model update는 0이다.",
        "",
        "아래는 v5의 provenance, residual trajectory, A/Y/E/rho/tau, inherited debt, recurrence, layer update, online/current-B100 diagnostics, cache fork, comparisons, compute, exclusions와 artifact inventory를 하나의 본문으로 계승한 것이다. 그 안의 과거 headline 이름은 secondary accuracy 의미로 재표기했다.",
        "",
        _relabel_v5_body(source_root),
        "",
        "## 32. FACT / boundary",
        "",
        "- FACT: RS/PS는 기존 sealed NLL pairs에서 재계산했고, NS에 필요한 locality target-new만 frozen checkpoint evaluation으로 보충했다.",
        "- FACT: 기존 v3/v4-r2/v5 bytes와 edit trajectory는 변경하지 않았다.",
        "- FACT: primary success bit는 strict inequality이며 tie는 failure다. 누락/중복/imputation은 0이다.",
        "- BOUNDARY: accuracy, current B100, online-at-write, cumulative final-W, mechanism telemetry는 서로 다른 지표/분모다.",
        "- DECISION: `scientific_promotion=false`; 자동 후속 GPU experiment=0.",
        "",
    ]
    return "\n".join(sections)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink() or args.local_output.exists() or args.local_output.is_symlink():
        raise CounterFactMetricBoundary("refusing existing v6 output root")
    head = _git(args.source_root, "rev-parse", "HEAD")
    tree = _git(args.source_root, "rev-parse", "HEAD^{tree}")
    if head != args.expected_head or _git(args.source_root, "status", "--porcelain", "--untracked-files=no"):
        raise CounterFactMetricBoundary("v6 source identity differs")
    v4_manifest, _, v4_members = _verify_package(
        args.source_root / V4_RELATIVE,
        report_name="official-layer-debt-lifelong-finalw-full10k-factual-ko.md",
        report_sha=V4_REPORT_SHA256,
        manifest_sha=V4_MANIFEST_SHA256,
        receipt_sha=V4_RECEIPT_SHA256,
    )
    v5_manifest, _, v5_members = _verify_package(
        args.source_root / V5_RELATIVE,
        report_name="official-layer-realization-debt-lifelong-v5-exhaustive-cumulative-factual-ko.md",
        report_sha=V5_REPORT_SHA256,
        manifest_sha=V5_MANIFEST_SHA256,
        receipt_sha=V5_RECEIPT_SHA256,
    )
    if sha256_file(AVAILABILITY_AUDIT) != AVAILABILITY_AUDIT_SHA256:
        raise CounterFactMetricBoundary("v6 availability audit SHA differs")
    for path, expected, label in (
        (SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256, "v6 source manifest"),
        (PREGPU_RECEIPT, PREGPU_RECEIPT_SHA256, "v6 pre-GPU receipt"),
    ):
        if sha256_file(path) != expected:
            raise CounterFactMetricBoundary(f"{label} SHA differs")
        _identity(_object(path), label)
    availability = _object(AVAILABILITY_AUDIT)
    _identity(availability, str(AVAILABILITY_AUDIT))
    if availability.get("status") != "MISSING_REQUIRES_EVALUATION_ONLY_BACKFILL":
        raise CounterFactMetricBoundary("v6 availability classification differs")
    indexed, backfill_members, compute = _terminal_and_receipts(args.backfill_root)
    args.output.mkdir(parents=True, mode=0o755)
    args.local_output.mkdir(parents=True, mode=0o700)
    prompt_path = args.local_output / "counterfact-primary-prompt-pairs.jsonl.gz"
    core, distributions, age, age_distributions, prompt_pairs = _aggregate(indexed, prompt_path)
    transitions = _transition_rows(core)
    paired = _paired_rows(core)
    associations = _mechanism_associations(args.source_root, core)
    if not (
        len(core) == 28
        and len(distributions) == 84
        and len(age) == 84
        and len(age_distributions) == 252
        and len(transitions) == 196
        and len(paired) == 168
        and len(associations) > 0
    ):
        raise CounterFactMetricBoundary("v6 derived table denominator differs")
    tables: dict[str, dict[str, Any]] = {}
    table_rows = (
        ("counterfact-primary-checkpoints.csv", core),
        ("counterfact-primary-distributions.csv", distributions),
        ("counterfact-primary-age-strata.csv", age),
        ("counterfact-primary-age-distributions.csv", age_distributions),
        ("counterfact-primary-transitions.csv", transitions),
        ("counterfact-primary-paired-deltas.csv", paired),
        ("counterfact-mechanism-associations.csv", associations),
        ("counterfact-backfill-compute.csv", compute),
        ("counterfact-backfill-member-inventory.csv", backfill_members),
        ("counterfact-existing-raw-member-inventory.csv", availability["raw_members"]),
        ("counterfact-source-audit.csv", availability["source_audit"]["members"]),
    )
    for name, values in table_rows:
        tables[name] = _write_csv(args.output / name, values)
    final = [row for row in core if int(row["accepted_edit_count"]) == 10_000]
    tables["counterfact-finalw-full10k.csv"] = _write_csv(args.output / "counterfact-finalw-full10k.csv", final)
    bit_roots = [
        {
            "arm": row["arm"],
            "accepted_edit_count": row["accepted_edit_count"],
            "metric": row["metric"],
            "numerator": row["numerator"],
            "denominator": row["denominator"],
            "bit_vector_sha256": row["bit_vector_sha256"],
            "prompt_order_sha256": row["prompt_order_sha256"],
            "ordered_nll_pair_sha256": row["ordered_nll_pair_sha256"],
        }
        for row in distributions
    ]
    tables["counterfact-primary-bit-roots.csv"] = _write_csv(args.output / "counterfact-primary-bit-roots.csv", bit_roots)
    figure_paths = generate_figures(args.output, args.output)
    plot_source = Path(__file__).with_name("lifelong_counterfact_figures.py")
    plot_inputs = {
        "counterfact-finalw-full10k-primary.png": "counterfact-finalw-full10k.csv",
        "counterfact-cumulative-primary.png": "counterfact-primary-checkpoints.csv",
        "counterfact-secondary-accuracy.png": "counterfact-primary-checkpoints.csv",
        "counterfact-age-strata-primary.png": "counterfact-primary-age-strata.csv",
        "counterfact-cumulative-nll-advantage.png": "counterfact-primary-distributions.csv",
        "counterfact-mechanism-associations.png": "counterfact-mechanism-associations.csv",
    }
    plot_rows = [
        {
            "figure": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "input_table": plot_inputs[path.name],
            "input_table_sha256": sha256_file(args.output / plot_inputs[path.name]),
            "plot_source": str(plot_source.relative_to(args.source_root)),
            "plot_source_sha256": sha256_file(plot_source),
            "command": (
                "MPLCONFIGDIR=/tmp/odeedit-counterfact-v6-mpl "
                "/data/janghj/EasyEdit/.venv/bin/python -m "
                "project.run_scripts.official_layer_realization_debt.lifelong_counterfact_figures "
                "--tables V6_PACKAGE --output CREATE_ONCE_OUTPUT"
            ),
            "backend": "Agg",
            "dpi": 180,
            "missing_policy": "NO_INTERPOLATION_NO_IMPUTATION",
        }
        for path in figure_paths
    ]
    tables["plot-reproduction.csv"] = _write_csv(args.output / "plot-reproduction.csv", plot_rows)
    report_path = args.output / REPORT_NAME
    report_path.write_text(
        _report(
            args.source_root,
            head,
            tree,
            core,
            distributions,
            age,
            transitions,
            paired,
            compute,
            prompt_pairs,
            backfill_members,
        ),
        encoding="utf-8",
    )
    members = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(args.output.iterdir(), key=lambda value: value.name)
        if path.name not in {"analysis-manifest.json", "rooted-analysis-receipt.json"}
    ]
    external = [
        {
            "kind": "IMMUTABLE_V4_R2_FINALW_EVALUATION",
            "root": str(V4_RELATIVE),
            "report_sha256": V4_REPORT_SHA256,
            "manifest_sha256": V4_MANIFEST_SHA256,
            "receipt_sha256": V4_RECEIPT_SHA256,
            "member_root": v4_manifest["member_root"],
            "member_count": len(v4_members),
        },
        {
            "kind": "IMMUTABLE_V5_EXHAUSTIVE_CUMULATIVE",
            "root": str(V5_RELATIVE),
            "report_sha256": V5_REPORT_SHA256,
            "manifest_sha256": V5_MANIFEST_SHA256,
            "receipt_sha256": V5_RECEIPT_SHA256,
            "member_root": v5_manifest["member_root"],
            "member_count": len(v5_members),
        },
        {
            "kind": "SEALED_V6_LOCALITY_TARGET_NEW_BACKFILL",
            "root": str(args.backfill_root),
            "slurm_job_id": BACKFILL_JOB_ID,
            "array_mapping": {"0": "LM", "1": "LA", "2": "QM", "3": "QA"},
            "array_throttle": 2,
            "memory_mib_per_cell": 60416,
            "member_root": canonical_hash(backfill_members),
            "member_count": len(backfill_members),
        },
        {
            "kind": "V6_AVAILABILITY_AUDIT",
            "path": str(AVAILABILITY_AUDIT),
            "bytes": AVAILABILITY_AUDIT.stat().st_size,
            "sha256": AVAILABILITY_AUDIT_SHA256,
            "identity_sha256": availability["identity_sha256"],
            "raw_member_root": availability["raw_member_root"],
        },
        {
            "kind": "V6_SOURCE_MANIFEST",
            "path": str(SOURCE_MANIFEST),
            "bytes": SOURCE_MANIFEST.stat().st_size,
            "sha256": SOURCE_MANIFEST_SHA256,
            "identity_sha256": _object(SOURCE_MANIFEST)["identity_sha256"],
        },
        {
            "kind": "V6_PREGPU_RECEIPT",
            "path": str(PREGPU_RECEIPT),
            "bytes": PREGPU_RECEIPT.stat().st_size,
            "sha256": PREGPU_RECEIPT_SHA256,
            "identity_sha256": _object(PREGPU_RECEIPT)["identity_sha256"],
        },
        {
            "kind": "LOCAL_PER_PROMPT_PRIMARY_TABLE",
            **prompt_pairs,
            "git_included": False,
        },
    ]
    manifest: dict[str, Any] = {
        "schema": f"{SCHEMA}.analysis-manifest",
        "instruction_id": INSTRUCTION_ID,
        "status": "TASK_COMPLETE_STOP",
        "source": {"head": head, "tree": tree, "tracked_clean_before_output": True},
        "denominators": {
            "arms": 4,
            "checkpoints": 28,
            "request_states": 120_000,
            "rs_prompts": 120_000,
            "ps_prompts": 240_000,
            "ns_prompts": 1_200_000,
            "primary_prompt_pair_rows": 1_560_000,
            "age_rows": 84,
        },
        "execution": {
            "slurm_job_id": BACKFILL_JOB_ID,
            "array": "0-3%2",
            "terminal_cells": 4,
            "technical_exclusion_jobs": [],
            "memory_mib_per_cell": 60416,
            "gpu_per_cell": 1,
        },
        "metric_semantics": {
            "counterfact_context_and_pair_source": {
                "path": str(CANONICAL_COUNTERFACT_EVALUATOR),
                "sha256": CANONICAL_COUNTERFACT_EVALUATOR_SHA256,
                "line_span": "43-76,124-180",
            },
            "counterfact_aggregation_source": {
                "path": str(CANONICAL_COUNTERFACT_SUMMARIZER),
                "sha256": CANONICAL_COUNTERFACT_SUMMARIZER_SHA256,
                "line_span": "63-108",
            },
            "rs_ps_rule": "target_new_mean_nll < target_true_mean_nll; tie fails",
            "ns_rule": "target_true_mean_nll < target_new_mean_nll; tie fails",
            "secondary_accuracy_rule": "all target tokens top-1 exact per prompt",
        },
        "verification": {
            "focused_unittest": {
                "status": "PASS",
                "test_count": 9,
                "modules": [
                    "test_lifelong_counterfact_metrics",
                    "test_lifelong_counterfact_analysis",
                ],
            },
            "synthetic_plot_byte_stability": "PASS_TWO_CREATE_ONCE_OUTPUTS_EXACT_SHA",
            "package_member_rehash": "PASS_FULL_MEMBER_SHA_AND_CANONICAL_ROOT",
        },
        "external_inputs": external,
        "external_input_root": canonical_hash(external),
        "table_row_counts": {name: value["rows"] for name, value in tables.items()},
        "plot_reproduction": plot_rows,
        "members": members,
        "member_root": canonical_hash(members),
        "edit_replay_count": 0,
        "raw_model_checkpoint_cache_log_dataset_credential_git_count": 0,
        "raw_prompt_logit_generation_publish_count": 0,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_path = args.output / "analysis-manifest.json"
    _write_json(manifest_path, manifest)
    receipt: dict[str, Any] = {
        "schema": f"{SCHEMA}.rooted-receipt",
        "instruction_id": INSTRUCTION_ID,
        "status": manifest["status"],
        "report": {
            "path": REPORT_NAME,
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        },
        "manifest": {
            "path": "analysis-manifest.json",
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "identity_sha256": manifest["identity_sha256"],
        },
        "member_root": manifest["member_root"],
        "external_input_root": manifest["external_input_root"],
        "scientific_promotion": False,
        "next": "STOP",
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    receipt_path = args.output / "rooted-analysis-receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "status": manifest["status"],
        "report": {"path": str(report_path), "sha256": receipt["report"]["sha256"]},
        "manifest": {"path": str(manifest_path), "sha256": receipt["manifest"]["sha256"]},
        "receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
        "member_root": manifest["member_root"],
        "tables": len(tables),
        "figures": len(figure_paths),
        "final": final,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--backfill-root", type=Path, required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
