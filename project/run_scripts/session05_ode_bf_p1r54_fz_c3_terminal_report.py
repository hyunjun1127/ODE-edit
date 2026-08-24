#!/usr/bin/env python3
"""Build canonical Korean P1R54 FZ independent/sequential reports."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.contracts import ODEBFStateError, canonical_hash
from project.run_scripts.session05_ode_bf_p1r54_fz_c3_independent_analyze import (
    build_analysis as build_independent_analysis,
)
from project.run_scripts.session05_ode_bf_p1r54_fz_c3_sequential_analyze import (
    _target_telemetry,
    build_analysis as build_sequential_analysis,
)


STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
ORDERED_RECORD_ROOT = "af215235177d1ac07e82fadc62de7235244a82342a146c5867f673625486d0e1"
INDEPENDENT_BASELINE_SHA = {
    "report": "3ea9739fa9200fd2e9fbf36a7a6b9e86c0212279fc52f9c6b2c0aa6f5f42e217",
    "manifest": "b38d8b23d7d2876e72dcab35c8a64083ba6baa5951769998cf3c1726d9a77d09",
    "receipt": "e82fb1dd0f8c1596823979b63f66168682a15bf231621771dae95340e68c8356",
    "identity": "ea5261b1f4b08732fa7ae08895dbe655d53a630d3c4f5b7bb86a5a74b644ff60",
}
SEQUENTIAL_BASELINE_SHA = {
    "report": "b93391e5c344bd5635fdd1c2f6532d41e683a88d85dabb3be21e949f05f3891b",
    "manifest": "58ee504c70e2a3a5708a937a44f1d6cdb736240e38745e021d4eb1244ba97a70",
    "receipt": "6d269759f9cc8b458e9589c1ca64fcd5362681ec5c6cf455fe6376cf4e0a1f66",
    "identity": "fb9b1e9709ec8c86319e8f88a36a9d99726fd4a22177f5cf8c2cb2607347812e",
}


def _json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ODEBFStateError(f"regular JSON required: {path}")
    return json.loads(path.read_bytes())


def _require_sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ODEBFStateError(f"external baseline SHA differs: {path}")


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0 or not 0 <= numerator <= denominator:
        raise ODEBFStateError("performance denominator differs")
    return {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator}


def _dist_from_row(row: Mapping[str, Any], target: str) -> dict[str, Any]:
    return {
        "count": int(float(row.get(f"{target}_count") or 0)) or None,
        "mean": float(row[f"{target}_mean"]),
        "median": float(row[f"{target}_median"]),
        "p90_nearest_rank": float(row[f"{target}_p90"]),
        "max": float(row[f"{target}_max"]),
    }


def _analysis_endpoint(surface: Mapping[str, Any]) -> dict[str, Any]:
    performance = surface["performance"]
    return {
        "performance": performance,
        "rewrite": {
            "target_new": surface["rewrite_target_new_nll"],
            "target_true": surface["rewrite_target_true_nll"],
            "success": performance["Eff"],
            "accuracy": performance["Rewrite_accuracy"],
            "strict_success": performance["Eff"],
            "strict_accuracy": performance["Rewrite_accuracy"],
        },
        "rephrase": {
            "target_new": surface["rephrase_target_new_nll"],
            "target_true": surface["rephrase_target_true_nll"],
            "success": performance["Gen"],
            "accuracy": performance["Rephrase_accuracy"],
            "strict_success": performance["Gen_strict"],
            "strict_accuracy": performance["Rephrase_strict_accuracy"],
        },
    }


def _independent_baselines(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    names = {
        "report": root / "p1r52-joint-pc-full-fp32-independent-b100x10-factual-ko.md",
        "manifest": root / "analysis-manifest.json",
        "receipt": root / "rooted-analysis-receipt.json",
    }
    for name, path in names.items():
        _require_sha(path, INDEPENDENT_BASELINE_SHA[name])
    receipt = _json(names["receipt"])
    summary = _json(root / "analysis-summary.json")
    if (
        receipt.get("identity_sha256") != INDEPENDENT_BASELINE_SHA["identity"]
        or summary.get("source_head") != "27a5ea828af26e30887b9546b31f7f9584ef1e7c"
        or summary.get("stream_root") != STREAM_ROOT
        or summary.get("stream_order") != STREAM_ORDER
        or summary.get("valid_endpoint_count") != 60
        or summary.get("valid_request_count") != 6000
        or summary.get("dtype_status") != "FULL_FP32_PASS_ALL_METHODS"
    ):
        raise ODEBFStateError("independent native baseline common seal differs")
    aggregates = _json(root / "method-aggregates.json")
    distributions = _json(root / "nll-prompt-distribution-aggregates.json")
    per_case = _json(root / "per-case.json")
    methods = {}
    for method, label in (("OFFICIAL-ALPHAEDIT", "Official AlphaEdit"), ("OFFICIAL-MEMIT", "Native MEMIT")):
        aggregate = aggregates[method]
        endpoints = {}
        for endpoint, prefix in (("Z", "z"), ("W", "W")):
            perf = {
                "Eff": _rate(aggregate[f"{prefix}_rewrite_success_numerator"], aggregate[f"{prefix}_rewrite_success_denominator"]),
                "Rewrite_accuracy": _rate(aggregate[f"{prefix}_rewrite_accuracy_numerator"], aggregate[f"{prefix}_rewrite_accuracy_denominator"]),
                "Gen": _rate(aggregate[f"{prefix}_rephrase_success_numerator"], aggregate[f"{prefix}_rephrase_success_denominator"]),
                "Gen_strict": _rate(aggregate[f"{prefix}_rephrase_strict_success_numerator"], aggregate[f"{prefix}_rephrase_strict_success_denominator"]),
                "Rephrase_accuracy": _rate(aggregate[f"{prefix}_rephrase_accuracy_numerator"], aggregate[f"{prefix}_rephrase_accuracy_denominator"]),
                "Rephrase_strict_accuracy": _rate(aggregate[f"{prefix}_rephrase_strict_accuracy_numerator"], aggregate[f"{prefix}_rephrase_strict_accuracy_denominator"]),
                "Loc": _rate(aggregate[f"{prefix}_locality_numerator"], aggregate[f"{prefix}_locality_denominator"]),
            }
            payload = {"performance": perf}
            for prompt in ("rewrite", "rephrase"):
                prompt_payload = {
                    "success": perf["Eff" if prompt == "rewrite" else "Gen"],
                    "accuracy": perf["Rewrite_accuracy" if prompt == "rewrite" else "Rephrase_accuracy"],
                    "strict_success": perf["Eff" if prompt == "rewrite" else "Gen_strict"],
                    "strict_accuracy": perf["Rewrite_accuracy" if prompt == "rewrite" else "Rephrase_strict_accuracy"],
                }
                for target in ("target-new", "target-true"):
                    rows = [
                        row for row in distributions
                        if row["method"] == method and row["endpoint"] == endpoint
                        and row["family"] == prompt and row["target"] == target
                    ]
                    if len(rows) != 1:
                        raise ODEBFStateError("independent native NLL distribution differs")
                    prompt_payload[target.replace("-", "_")] = {
                        "count": rows[0]["prompt_denominator"],
                        **{
                            ("p90_nearest_rank" if key == "p90" else key): value
                            for key, value in rows[0]["cross_prompt_summary"].items()
                        },
                    }
                payload[prompt] = prompt_payload
            endpoints[endpoint] = payload
        methods[method] = {
            "label": label,
            "source_head": "27a5ea828af26e30887b9546b31f7f9584ef1e7c",
            "z_provenance": "NATIVE_ALPHAEDIT_Z" if method == "OFFICIAL-ALPHAEDIT" else "NATIVE_MEMIT_LATENT_Z",
            "accepted_z": endpoints["Z"],
            "post_W": endpoints["W"],
            "final_W": endpoints["W"],
            "per_case": [row for row in per_case if row["method"] == method],
            "compute": {
                "case_count": aggregate["case_count"],
                "case_total_seconds": aggregate["case_total_seconds_stats"],
                "target_accepted_z_seconds": aggregate["target_accepted_z_generation_seconds_stats"],
                "writer_edit_core_seconds": aggregate["writer_edit_core_seconds_stats"],
                "model_forward_count_per_case_mean": aggregate["model_forward_count"],
                "backward_count_per_case_mean": aggregate["backward_count"],
                "processed_tokens_per_case_mean": aggregate["processed_tokens"],
                "total_actual_fp32_update_energy": aggregate["total_actual_fp32_update_energy"],
            },
        }
    inputs = {
        "mode": "INDEPENDENT_W0_COLD_PER_SLICE",
        "source_head": "27a5ea828af26e30887b9546b31f7f9584ef1e7c",
        "job_group": "job22541",
        "report_path": str(names["report"]),
        "report_sha256": INDEPENDENT_BASELINE_SHA["report"],
        "manifest_path": str(names["manifest"]),
        "manifest_sha256": INDEPENDENT_BASELINE_SHA["manifest"],
        "receipt_path": str(names["receipt"]),
        "receipt_sha256": INDEPENDENT_BASELINE_SHA["receipt"],
        "receipt_identity": INDEPENDENT_BASELINE_SHA["identity"],
        "structured_members": {
            name: {"path": str(root / name), "sha256": sha256_file(root / name)}
            for name in ("analysis-summary.json", "method-aggregates.json", "nll-prompt-distribution-aggregates.json", "per-case.json")
        },
    }
    return methods, inputs


def _csv_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def _sequential_baselines(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    names = {
        "report": root / "p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-exhaustive-factual-ko.md",
        "manifest": root / "analysis-manifest.json",
        "receipt": root / "rooted-analysis-receipt.json",
    }
    for name, path in names.items():
        _require_sha(path, SEQUENTIAL_BASELINE_SHA[name])
    receipt = _json(names["receipt"])
    report_text = names["report"].read_text(encoding="utf-8")
    if (
        receipt.get("identity_sha256") != SEQUENTIAL_BASELINE_SHA["identity"]
        or STREAM_ROOT not in report_text
        or STREAM_ORDER not in report_text
        or "10/10 batches" not in report_text
    ):
        raise ODEBFStateError("sequential native baseline common seal differs")
    rewrite = _csv_rows(root / "tables/rewrite-endpoint-summary.csv")
    rephrase = _csv_rows(root / "tables/rephrase-endpoint-summary.csv")
    final_rows = {row["arm"]: row for row in _json(root / "tables/terminal-w10-performance-comparison.json")["rows"]}
    methods = {}
    for arm, key, label in (("alphaedit", "OFFICIAL-ALPHAEDIT-CACHE", "Official AlphaEdit-cache"), ("memit", "OFFICIAL-MEMIT", "Native MEMIT")):
        endpoints = {}
        for endpoint_name, endpoint_label in (("accepted/native-z", "accepted_z"), ("immediate W", "post_W")):
            selected = {
                "rewrite": next(row for row in rewrite if row["arm"] == arm and row["endpoint"] == endpoint_name),
                "rephrase": next(row for row in rephrase if row["arm"] == arm and row["endpoint"] == endpoint_name),
            }
            performance = {
                "Eff": _rate(int(selected["rewrite"]["success_numerator"]), int(selected["rewrite"]["success_denominator"])),
                "Rewrite_accuracy": _rate(int(selected["rewrite"]["accuracy_numerator"]), int(selected["rewrite"]["accuracy_denominator"])),
                "Gen": _rate(int(selected["rephrase"]["success_numerator"]), int(selected["rephrase"]["success_denominator"])),
                "Gen_strict": _rate(int(selected["rephrase"]["success_strict_numerator"]), int(selected["rephrase"]["success_strict_denominator"])),
                "Rephrase_accuracy": _rate(int(selected["rephrase"]["accuracy_numerator"]), int(selected["rephrase"]["accuracy_denominator"])),
                "Rephrase_strict_accuracy": _rate(int(selected["rephrase"]["accuracy_strict_numerator"]), int(selected["rephrase"]["accuracy_strict_denominator"])),
                "Loc": {"numerator": None, "denominator": None, "rate": None},
            }
            endpoints[endpoint_label] = {
                "performance": performance,
                **{
                    prompt: {
                        "success": performance["Eff" if prompt == "rewrite" else "Gen"],
                        "accuracy": performance["Rewrite_accuracy" if prompt == "rewrite" else "Rephrase_accuracy"],
                        "strict_success": performance["Eff" if prompt == "rewrite" else "Gen_strict"],
                        "strict_accuracy": performance["Rewrite_accuracy" if prompt == "rewrite" else "Rephrase_strict_accuracy"],
                        "target_new": _dist_from_row(selected[prompt], "target_new"),
                        "target_true": _dist_from_row(selected[prompt], "target_true"),
                    }
                    for prompt in ("rewrite", "rephrase")
                },
            }
        final = final_rows[arm]
        final_performance = {
            "Eff": _rate(final["eff_numerator"], final["eff_denominator"]),
            "Rewrite_accuracy": _rate(final["rewrite_accuracy_numerator"], final["rewrite_accuracy_denominator"]),
            "Gen": _rate(final["gen_numerator"], final["gen_denominator"]),
            "Gen_strict": _rate(final["gen_strict_numerator"], final["gen_strict_denominator"]),
            "Rephrase_accuracy": _rate(final["rephrase_accuracy_numerator"], final["rephrase_accuracy_denominator"]),
            "Loc": _rate(final["locality_numerator"], final["locality_denominator"]),
        }
        match = re.search(rf"^\|{arm}\|([0-9.]+)\|", report_text, flags=re.MULTILINE)
        methods[key] = {
            "label": label,
            "source_head": "251e616cf95972e230ce40718e1e390ef7dc9eb4",
            "z_provenance": "NATIVE_ALPHAEDIT_Z" if arm == "alphaedit" else "NATIVE_MEMIT_LATENT_Z",
            **endpoints,
            "final_W": {"performance": final_performance, "NLL": "NOT_RECORDED_IN_PINNED_FINAL_W10_PACKAGE"},
            "compute": {
                "job_total_seconds": None if match is None else float(match.group(1)),
                "field_forward_selector_writer_materialization_counts": "NOT_RECORDED_IN_PINNED_V5_PACKAGE",
            },
            "per_batch": "NOT_RECORDED_IN_PINNED_V5_PACKAGE",
        }
    inputs = {
        "mode": "SEQUENTIAL_B1_TO_B10_W_AND_APPLICABLE_CACHE_CONTINUITY",
        "source_head": "251e616cf95972e230ce40718e1e390ef7dc9eb4",
        "job_group": "job22759_PHASE1_GROUP",
        "report_path": str(names["report"]),
        "report_sha256": SEQUENTIAL_BASELINE_SHA["report"],
        "manifest_path": str(names["manifest"]),
        "manifest_sha256": SEQUENTIAL_BASELINE_SHA["manifest"],
        "receipt_path": str(names["receipt"]),
        "receipt_sha256": SEQUENTIAL_BASELINE_SHA["receipt"],
        "receipt_identity": SEQUENTIAL_BASELINE_SHA["identity"],
        "structured_members": {
            name: {"path": str(root / "tables" / name), "sha256": sha256_file(root / "tables" / name)}
            for name in ("rewrite-endpoint-summary.csv", "rephrase-endpoint-summary.csv", "terminal-w10-performance-comparison.json")
        },
    }
    return methods, inputs


def _fmt_rate(value: Mapping[str, Any] | None) -> str:
    if not value or value.get("rate") is None:
        return "NOT_RECORDED"
    return f"{100.0 * float(value['rate']):.2f}% ({value['numerator']}/{value['denominator']})"


def _fmt_dist(value: Mapping[str, Any] | str | None) -> str:
    if not isinstance(value, Mapping):
        return "NOT_RECORDED"
    return f"{float(value['mean']):.6f} / {float(value['median']):.6f} / {float(value['p90_nearest_rank']):.6f} / {float(value['max']):.6f}"


def _endpoint_rows(methods: Mapping[str, Any], prompt: str) -> list[dict[str, Any]]:
    rows = []
    for method, value in methods.items():
        for endpoint in ("accepted_z", "post_W"):
            item = value[endpoint][prompt]
            rows.append({
                "method": method,
                "label": value["label"],
                "endpoint": endpoint,
                "prompt": prompt,
                "success": item["success"],
                "accuracy": item["accuracy"],
                "strict_success": item["strict_success"],
                "strict_accuracy": item["strict_accuracy"],
                "target_new": item["target_new"],
                "target_true": item["target_true"],
            })
    return rows


def _headline(methods: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "label": value["label"],
            "final_performance": value["final_W"]["performance"],
            "z_rewrite": value["accepted_z"]["rewrite"]["target_new"],
            "z_rephrase": value["accepted_z"]["rephrase"]["target_new"],
            "W_rewrite": value["post_W"]["rewrite"]["target_new"],
            "W_rephrase": value["post_W"]["rephrase"]["target_new"],
        }
        for method, value in methods.items()
    ]


def _independent_native_paired(
    analysis: Mapping[str, Any], methods: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for method in ("OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT"):
        baseline = {int(row["case_index"]): row for row in methods[method]["per_case"]}
        if sorted(baseline) != list(range(1, 11)):
            raise ODEBFStateError("independent native per-case denominator differs")
        for batch_index in range(1, 11):
            native = baseline[batch_index]
            z = analysis["accepted_z"]["by_batch"][batch_index - 1]
            w = analysis["post_W"]["by_batch"][batch_index - 1]
            rows.append({
                "baseline": method,
                "batch_index": batch_index,
                "FZ_minus_native_z_rewrite_target_new_nll_mean": z["rewrite_target_new_nll"]["mean"] - float(native["z_rewrite_target_new_nll_mean"]),
                "FZ_minus_native_z_rephrase_target_new_nll_mean": z["rephrase_target_new_nll"]["mean"] - float(native["z_rephrase_target_new_nll_mean"]),
                "FZ_minus_native_W_rewrite_target_new_nll_mean": w["rewrite_target_new_nll"]["mean"] - float(native["W_rewrite_target_new_nll_mean"]),
                "FZ_minus_native_W_rephrase_target_new_nll_mean": w["rephrase_target_new_nll"]["mean"] - float(native["W_rephrase_target_new_nll_mean"]),
            })
    return rows


def _cache_layer_load(writer: Mapping[str, Any]) -> list[dict[str, Any]]:
    loads = [[] for _ in range(5)]
    shares = [[] for _ in range(5)]
    for batch in writer["K1_to_K8_target_writer_telemetry"]:
        for step in batch["K1_to_K8"]:
            for index, value in enumerate(step["cache_layer_load"]):
                loads[index].append(float(value))
                shares[index].append(float(step["cache_layer_load_share"][index]))
    if any(len(values) != 80 for values in loads):
        raise ODEBFStateError("writer cache layer load denominator differs")
    return [
        {
            "layer": 4 + index,
            "cache_load_mean": sum(loads[index]) / len(loads[index]),
            "cache_load_max": max(loads[index]),
            "cache_load_share_mean": sum(shares[index]) / len(shares[index]),
            "actual_update_energy": "NOT_RECORDED_NATIVE_OFFICIAL_APPLY",
        }
        for index in range(5)
    ]


def _compute_comparison(
    mode: str, analysis: Mapping[str, Any], methods: Mapping[str, Any]
) -> list[dict[str, Any]]:
    writer = analysis["writer_cache_target_telemetry"] if mode == "independent" else analysis["writer_cache_telemetry"]
    fz_total = analysis["cell_wall_seconds"]["sum"] if mode == "independent" else analysis["job_runtime_after_model_preflight_seconds"]
    fz_unit_mean = analysis["cell_wall_seconds"]["mean"] if mode == "independent" else analysis["batch_wall_seconds"]["mean"]
    rows = [{
        "method": "P1R54-FZ",
        "scope": "SUM_OF_10_INDEPENDENT_CELLS_POST_MODEL_PREFLIGHT" if mode == "independent" else "ONE_SEQUENTIAL_JOB_POST_MODEL_PREFLIGHT",
        "total_or_sum_seconds": fz_total,
        "case_or_batch_seconds_mean": fz_unit_mean,
        "target_seconds_mean": "NOT_SEPARATELY_RECORDED",
        "writer_seconds_mean": writer["writer_core_wall_seconds"]["mean"],
        "field_evaluation_count": 80,
        "writer_call_count": 80,
        "layer_apply_count": 400,
        "model_forward_count": "RECORDED_IN_JOB_LEDGER_SEPARATE",
        "ratio_to_FZ_matching_scope": 1.0,
    }]
    for method in ("OFFICIAL-ALPHAEDIT-CACHE", "OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT"):
        if method not in methods:
            continue
        compute = methods[method]["compute"]
        if mode == "independent":
            baseline_total = float(compute["case_total_seconds"]["mean"]) * 10
            rows.append({
                "method": method,
                "scope": "SUM_OF_10_INDEPENDENT_CASES_POST_MODEL_PREFLIGHT",
                "total_or_sum_seconds": baseline_total,
                "case_or_batch_seconds_mean": compute["case_total_seconds"]["mean"],
                "target_seconds_mean": compute["target_accepted_z_seconds"]["mean"],
                "writer_seconds_mean": compute["writer_edit_core_seconds"]["mean"],
                "field_evaluation_count": "NATIVE_COMPUTE_Z_NOT_FZ_FIELD",
                "writer_call_count": 10,
                "layer_apply_count": 50,
                "model_forward_count": compute["model_forward_count_per_case_mean"],
                "ratio_to_FZ_matching_scope": fz_total / baseline_total,
            })
        else:
            baseline_total = compute["job_total_seconds"]
            rows.append({
                "method": method,
                "scope": "JOB_TOTAL_INCLUDES_SETUP_MODEL_LOAD_EVALUATOR",
                "total_or_sum_seconds": baseline_total,
                "case_or_batch_seconds_mean": "NOT_RECORDED",
                "target_seconds_mean": "NOT_RECORDED",
                "writer_seconds_mean": "NOT_RECORDED",
                "field_evaluation_count": "NOT_RECORDED_IN_PINNED_V5_PACKAGE",
                "writer_call_count": "NOT_RECORDED_IN_PINNED_V5_PACKAGE",
                "layer_apply_count": "NOT_RECORDED_IN_PINNED_V5_PACKAGE",
                "model_forward_count": "NOT_RECORDED_IN_PINNED_V5_PACKAGE",
                "ratio_to_FZ_matching_scope": "NOT_COMPARABLE_TIMER_SCOPE",
            })
    return rows


def _sequential_independent_comparison(
    sequential: Mapping[str, Any], independent_path: Path
) -> dict[str, Any]:
    independent = _json(independent_path)
    if (
        independent.get("status") != "TERMINAL_ANALYSIS_VALID"
        or independent.get("completeness") != "10_OF_10_ENDPOINTS_1000_OF_1000_REQUESTS"
        or independent.get("stream_root") != STREAM_ROOT
        or independent.get("stream_order") != STREAM_ORDER
        or independent.get("cache_W0_integrity", {}).get("W0_restored_count") != 10
    ):
        raise ODEBFStateError("independent comparison analysis differs")
    rows = []
    for batch_index in range(1, 11):
        row: dict[str, Any] = {"batch_index": batch_index}
        for surface, seq_key, ind_key in (
            ("z", "accepted_z_immediate", "accepted_z"),
            ("W", "post_W_immediate", "post_W"),
        ):
            seq = sequential[seq_key]["by_batch"][batch_index - 1]
            ind = independent[ind_key]["by_batch"][batch_index - 1]
            for prompt in ("rewrite", "rephrase"):
                row[f"sequential_minus_independent_{surface}_{prompt}_target_new_nll_mean"] = (
                    float(seq[f"{prompt}_target_new_nll"]["mean"])
                    - float(ind[f"{prompt}_target_new_nll"]["mean"])
                )
        rows.append(row)
    aggregate = {}
    for surface, seq_key, ind_key in (
        ("z", "accepted_z_immediate", "accepted_z"),
        ("W", "post_W_immediate", "post_W"),
    ):
        for prompt in ("rewrite", "rephrase"):
            aggregate[f"sequential_minus_independent_{surface}_{prompt}_target_new_nll_mean"] = (
                float(sequential[seq_key][f"{prompt}_target_new_nll"]["mean"])
                - float(independent[ind_key][f"{prompt}_target_new_nll"]["mean"])
            )
    rate_delta = {
        label: float(sequential["final_W10"]["performance"][label]["rate"])
        - float(independent["post_W"]["performance"][label]["rate"])
        for label in ("Eff", "Gen", "Gen_strict", "Loc")
    }
    receipt_path = independent_path.parent / "rooted-receipt.json"
    manifest_path = independent_path.parent / "analysis-manifest.json"
    return {
        "status": "EXACT_STREAM_DESCRIPTIVE_CROSS_EXECUTION_COMPARISON",
        "boundary": "cross-batch W/cache continuity differs; causal claim0",
        "independent_analysis_path": str(independent_path),
        "independent_analysis_sha256": sha256_file(independent_path),
        "independent_analysis_identity": independent["identity_sha256"],
        "independent_manifest_sha256": sha256_file(manifest_path),
        "independent_receipt_sha256": sha256_file(receipt_path),
        "aggregate": aggregate,
        "sequential_final_W10_minus_independent_endpoint_rate": rate_delta,
        "by_batch": rows,
    }


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("|" + "|".join(str(value) for value in row) + "|" for row in rows)
    return lines


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _report_text(mode: str, methods: Mapping[str, Any], analysis: Mapping[str, Any], baseline_inputs: Mapping[str, Any]) -> str:
    headline = _headline(methods)
    title = "P1R54 FZ-C3 Independent 10×B100" if mode == "independent" else "P1R54 FZ-C3 Sequential 10×B100"
    lines = [f"# {title} — Llama FULL-FP32 native baseline 포함 사실 보고서", "", "> **해석 경계:** 동일 sealed B1–B10×B100, 1,000 requests를 사용한다. Independent는 각 slice가 W0/cold cache에서 독립 시작하고, Sequential은 B1→B10 W 및 Alpha-cache 연속성을 유지한다. 두 실행 의미를 섞은 직접 paired 인과 비교는 하지 않는다. `scientific_promotion=false`.", "", "## 1. 최종 한눈표", ""]
    lines += _markdown_table(
        ["method", "final W Eff", "final W Gen", "strict Gen", "final W Loc", "z rewrite NLL mean/median/p90/max", "z rephrase NLL mean/median/p90/max", "post-W rewrite NLL mean/median/p90/max", "post-W rephrase NLL mean/median/p90/max"],
        [
            [row["label"], _fmt_rate(row["final_performance"].get("Eff")), _fmt_rate(row["final_performance"].get("Gen")), _fmt_rate(row["final_performance"].get("Gen_strict")), _fmt_rate(row["final_performance"].get("Loc")), _fmt_dist(row["z_rewrite"]), _fmt_dist(row["z_rephrase"]), _fmt_dist(row["W_rewrite"]), _fmt_dist(row["W_rephrase"])]
            for row in headline
        ],
    )
    if mode == "sequential":
        lines += ["", "Sequential 표의 Eff/Gen/Loc는 **최종 W10 전체 1,000-request 재평가**다. z와 post-W NLL은 각 batch의 K8 accepted-z 및 immediate post-W pooled 분포다. Pinned native package에는 final-W10 NLL 분포가 없어 이를 immediate post-W 값으로 대체하지 않고 분리 표기한다."]
    for prompt, heading in (("rewrite", "Rewrite"), ("rephrase", "Rephrase")):
        lines += ["", f"## {2 if prompt == 'rewrite' else 3}. {heading} 세부", ""]
        rows = _endpoint_rows(methods, prompt)
        lines += _markdown_table(
            ["method", "endpoint", "success", "accuracy", "strict success", "strict accuracy", "target-new mean/median/p90/max", "target-true mean/median/p90/max"],
            [[row["label"], row["endpoint"], _fmt_rate(row["success"]), _fmt_rate(row["accuracy"]), _fmt_rate(row["strict_success"]), _fmt_rate(row["strict_accuracy"]), _fmt_dist(row["target_new"]), _fmt_dist(row["target_true"])] for row in rows],
        )
    lines += ["", "## 4. FZ 고유 실행 진단", ""]
    telemetry = analysis["target_telemetry"]
    lines += [f"- target field: {analysis.get('target_field_evaluation_count', 80)}회; writer: {analysis.get('writer_call_count', 80)}회; layer apply: {analysis.get('writer_layer_apply_count', 400)}회.", f"- clamp: {telemetry['clamp_hit_numerator']}/{telemetry['clamp_hit_denominator']} ({telemetry['clamp_hit_numerator']/telemetry['clamp_hit_denominator']:.6f}).", f"- forbidden decision access={telemetry['forbidden_decision_access_count']}, additional F/B={telemetry['additional_forward_backward_count']}.", "- writer actual per-layer update energy/share는 Official apply receipt에 저장되지 않아 `NOT_RECORDED`; cache layer Frobenius load만 별도 JSON/CSV에 유지한다."]
    lines += ["", "|K|clamp|top1 energy share mean|top10 mean|bottom90 mean|allocation energy mean|", "|---:|---:|---:|---:|---:|---:|"]
    for row in telemetry["by_K"]:
        lines.append(f"|{row['K']}|{row['clamp_hit_numerator']}/{row['clamp_hit_denominator']}|{row['top1_energy_share']['mean']:.6f}|{row['top10_energy_share']['mean']:.6f}|{row['bottom90_energy_share']['mean']:.6f}|{row['allocation_energy']['mean']:.6f}|")
    if mode == "independent":
        lines += ["", "## 5. Independent 고유 표", "", "각 B1–B10은 동일 W0/cold Alpha-cache width0에서 시작했고, cell-local append는 다른 cell로 전달되지 않았다. W0 restore=10/10, cross-batch state consumption=0."]
        per_batch = analysis["post_W"]["by_batch"]
        lines += ["", "|B|W rewrite NLL mean|W rephrase NLL mean|Eff|Gen|Loc|", "|---:|---:|---:|---:|---:|---:|"]
        for row in per_batch:
            lines.append(f"|{row['batch_index']}|{row['rewrite_target_new_nll']['mean']:.6f}|{row['rephrase_target_new_nll']['mean']:.6f}|{_fmt_rate(row['performance']['Eff'])}|{_fmt_rate(row['performance']['Gen'])}|{_fmt_rate(row['performance']['Loc'])}|")
        lines += ["", "### Exact native baseline 대비 B별 paired NLL delta (FZ−native; 음수일수록 FZ NLL이 낮음)", "", "|baseline|B|z rewrite Δ|z rephrase Δ|W rewrite Δ|W rephrase Δ|", "|---|---:|---:|---:|---:|---:|"]
        for row in analysis["native_paired_by_batch"]:
            lines.append(f"|{row['baseline']}|{row['batch_index']}|{row['FZ_minus_native_z_rewrite_target_new_nll_mean']:+.6f}|{row['FZ_minus_native_z_rephrase_target_new_nll_mean']:+.6f}|{row['FZ_minus_native_W_rewrite_target_new_nll_mean']:+.6f}|{row['FZ_minus_native_W_rephrase_target_new_nll_mean']:+.6f}|")
    else:
        lines += ["", "## 5. Sequential 고유 표 — immediate→final W10 forgetting", ""]
        lines += ["|B|final−immediate rewrite target-new NLL mean|final−immediate rephrase target-new NLL mean|rewrite success→failure|rephrase success→failure|", "|---:|---:|---:|---:|---:|"]
        for row in analysis["forgetting_immediate_to_final_W10"]["rows"]:
            lines.append(f"|{row['batch_index']}|{row['rewrite_final_W10_minus_immediate_W_target_new_nll']['mean']:.6f}|{row['rephrase_final_W10_minus_immediate_W_target_new_nll']['mean']:.6f}|{row['rewrite_immediate_success_to_final_failure']}|{row['rephrase_immediate_success_to_final_failure']}|")
        lines += ["", "Cache entry widths는 0→900, exit widths는 100→1000이며 성공 batch당 append1, 총 append10/consume80이다. B_r commit hash와 B_(r+1) entry hash는 9/9 exact 일치했고 종료 후 W0 bytes를 복원했다."]
        lines += ["", "### 최종 W10 NLL 분포", "", "|method|prompt|target-new mean/median/p90/max|target-true mean/median/p90/max|", "|---|---|---:|---:|", "|Official AlphaEdit-cache|rewrite/rephrase|NOT_RECORDED_IN_PINNED_FINAL_W10_PACKAGE|NOT_RECORDED_IN_PINNED_FINAL_W10_PACKAGE|", "|Native MEMIT|rewrite/rephrase|NOT_RECORDED_IN_PINNED_FINAL_W10_PACKAGE|NOT_RECORDED_IN_PINNED_FINAL_W10_PACKAGE|"]
        for prompt in ("rewrite", "rephrase"):
            lines.append(f"|P1R54 FZ-SEQUENTIAL|{prompt}|{_fmt_dist(analysis['final_W10'][f'{prompt}_target_new_nll'])}|{_fmt_dist(analysis['final_W10'][f'{prompt}_target_true_nll'])}|")
        comparison = analysis["independent_comparison"]
        lines += ["", "### 동일 FZ Independent 대비 descriptive delta (Sequential−Independent)", "", "|B|z rewrite Δ|z rephrase Δ|immediate W rewrite Δ|immediate W rephrase Δ|", "|---:|---:|---:|---:|---:|"]
        for row in comparison["by_batch"]:
            lines.append(f"|{row['batch_index']}|{row['sequential_minus_independent_z_rewrite_target_new_nll_mean']:+.6f}|{row['sequential_minus_independent_z_rephrase_target_new_nll_mean']:+.6f}|{row['sequential_minus_independent_W_rewrite_target_new_nll_mean']:+.6f}|{row['sequential_minus_independent_W_rephrase_target_new_nll_mean']:+.6f}|")
        lines += ["", "|final endpoint rate|Sequential final W10 − Independent endpoint|", "|---|---:|"]
        for label, delta in comparison["sequential_final_W10_minus_independent_endpoint_rate"].items():
            lines.append(f"|{label}|{100.0 * delta:+.3f} percentage points|")
    lines += ["", "## 6. z→W gap 및 writer/cache load", ""]
    gap = analysis["z_to_W_gaps"] if mode == "independent" else analysis["z_to_W_immediate"]
    lines += ["|prompt/target|W−z NLL mean/median/p90/max|", "|---|---:|"]
    for key, value in gap.items():
        if isinstance(value, Mapping) and "mean" in value:
            lines.append(f"|{key}|{_fmt_dist(value)}|")
    lines += ["", "|layer|Alpha-cache load mean|max|mean share|actual writer update energy|", "|---:|---:|---:|---:|---|"]
    for row in analysis["cache_layer_load"]:
        lines.append(f"|L{row['layer']}|{row['cache_load_mean']:.6f}|{row['cache_load_max']:.6f}|{row['cache_load_share_mean']:.6f}|{row['actual_update_energy']}|")
    lines += ["", "## 7. Native baseline 및 compute/overhead", "", f"- baseline source `{baseline_inputs['source_head']}`, group `{baseline_inputs['job_group']}`.", f"- baseline report/manifest/receipt SHA: `{baseline_inputs['report_sha256']}` / `{baseline_inputs['manifest_sha256']}` / `{baseline_inputs['receipt_sha256']}`; receipt identity `{baseline_inputs['receipt_identity']}`.", "- Official AlphaEdit와 MEMIT의 z는 각각 native compute_z/latent-z이며 FZ accepted-z와 같은 알고리즘으로 해석하지 않는다.", "- setup/model-load/evaluator 포함 wall time과 edit-core time을 혼합하지 않는다. 없는 count/timer는 역추정하지 않고 `NOT_RECORDED`로 둔다.", "", "|method|timer scope|total/sum s|case/batch mean s|target mean s|writer mean s|field eval|writer calls|layer apply|FZ/native ratio|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in analysis["compute_comparison"]:
        lines.append("|{method}|{scope}|{total}|{unit}|{target}|{writer}|{field}|{calls}|{layers}|{ratio}|".format(
            method=row["method"], scope=row["scope"],
            total=(f"{row['total_or_sum_seconds']:.3f}" if isinstance(row["total_or_sum_seconds"], (int, float)) else row["total_or_sum_seconds"]),
            unit=(f"{row['case_or_batch_seconds_mean']:.3f}" if isinstance(row["case_or_batch_seconds_mean"], (int, float)) else row["case_or_batch_seconds_mean"]),
            target=(f"{row['target_seconds_mean']:.3f}" if isinstance(row["target_seconds_mean"], (int, float)) else row["target_seconds_mean"]),
            writer=(f"{row['writer_seconds_mean']:.3f}" if isinstance(row["writer_seconds_mean"], (int, float)) else row["writer_seconds_mean"]),
            field=row["field_evaluation_count"], calls=row["writer_call_count"], layers=row["layer_apply_count"],
            ratio=(f"{row['ratio_to_FZ_matching_scope']:.3f}" if isinstance(row["ratio_to_FZ_matching_scope"], (int, float)) else row["ratio_to_FZ_matching_scope"]),
        ))
    lines += ["", "## 8. 판정", "", "- 결과는 Llama 단일 stream의 factual possibility evidence다. 새 hard threshold, imputation, baseline rerun, automatic promotion은 0이다.", "- Independent와 Sequential 차이는 cross-batch W/cache continuity가 함께 달라지는 descriptive comparison이며 causal claim은 하지 않는다.", "- P1R52 Phase2/Phase3 C3는 필요 시 same-sample non-native accepted-z 보조 reference일 뿐 native baseline이 아니다.", "", "## 9. 재현성", "", f"- stream/order/ordered-record root: `{STREAM_ROOT}` / `{STREAM_ORDER}` / `{ORDERED_RECORD_ROOT}`.", f"- FZ source HEAD: `{analysis['source_head']}`; canonical Slurm job: `{analysis['canonical_job']}`.", f"- excluded technical attempts: `{json.dumps(analysis['technical_attempts_excluded'], ensure_ascii=False, sort_keys=True)}`.", f"- FZ analysis identity: `{analysis['identity_sha256']}`.", "- raw results는 immutable이며 report 생성 중 mutation count=0. 모든 small member와 external input SHA는 analysis manifest/rooted receipt에 결속한다."]
    return "\n".join(lines) + "\n"


def build_package(
    mode: str,
    results_root: Path,
    baseline_root: Path,
    output_dir: Path,
    *,
    independent_analysis: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ODEBFStateError("create-once report directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "independent":
        analysis = dict(build_independent_analysis(independent_results_root=results_root))
        terminals = [_json(Path(row["path"])) for row in analysis["terminal_files"]]
        analysis["target_telemetry"] = _target_telemetry(terminals)
        analysis["target_field_evaluation_count"] = 80
        analysis["writer_call_count"] = 80
        analysis["writer_layer_apply_count"] = 400
        analysis["source_head"] = _json(Path(analysis["terminal_files"][0]["path"]))["source_head"]
        analysis["canonical_job"] = "23787_[0-9] TECH-R2"
        analysis["technical_attempts_excluded"] = [
            {"job": "23768", "class": "PURE_TECHNICAL_PRE_MODEL", "reason": "local session config missing", "scientific_denominator_influence_count": 0},
            {"job": "23777", "class": "PURE_TECHNICAL_PRE_MODEL", "reason": "logs parent missing", "scientific_denominator_influence_count": 0},
        ]
        baselines, baseline_inputs = _independent_baselines(baseline_root)
        fz = {
            "label": "P1R54 FZ-INDEPENDENT",
            "source_head": _json(Path(analysis["terminal_files"][0]["path"]))["source_head"],
            "z_provenance": "P1R54_FZ_ACCEPTED_Z",
            "accepted_z": _analysis_endpoint(analysis["accepted_z"]),
            "post_W": _analysis_endpoint(analysis["post_W"]),
            "final_W": _analysis_endpoint(analysis["post_W"]),
        }
        methods = {**baselines, "P1R54-FZ-INDEPENDENT": fz}
        analysis["native_paired_by_batch"] = _independent_native_paired(analysis, methods)
    else:
        analysis = dict(build_sequential_analysis(results_root))
        if independent_analysis is None:
            raise ODEBFStateError("sequential report requires independent analysis")
        analysis["independent_comparison"] = _sequential_independent_comparison(
            analysis, independent_analysis
        )
        analysis["canonical_job"] = "23682"
        analysis["technical_attempts_excluded"] = []
        baselines, baseline_inputs = _sequential_baselines(baseline_root)
        fz = {
            "label": "P1R54 FZ-SEQUENTIAL",
            "source_head": analysis["source_head"],
            "z_provenance": "P1R54_FZ_ACCEPTED_Z",
            "accepted_z": _analysis_endpoint(analysis["accepted_z_immediate"]),
            "post_W": _analysis_endpoint(analysis["post_W_immediate"]),
            "final_W": {"performance": analysis["final_W10"]["performance"], "NLL": {key: analysis["final_W10"][key] for key in analysis["final_W10"] if key.endswith("_nll")}},
        }
        methods = {**baselines, "P1R54-FZ-SEQUENTIAL": fz}
    writer_key = "writer_cache_target_telemetry" if mode == "independent" else "writer_cache_telemetry"
    analysis["cache_layer_load"] = _cache_layer_load(analysis[writer_key])
    analysis["compute_comparison"] = _compute_comparison(mode, analysis, methods)
    analysis["external_native_baseline"] = baseline_inputs
    analysis["common_seal"] = {"stream_root": STREAM_ROOT, "order_root": STREAM_ORDER, "ordered_record_root": ORDERED_RECORD_ROOT, "request_count": 1000, "endpoint_count": 10, "model": "Llama-3-8B-Instruct", "dtype": "FULL_FP32"}
    analysis["identity_sha256"] = canonical_hash({key: value for key, value in analysis.items() if key != "identity_sha256"})
    report = _report_text(mode, methods, analysis, baseline_inputs)

    _write_once(output_dir / "analysis.json", json.dumps(analysis, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    _write_once(output_dir / "native-baselines.json", json.dumps({"methods": methods, "inputs": baseline_inputs}, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    _write_once(output_dir / "report-ko.md", report)
    headline_csv = []
    for row in _headline(methods):
        headline_csv.append({
            "method": row["label"],
            "final_W_Eff": row["final_performance"].get("Eff", {}).get("rate"),
            "final_W_Gen": row["final_performance"].get("Gen", {}).get("rate"),
            "final_W_Gen_strict": row["final_performance"].get("Gen_strict", {}).get("rate"),
            "final_W_Loc": row["final_performance"].get("Loc", {}).get("rate"),
            "z_rewrite_new_mean": row["z_rewrite"]["mean"],
            "z_rewrite_new_median": row["z_rewrite"]["median"],
            "z_rewrite_new_p90": row["z_rewrite"]["p90_nearest_rank"],
            "z_rephrase_new_mean": row["z_rephrase"]["mean"],
            "z_rephrase_new_median": row["z_rephrase"]["median"],
            "z_rephrase_new_p90": row["z_rephrase"]["p90_nearest_rank"],
            "post_W_rewrite_new_mean": row["W_rewrite"]["mean"],
            "post_W_rewrite_new_median": row["W_rewrite"]["median"],
            "post_W_rewrite_new_p90": row["W_rewrite"]["p90_nearest_rank"],
            "post_W_rephrase_new_mean": row["W_rephrase"]["mean"],
            "post_W_rephrase_new_median": row["W_rephrase"]["median"],
            "post_W_rephrase_new_p90": row["W_rephrase"]["p90_nearest_rank"],
        })
    _write_once(output_dir / "headline.csv", _csv_text(headline_csv))
    detail_csv = []
    for prompt in ("rewrite", "rephrase"):
        for row in _endpoint_rows(methods, prompt):
            detail_csv.append({
                "method": row["label"], "endpoint": row["endpoint"], "prompt": prompt,
                "success_numerator": row["success"]["numerator"], "success_denominator": row["success"]["denominator"],
                "accuracy_numerator": row["accuracy"]["numerator"], "accuracy_denominator": row["accuracy"]["denominator"],
                "strict_success_numerator": row["strict_success"]["numerator"], "strict_success_denominator": row["strict_success"]["denominator"],
                "strict_accuracy_numerator": row["strict_accuracy"]["numerator"], "strict_accuracy_denominator": row["strict_accuracy"]["denominator"],
                **{f"target_new_{key}": row["target_new"].get("p90_nearest_rank" if key == "p90" else key) for key in ("mean", "median", "p90", "max")},
                **{f"target_true_{key}": row["target_true"].get("p90_nearest_rank" if key == "p90" else key) for key in ("mean", "median", "p90", "max")},
            })
    _write_once(output_dir / "endpoint-detail.csv", _csv_text(detail_csv))
    _write_once(output_dir / "target-K.json", json.dumps(analysis["target_telemetry"], ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    _write_once(output_dir / "compute.csv", _csv_text(analysis["compute_comparison"]))
    _write_once(output_dir / "cache-layer-load.csv", _csv_text(analysis["cache_layer_load"]))
    if mode == "independent":
        _write_once(
            output_dir / "native-paired-by-batch.csv",
            _csv_text(analysis["native_paired_by_batch"]),
        )
    if mode == "sequential":
        _write_once(output_dir / "forgetting.json", json.dumps(analysis["forgetting_immediate_to_final_W10"], ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        forgetting_rows = []
        for row in analysis["forgetting_immediate_to_final_W10"]["rows"]:
            forgetting_rows.append({
                "batch_index": row["batch_index"],
                "rewrite_final_minus_immediate_target_new_nll_mean": row["rewrite_final_W10_minus_immediate_W_target_new_nll"]["mean"],
                "rephrase_final_minus_immediate_target_new_nll_mean": row["rephrase_final_W10_minus_immediate_W_target_new_nll"]["mean"],
                "rewrite_success_to_failure": row["rewrite_immediate_success_to_final_failure"],
                "rephrase_success_to_failure": row["rephrase_immediate_success_to_final_failure"],
            })
        _write_once(output_dir / "forgetting.csv", _csv_text(forgetting_rows))
        _write_once(
            output_dir / "independent-comparison.csv",
            _csv_text(analysis["independent_comparison"]["by_batch"]),
        )

    members = []
    for path in sorted(output_dir.iterdir()):
        members.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest: dict[str, Any] = {"schema": f"ode-edit-s05-p1r54-fz-c3-{mode}-analysis-manifest/v1", "members": members, "member_root_sha256": canonical_hash([[row["path"], row["bytes"], row["sha256"]] for row in members]), "external_native_baseline": baseline_inputs, "scientific_promotion": False}
    if mode == "sequential":
        comparison = analysis["independent_comparison"]
        manifest["external_independent_analysis"] = {
            key: comparison[key]
            for key in (
                "independent_analysis_path",
                "independent_analysis_sha256",
                "independent_analysis_identity",
                "independent_manifest_sha256",
                "independent_receipt_sha256",
            )
        }
    manifest["identity_sha256"] = canonical_hash(manifest)
    _write_once(output_dir / "analysis-manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    receipt: dict[str, Any] = {"schema": f"ode-edit-s05-p1r54-fz-c3-{mode}-rooted-analysis-receipt/v1", "status": "TERMINAL_NATIVE_BASELINE_REPORT_VALID", "analysis_identity_sha256": analysis["identity_sha256"], "manifest_path": str(output_dir / "analysis-manifest.json"), "manifest_sha256": sha256_file(output_dir / "analysis-manifest.json"), "manifest_identity_sha256": manifest["identity_sha256"], "report_path": str(output_dir / "report-ko.md"), "report_sha256": sha256_file(output_dir / "report-ko.md"), "external_native_baseline": baseline_inputs, "common_seal": analysis["common_seal"], "raw_result_mutation_count": 0, "baseline_rerun_count": 0, "imputation_count": 0, "scientific_promotion": False}
    if mode == "sequential":
        receipt["external_independent_analysis"] = manifest[
            "external_independent_analysis"
        ]
    receipt["identity_sha256"] = canonical_hash(receipt)
    _write_once(output_dir / "rooted-receipt.json", json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return {"status": receipt["status"], "output_dir": str(output_dir), "report_sha256": receipt["report_sha256"], "manifest_sha256": receipt["manifest_sha256"], "receipt_sha256": sha256_file(output_dir / "rooted-receipt.json"), "receipt_identity_sha256": receipt["identity_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", required=True, choices=("independent", "sequential"))
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--independent-analysis", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_package(
        args.mode,
        args.results_root,
        args.baseline_root,
        args.output_dir,
        independent_analysis=args.independent_analysis,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
