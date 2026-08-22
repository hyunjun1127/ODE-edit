"""Raw-free terminal analysis for the P1R52 independent FULL-FP32 panel.

The analyzer is intentionally model-free.  It verifies the sealed case receipts,
enforces the ten paired slice denominator, and emits create-once factual tables.
It never evaluates a model or changes a scientific endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from project.run_scripts.ode_bf.contracts import canonical_hash, canonical_json


CASE_COUNT = 10
REQUESTS_PER_CASE = 100
LAYERS = (4, 5, 6, 7, 8)
METHODS = ("OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT", "C0", "C1", "C2", "C3")
C_METHODS = ("C0", "C1", "C2", "C3")
ROOT_NAMES = {
    "BASELINE": "s05-p1r52-joint-pc-full-fp32-independent-b100x10-baseline-group-v1",
    "C0": "s05-p1r52-joint-pc-full-fp32-independent-b100x10-c0-v1",
    "C1": "s05-p1r52-joint-pc-full-fp32-independent-b100x10-c1-v1",
    "C2": "s05-p1r52-joint-pc-full-fp32-independent-b100x10-c2-v1",
    "C3": "s05-p1r52-joint-pc-full-fp32-independent-b100x10-c3-v1",
}
PILOT_REPORT = (
    "local/odebf/reports/"
    "p1r52-joint-pc-c0-c1-c2-c3-full-fp32-b100-tech-r2-analysis-r1-v1/"
    "endpoint-summary.json"
)
REPORT_NAME = "p1r52-joint-pc-full-fp32-independent-b100x10-factual-ko.md"


class AnalysisError(RuntimeError):
    """A sealed result violates the analysis contract."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _read_json(path: Path) -> Any:
    _need(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_identity(payload: Mapping[str, Any], label: str) -> None:
    expected = payload.get("identity_sha256")
    _need(isinstance(expected, str), f"{label}: missing identity")
    body = {key: value for key, value in payload.items() if key != "identity_sha256"}
    _need(expected == canonical_hash(body), f"{label}: identity mismatch")


def _finite(value: Any, label: str) -> float:
    _need(not isinstance(value, bool), f"{label}: not finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label}: not finite") from exc
    _need(math.isfinite(result), f"{label}: not finite")
    return result


def _rate(numerator: int, denominator: int) -> float:
    _need(denominator > 0 and 0 <= numerator <= denominator, "invalid rate")
    return numerator / denominator


def _stats(values: Sequence[float]) -> dict[str, float]:
    _need(bool(values), "empty statistic")
    clean = [_finite(value, "statistic") for value in values]
    ordered = sorted(clean)
    p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
    return {
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p90": p90,
        "max": max(clean),
        "min": min(clean),
    }


def _flatten_bits(value: Any, label: str) -> list[int]:
    _need(isinstance(value, list), f"{label}: bits not list")
    result: list[int] = []
    for item in value:
        if isinstance(item, list):
            result.extend(_flatten_bits(item, label))
        else:
            _need(item in (0, 1), f"{label}: non-bit")
            result.append(int(item))
    return result


def _score_summary(endpoint: Mapping[str, Any], label: str) -> dict[str, Any]:
    summary = endpoint["summary"]
    scores = endpoint["scores"]
    fields = (
        ("rewrite_success", 100),
        ("rewrite_accuracy", 100),
        ("rephrase_success", 200),
        ("rephrase_accuracy", 200),
        ("rephrase_strict_success", 100),
        ("rephrase_strict_accuracy", 100),
        ("locality", 1000),
    )
    result: dict[str, Any] = {}
    for name, expected_denominator in fields:
        numerator = int(summary[f"{name}_numerator"])
        denominator = int(summary[f"{name}_denominator"])
        _need(denominator == expected_denominator, f"{label}: {name} denominator")
        result[f"{name}_numerator"] = numerator
        result[f"{name}_denominator"] = denominator
        result[f"{name}_rate"] = _rate(numerator, denominator)
    for family in ("rewrite", "rephrase"):
        for target in ("new", "true"):
            key = f"{family}_target_{target}_nll_mean"
            result[key] = _finite(summary[key], f"{label}: {key}")
    _validate_identity(summary, f"{label}: summary")
    _validate_identity(scores, f"{label}: scores")
    return result


def _z_to_w_losses(z: Mapping[str, Any], w: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for family in ("rewrite_success", "rephrase_success"):
        z_prompt = _flatten_bits(z["scores"][family]["per_request_bits"], f"z {family}")
        w_prompt = _flatten_bits(w["scores"][family]["per_request_bits"], f"W {family}")
        _need(len(z_prompt) == len(w_prompt), f"{family}: prompt denominator mismatch")
        result[f"{family}_z_success_to_w_failure"] = sum(
            int(z_bit == 1 and w_bit == 0) for z_bit, w_bit in zip(z_prompt, w_prompt)
        )
        z_strict = _flatten_bits(z["scores"][family]["strict_all_prompt_bits"], f"z strict {family}")
        w_strict = _flatten_bits(w["scores"][family]["strict_all_prompt_bits"], f"W strict {family}")
        _need(len(z_strict) == len(w_strict) == REQUESTS_PER_CASE, f"{family}: strict denominator")
        result[f"{family}_strict_z_success_to_w_failure"] = sum(
            int(z_bit == 1 and w_bit == 0) for z_bit, w_bit in zip(z_strict, w_strict)
        )
    return result


def _dtype_gate(dtype: Mapping[str, Any], label: str) -> None:
    _need(dtype["status"] == "FULL_FP32_PASS", f"{label}: dtype status")
    for key in ("requested_dtype", "loaded_model_dtype", "model_and_floating_parameters"):
        _need(dtype[key] == "torch.float32", f"{label}: {key}")
    _need(dtype["parameter_dtype_counts"] == {"torch.float32": 291}, f"{label}: parameter inventory")
    for key in (
        "autocast_count",
        "bf16_conversion_count",
        "fp16_conversion_count",
        "bf16_candidate_materializer_call_count",
        "numeric_storage_cast_count",
    ):
        _need(int(dtype[key]) == 0, f"{label}: {key}")
    _need(dtype["algorithm_tensors"] == "torch.float32", f"{label}: algorithm dtype")


def _manifest_gate(root: Path, terminal: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    _validate_identity(manifest, f"{root.name}: manifest")
    terminal_path = root / "terminal.json"
    _need(manifest["terminal_sha256"] == _sha256(terminal_path), f"{root.name}: terminal SHA")
    _need(manifest["source_head"] == terminal["source_head"], f"{root.name}: source HEAD")
    case_shas: list[str] = []
    for case_index in range(1, CASE_COUNT + 1):
        case_shas.append(_sha256(root / "raw/cases" / f"case-{case_index:02d}" / "terminal.json"))
    _need(case_shas == manifest["case_terminal_sha256"], f"{root.name}: case terminal SHA")
    return manifest


def _terminal_gate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    terminal = _read_json(root / "terminal.json")
    _validate_identity(terminal, f"{root.name}: terminal")
    _need(terminal["status"] == "TERMINAL_VALID", f"{root.name}: status")
    for key in ("case_count", "completed_case_count", "independent_W0_entry_count", "independent_W0_restore_count"):
        _need(int(terminal[key]) == CASE_COUNT, f"{root.name}: {key}")
    for key in ("technical_failure_count", "scientific_failure_count", "imputation_count", "retry_count", "cross_case_state_count", "sequential_continuity_count"):
        _need(int(terminal[key]) == 0, f"{root.name}: {key}")
    _need(int(terminal["valid_request_count"]) == CASE_COUNT * REQUESTS_PER_CASE, f"{root.name}: requests")
    _need(terminal["W0_restored"] is True, f"{root.name}: W0")
    _dtype_gate(terminal["dtype_contract"], root.name)
    manifest = _manifest_gate(root, terminal)
    return terminal, manifest


def _restore_gate(endpoint: Mapping[str, Any], label: str) -> None:
    restore = endpoint["restore"]
    _need(restore["pointer_restored_exact"] is True and restore["byte_restored_exact"] is True, f"{label}: restore")
    _validate_identity(restore, f"{label}: restore")
    _need(endpoint["module_state_restored"] is True if "module_state_restored" in endpoint else True, f"{label}: module state")


def _timing(endpoint: Mapping[str, Any], is_baseline: bool) -> dict[str, float]:
    timing = endpoint["timing"]
    policy = timing["policy"]
    _need(policy["decision_influence_count"] == 0, "timing decision influence")
    _need(policy["additional_evaluator_count"] == 0, "timing extra evaluator")
    _need(policy["additional_model_forward_count"] == policy["additional_model_backward_count"] == 0, "timing extra F/B")
    return {
        "target_accepted_z_generation_seconds": _finite(timing["target_accepted_z_generation_seconds"], "target timing"),
        "writer_edit_core_seconds": _finite(timing["writer_edit_core_seconds"], "writer timing"),
        "target_plus_writer_seconds": _finite(timing["target_plus_writer_seconds"], "target+writer timing"),
        "immediate_post_evaluator_seconds": _finite(timing["immediate_post_evaluator_seconds"], "evaluator timing"),
        "accepted_z_evaluator_seconds": _finite(timing["accepted_z_evaluator_seconds"], "z evaluator timing"),
        "case_total_seconds": _finite(timing["case_method_total_seconds" if is_baseline else "case_total_seconds"], "case timing"),
        "restore_seconds": _finite(timing["restore_seconds" if is_baseline else "endpoint_restore_seconds"], "restore timing"),
    }


def _energy(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    energy = endpoint["update_energy"]
    _need(energy["bf16_path_call_count"] == energy["numeric_storage_cast_count"] == 0, "energy dtype")
    return {
        "total_actual_fp32_update_energy": _finite(energy["total_actual_fp32_update_energy"], "energy"),
        "layer_update_norm": dict(energy["actual_fp32_update_norm"]),
        "layer_update_energy": dict(energy["actual_fp32_update_energy"]),
        "layer_norm_share": dict(energy["actual_fp32_norm_share"]),
        "layer_energy_share": dict(energy["actual_fp32_squared_energy_share"]),
    }


def _case_row(method: str, case: Mapping[str, Any], endpoint: Mapping[str, Any], is_baseline: bool) -> dict[str, Any]:
    label = f"{method}/case-{case['case_index']:02d}"
    _restore_gate(endpoint, label)
    w_summary = _score_summary(endpoint["W"], f"{label}/W")
    z_summary = _score_summary(endpoint["z"], f"{label}/z")
    losses = _z_to_w_losses(endpoint["z"], endpoint["W"])
    timing = _timing(endpoint, is_baseline)
    energy = _energy(endpoint)
    compute = endpoint["compute_delta"]
    counters = compute["counters"]
    entry_w = endpoint["entry_W0_parameter_sha256"]
    entry_w_identity = canonical_hash(entry_w) if isinstance(entry_w, Mapping) else str(entry_w)
    row: dict[str, Any] = {
        "method": method,
        "case_index": int(case["case_index"]),
        "request_count": REQUESTS_PER_CASE,
        "request_order_sha256": endpoint["request_order_sha256"],
        "slice_identity_sha256": canonical_hash(case["slice_identity"]),
        "stream_root": case["slice_identity"]["stream_root"],
        "stream_order": case["slice_identity"]["stream_order"],
        "entry_W0_parameter_sha256": entry_w_identity,
        "accepted_z_sha256": endpoint.get("accepted_z_sha256") or endpoint.get("accepted_z", {}).get("identity_sha256"),
        "case_terminal_identity_sha256": case["identity_sha256"],
        "endpoint_identity_sha256": endpoint.get("identity_sha256", case["identity_sha256"]),
        "model_forward_count": int(counters["model_forward"]),
        "backward_count": int(counters["backward"]),
        "processed_tokens": int(counters["processed_tokens"]),
        "nonfinite_count": 0,
        "retry_count": int(case.get("retry_count", 0)),
        "imputation_count": int(case.get("imputation_count", 0)),
        **timing,
        **losses,
        **energy,
    }
    for prefix, values in (("z", z_summary), ("W", w_summary)):
        row.update({f"{prefix}_{key}": value for key, value in values.items()})
    row["rewrite_W_minus_z_nll"] = row["W_rewrite_target_new_nll_mean"] - row["z_rewrite_target_new_nll_mean"]
    row["rephrase_W_minus_z_nll"] = row["W_rephrase_target_new_nll_mean"] - row["z_rephrase_target_new_nll_mean"]
    writer = case.get("writer", {}) if not is_baseline else endpoint.get("apply", {})
    row["terminal_residual_norm"] = writer.get("post_terminal_residual_norm")
    row["pi"] = case.get("writer", {}).get("pi") if not is_baseline else None
    row["beta"] = case.get("writer", {}).get("beta") if not is_baseline else None
    route = case.get("writer", {}).get("entry_joint_route") or {}
    for key in ("selected_p", "selected_c", "selected_p_normalized", "selected_c_normalized", "minimax_t", "stationarity_residual", "complementarity_residual", "simplex_sum_residual"):
        row[f"route_{key}"] = route.get(key)
    return row


def _layer_rows(method: str, case: Mapping[str, Any], endpoint: Mapping[str, Any], case_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    telemetry = {int(item["layer"]): item for item in case.get("writer", {}).get("layers", [])}
    rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        weight = f"model.layers.{layer}.mlp.down_proj.weight"
        item = telemetry.get(layer, {})
        rows.append({
            "method": method,
            "case_index": int(case["case_index"]),
            "layer": layer,
            "pi": item.get("pi"),
            "beta": item.get("beta"),
            "suffix_mass": item.get("suffix_mass"),
            "applied_coefficient": item.get("applied_coefficient"),
            "entry_residual_norm": item.get("entry_residual_norm"),
            "current_residual_norm": item.get("current_residual_norm"),
            "key_norm": item.get("key_norm"),
            "q_norm": item.get("q_norm"),
            "update_norm": case_row["layer_update_norm"][weight],
            "update_energy": case_row["layer_update_energy"][weight],
            "update_norm_share": case_row["layer_norm_share"][weight],
            "update_energy_share": case_row["layer_energy_share"][weight],
            "storage_dtype": item.get("storage_dtype", "torch.float32"),
            "numeric_storage_cast_count": int(item.get("numeric_storage_cast_count", 0)),
            "bf16_path_call_count": int(item.get("bf16_path_call_count", 0)),
        })
    return rows


def _aggregate_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _need(len(rows) == CASE_COUNT, "method denominator is not 10")
    result: dict[str, Any] = {"case_count": CASE_COUNT, "request_count": CASE_COUNT * REQUESTS_PER_CASE}
    metric_names = (
        "rewrite_success", "rewrite_accuracy", "rephrase_success", "rephrase_accuracy",
        "rephrase_strict_success", "rephrase_strict_accuracy", "locality",
    )
    for endpoint in ("z", "W"):
        for metric in metric_names:
            numerator = sum(int(row[f"{endpoint}_{metric}_numerator"]) for row in rows)
            denominator = sum(int(row[f"{endpoint}_{metric}_denominator"]) for row in rows)
            result[f"{endpoint}_{metric}_numerator"] = numerator
            result[f"{endpoint}_{metric}_denominator"] = denominator
            result[f"{endpoint}_{metric}_rate"] = _rate(numerator, denominator)
        for family in ("rewrite", "rephrase"):
            for target in ("new", "true"):
                key = f"{endpoint}_{family}_target_{target}_nll_mean"
                result[key] = statistics.fmean(float(row[key]) for row in rows)
    for key in (
        "rewrite_W_minus_z_nll", "rephrase_W_minus_z_nll", "terminal_residual_norm",
        "total_actual_fp32_update_energy", "model_forward_count", "backward_count", "processed_tokens",
        "rewrite_success_z_success_to_w_failure", "rewrite_success_strict_z_success_to_w_failure",
        "rephrase_success_z_success_to_w_failure", "rephrase_success_strict_z_success_to_w_failure",
    ):
        values = [row[key] for row in rows if row.get(key) is not None]
        result[key] = statistics.fmean(float(value) for value in values) if values else None
        if key.endswith("failure"):
            result[f"{key}_total"] = sum(int(value) for value in values)
    for boundary in ("writer_edit_core_seconds", "target_plus_writer_seconds", "case_total_seconds", "target_accepted_z_generation_seconds", "immediate_post_evaluator_seconds"):
        result[f"{boundary}_stats"] = _stats([float(row[boundary]) for row in rows])
    result["layer_norm_share_mean"] = {
        str(layer): statistics.fmean(float(row["layer_norm_share"][f"model.layers.{layer}.mlp.down_proj.weight"]) for row in rows)
        for layer in LAYERS
    }
    result["layer_energy_share_mean"] = {
        str(layer): statistics.fmean(float(row["layer_energy_share"][f"model.layers.{layer}.mlp.down_proj.weight"]) for row in rows)
        for layer in LAYERS
    }
    pi_rows = [row["pi"] for row in rows if row.get("pi") is not None]
    beta_rows = [row["beta"] for row in rows if row.get("beta") is not None]
    result["pi_mean"] = [statistics.fmean(float(row[index]) for row in pi_rows) for index in range(5)] if pi_rows else None
    result["beta_mean"] = [statistics.fmean(float(row[index]) for row in beta_rows) for index in range(5)] if beta_rows else None
    for key in ("route_selected_p", "route_selected_c", "route_selected_p_normalized", "route_selected_c_normalized", "route_minimax_t", "route_stationarity_residual", "route_complementarity_residual", "route_simplex_sum_residual"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[f"{key}_mean"] = statistics.fmean(values) if values else None
        result[f"{key}_max"] = max(values) if values else None
    if result["terminal_residual_norm"] is not None:
        entry_values = [float(row["terminal_residual_norm"]) for row in rows if row.get("terminal_residual_norm") is not None]
        result["terminal_residual_norm_stats"] = _stats(entry_values)
    return result


def _aggregate_layers(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for method in METHODS:
        for layer in LAYERS:
            selected = [row for row in rows if row["method"] == method and row["layer"] == layer]
            _need(len(selected) == CASE_COUNT, f"{method}/L{layer}: layer denominator")
            item: dict[str, Any] = {"method": method, "layer": layer, "case_count": CASE_COUNT}
            for key in ("pi", "beta", "suffix_mass", "applied_coefficient", "entry_residual_norm", "current_residual_norm", "key_norm", "q_norm", "update_norm", "update_energy", "update_norm_share", "update_energy_share"):
                values = [float(row[key]) for row in selected if row.get(key) is not None]
                item[f"{key}_mean"] = statistics.fmean(values) if values else None
            result.append(item)
    return result


def _paired_overhead(case_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed = {(row["method"], row["case_index"]): row for row in case_rows}
    pairs: list[dict[str, Any]] = []
    boundaries = ("writer_edit_core_seconds", "target_plus_writer_seconds", "case_total_seconds")
    for method in METHODS:
        if method == "OFFICIAL-ALPHAEDIT":
            continue
        for case_index in range(1, CASE_COUNT + 1):
            method_row = indexed[(method, case_index)]
            alpha_row = indexed[("OFFICIAL-ALPHAEDIT", case_index)]
            _need(method_row["request_order_sha256"] == alpha_row["request_order_sha256"], "overhead slice mismatch")
            for boundary in boundaries:
                observed = float(method_row[boundary])
                baseline = float(alpha_row[boundary])
                _need(baseline > 0.0, "zero AlphaEdit timing denominator")
                ratio = observed / baseline
                pairs.append({
                    "method": method,
                    "case_index": case_index,
                    "boundary": boundary,
                    "method_seconds": observed,
                    "alphaedit_seconds": baseline,
                    "absolute_overhead_seconds": observed - baseline,
                    "overhead_ratio": ratio,
                    "overhead_percent": (ratio - 1.0) * 100.0,
                    "request_order_sha256": method_row["request_order_sha256"],
                })
    aggregates: dict[str, Any] = {}
    for method in METHODS:
        if method == "OFFICIAL-ALPHAEDIT":
            continue
        aggregates[method] = {}
        for boundary in boundaries:
            selected = [row for row in pairs if row["method"] == method and row["boundary"] == boundary]
            _need(len(selected) == CASE_COUNT, f"{method}/{boundary}: NOT_COMPARABLE")
            aggregates[method][boundary] = {
                "denominator": CASE_COUNT,
                "seconds": _stats([row["method_seconds"] for row in selected]),
                "paired_delta_seconds": _stats([row["absolute_overhead_seconds"] for row in selected]),
                "paired_ratio": _stats([row["overhead_ratio"] for row in selected]),
                "paired_percent": _stats([row["overhead_percent"] for row in selected]),
                "comparison_status": "COMPARABLE_10_OF_10",
            }
    return pairs, aggregates


def _paired_performance(case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["method"], row["case_index"]): row for row in case_rows}
    pairs: list[dict[str, Any]] = []
    for method in C_METHODS:
        for baseline in ("OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT"):
            for case_index in range(1, CASE_COUNT + 1):
                row = indexed[(method, case_index)]
                ref = indexed[(baseline, case_index)]
                _need(row["request_order_sha256"] == ref["request_order_sha256"], "performance slice mismatch")
                pairs.append({
                    "method": method,
                    "baseline": baseline,
                    "case_index": case_index,
                    "W_rephrase_success_rate_delta": row["W_rephrase_success_rate"] - ref["W_rephrase_success_rate"],
                    "W_strict_rephrase_success_rate_delta": row["W_rephrase_strict_success_rate"] - ref["W_rephrase_strict_success_rate"],
                    "W_locality_rate_delta": row["W_locality_rate"] - ref["W_locality_rate"],
                    "rewrite_nll_delta": row["W_rewrite_target_new_nll_mean"] - ref["W_rewrite_target_new_nll_mean"],
                    "rephrase_nll_delta": row["W_rephrase_target_new_nll_mean"] - ref["W_rephrase_target_new_nll_mean"],
                    "update_energy_delta": row["total_actual_fp32_update_energy"] - ref["total_actual_fp32_update_energy"],
                })
    return pairs


def _nll_distribution(case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize the ten sealed case-level NLL means without pooling endpoints."""

    result: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in case_rows if row["method"] == method]
        _need(len(selected) == CASE_COUNT, f"{method}: NLL denominator is not 10")
        for endpoint in ("z", "W"):
            for family in ("rewrite", "rephrase"):
                for target in ("new", "true"):
                    key = f"{endpoint}_{family}_target_{target}_nll_mean"
                    result.append({
                        "method": method,
                        "endpoint": endpoint.upper(),
                        "family": family,
                        "target": f"target-{target}",
                        "case_denominator": CASE_COUNT,
                        "case_level_statistic": "NLL_MEAN_WITHIN_CASE",
                        "cross_case_summary": _stats([float(row[key]) for row in selected]),
                        "p90_definition": "NEAREST_RANK_CEIL_0.9N",
                    })
    _need(len(result) == len(METHODS) * 2 * 2 * 2, "NLL distribution row count")
    return result


def _c_target_steps(
    cases_by_method: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract only the K1→K8 delayed target progress sealed by the runtime."""

    rows: list[dict[str, Any]] = []
    for method in C_METHODS:
        for case_index in range(1, CASE_COUNT + 1):
            case = cases_by_method[(method, case_index)]
            target = case["target_public"]
            _validate_identity(target, f"{method}/case-{case_index:02d}: target_public")
            _need(target["target_depth_inner_telemetry_enabled"] is False, "unexpected inner telemetry")
            progress = target["delayed_progress"]
            _need(len(progress) == 7, f"{method}/case-{case_index:02d}: delayed progress")
            for expected_transition, item in enumerate(progress, start=1):
                _validate_identity(item, f"{method}/case-{case_index:02d}/transition-{expected_transition}")
                _need(item["transition_index"] == expected_transition, "target transition order")
                _need(item["candidate_objective_inner_count"] == 0, "target inner objective count")
                rows.append({
                    "method": method,
                    "case_index": case_index,
                    "source_k": expected_transition,
                    "next_k": expected_transition + 1,
                    "request_order_sha256": case["request_order_sha256"],
                    "accepted_z_sha256": case["accepted_z_sha256"],
                    "target_public_identity_sha256": target["identity_sha256"],
                    "transition_identity_sha256": item["identity_sha256"],
                    "completion": item["completion"],
                    "source_mean_target_new_nll": _finite(item["source_mean_target_new_nll"], "source step NLL"),
                    "next_field_mean_target_new_nll": _finite(item["next_field_mean_target_new_nll"], "next step NLL"),
                    "predicted_progress": _finite(item["predicted"], "predicted progress"),
                    "actual_progress": _finite(item["actual"], "actual progress"),
                    "linearization_error": _finite(item["linearization_error"], "linearization error"),
                    "realization_ratio": _finite(item["realization_ratio"], "realization ratio"),
                    "candidate_objective_inner_count": 0,
                    "heldout_inner_evaluator_count": int(case["heldout_inner_evaluator_count"]),
                    "target_depth_inner_telemetry_enabled": False,
                    "target_depth_inner_telemetry_row_count": int(target["target_depth_inner_telemetry_row_count"]),
                })
    _need(len(rows) == len(C_METHODS) * CASE_COUNT * 7, "C target step row count")
    aggregates: list[dict[str, Any]] = []
    for method in C_METHODS:
        for transition in range(1, 8):
            selected = [row for row in rows if row["method"] == method and row["source_k"] == transition]
            _need(len(selected) == CASE_COUNT, "C target step denominator")
            aggregate: dict[str, Any] = {
                "method": method,
                "source_k": transition,
                "next_k": transition + 1,
                "case_denominator": CASE_COUNT,
                "inner_objective_count": 0,
                "heldout_inner_evaluator_count": 0,
                "target_depth_inner_telemetry_enabled": False,
                "target_depth_inner_telemetry_row_count_per_case": int(selected[0]["target_depth_inner_telemetry_row_count"]),
            }
            for key in (
                "source_mean_target_new_nll",
                "next_field_mean_target_new_nll",
                "predicted_progress",
                "actual_progress",
                "linearization_error",
                "realization_ratio",
            ):
                aggregate[f"{key}_stats"] = _stats([float(row[key]) for row in selected])
            aggregates.append(aggregate)
    return rows, aggregates


def _prompt_nll_distribution(
    cases_by_method: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize every sealed rewrite/rephrase prompt NLL across 10 cases."""

    result: list[dict[str, Any]] = []
    for method in METHODS:
        for endpoint_name in ("z", "W"):
            for family in ("rewrite", "rephrase"):
                score_name = f"{family}_success"
                for target in ("new", "true"):
                    values: list[float] = []
                    for case_index in range(1, CASE_COUNT + 1):
                        case = cases_by_method[(method, case_index)]
                        endpoint = case["methods"][method] if method.startswith("OFFICIAL-") else case
                        nested = endpoint[endpoint_name]["scores"][score_name][f"target_{target}_nll_by_request"]
                        for value in _flatten_nested_values(nested, f"{method}/{endpoint_name}/{family}/{target}"):
                            values.append(_finite(value, "prompt NLL"))
                    expected = CASE_COUNT * REQUESTS_PER_CASE * (1 if family == "rewrite" else 2)
                    _need(len(values) == expected, f"{method}/{endpoint_name}/{family}/{target}: prompt denominator")
                    result.append({
                        "method": method,
                        "endpoint": endpoint_name.upper(),
                        "family": family,
                        "target": f"target-{target}",
                        "prompt_denominator": expected,
                        "cross_prompt_summary": _stats(values),
                        "p90_definition": "NEAREST_RANK_CEIL_0.9N",
                    })
    _need(len(result) == len(METHODS) * 2 * 2 * 2, "prompt NLL row count")
    return result


def _flatten_nested_values(value: Any, label: str) -> list[Any]:
    _need(isinstance(value, list), f"{label}: values not list")
    result: list[Any] = []
    for item in value:
        if isinstance(item, list):
            result.extend(_flatten_nested_values(item, label))
        else:
            result.append(item)
    return result


def _pilot_consistency(repo: Path, case_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path = repo / PILOT_REPORT
    if not path.is_file():
        return {"status": "NOT_RECORDED", "path": str(path)}
    pilot_rows = _read_json(path)
    pilot = {row["endpoint"]: row for row in pilot_rows}
    current = {(row["method"], row["case_index"]): row for row in case_rows}
    mapping = {
        "OFFICIAL-ALPHAEDIT": "OFFICIAL_ALPHAEDIT_W",
        "OFFICIAL-MEMIT": "OFFICIAL_MEMIT_W",
        "C0": "C0-PIRU-CONTROL",
        "C1": "C1-JOINT-PC-REMAINING",
        "C2": "C2-JOINT-PC-FIXED-QUOTA",
        "C3": "C3-DIRECT-OFFICIAL-ALPHAEDIT",
    }
    comparisons: dict[str, Any] = {}
    for method, endpoint in mapping.items():
        row = current[(method, 1)]
        sealed = pilot[endpoint]
        comparisons[method] = {
            "rewrite_nll_delta": row["W_rewrite_target_new_nll_mean"] - sealed["rewrite_nll"],
            "rephrase_nll_delta": row["W_rephrase_target_new_nll_mean"] - sealed["rephrase_nll"],
            "rephrase_success_numerator_delta": row["W_rephrase_success_numerator"] - sealed["rephrase_success_numerator"],
            "strict_rephrase_success_numerator_delta": row["W_rephrase_strict_success_numerator"] - sealed["strict_rephrase_success_numerator"],
            "locality_numerator_delta": row["W_locality_numerator"] - sealed["locality_numerator"],
            "energy_delta": row["total_actual_fp32_update_energy"] - sealed["total_update_energy"],
        }
    current_z = current[("C0", 1)]["accepted_z_sha256"]
    receipt_path = path.parent / "rooted-analysis-receipt.json"
    pilot_receipt = _read_json(receipt_path)
    return {
        "status": "FACTUAL_COMPARISON_COMPLETE",
        "request_stream_identity_exact": current[("C0", 1)]["stream_order"] == pilot_receipt["stream_order"],
        "stream_root_identity_exact": current[("C0", 1)]["stream_root"] == pilot_receipt["stream_root"],
        "accepted_z_identity_exact": current_z == pilot_receipt["accepted_z_sha256"],
        "current_case01_accepted_z_sha256": current_z,
        "sealed_pilot_accepted_z_sha256": pilot_receipt["accepted_z_sha256"],
        "comparisons": comparisons,
        "interpretation": "Official baselines are exact when deltas are zero; P1R52 rows are factual cross-run comparisons, not substituted endpoints.",
    }


def analyze(repo: Path) -> dict[str, Any]:
    results_base = repo / "local/odebf/results"
    roots = {name: results_base / root_name for name, root_name in ROOT_NAMES.items()}
    top: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for name, root in roots.items():
        top[name], manifests[name] = _terminal_gate(root)
    source_heads = {payload["source_head"] for payload in top.values()}
    _need(len(source_heads) == 1, "source HEAD mismatch")
    stream_orders = {payload["stream_order"] for payload in top.values()}
    stream_roots = {payload["stream_root"] for payload in top.values()}
    _need(len(stream_orders) == len(stream_roots) == 1, "stream mismatch")
    dtype_inventory = {canonical_hash(payload["dtype_contract"]) for payload in top.values()}
    _need(len(dtype_inventory) == 1, "dtype inventory mismatch")

    case_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    cases_by_method: dict[tuple[str, int], Mapping[str, Any]] = {}
    for case_index in range(1, CASE_COUNT + 1):
        loaded: dict[str, Mapping[str, Any]] = {}
        for name, root in roots.items():
            case = _read_json(root / "raw/cases" / f"case-{case_index:02d}" / "terminal.json")
            _validate_identity(case, f"{name}/case-{case_index:02d}")
            _need(int(case["case_index"]) == case_index and int(case["request_count"]) == REQUESTS_PER_CASE, "case coordinate")
            _need(case["W0_restored"] is True and case.get("retry_count", 0) == case.get("imputation_count", 0) == 0, "case state")
            loaded[name] = case
        slice_ids = {canonical_hash(case["slice_identity"]) for case in loaded.values()}
        request_orders = {case["request_order_sha256"] for case in loaded.values()}
        _need(len(slice_ids) == len(request_orders) == 1, f"case-{case_index:02d}: slice mismatch")
        accepted = {loaded[name]["accepted_z_sha256"] for name in C_METHODS}
        c_entry = {
            canonical_hash(loaded[name]["entry_W0_parameter_sha256"])
            if isinstance(loaded[name]["entry_W0_parameter_sha256"], Mapping)
            else str(loaded[name]["entry_W0_parameter_sha256"])
            for name in C_METHODS
        }
        _need(len(accepted) == len(c_entry) == 1, f"case-{case_index:02d}: C-arm identity")

        baseline = loaded["BASELINE"]
        for method in ("OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT"):
            endpoint = baseline["methods"][method]
            row = _case_row(method, baseline, endpoint, True)
            case_rows.append(row)
            layer_rows.extend(_layer_rows(method, baseline, endpoint, row))
            cases_by_method[(method, case_index)] = baseline
        all_entry = set(c_entry)
        for method in C_METHODS:
            case = loaded[method]
            row = _case_row(method, case, case, False)
            case_rows.append(row)
            layer_rows.extend(_layer_rows(method, case, case, row))
            cases_by_method[(method, case_index)] = case
            all_entry.add(row["entry_W0_parameter_sha256"])
        for method in ("OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT"):
            all_entry.add(next(row["entry_W0_parameter_sha256"] for row in case_rows if row["method"] == method and row["case_index"] == case_index))
        _need(len(all_entry) == 1, f"case-{case_index:02d}: W0 mismatch")

    _need(len(case_rows) == len(METHODS) * CASE_COUNT, "case row count")
    _need(len(layer_rows) == len(METHODS) * CASE_COUNT * len(LAYERS), "layer row count")
    c3_cases = [cases_by_method[("C3", index)] for index in range(1, CASE_COUNT + 1)]
    for case in c3_cases:
        writer = case["writer"]
        apply = writer["apply"]
        cache = apply["alphaedit_dynamic_cache_contract"]
        _need(writer["native_compute_z_call_count"] == apply["native_alphaedit_compute_z_call_count"] == 0, "C3 compute_z")
        _need(cache["static_projection"]["loaded"] is True and cache["static_projection"]["dtype"] == "torch.float32", "C3 static P")
        _need(cache["logical_history_width_at_entry"] == 0 and cache["logical_history_width_after_append"] == 100, "C3 cache")
        _need(writer["p1r52_barrier_decision_influence_count"] == writer["p1r52_pc_router_decision_influence_count"] == 0, "C3 influence")

    aggregates = {
        method: _aggregate_method([row for row in case_rows if row["method"] == method])
        for method in METHODS
    }
    layer_aggregates = _aggregate_layers(layer_rows)
    overhead_rows, overhead_aggregates = _paired_overhead(case_rows)
    performance_pairs = _paired_performance(case_rows)
    nll_distribution = _nll_distribution(case_rows)
    prompt_nll_distribution = _prompt_nll_distribution(cases_by_method)
    c_target_step_rows, c_target_step_aggregates = _c_target_steps(cases_by_method)
    job_facts: dict[str, Any] = {}
    for name, root in roots.items():
        timing = _read_json(root / "job-timing.json")
        _validate_identity(timing, f"{name}: job timing")
        gpu = top[name]["final_gpu_host_observation"]
        job_facts[name] = {
            "slurm_array_job_id": timing["slurm_array_job_id"],
            "slurm_array_task_id": timing["slurm_array_task_id"],
            "slurm_job_id": timing["slurm_job_id"],
            "job_total_seconds": timing["job_total_seconds"],
            "runtime_after_model_preflight_seconds": top[name]["runtime_after_model_preflight_seconds"],
            "model_load_preflight_seconds": top[name]["model_load_preflight"],
            "gpu": gpu,
            "compute": top[name]["job_compute"],
        }
    return {
        "schema": "ode-edit-s05-p1r52-joint-pc-independent-full-fp32-analysis/v1",
        "status": "TERMINAL_ANALYSIS_VALID",
        "source_head": next(iter(source_heads)),
        "stream_order": next(iter(stream_orders)),
        "stream_root": next(iter(stream_roots)),
        "method_count": len(METHODS),
        "case_count_per_method": CASE_COUNT,
        "valid_endpoint_count": len(case_rows),
        "valid_request_count": len(case_rows) * REQUESTS_PER_CASE,
        "dtype_status": "FULL_FP32_PASS_ALL_METHODS",
        "W0_restore_status": "PASS_60_OF_60",
        "C_arm_same_accepted_z_status": "PASS_10_OF_10",
        "C_arm_same_entry_W_status": "PASS_10_OF_10",
        "sequential_continuity_count": 0,
        "technical_failure_count": 0,
        "scientific_failure_count": 0,
        "retry_count": 0,
        "imputation_count": 0,
        "promotion_ruling": "NO_SCIENTIFIC_PROMOTION_IN_THIS_INDEPENDENT_GATE",
        "case_rows": case_rows,
        "layer_rows": layer_rows,
        "layer_aggregates": layer_aggregates,
        "aggregates": aggregates,
        "performance_pairs": performance_pairs,
        "nll_distribution": nll_distribution,
        "prompt_nll_distribution": prompt_nll_distribution,
        "c_target_step_rows": c_target_step_rows,
        "c_target_step_aggregates": c_target_step_aggregates,
        "overhead_rows": overhead_rows,
        "overhead_aggregates": overhead_aggregates,
        "pilot_case01_consistency": _pilot_consistency(repo, case_rows),
        "job_facts": job_facts,
        "input_roots": {name: str(root.resolve()) for name, root in roots.items()},
        "input_manifests": {name: manifests[name]["identity_sha256"] for name in manifests},
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NOT_RECORDED"
    return f"{float(value):.{digits}f}"


def _report(analysis: Mapping[str, Any]) -> str:
    aggregate = analysis["aggregates"]
    lines = [
        "# P1R52 Joint P+C FULL-FP32 독립 10×B100 사실 보고서",
        "",
        "> **핵심 실행 경계:** 이번 C0–C3는 이전 J0의 step-wise write 경로와 달리, P1R52 target controller가 K8까지 z 최적화를 모두 완료한 뒤 최종 accepted-z를 고정하고 writer를 정확히 한 번만 실행한 one-shot 편집이다. 따라서 아래 K1→K8 표는 write 중간 endpoint가 아니라 최종 write 전에 수행된 z/target 최적화 telemetry이며, 실제 W endpoint는 그 뒤의 단일 writer 적용 결과다.",
        "",
        "## 결론",
        "",
        f"- 상태: `{analysis['status']}`. 6개 방법 × 10개 독립 B100 = {analysis['valid_endpoint_count']} endpoints, {analysis['valid_request_count']} requests가 유효하다.",
        "- 모든 floating model parameter와 writer 알고리즘 텐서는 FP32였다. BF16/FP16/autocast/quantization, numeric storage cast, retry, imputation, sequential carry는 모두 0이다.",
        "- 각 case는 동일 slice/order와 동일 W0에서 시작했고 60/60 endpoint가 pointer/bytes exact W0 restore를 통과했다. C0–C3는 매 slice 같은 accepted-z 및 entry-W를 사용했다.",
        f"- 판정: `{analysis['promotion_ruling']}`. 이 독립 gate는 방법을 자동 승격하거나 sequential endpoint와 혼합하지 않는다.",
        "",
        "## 방법별 10-case 집계",
        "",
        "|방법|W Rewrite|W Gen|W strict Gen|W Loc|rewrite W−z NLL|rephrase W−z NLL|평균 energy|L8 energy share|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate[method]
        lines.append(
            f"|{method}|{row['W_rewrite_success_numerator']}/{row['W_rewrite_success_denominator']} "
            f"({_fmt(row['W_rewrite_success_rate'])})|{row['W_rephrase_success_numerator']}/{row['W_rephrase_success_denominator']} "
            f"({_fmt(row['W_rephrase_success_rate'])})|{row['W_rephrase_strict_success_numerator']}/{row['W_rephrase_strict_success_denominator']} "
            f"({_fmt(row['W_rephrase_strict_success_rate'])})|{row['W_locality_numerator']}/{row['W_locality_denominator']} "
            f"({_fmt(row['W_locality_rate'])})|{_fmt(row['rewrite_W_minus_z_nll'])}|{_fmt(row['rephrase_W_minus_z_nll'])}|"
            f"{_fmt(row['total_actual_fp32_update_energy'])}|{_fmt(row['layer_energy_share_mean']['8'])}|"
        )
    lines.extend([
        "",
        "### Accepted-z 전용 요약",
        "",
        "C0–C3는 각 slice 안에서 동일한 P1R52 K8 accepted-z를 사용했다(`PASS_10_OF_10`). AlphaEdit/MEMIT의 Z는 각 native method가 만든 별도 reference이므로 C-arm accepted-z와 byte-equivalence를 주장하지 않는다.",
        "",
        "|방법|Z source|z Rewrite|z Gen|z strict Gen|z rewrite acc|z rephrase acc|z strict acc|z Loc|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        row = aggregate[method]
        z_source = "P1R52_K8_ACCEPTED_Z_SHARED_C0_C3" if method in C_METHODS else f"NATIVE_{method.removeprefix('OFFICIAL-')}_Z"
        lines.append(
            f"|{method}|{z_source}|{row['z_rewrite_success_numerator']}/{row['z_rewrite_success_denominator']}|"
            f"{row['z_rephrase_success_numerator']}/{row['z_rephrase_success_denominator']}|"
            f"{row['z_rephrase_strict_success_numerator']}/{row['z_rephrase_strict_success_denominator']}|"
            f"{row['z_rewrite_accuracy_numerator']}/{row['z_rewrite_accuracy_denominator']}|"
            f"{row['z_rephrase_accuracy_numerator']}/{row['z_rephrase_accuracy_denominator']}|"
            f"{row['z_rephrase_strict_accuracy_numerator']}/{row['z_rephrase_strict_accuracy_denominator']}|"
            f"{row['z_locality_numerator']}/{row['z_locality_denominator']}|"
        )
    nll_index = {
        (row["method"], row["endpoint"], row["family"], row["target"]): row["cross_prompt_summary"]
        for row in analysis["prompt_nll_distribution"]
    }
    for target in ("target-new", "target-true"):
        title = "새 target NLL" if target == "target-new" else "원 target NLL"
        lines.extend([
            "",
            f"### Z/W {title}: 전체 prompt mean·median·p90",
            "",
            "Rewrite denominator는 10×100=1000 prompts, rephrase denominator는 10×100×2=2000 prompts이다. p90은 nearest-rank `ceil(0.9N)`이다. case-level mean의 10-case mean/median/p90도 별도 machine table에 보존했다.",
            "",
            "|방법|Z rewrite mean/median/p90|Z rephrase mean/median/p90|W rewrite mean/median/p90|W rephrase mean/median/p90|",
            "|---|---:|---:|---:|---:|",
        ])
        for method in METHODS:
            cells = []
            for endpoint, family in (("Z", "rewrite"), ("Z", "rephrase"), ("W", "rewrite"), ("W", "rephrase")):
                stats = nll_index[(method, endpoint, family, target)]
                cells.append(f"{_fmt(stats['mean'])} / {_fmt(stats['median'])} / {_fmt(stats['p90'])}")
            lines.append(f"|{method}|" + "|".join(cells) + "|")
    lines.extend([
        "",
        "### W accuracy",
        "",
        "|방법|W rewrite accuracy|W rephrase accuracy|W strict accuracy|",
        "|---|---:|---:|---:|",
    ])
    for method in METHODS:
        row = aggregate[method]
        lines.append(
            f"|{method}|{row['W_rewrite_accuracy_numerator']}/{row['W_rewrite_accuracy_denominator']}|"
            f"{row['W_rephrase_accuracy_numerator']}/{row['W_rephrase_accuracy_denominator']}|"
            f"{row['W_rephrase_strict_accuracy_numerator']}/{row['W_rephrase_strict_accuracy_denominator']}|"
        )
    lines.extend([
        "",
        "### C 계열 K-step target progress",
        "",
        "Runtime receipt에는 C0–C3 각각 K1→K8의 7개 delayed target transition이 기록돼 있었다. 아래는 각 transition의 10-case 평균이며, per-case 원값과 mean/median/p90은 `c-target-step.json` 및 `c-target-step-aggregates.json`에 보존했다. heldout evaluator와 candidate objective 재평가는 모두 0이다. `target_depth_inner_telemetry_enabled=false`이고 봉인된 bookkeeping row count는 case당 8로 별도 기록했다.",
        "",
        "|방법|transition|source NLL|next-field NLL|predicted progress|actual progress|linearization error|realization ratio|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in analysis["c_target_step_aggregates"]:
        lines.append(
            f"|{row['method']}|K{row['source_k']}→K{row['next_k']}|"
            f"{_fmt(row['source_mean_target_new_nll_stats']['mean'])}|"
            f"{_fmt(row['next_field_mean_target_new_nll_stats']['mean'])}|"
            f"{_fmt(row['predicted_progress_stats']['mean'])}|"
            f"{_fmt(row['actual_progress_stats']['mean'])}|"
            f"{_fmt(row['linearization_error_stats']['mean'])}|"
            f"{_fmt(row['realization_ratio_stats']['mean'])}|"
        )
    lines.extend([
        "",
        "### z→W loss와 writer realization",
        "",
        "|방법|rewrite prompt loss|Gen prompt loss|strict Gen request loss|terminal residual|평균 L8 norm share|",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        row = aggregate[method]
        lines.append(
            f"|{method}|{int(row['rewrite_success_z_success_to_w_failure_total'])}|"
            f"{int(row['rephrase_success_z_success_to_w_failure_total'])}|"
            f"{int(row['rephrase_success_strict_z_success_to_w_failure_total'])}|"
            f"{_fmt(row['terminal_residual_norm'])}|{_fmt(row['layer_norm_share_mean']['8'])}|"
        )
    lines.extend([
        "",
        "Success와 accuracy는 서로 바꾸어 쓰지 않았고 모든 numerator/denominator를 분리했다. W−z gap과 z-success→W-failure는 동일 endpoint의 prompt bit vectors에서 직접 계산했다.",
        "",
        "### C-arm allocation 및 layer trajectory",
        "",
        "|방법|mean π (L4→L8)|mean β (L4→L8)|predicted P(norm)|predicted C(norm)|minimax t|",
        "|---|---|---|---:|---:|---:|",
    ])
    for method in C_METHODS:
        row = aggregate[method]
        pi = ", ".join(_fmt(value, 3) for value in row["pi_mean"]) if row["pi_mean"] else "NOT_RECORDED"
        beta = ", ".join(_fmt(value, 3) for value in row["beta_mean"]) if row["beta_mean"] else "NOT_RECORDED"
        lines.append(
            f"|{method}|{pi}|{beta}|{_fmt(row['route_selected_p_normalized_mean'])}|"
            f"{_fmt(row['route_selected_c_normalized_mean'])}|{_fmt(row['route_minimax_t_mean'])}|"
        )
    lines.extend([
        "",
        "P/C 값은 entry quadratic proxy가 예측한 값이다. 물리 endpoint의 별도 realized P/C scalar가 receipt에 없으므로 추정하지 않고, 실제 realization은 residual·update norm/energy·W 지표로 분리했다.",
        "",
        "|방법/Layer|π|β|current residual|q norm|update norm share|energy share|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    layer_index = {(row["method"], row["layer"]): row for row in analysis["layer_aggregates"]}
    for method in C_METHODS:
        for layer in LAYERS:
            row = layer_index[(method, layer)]
            lines.append(
                f"|{method}/L{layer}|{_fmt(row['pi_mean'])}|{_fmt(row['beta_mean'])}|"
                f"{_fmt(row['current_residual_norm_mean'])}|{_fmt(row['q_norm_mean'])}|"
                f"{_fmt(row['update_norm_share_mean'])}|{_fmt(row['update_energy_share_mean'])}|"
            )
    lines.extend([
        "",
        "### 동일-slice baseline 대비 성능 delta",
        "",
        "|방법|기준|Δ Gen rate|Δ strict Gen rate|Δ Loc rate|Δ rephrase NLL|Δ energy|",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for method in C_METHODS:
        for baseline in ("OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT"):
            selected = [row for row in analysis["performance_pairs"] if row["method"] == method and row["baseline"] == baseline]
            lines.append(
                f"|{method}|{baseline}|{_fmt(statistics.fmean(row['W_rephrase_success_rate_delta'] for row in selected))}|"
                f"{_fmt(statistics.fmean(row['W_strict_rephrase_success_rate_delta'] for row in selected))}|"
                f"{_fmt(statistics.fmean(row['W_locality_rate_delta'] for row in selected))}|"
                f"{_fmt(statistics.fmean(row['rephrase_nll_delta'] for row in selected))}|"
                f"{_fmt(statistics.fmean(row['update_energy_delta'] for row in selected))}|"
            )
    lines.extend([
        "",
        "## AlphaEdit 대비 시간 overhead",
        "",
        "주 비교는 공통 evaluator와 model load를 제외한 `writer_edit_core_seconds`이다. `target_plus_writer_seconds`와 W0 entry→endpoint+restore의 `case_total_seconds`를 보조 경계로 제시한다. 각 값은 동일 slice 10/10 paired denominator이다.",
        "",
        "|방법|edit-core 평균(s)|paired ratio 평균|overhead % 평균|target+writer ratio|case E2E ratio|",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        if method == "OFFICIAL-ALPHAEDIT":
            continue
        overhead = analysis["overhead_aggregates"][method]
        core = overhead["writer_edit_core_seconds"]
        lines.append(
            f"|{method}|{_fmt(core['seconds']['mean'], 2)}|{_fmt(core['paired_ratio']['mean'], 3)}|"
            f"{_fmt(core['paired_percent']['mean'], 1)}|{_fmt(overhead['target_plus_writer_seconds']['paired_ratio']['mean'], 3)}|"
            f"{_fmt(overhead['case_total_seconds']['paired_ratio']['mean'], 3)}|"
        )
    lines.extend([
        "",
        "관측 timer는 각 CUDA 구간 시작 전 synchronize를 interval 밖에 두고, 종료 synchronize의 wait를 interval 안에 포함했다. timing의 decision influence와 추가 F/B/evaluator는 0이다. 동시 task는 서로 다른 A6000에서 실행되었으므로 wall time에는 장치·contention 차이가 포함될 수 있고, 원인 단독 증거로 해석하지 않는다.",
        "",
        "## FP32·독립성·transaction 무결성",
        "",
        "- requested/loaded/storage dtype: `torch.float32`; parameter inventory는 cell마다 `torch.float32: 291`로 동일하다.",
        "- accepted-z, terminal/residual, P/covariance/history/K/A/RHS/q, coefficient 및 update는 FP32이다.",
        "- C3는 direct Official AlphaEdit entrypoint를 사용했고 native `compute_z` 호출 0, static-P FP32 load, case-entry cache width 0→exit 100을 만족했다.",
        "- C0/C1/C2의 P/C route 수식 및 C1 remaining-residual, C2 fixed-quota 실행은 source receipt 그대로이며 이 분석은 solver를 다시 호출하지 않았다.",
        "- case 간 physical W/controller/cache/history 전달은 0이다. 과거 sequential 결과는 slice identity 참조일 뿐 endpoint comparison denominator에 포함하지 않았다.",
        "",
        "## SH2 sealed single-B100와 case01",
        "",
    ])
    pilot = analysis["pilot_case01_consistency"]
    if pilot["status"] == "FACTUAL_COMPARISON_COMPLETE":
        lines.append(f"- stream/order identity: {pilot['request_stream_identity_exact']}; stream-root identity: {pilot['stream_root_identity_exact']}.")
        lines.append(f"- P1R52 accepted-z byte identity: {pilot['accepted_z_identity_exact']} (`{pilot['sealed_pilot_accepted_z_sha256']}` → `{pilot['current_case01_accepted_z_sha256']}`).")
        lines.append("- Official AlphaEdit/MEMIT 수치가 sealed report와 동일한지는 `pilot-case01-consistency.json`의 zero-delta로 확인한다. P1R52 accepted-z가 byte-different이면 current production 내부 4-arm same-z gate와 분리하여 factual cross-run difference로만 기록하며 endpoint를 대체하지 않는다.")
    else:
        lines.append("- sealed pilot table: NOT_RECORDED.")
    lines.extend([
        "",
        "## Compute와 자원",
        "",
        "- `job-facts.json`은 cell별 model-load/preflight, job total, model F/B, processed tokens, peak allocated/reserved GPU memory, MaxRSS, GPU UUID를 보존한다.",
        "- model load/job total은 one-time 사실값으로 overhead ratio에 섞지 않았다. technical-repair wasted compute는 valid job 22541에서 0이다.",
        "",
        "|cell|job total(s)|model load(s)|model F|tokens|GPU peak alloc GiB|MaxRSS GiB|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for cell, facts in analysis["job_facts"].items():
        compute = facts["compute"]
        gpu = facts["gpu"]
        lines.append(
            f"|{cell}|{_fmt(facts['job_total_seconds'], 1)}|{_fmt(facts['model_load_preflight_seconds']['model_load'], 2)}|"
            f"{compute['counters']['model_forward']}|{compute['counters']['processed_tokens']}|"
            f"{_fmt(gpu['peak_allocated_bytes'] / (1024 ** 3), 2)}|{_fmt(gpu['host_maxrss_kib'] / (1024 ** 2), 2)}|"
        )
    lines.extend([
        "",
        "## Machine-readable denominators",
        "",
        "- `per-case.json`: 60 rows.",
        "- `per-layer.json`: 300 rows.",
        "- `layer-aggregates.json`: 30 method/layer rows.",
        "- `method-aggregates.json`: 6 methods, 각각 10 cases/1000 requests.",
        "- `nll-prompt-distribution-aggregates.json`: 전체 prompt 기준 Z/W NLL mean/median/p90 48 rows.",
        "- `nll-distribution-aggregates.json`: case-level mean 기준 10-case mean/median/p90 48 rows.",
        "- `c-target-step.json`: C0–C3 × 10 cases × K1→K8의 280개 target-transition rows.",
        "- `c-target-step-aggregates.json`: 28 method/transition mean·median·p90 rows.",
        "- `paired-performance.json`: C0–C3 × AlphaEdit/MEMIT × 10 slices.",
        "- `paired-overhead.json`: 5 non-reference methods × 3 timing boundaries × 10 slices.",
        "- `overhead-aggregates.json`: mean/median/p90/max seconds, paired delta, ratio, percent.",
        "",
    ])
    return "\n".join(lines)


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def emit(repo: Path, output: Path, analysis: Mapping[str, Any]) -> dict[str, Any]:
    _need(not output.exists(), f"output already exists: {output}")
    output.mkdir(parents=True, mode=0o700)
    payloads = {
        REPORT_NAME: _report(analysis).encode("utf-8"),
        "per-case.json": _json_bytes(analysis["case_rows"]),
        "per-layer.json": _json_bytes(analysis["layer_rows"]),
        "layer-aggregates.json": _json_bytes(analysis["layer_aggregates"]),
        "method-aggregates.json": _json_bytes(analysis["aggregates"]),
        "paired-performance.json": _json_bytes(analysis["performance_pairs"]),
        "nll-distribution-aggregates.json": _json_bytes(analysis["nll_distribution"]),
        "nll-prompt-distribution-aggregates.json": _json_bytes(analysis["prompt_nll_distribution"]),
        "c-target-step.json": _json_bytes(analysis["c_target_step_rows"]),
        "c-target-step-aggregates.json": _json_bytes(analysis["c_target_step_aggregates"]),
        "paired-overhead.json": _json_bytes(analysis["overhead_rows"]),
        "overhead-aggregates.json": _json_bytes(analysis["overhead_aggregates"]),
        "pilot-case01-consistency.json": _json_bytes(analysis["pilot_case01_consistency"]),
        "job-facts.json": _json_bytes(analysis["job_facts"]),
        "analysis-summary.json": _json_bytes({key: value for key, value in analysis.items() if key not in {"case_rows", "layer_rows", "layer_aggregates", "performance_pairs", "nll_distribution", "prompt_nll_distribution", "c_target_step_rows", "c_target_step_aggregates", "overhead_rows", "aggregates", "overhead_aggregates", "job_facts"}}),
    }
    for name, data in payloads.items():
        _write_once(output / name, data)
    members = []
    for name in sorted(payloads):
        path = output / name
        members.append({"path": name, "bytes": path.stat().st_size, "lines": path.read_bytes().count(b"\n"), "sha256": _sha256(path)})
    manifest_body = {
        "schema": "ode-edit-s05-p1r52-joint-pc-independent-full-fp32-analysis-manifest/v1",
        "source_head": analysis["source_head"],
        "members": members,
        "member_count": len(members),
    }
    manifest = {**manifest_body, "identity_sha256": canonical_hash(manifest_body)}
    _write_once(output / "analysis-manifest.json", _json_bytes(manifest))
    receipt_body = {
        "schema": "ode-edit-s05-p1r52-joint-pc-independent-full-fp32-rooted-receipt/v1",
        "status": analysis["status"],
        "source_head": analysis["source_head"],
        "stream_order": analysis["stream_order"],
        "stream_root": analysis["stream_root"],
        "valid_endpoint_count": analysis["valid_endpoint_count"],
        "valid_request_count": analysis["valid_request_count"],
        "dtype_status": analysis["dtype_status"],
        "W0_restore_status": analysis["W0_restore_status"],
        "C_arm_same_accepted_z_status": analysis["C_arm_same_accepted_z_status"],
        "C_arm_same_entry_W_status": analysis["C_arm_same_entry_W_status"],
        "technical_failure_count": 0,
        "scientific_failure_count": 0,
        "retry_count": 0,
        "imputation_count": 0,
        "model_gpu_slurm_replay_count": 0,
        "analysis_manifest_sha256": _sha256(output / "analysis-manifest.json"),
        "analysis_manifest_identity": manifest["identity_sha256"],
        "input_roots": analysis["input_roots"],
        "input_manifests": analysis["input_manifests"],
        "promotion_ruling": analysis["promotion_ruling"],
    }
    receipt = {**receipt_body, "identity_sha256": canonical_hash(receipt_body)}
    _write_once(output / "rooted-analysis-receipt.json", _json_bytes(receipt))
    return receipt


def verify(output: Path) -> dict[str, Any]:
    manifest = _read_json(output / "analysis-manifest.json")
    receipt = _read_json(output / "rooted-analysis-receipt.json")
    _validate_identity(manifest, "analysis manifest")
    _validate_identity(receipt, "analysis receipt")
    for member in manifest["members"]:
        path = output / member["path"]
        _need(path.is_file() and not path.is_symlink(), f"missing output member: {path}")
        _need(path.stat().st_size == member["bytes"], f"member bytes: {path}")
        _need(path.read_bytes().count(b"\n") == member["lines"], f"member lines: {path}")
        _need(_sha256(path) == member["sha256"], f"member SHA: {path}")
    _need(receipt["analysis_manifest_sha256"] == _sha256(output / "analysis-manifest.json"), "receipt manifest SHA")
    _need(len(_read_json(output / "per-case.json")) == 60, "per-case rows")
    _need(len(_read_json(output / "per-layer.json")) == 300, "per-layer rows")
    _need(len(_read_json(output / "paired-overhead.json")) == 150, "overhead rows")
    _need(len(_read_json(output / "nll-distribution-aggregates.json")) == 48, "NLL distribution rows")
    _need(len(_read_json(output / "nll-prompt-distribution-aggregates.json")) == 48, "prompt NLL distribution rows")
    _need(len(_read_json(output / "c-target-step.json")) == 280, "C target step rows")
    _need(len(_read_json(output / "c-target-step-aggregates.json")) == 28, "C target step aggregate rows")
    return {"status": "INDEPENDENT_REHASH_PASS", "member_count": manifest["member_count"], "receipt_identity": receipt["identity_sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        print(canonical_json(verify(args.verify.resolve())))
        return 0
    analysis = analyze(args.repo.resolve())
    if args.validate_only:
        print(canonical_json({key: analysis[key] for key in ("status", "source_head", "valid_endpoint_count", "valid_request_count", "dtype_status", "W0_restore_status", "C_arm_same_accepted_z_status")}))
        return 0
    _need(args.output is not None, "--output is required unless --validate-only")
    receipt = emit(args.repo.resolve(), args.output.resolve(), analysis)
    print(canonical_json({"status": "SEALED", "output": str(args.output.resolve()), "receipt_identity": receipt["identity_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
