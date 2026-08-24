#!/usr/bin/env python3
"""Fail-closed endpoint aggregation and optional sequential pairing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.contracts import ODEBFStateError, canonical_hash
from project.run_scripts.ode_bf.p1_runtime import _atomic_write_once
from project.run_scripts.ode_bf.p1r54_fz_c3_independent import (
    INSTRUCTION_ID,
    STREAM_ORDER,
    STREAM_ROOT,
    cell_config,
)


def _verified_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ODEBFStateError(f"regular rooted JSON required: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ODEBFStateError(f"JSON object required: {path}")
    claimed = value.get("identity_sha256")
    payload = dict(value)
    payload.pop("identity_sha256", None)
    if claimed != canonical_hash(payload):
        raise ODEBFStateError(f"rooted JSON identity differs: {path}")
    return value


def _flatten(values: Any) -> list[float]:
    if isinstance(values, bool):
        raise ODEBFStateError("boolean entered numeric endpoint distribution")
    if isinstance(values, (int, float)):
        value = float(values)
        if not math.isfinite(value):
            raise ODEBFStateError("nonfinite endpoint value")
        return [value]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return [item for nested in values for item in _flatten(nested)]
    raise ODEBFStateError("endpoint distribution shape differs")


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    observed = sorted(float(value) for value in values)
    if not observed or not all(math.isfinite(value) for value in observed):
        raise ODEBFStateError("finite nonempty distribution required")
    p90_index = max(0, math.ceil(0.9 * len(observed)) - 1)
    return {
        "count": len(observed),
        "mean": statistics.fmean(observed),
        "median": statistics.median(observed),
        "p90_nearest_rank": observed[p90_index],
        "max": observed[-1],
        "nonfinite_count": 0,
    }


def _metric_rates(summary: Mapping[str, Any]) -> dict[str, Any]:
    pairs = {
        "Eff": ("rewrite_success_numerator", "rewrite_success_denominator"),
        "Gen": ("rephrase_success_numerator", "rephrase_success_denominator"),
        "Loc": ("locality_numerator", "locality_denominator"),
    }
    result = {}
    for label, (numerator_key, denominator_key) in pairs.items():
        numerator = int(summary[numerator_key])
        denominator = int(summary[denominator_key])
        if denominator <= 0 or not 0 <= numerator <= denominator:
            raise ODEBFStateError("endpoint Eff/Gen/Loc denominator differs")
        result[label] = {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator,
        }
    return result


def _surface(terminal: Mapping[str, Any], surface: str) -> Mapping[str, Any]:
    metrics = terminal["kstep_executions"][7]["metrics"]
    if surface == "accepted_z":
        return metrics["accepted_z"]
    if surface == "post_W":
        return metrics["post_writer_W"]
    raise ODEBFStateError("endpoint surface differs")


def _nll_values(endpoint: Mapping[str, Any], prompt: str) -> list[float]:
    score_key = "rewrite_success" if prompt == "rewrite" else "rephrase_success"
    return _flatten(endpoint["scores"][score_key]["target_new_nll_by_request"])


def _aggregate_surface(
    terminals: Sequence[Mapping[str, Any]], surface: str
) -> dict[str, Any]:
    endpoints = [_surface(terminal, surface) for terminal in terminals]
    rates = [_metric_rates(endpoint["summary"]) for endpoint in endpoints]
    aggregate_rates = {}
    for label in ("Eff", "Gen", "Loc"):
        numerator = sum(int(row[label]["numerator"]) for row in rates)
        denominator = sum(int(row[label]["denominator"]) for row in rates)
        aggregate_rates[label] = {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator,
        }
    return {
        "rewrite_target_new_nll": _distribution(
            item
            for endpoint in endpoints
            for item in _nll_values(endpoint, "rewrite")
        ),
        "rephrase_target_new_nll": _distribution(
            item
            for endpoint in endpoints
            for item in _nll_values(endpoint, "rephrase")
        ),
        "Eff_Gen_Loc": aggregate_rates,
        "by_batch": [
            {
                "batch_index": index,
                "rewrite_target_new_nll": _distribution(
                    _nll_values(endpoint, "rewrite")
                ),
                "rephrase_target_new_nll": _distribution(
                    _nll_values(endpoint, "rephrase")
                ),
                "Eff_Gen_Loc": rate,
            }
            for index, (endpoint, rate) in enumerate(zip(endpoints, rates, strict=True), start=1)
        ],
    }


def _writer_telemetry(terminals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    writer_wall = []
    allocation_energy = []
    clamp_hits = 0
    layer_cache_load = []
    k_telemetry = []
    for batch_index, terminal in enumerate(terminals, start=1):
        per_k = []
        for k_index, (execution, microstep) in enumerate(
            zip(terminal["kstep_executions"], terminal["microstep_trajectory"], strict=True),
            start=1,
        ):
            writer = execution["writer_receipt"]["writer"]
            dynamic = writer["alphaedit_dynamic_cache_contract"]
            field = microstep["field_receipt"]
            wall = float(writer["edit_core_wall_seconds"])
            energy = float(field["allocation_energy"])
            writer_wall.append(wall)
            allocation_energy.append(energy)
            clamp_hits += int(field["clamp_hit_count"])
            loads = [float(value) for value in dynamic["exit"]["layer_frobenius_norm"]]
            total = sum(loads)
            shares = [value / total if total > 0.0 else 0.0 for value in loads]
            layer_cache_load.extend(loads)
            per_k.append(
                {
                    "K": k_index,
                    "selected_target_sha256": execution["selected_target_sha256"],
                    "allocation_energy": energy,
                    "allocation_entropy": float(field["allocation_entropy"]),
                    "top1_allocation_share": float(field["top1_allocation_share"]),
                    "top3_allocation_share": float(field["top3_allocation_share"]),
                    "clamp_hit_count": int(field["clamp_hit_count"]),
                    "cache_layer_load": loads,
                    "cache_layer_load_share": shares,
                    "writer_wall_seconds": wall,
                }
            )
        k_telemetry.append({"batch_index": batch_index, "K1_to_K8": per_k})
    return {
        "official_writer_call_count": len(writer_wall),
        "writer_layer_apply_count": len(writer_wall) * 5,
        "writer_core_wall_seconds": _distribution(writer_wall),
        "target_allocation_energy": _distribution(allocation_energy),
        "cache_layer_load_frobenius_norm": _distribution(layer_cache_load),
        "clamp_hit_count": clamp_hits,
        "K1_to_K8_target_writer_telemetry": k_telemetry,
    }


def _paired_sequential(
    terminals: Sequence[Mapping[str, Any]], sequential_root: Path
) -> Mapping[str, Any]:
    sequential_terminal = _verified_json(sequential_root / "terminal.json")
    final_w10 = _verified_json(sequential_root / "raw/final-w10.json")
    if (
        sequential_terminal.get("status") != "TERMINAL_VALID"
        or sequential_terminal.get("stream_root") != STREAM_ROOT
        or sequential_terminal.get("stream_order") != STREAM_ORDER
        or sequential_terminal.get("completed_batch_count") != 10
    ):
        raise ODEBFStateError("paired sequential terminal differs")
    rows = []
    for batch_index, independent in enumerate(terminals, start=1):
        sequential_batch = _verified_json(
            sequential_root / f"raw/batches/b{batch_index:02d}/terminal.json"
        )
        independent_summary = _surface(independent, "post_W")["summary"]
        sequential_summary = sequential_batch["immediate_post_W"]["summary"]
        final_summary = final_w10["cohorts"][batch_index - 1]["summary"]
        keys = (
            "rewrite_target_new_nll_mean",
            "rephrase_target_new_nll_mean",
        )
        rows.append(
            {
                "batch_index": batch_index,
                "independent_minus_sequential_immediate": {
                    key: float(independent_summary[key]) - float(sequential_summary[key])
                    for key in keys
                },
                "sequential_final_w10_minus_immediate": {
                    key: float(final_summary[key]) - float(sequential_summary[key])
                    for key in keys
                },
                "sequential_final_w10_minus_independent": {
                    key: float(final_summary[key]) - float(independent_summary[key])
                    for key in keys
                },
            }
        )
    return {
        "status": "PAIRED_B1_B10_COMPLETE",
        "interpretation_boundary": (
            "same sealed stream; raw arms unchanged; differences jointly expose "
            "accumulated W/cache continuity and final-stream forgetting"
        ),
        "sequential_terminal_sha256": sha256_file(sequential_root / "terminal.json"),
        "rows": rows,
    }


def build_analysis(
    *, independent_results_root: Path, sequential_root: Path | None = None
) -> Mapping[str, Any]:
    terminals = []
    terminal_files = []
    for cell in range(10):
        config = cell_config(cell)
        path = independent_results_root / config.result_name / "terminal.json"
        terminal = _verified_json(path)
        if (
            terminal.get("status") != "TERMINAL_VALID"
            or terminal.get("role") != config.role
            or terminal.get("canonical_batch_index") != config.batch_index
            or terminal.get("request_count") != 100
            or terminal.get("request_order_sha256")
            != terminal.get("canonical_batch_order_sha256")
            or terminal.get("stream_root") != STREAM_ROOT
            or terminal.get("stream_order") != STREAM_ORDER
            or terminal.get("W0_restored") is not True
            or terminal.get("target_field_evaluation_count") != 8
            or terminal.get("writer_call_count") != 8
            or terminal.get("writer_layer_apply_count") != 40
            or terminal.get("alpha_cache", {}).get("entry_width") != 0
            or terminal.get("alpha_cache", {}).get("exit_width") != 100
            or terminal.get("alpha_cache", {}).get("append_count") != 1
            or terminal.get(
                "cross_batch_W_cache_history_factor_materializer_consumption_count"
            )
            != 0
            or terminal.get("scientific_promotion") is not False
        ):
            raise ODEBFStateError(f"independent B{config.batch_index} terminal differs")
        terminals.append(terminal)
        terminal_files.append(
            {"batch_index": config.batch_index, "path": str(path), "sha256": sha256_file(path)}
        )
    accepted_z = _aggregate_surface(terminals, "accepted_z")
    post_w = _aggregate_surface(terminals, "post_W")
    gaps = {}
    for prompt in ("rewrite", "rephrase"):
        z_values = [
            item
            for terminal in terminals
            for item in _nll_values(_surface(terminal, "accepted_z"), prompt)
        ]
        w_values = [
            item
            for terminal in terminals
            for item in _nll_values(_surface(terminal, "post_W"), prompt)
        ]
        if len(z_values) != len(w_values):
            raise ODEBFStateError("z to W paired NLL shape differs")
        gaps[f"{prompt}_post_W_minus_accepted_z_nll"] = _distribution(
            right - left for left, right in zip(z_values, w_values, strict=True)
        )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-c3-independent-analysis/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_ANALYSIS_VALID",
        "endpoint_count": len(terminals),
        "request_count": len(terminals) * 100,
        "completeness": "10_OF_10_ENDPOINTS_1000_OF_1000_REQUESTS",
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "terminal_files": terminal_files,
        "accepted_z": accepted_z,
        "post_W": post_w,
        "z_to_W_gaps": gaps,
        "writer_cache_target_telemetry": _writer_telemetry(terminals),
        "cell_wall_seconds": _distribution(
            float(terminal["runtime_after_model_preflight_seconds"])
            for terminal in terminals
        ),
        "cache_W0_integrity": {
            "cold_entry_widths": [terminal["alpha_cache"]["entry_width"] for terminal in terminals],
            "cell_local_exit_widths": [terminal["alpha_cache"]["exit_width"] for terminal in terminals],
            "W0_restored_count": sum(terminal["W0_restored"] is True for terminal in terminals),
            "cross_batch_state_consumption_count": 0,
        },
        "paired_sequential": (
            {"status": "DEFERRED_UNTIL_SEQUENTIAL_TERMINAL"}
            if sequential_root is None
            else _paired_sequential(terminals, sequential_root)
        ),
        "nonfinite_count": 0,
        "raw_result_mutation_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--independent-results-root", required=True, type=Path)
    parser.add_argument("--sequential-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    analysis = build_analysis(
        independent_results_root=args.independent_results_root,
        sequential_root=args.sequential_root,
    )
    output_sha = _atomic_write_once(args.output, analysis)
    print(json.dumps({"status": analysis["status"], "output_sha256": output_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
