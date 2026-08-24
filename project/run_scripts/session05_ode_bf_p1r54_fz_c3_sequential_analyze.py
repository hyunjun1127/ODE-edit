#!/usr/bin/env python3
"""Fail-closed P1R54 FZ sequential endpoint and forgetting analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.contracts import ODEBFStateError, canonical_hash
from project.run_scripts.ode_bf.p1_runtime import _atomic_write_once
from project.run_scripts.ode_bf.p1r54_fz_c3_sequential import (
    INSTRUCTION_ID,
    ROLE,
    STREAM_ORDER,
    STREAM_ROOT,
)
from project.run_scripts.session05_ode_bf_p1r54_fz_c3_independent_analyze import (
    _aggregate_surface,
    _distribution,
    _flatten,
    _metric_rates,
    _nll_values,
    _surface,
    _verified_json,
    _writer_telemetry,
)


def _aggregate_endpoints(endpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(endpoints) != 10:
        raise ODEBFStateError("sequential final W10 cohort count differs")
    rates = [_metric_rates(endpoint["summary"]) for endpoint in endpoints]
    aggregate_rates: dict[str, Any] = {}
    for label in rates[0]:
        numerator = sum(int(row[label]["numerator"]) for row in rates)
        denominator = sum(int(row[label]["denominator"]) for row in rates)
        aggregate_rates[label] = {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator,
        }
    payload: dict[str, Any] = {"performance": aggregate_rates}
    for prompt in ("rewrite", "rephrase"):
        for target in ("new", "true"):
            payload[f"{prompt}_target_{target}_nll"] = _distribution(
                item
                for endpoint in endpoints
                for item in _nll_values(endpoint, prompt, target=target)
            )
    payload["by_batch"] = [
        {
            "batch_index": index,
            "performance": rate,
            **{
                f"{prompt}_target_{target}_nll": _distribution(
                    _nll_values(endpoint, prompt, target=target)
                )
                for prompt in ("rewrite", "rephrase")
                for target in ("new", "true")
            },
        }
        for index, (endpoint, rate) in enumerate(zip(endpoints, rates, strict=True), start=1)
    ]
    return payload


def _prompt_bits(endpoint: Mapping[str, Any], prompt: str) -> list[bool]:
    score_key = "rewrite_success" if prompt == "rewrite" else "rephrase_success"
    values = endpoint["scores"][score_key]["per_request_bits"]
    result: list[bool] = []
    for request in values:
        if not isinstance(request, Sequence) or isinstance(request, (str, bytes)):
            raise ODEBFStateError("per-request prompt bits differ")
        result.extend(bool(item) for item in request)
    return result


def _strict_bits(endpoint: Mapping[str, Any], prompt: str) -> list[bool]:
    score_key = "rewrite_success" if prompt == "rewrite" else "rephrase_success"
    return [bool(item) for item in endpoint["scores"][score_key]["strict_all_prompt_bits"]]


def _paired_failure_count(
    left: Mapping[str, Any], right: Mapping[str, Any], prompt: str, *, strict: bool
) -> int:
    left_bits = _strict_bits(left, prompt) if strict else _prompt_bits(left, prompt)
    right_bits = _strict_bits(right, prompt) if strict else _prompt_bits(right, prompt)
    if len(left_bits) != len(right_bits):
        raise ODEBFStateError("paired success bit denominator differs")
    return sum(a and not b for a, b in zip(left_bits, right_bits, strict=True))


def _paired_forgetting(
    batch_terminals: Sequence[Mapping[str, Any]],
    final_endpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    pooled: dict[str, list[float]] = {
        f"{prompt}_final_W10_minus_immediate_W_target_{target}_nll": []
        for prompt in ("rewrite", "rephrase")
        for target in ("new", "true")
    }
    for batch_index, (terminal, final_endpoint) in enumerate(
        zip(batch_terminals, final_endpoints, strict=True), start=1
    ):
        immediate = terminal["immediate_post_W"]
        row: dict[str, Any] = {"batch_index": batch_index}
        for prompt in ("rewrite", "rephrase"):
            for target in ("new", "true"):
                left = _nll_values(immediate, prompt, target=target)
                right = _nll_values(final_endpoint, prompt, target=target)
                if len(left) != len(right):
                    raise ODEBFStateError("forgetting request denominator differs")
                values = [b - a for a, b in zip(left, right, strict=True)]
                key = f"{prompt}_final_W10_minus_immediate_W_target_{target}_nll"
                row[key] = _distribution(values)
                pooled[key].extend(values)
            row[f"{prompt}_immediate_success_to_final_failure"] = _paired_failure_count(
                immediate, final_endpoint, prompt, strict=False
            )
            row[f"{prompt}_immediate_strict_success_to_final_failure"] = (
                _paired_failure_count(immediate, final_endpoint, prompt, strict=True)
            )
        immediate_rates = _metric_rates(immediate["summary"])
        final_rates = _metric_rates(final_endpoint["summary"])
        row["final_minus_immediate_rate"] = {
            label: final_rates[label]["rate"] - immediate_rates[label]["rate"]
            for label in immediate_rates
        }
        rows.append(row)
    return {
        "rows": rows,
        "pooled_request_distributions": {
            key: _distribution(values) for key, values in pooled.items()
        },
    }


def _z_to_w(
    batch_terminals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for prompt in ("rewrite", "rephrase"):
        for target in ("new", "true"):
            z_values = [
                item
                for terminal in batch_terminals
                for item in _nll_values(_surface(terminal, "accepted_z"), prompt, target=target)
            ]
            w_values = [
                item
                for terminal in batch_terminals
                for item in _nll_values(_surface(terminal, "post_W"), prompt, target=target)
            ]
            if len(z_values) != len(w_values):
                raise ODEBFStateError("sequential z to W request denominator differs")
            payload[f"{prompt}_post_W_minus_accepted_z_target_{target}_nll"] = (
                _distribution(b - a for a, b in zip(z_values, w_values, strict=True))
            )
    for prompt in ("rewrite", "rephrase"):
        payload[f"{prompt}_z_success_to_W_failure"] = sum(
            _paired_failure_count(
                _surface(terminal, "accepted_z"),
                _surface(terminal, "post_W"),
                prompt,
                strict=False,
            )
            for terminal in batch_terminals
        )
        payload[f"{prompt}_z_strict_success_to_W_failure"] = sum(
            _paired_failure_count(
                _surface(terminal, "accepted_z"),
                _surface(terminal, "post_W"),
                prompt,
                strict=True,
            )
            for terminal in batch_terminals
        )
    return payload


def _target_telemetry(batch_terminals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for batch_index, terminal in enumerate(batch_terminals, start=1):
        for microstep in terminal["microstep_trajectory"]:
            field = microstep["field_receipt"]
            shares = sorted((float(value) for value in field["allocation_energy_share_by_request"]), reverse=True)
            if len(shares) != 100 or not math.isclose(sum(shares), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ODEBFStateError("FZ allocation share denominator differs")
            rows.append(
                {
                    "batch_index": batch_index,
                    "K": int(microstep["outer_step_index"]) + 1,
                    "rho": _distribution(_flatten(field["p1r54_rho_by_request"])),
                    "velocity": _distribution(_flatten(field["p1r54_velocity_norm_by_request"])),
                    "target_new_nll_current": _distribution(_flatten(field["target_new_nll_current_by_request"])),
                    "clamp_hit_count": int(field["clamp_hit_count"]),
                    "clamp_denominator": 100,
                    "top1_energy_share": shares[0],
                    "top10_energy_share": sum(shares[:10]),
                    "bottom90_energy_share": sum(shares[10:]),
                    "allocation_entropy": float(field["allocation_entropy"]),
                    "allocation_energy": float(field["allocation_energy"]),
                    "selection_counts": {
                        label: sum(value == label for value in microstep["selection_by_request"])
                        for label in ("PRIMARY", "RESCUE", "CURRENT")
                    },
                    "forbidden_decision_access_count": sum(
                        int(field[key])
                        for key in field
                        if key.startswith("p1r54_") and key.endswith("decision_access_count")
                    ),
                    "added_forward_backward_count": int(field["p1r54_added_model_forward_count"])
                    + int(field["p1r54_added_backward_count"]),
                }
            )
    if len(rows) != 80:
        raise ODEBFStateError("sequential target telemetry row count differs")
    return {
        "rows": rows,
        "clamp_hit_numerator": sum(row["clamp_hit_count"] for row in rows),
        "clamp_hit_denominator": 8000,
        "forbidden_decision_access_count": sum(
            row["forbidden_decision_access_count"] for row in rows
        ),
        "additional_forward_backward_count": sum(
            row["added_forward_backward_count"] for row in rows
        ),
        "by_K": [
            {
                "K": k,
                "clamp_hit_numerator": sum(row["clamp_hit_count"] for row in rows if row["K"] == k),
                "clamp_hit_denominator": 1000,
                "top1_energy_share": _distribution(row["top1_energy_share"] for row in rows if row["K"] == k),
                "top10_energy_share": _distribution(row["top10_energy_share"] for row in rows if row["K"] == k),
                "bottom90_energy_share": _distribution(row["bottom90_energy_share"] for row in rows if row["K"] == k),
                "allocation_energy": _distribution(row["allocation_energy"] for row in rows if row["K"] == k),
            }
            for k in range(1, 9)
        ],
    }


def build_analysis(results_root: Path) -> Mapping[str, Any]:
    terminal = _verified_json(results_root / "terminal.json")
    manifest = _verified_json(results_root / "manifest.json")
    final_w10 = _verified_json(results_root / "raw/final-w10.json")
    if (
        terminal.get("status") != "TERMINAL_VALID"
        or terminal.get("instruction_id") != INSTRUCTION_ID
        or terminal.get("role") != ROLE
        or terminal.get("stream_root") != STREAM_ROOT
        or terminal.get("stream_order") != STREAM_ORDER
        or terminal.get("completed_batch_count") != 10
        or terminal.get("valid_request_count") != 1000
        or terminal.get("K_writer_call_count") != 80
        or terminal.get("target_field_evaluation_count") != 80
        or terminal.get("writer_layer_apply_count") != 400
        or terminal.get("W0_restored") is not True
        or terminal.get("technical_failure_count") != 0
        or terminal.get("scientific_failure_count") != 0
        or terminal.get("scientific_promotion") is not False
    ):
        raise ODEBFStateError("P1R54 FZ sequential terminal differs")
    if (
        manifest.get("terminal_sha256") != sha256_file(results_root / "terminal.json")
        or manifest.get("final_w10_sha256") != sha256_file(results_root / "raw/final-w10.json")
        or final_w10.get("cohort_count") != 10
        or final_w10.get("request_count") != 1000
    ):
        raise ODEBFStateError("P1R54 FZ sequential manifest/final differs")

    batch_terminals = []
    batch_files = []
    previous_commit = None
    for batch_index in range(1, 11):
        path = results_root / f"raw/batches/b{batch_index:02d}/terminal.json"
        batch = _verified_json(path)
        cache = batch["alpha_cache"]
        if (
            batch.get("batch_index") != batch_index
            or batch.get("request_count") != 100
            or batch.get("stream_root") != STREAM_ROOT
            or batch.get("stream_order") != STREAM_ORDER
            or batch.get("target_field_evaluation_count") != 8
            or batch.get("K_writer_call_count") != 8
            or batch.get("writer_layer_apply_count") != 40
            or cache.get("entry_width") != (batch_index - 1) * 100
            or cache.get("exit_width") != batch_index * 100
            or cache.get("append_count") != 1
            or cache.get("consume_count") != 8
            or cache.get("current_batch_k_key_history_inclusion_count") != 0
            or batch.get("rollback_count") != 0
            or batch.get("retry_count") != 0
            or batch.get("imputation_count") != 0
        ):
            raise ODEBFStateError(f"P1R54 FZ sequential B{batch_index} differs")
        if previous_commit is not None and batch.get("entry_weight_sha256") != previous_commit:
            raise ODEBFStateError("sequential commit to next entry W chain differs")
        previous_commit = batch.get("commit_weight_sha256")
        batch_terminals.append(batch)
        batch_files.append(
            {"batch_index": batch_index, "path": str(path), "sha256": sha256_file(path)}
        )
    if terminal.get("batch_terminal_sha256") != [row["sha256"] for row in batch_files]:
        raise ODEBFStateError("sequential terminal batch SHA list differs")

    accepted_z = _aggregate_surface(batch_terminals, "accepted_z")
    immediate_w = _aggregate_surface(batch_terminals, "post_W")
    final_endpoints = final_w10["cohorts"]
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-c3-sequential-analysis/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_ANALYSIS_VALID",
        "source_head": terminal["source_head"],
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "completeness": "10_OF_10_BATCHES_1000_OF_1000_REQUESTS",
        "terminal_sha256": sha256_file(results_root / "terminal.json"),
        "manifest_sha256": sha256_file(results_root / "manifest.json"),
        "final_w10_sha256": sha256_file(results_root / "raw/final-w10.json"),
        "batch_terminal_files": batch_files,
        "accepted_z_immediate": accepted_z,
        "post_W_immediate": immediate_w,
        "final_W10": _aggregate_endpoints(final_endpoints),
        "z_to_W_immediate": _z_to_w(batch_terminals),
        "forgetting_immediate_to_final_W10": _paired_forgetting(
            batch_terminals, final_endpoints
        ),
        "target_telemetry": _target_telemetry(batch_terminals),
        "writer_cache_telemetry": _writer_telemetry(batch_terminals),
        "batch_wall_seconds": _distribution(
            float(batch["batch_total_seconds"]) for batch in batch_terminals
        ),
        "job_runtime_after_model_preflight_seconds": float(
            terminal["runtime_after_model_preflight_seconds"]
        ),
        "job_compute": terminal["job_compute"],
        "integrity": {
            "W0_restored": True,
            "batch_commit_to_next_entry_chain": True,
            "cache_entry_widths": list(range(0, 1000, 100)),
            "cache_exit_widths": list(range(100, 1100, 100)),
            "cache_append_count": 10,
            "cache_consume_count": 80,
            "same_batch_current_key_history_inclusion_count": 0,
            "full_fp32": True,
            "nonfinite_count": 0,
            "technical_failure_count": 0,
            "scientific_failure_count": 0,
        },
        "raw_result_mutation_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    analysis = build_analysis(args.results_root)
    output_sha = _atomic_write_once(args.output, analysis)
    print(json.dumps({"status": analysis["status"], "output_sha256": output_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
