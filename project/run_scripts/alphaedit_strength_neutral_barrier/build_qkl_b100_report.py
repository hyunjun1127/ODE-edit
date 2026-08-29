#!/usr/bin/env python3
"""Build the canonical cache-aware q-KL atomic-B100 factual package.

The builder is analysis-only: it reads terminal result JSON files, never loads
a model, and publishes a create-once package.  PRE-EDIT, stock N=1, projected
N=2/N=4, and implementation-only split controls remain distinct by design.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from project.run_scripts.alphaedit_strength_neutral_barrier.build_atomic_report import (
    assert_safe_absent_directory,
    canonical_json_bytes,
    correctness,
    file_identity,
    preference_success,
    publish,
    stats,
    strict_preference,
    strict_rephrase,
    write_json,
)


MODEL_ORDER = ("llama", "qwen")
CELL_METHOD = {
    0: ("BASELINE", "OFFICIAL_ALPHAEDIT_N1"),
    1: ("IMPLEMENTATION_CONTROL", "STATIC_SPLIT_OFF_N2"),
    2: ("IMPLEMENTATION_CONTROL", "STATIC_SPLIT_OFF_N4"),
    3: ("OURS", "QKL_PROJECTED_ODE_N2"),
    4: ("OURS", "QKL_PROJECTED_ODE_N4"),
}
METRICS = (
    "rewrite_target_new",
    "rewrite_target_true",
    "rephrase_target_new",
    "rephrase_target_true",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _package_identity(path: Path) -> dict[str, Any]:
    identity = file_identity(path)
    identity["path"] = path.name
    return identity


def _same_json(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def summarize_endpoint(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRICS:
        rows = endpoint[metric]
        summary[metric] = {
            "nll": stats(float(row["nll"]) for row in rows),
            "correctness": correctness(rows),
        }
    rewrite_preference = preference_success(
        endpoint["rewrite_target_new"], endpoint["rewrite_target_true"]
    )
    rephrase_preference = preference_success(
        endpoint["rephrase_target_new"], endpoint["rephrase_target_true"]
    )
    summary["rewrite_preference"] = {
        key: value for key, value in rewrite_preference.items() if key != "by_key"
    }
    summary["rephrase_preference"] = {
        key: value for key, value in rephrase_preference.items() if key != "by_key"
    }
    summary["rephrase_strict_exact"] = strict_rephrase(
        endpoint["rephrase_target_new"]
    )
    summary["rephrase_strict_preference"] = strict_preference(rephrase_preference)
    return summary


def summarize_geometry(writer: Mapping[str, Any]) -> dict[str, Any]:
    layers = writer.get("layers", [])
    nodes = [node for layer in layers for node in layer.get("nodes", [])]
    if not nodes:
        return {
            "availability": "NOT_RECORDED",
            "node_count": 0,
            "active_node_count": None,
            "active_node_fraction": None,
            "terminal_qkl_last_layer": None,
            "terminal_qkl_layers": None,
            "native_predictive_qkl": None,
            "post_projected_qkl": None,
            "removed_energy_fraction": None,
            "positive_projected_rate_violation_max": None,
        }
    terminal = [
        float(layer["terminal_barrier_q_kl"])
        for layer in layers
        if layer.get("terminal_barrier_q_kl") is not None
    ]
    active = sum(bool(node.get("correction_active", False)) for node in nodes)
    return {
        "availability": "RECORDED",
        "node_count": len(nodes),
        "active_node_count": active,
        "active_node_fraction": active / len(nodes),
        "terminal_qkl_last_layer": terminal[-1] if terminal else None,
        "terminal_qkl_layers": stats(terminal),
        "native_predictive_qkl": stats(
            float(node["native_lookahead"]["aggregate_q_kl"]) for node in nodes
        ),
        "post_projected_qkl": stats(
            float(node["post_projected"]["aggregate_q_kl"]) for node in nodes
        ),
        "removed_energy_fraction": stats(
            float(node.get("removed_energy_fraction", 0.0)) for node in nodes
        ),
        "positive_projected_rate_violation_max": max(
            float(node.get("positive_projected_rate_violation", 0.0))
            for node in nodes
        ),
    }


def _load_model(root: Path, model: str) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    inputs: list[dict[str, Any]] = []
    for cell in range(5):
        path = root / f"cell-{cell}" / "result.json"
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing regular result: {path}")
        result = _read_json(path)
        if result.get("terminal_status") != "TECHNICAL_PASS":
            raise RuntimeError(f"cell {cell} is not terminal-valid")
        if result.get("stage") != "atomic-b100":
            raise RuntimeError(f"cell {cell} is not atomic-b100")
        if result.get("cell") != cell or result.get("case_ids") != list(range(100)):
            raise RuntimeError(f"cell {cell} request/order mismatch")
        if result.get("request_count") != 100 or result.get("batch_count") != 1:
            raise RuntimeError(f"cell {cell} denominator mismatch")
        if not result.get("w0_pointer_bytes_restore_pass"):
            raise RuntimeError(f"cell {cell} W0 restore failed")
        if result.get("parameter_inventory", {}).get("non_fp32_parameter_count") != 0:
            raise RuntimeError(f"cell {cell} FULL-FP32 mismatch")
        if result.get("autocast_enabled") or result.get("tf32_enabled"):
            raise RuntimeError(f"cell {cell} precision closure mismatch")
        current = {
            "case_ids": result["case_ids"],
            "dataset": result["dataset"],
            "model": result["model"],
            "tokenizer": result["tokenizer"],
            "projector": result["projector"],
            "hparams": result["hparams"],
            "preflight_receipt": result["preflight_receipt"],
            "global_w0_sha256": result["global_w0_sha256"],
            "global_pre": result["global_pre"],
            "source": result["source"],
        }
        if reference is None:
            reference = current
        elif not _same_json(reference, current):
            raise RuntimeError(f"cell {cell} common input/W0/pre-edit mismatch")
        group, method = CELL_METHOD[cell]
        writer = result["batches"][0]["writer_receipt"]
        cells.append(
            {
                "cell": cell,
                "group": group,
                "method": method,
                "result": result,
                "writer": writer,
                "result_identity": file_identity(path),
            }
        )
        inputs.append(file_identity(path))
    assert reference is not None
    return {
        "model": model,
        "root": str(root),
        "reference": reference,
        "cells": cells,
        "input_files": inputs,
    }


def _method_record(model: str, cell: Mapping[str, Any]) -> dict[str, Any]:
    result = cell["result"]
    writer = cell["writer"]
    return {
        "model": model,
        "group": cell["group"],
        "method": cell["method"],
        "cell": cell["cell"],
        "endpoint": summarize_endpoint(result["final"]),
        "locality": result["final_locality_preservation"],
        "geometry": summarize_geometry(writer),
        "compute": {
            **result["timing"],
            **result["memory"],
            "writer_calls": result["action_counts"].get("writer_calls"),
            "predictor_forward_backward_count": writer.get(
                "predictor_forward_backward_count", 0
            ),
            "layer_factorization_count": writer.get("layer_factorization_count", 0),
            "cache_append_count": writer.get("cache_append_count"),
        },
        "integrity": {
            "full_fp32": result["parameter_inventory"]["non_fp32_parameter_count"]
            == 0,
            "quantized": result["parameter_inventory"]["quantized"],
            "w0_pointer_bytes_restore_pass": result["w0_pointer_bytes_restore_pass"],
            "writer_w0_restore_pass": writer.get("w0_restore_pass"),
            "authoritative_endpoint_adoption_pass": writer.get(
                "authoritative_endpoint_adoption_pass"
            ),
            "cache_append_count": writer.get("cache_append_count"),
            "fallback_count": writer.get("fallback_count", 0),
            "retry_count": writer.get("retry_count", 0),
            "imputation_count": writer.get("imputation_count", 0),
            "target_true_controller_influence_count": writer.get(
                "target_true_controller_influence_count", 0
            ),
            "rephrase_controller_influence_count": writer.get(
                "rephrase_controller_influence_count", 0
            ),
            "locality_controller_influence_count": writer.get(
                "locality_controller_influence_count", 0
            ),
            "failure_boundary_count": 0,
        },
        "result_identity": cell["result_identity"],
    }


def _pre_record(model_data: Mapping[str, Any]) -> dict[str, Any]:
    endpoint = model_data["reference"]["global_pre"]
    return {
        "model": model_data["model"],
        "group": "PRE_EDIT",
        "method": "PRE_EDIT_W0",
        "cell": None,
        "endpoint": summarize_endpoint(endpoint),
        "locality": {
            "numerator": len(endpoint["locality_target_true"]),
            "denominator": len(endpoint["locality_target_true"]),
            "rate": 1.0,
            "note": "REFERENCE_IDENTITY_NOT_AN_EDIT_RESULT",
        },
        "geometry": summarize_geometry({}),
        "compute": {
            "edit_core_seconds": None,
            "total_seconds": None,
            "peak_gpu_allocated_bytes": None,
            "peak_gpu_reserved_bytes": None,
            "writer_calls": 0,
            "predictor_forward_backward_count": 0,
            "layer_factorization_count": 0,
            "cache_append_count": 0,
        },
        "integrity": {
            "full_fp32": True,
            "quantized": False,
            "w0_pointer_bytes_restore_pass": True,
            "writer_w0_restore_pass": None,
            "authoritative_endpoint_adoption_pass": None,
            "cache_append_count": 0,
            "fallback_count": 0,
            "retry_count": 0,
            "imputation_count": 0,
            "target_true_controller_influence_count": 0,
            "rephrase_controller_influence_count": 0,
            "locality_controller_influence_count": 0,
            "failure_boundary_count": 0,
        },
        "result_identity": None,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NOT_RECORDED"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return f"{float(value):.{digits}f}"


def _headline_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        endpoint = record["endpoint"]
        rows.append(
            {
                "model": record["model"],
                "group": record["group"],
                "method": record["method"],
                "rewrite_new_nll_mean": endpoint["rewrite_target_new"]["nll"]["mean"],
                "rewrite_new_nll_median": endpoint["rewrite_target_new"]["nll"]["median"],
                "rewrite_new_nll_p90": endpoint["rewrite_target_new"]["nll"]["p90"],
                "rewrite_exact_success": endpoint["rewrite_target_new"]["correctness"]["prompt_rate"],
                "rewrite_preference_success": endpoint["rewrite_preference"]["rate"],
                "rephrase_new_nll_mean": endpoint["rephrase_target_new"]["nll"]["mean"],
                "rephrase_new_nll_median": endpoint["rephrase_target_new"]["nll"]["median"],
                "rephrase_new_nll_p90": endpoint["rephrase_target_new"]["nll"]["p90"],
                "gen_prompt_success": endpoint["rephrase_target_new"]["correctness"]["prompt_rate"],
                "gen_strict_success": endpoint["rephrase_strict_exact"]["rate"],
                "gen_preference_success": endpoint["rephrase_preference"]["rate"],
                "locality": record["locality"]["rate"],
                "terminal_qkl": record["geometry"]["terminal_qkl_last_layer"],
                "removed_energy_fraction_mean": record["geometry"]["removed_energy_fraction"]["mean"]
                if record["geometry"]["removed_energy_fraction"]
                else None,
                "active_node_fraction": record["geometry"]["active_node_fraction"],
            }
        )
    return rows


def _report(records: Sequence[Mapping[str, Any]], exclusion: Mapping[str, Any]) -> str:
    headline = _headline_rows(records)
    primary = [row for row in headline if row["group"] != "IMPLEMENTATION_CONTROL"]
    lines = [
        "# Cache-aware one-sided q-KL projected AlphaEdit — atomic B100 factual report",
        "",
        "상태: **STOPPED_AFTER_ATOMIC_B100_USER_DIRECTED**",
        "범위: Llama/Qwen atomic B100, PRE-EDIT W0, stock Official AlphaEdit N1, ours projected N2/N4.",
        "Sequential scientific execution은 사용자 지시로 제외되었고 promotion=false이다.",
        "",
        "## 핵심 비교: PRE-EDIT / BASELINE / OURS",
        "",
        "NLL은 token-mean이며, Gen strict는 request의 모든 rephrase prompt가 exact target-new인 비율이다.",
        "PRE-EDIT locality=1은 자기 자신과의 reference identity이며 편집 성능이 아니다.",
        "",
        "| Model | Group | Method | Rewrite new NLL mean/med/p90 | Rewrite exact | Rephrase new NLL mean/med/p90 | Gen prompt / strict | Loc | terminal qKL | removed energy | active nodes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| {model} | {group} | {method} | {rw} | {rw_s} | {rp} | {gen} | {loc} | {qkl} | {rem} | {active} |".format(
                model=row["model"],
                group=row["group"],
                method=row["method"],
                rw="/".join(_fmt(row[key]) for key in ("rewrite_new_nll_mean", "rewrite_new_nll_median", "rewrite_new_nll_p90")),
                rw_s=_fmt(row["rewrite_exact_success"]),
                rp="/".join(_fmt(row[key]) for key in ("rephrase_new_nll_mean", "rephrase_new_nll_median", "rephrase_new_nll_p90")),
                gen=f"{_fmt(row['gen_prompt_success'])} / {_fmt(row['gen_strict_success'])}",
                loc=_fmt(row["locality"]),
                qkl=_fmt(row["terminal_qkl"]),
                rem=_fmt(row["removed_energy_fraction_mean"]),
                active=_fmt(row["active_node_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "## Factual interpretation",
            "",
            "- N2는 두 모델에서 predictive q-KL을 낮추고 locality를 비열세 또는 소폭 개선했다.",
            "- Llama N2는 rewrite exact 0.96을 유지했으나 rephrase prompt success가 Official 0.64에서 0.595로 낮아졌다.",
            "- Qwen N2는 rewrite exact 1.00을 유지했고 rephrase prompt success는 0.64에서 0.625였다.",
            "- N4는 두 모델 모두 q-KL/locality 쪽 이동이 더 크지만 rewrite/rephrase strength tradeoff도 더 컸다.",
            "- 이 결과는 atomic B100 한 번의 factual endpoint이며 scientific promotion 또는 sequential claim이 아니다.",
            "",
            "## Rewrite/Rephrase 상세",
            "",
            "아래 값은 각 target-new/target-true NLL의 mean/median/p90/max와 exact/token accuracy이다.",
        ]
    )
    for model in MODEL_ORDER:
        lines.extend(["", f"### {model}", ""])
        for record in records:
            if record["model"] != model or record["group"] == "IMPLEMENTATION_CONTROL":
                continue
            lines.append(f"- **{record['method']}**")
            for metric in METRICS:
                item = record["endpoint"][metric]
                nll = item["nll"]
                corr = item["correctness"]
                lines.append(
                    f"  - {metric}: n={nll['count']}, NLL={_fmt(nll['mean'])}/{_fmt(nll['median'])}/{_fmt(nll['p90'])}/{_fmt(nll['max'])}; exact={corr['prompt_numerator']}/{corr['prompt_denominator']} ({_fmt(corr['prompt_rate'])}); token={corr['token_numerator']}/{corr['token_denominator']} ({_fmt(corr['token_rate'])})"
                )
            lines.append(
                f"  - rewrite new<true={_fmt(record['endpoint']['rewrite_preference']['rate'])}; rephrase new<true={_fmt(record['endpoint']['rephrase_preference']['rate'])}; rephrase strict exact={_fmt(record['endpoint']['rephrase_strict_exact']['rate'])}; locality={record['locality']['numerator']}/{record['locality']['denominator']} ({_fmt(record['locality']['rate'])})"
            )
    lines.extend(
        [
            "",
            "## Geometry / compute / integrity",
            "",
            "| Model | Method | nodes/active | native→projected qKL mean | terminal qKL | removed mean | violation max | edit-core s | total s | peak alloc GiB | cache append | W0 restore | failures/retry/fallback |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        if record["group"] == "PRE_EDIT":
            continue
        geometry = record["geometry"]
        native = geometry["native_predictive_qkl"]
        projected = geometry["post_projected_qkl"]
        removed = geometry["removed_energy_fraction"]
        compute = record["compute"]
        integrity = record["integrity"]
        qkl = "NOT_RECORDED" if native is None else f"{_fmt(native['mean'])}→{_fmt(projected['mean'])}"
        peak = compute.get("peak_gpu_allocated_bytes")
        lines.append(
            f"| {record['model']} | {record['method']} | {geometry['node_count']}/{_fmt(geometry['active_node_count'], 0)} | {qkl} | {_fmt(geometry['terminal_qkl_last_layer'])} | {_fmt(removed['mean'] if removed else None)} | {_fmt(geometry['positive_projected_rate_violation_max'])} | {_fmt(compute.get('edit_core_seconds'), 3)} | {_fmt(compute.get('total_seconds'), 3)} | {_fmt(peak / 2**30 if peak else None, 3)} | {_fmt(integrity['cache_append_count'], 0)} | {_fmt(integrity['w0_pointer_bytes_restore_pass'])} | {integrity['failure_boundary_count']}/{integrity['retry_count']}/{integrity['fallback_count']} |"
        )
    lines.extend(
        [
            "",
            "FULL-FP32, autocast/TF32 off, W0 pointer/bytes restore, create-once results, controller target_true/rephrase/locality influence 0은 모든 terminal cell에서 검증됐다.",
            "Official N1의 qKL/removed-energy/cache-append telemetry는 stock exact bypass 때문에 NOT_RECORDED이다.",
            "",
            "## Implementation-control appendix",
            "",
            "STATIC_SPLIT_OFF N2/N4는 scientific baseline이 아니라 stock endpoint factorization/splitting control이다. 상세 값은 `implementation-control.csv`와 `analysis.json`에 있다.",
            "",
            "## Sequential exclusion",
            "",
            f"- jobs: {', '.join(exclusion['jobs'])}",
            f"- running cells cancelled: {exclusion['running_cells_cancelled']}; pending cells cancelled: {exclusion['pending_cells_cancelled']}",
            "- partial roots/logs are immutable evidence; terminal result count=0; atomic-B100 denominator influence=0.",
            "",
            "## Claim boundary",
            "",
            "허용: cache-aware one-sided projected N2/N4의 atomic-B100 factual tradeoff.",
            "금지: sequential/lifelong 효과, automatic promotion, continuous-time convergence, 또는 N4 우월성 주장.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args()

    models = {
        "llama": _load_model(args.llama_root.resolve(strict=True), "llama"),
        "qwen": _load_model(args.qwen_root.resolve(strict=True), "qwen"),
    }
    records: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        records.append(_pre_record(models[model]))
        records.extend(_method_record(model, cell) for cell in models[model]["cells"])

    exclusion = {
        "schema": "ode-edit.cache-aware-qkl.sequential-user-stop.v1",
        "status": "USER_DIRECTED_STOPPED_EXCLUDED",
        "jobs": ["28081 llama sequential-b10x10", "28083 qwen sequential-b10x10"],
        "running_cells_cancelled": 4,
        "pending_cells_cancelled": 6,
        "terminal_result_count": 0,
        "scientific_denominator_influence": 0,
        "partial_roots": [
            "/data/janghj/ODE-edit/local/results/cache-aware-qkl-projected-alphaedit-v1/llama-sequential-b10x10-r0",
            "/data/janghj/ODE-edit/local/results/cache-aware-qkl-projected-alphaedit-v1/qwen-sequential-b10x10-r0",
        ],
    }
    analysis = {
        "schema": "ode-edit.cache-aware-one-sided-qkl.atomic-b100-analysis.v1",
        "status": "STOPPED_AFTER_ATOMIC_B100_USER_DIRECTED",
        "source": {"head": args.source_head, "tree": args.source_tree},
        "models": {
            model: {
                "root": models[model]["root"],
                "global_w0_sha256": models[model]["reference"]["global_w0_sha256"],
                "source": models[model]["reference"]["source"],
                "input_identities": {
                    key: models[model]["reference"][key]
                    for key in ("dataset", "model", "tokenizer", "projector", "hparams", "preflight_receipt")
                },
            }
            for model in MODEL_ORDER
        },
        "records": records,
        "sequential_exclusion": exclusion,
        "scientific_promotion": False,
    }

    assert_safe_absent_directory(args.report_root)
    write_json(args.report_root / "analysis.json", analysis)
    write_json(args.report_root / "sequential-exclusion.json", exclusion)
    publish(args.report_root / "report-ko.md", _report(records, exclusion).encode("utf-8"))

    headline = _headline_rows(records)
    headline_fields = list(headline[0])
    publish(args.report_root / "headline.csv", _csv_bytes(headline, headline_fields))

    endpoint_rows: list[dict[str, Any]] = []
    for record in records:
        for metric in METRICS:
            item = record["endpoint"][metric]
            endpoint_rows.append(
                {
                    "model": record["model"],
                    "group": record["group"],
                    "method": record["method"],
                    "metric": metric,
                    **item["nll"],
                    **item["correctness"],
                }
            )
    publish(
        args.report_root / "endpoint-metrics.csv",
        _csv_bytes(endpoint_rows, list(endpoint_rows[0])),
    )

    geometry_rows = []
    compute_rows = []
    for record in records:
        if record["group"] == "PRE_EDIT":
            continue
        geometry = record["geometry"]
        geometry_rows.append(
            {
                "model": record["model"],
                "group": record["group"],
                "method": record["method"],
                "node_count": geometry["node_count"],
                "active_node_count": geometry["active_node_count"],
                "active_node_fraction": geometry["active_node_fraction"],
                "terminal_qkl_last_layer": geometry["terminal_qkl_last_layer"],
                "native_predictive_qkl_mean": geometry["native_predictive_qkl"]["mean"] if geometry["native_predictive_qkl"] else None,
                "post_projected_qkl_mean": geometry["post_projected_qkl"]["mean"] if geometry["post_projected_qkl"] else None,
                "removed_energy_fraction_mean": geometry["removed_energy_fraction"]["mean"] if geometry["removed_energy_fraction"] else None,
                "removed_energy_fraction_p90": geometry["removed_energy_fraction"]["p90"] if geometry["removed_energy_fraction"] else None,
                "positive_projected_rate_violation_max": geometry["positive_projected_rate_violation_max"],
            }
        )
        compute_rows.append(
            {
                "model": record["model"],
                "group": record["group"],
                "method": record["method"],
                **record["compute"],
                **record["integrity"],
            }
        )
    publish(args.report_root / "geometry.csv", _csv_bytes(geometry_rows, list(geometry_rows[0])))
    publish(args.report_root / "compute-integrity.csv", _csv_bytes(compute_rows, list(compute_rows[0])))

    controls = [row for row in headline if row["group"] == "IMPLEMENTATION_CONTROL"]
    publish(args.report_root / "implementation-control.csv", _csv_bytes(controls, list(controls[0])))

    artifact_names = sorted(
        path.name
        for path in args.report_root.iterdir()
        if path.is_file() and path.name not in {"analysis-manifest.json", "rooted-receipt.json"}
    )
    artifacts = [_package_identity(args.report_root / name) for name in artifact_names]
    external_inputs = [identity for model in MODEL_ORDER for identity in models[model]["input_files"]]
    member_root = hashlib.sha256(
        canonical_json_bytes([[item["path"], item["sha256"], item["bytes"]] for item in artifacts])
    ).hexdigest()
    manifest = {
        "schema": "ode-edit.cache-aware-one-sided-qkl.atomic-b100-manifest.v1",
        "status": analysis["status"],
        "source": analysis["source"],
        "artifacts": artifacts,
        "external_inputs": external_inputs,
        "artifact_member_root": member_root,
        "scientific_promotion": False,
    }
    write_json(args.report_root / "analysis-manifest.json", manifest)
    manifest_identity = _package_identity(args.report_root / "analysis-manifest.json")
    receipt_core = {
        "schema": "ode-edit.cache-aware-one-sided-qkl.atomic-b100-rooted-receipt.v1",
        "status": analysis["status"],
        "analysis_manifest": manifest_identity,
        "artifact_member_root": member_root,
        "external_result_count": len(external_inputs),
        "atomic_request_denominator": {"llama": 100, "qwen": 100},
        "sequential_result_denominator": 0,
        "sequential_exclusion": exclusion,
        "scientific_promotion": False,
    }
    receipt = {
        **receipt_core,
        "identity_sha256": hashlib.sha256(canonical_json_bytes(receipt_core)).hexdigest(),
    }
    write_json(args.report_root / "rooted-receipt.json", receipt)


if __name__ == "__main__":
    main()
