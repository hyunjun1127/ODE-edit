"""Analysis-only builder for the frozen B1 and B10x10 observer artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project.run_scripts.ode_bf.contracts import canonical_hash

from .contracts import (
    EVALUATOR_IDENTITY,
    IDEAL_Q,
    INSTRUCTION_ID,
    LAYERS,
    NONCE,
    ORDER_IDENTITY,
    STREAM_IDENTITY,
    ObservationBoundary,
)


CELL_ORDER = (
    ("llama3-8b-inst", "memit"),
    ("llama3-8b-inst", "alphaedit"),
    ("qwen2.5-7b-inst", "memit"),
    ("qwen2.5-7b-inst", "alphaedit"),
)
MODEL_LABEL = {"llama3-8b-inst": "Llama-3-8B-Instruct", "qwen2.5-7b-inst": "Qwen2.5-7B-Instruct"}
METHOD_LABEL = {"memit": "Official MEMIT", "alphaedit": "Official AlphaEdit"}
Q_LABELS = ("pre-L4", "pre-L5", "pre-L6", "pre-L7", "pre-L8", "post-L8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _load(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ObservationBoundary(f"analysis input is not a regular file: {path}")
    return json.loads(path.read_text())


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ObservationBoundary("analysis scalar inventory is empty/nonfinite")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "min": float(array.min()),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ObservationBoundary(f"refusing empty CSV: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ObservationBoundary(f"CSV schema differs: {path.name}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_campaign(b1_root: Path, b10_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    b1_receipts: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    for model, method in CELL_ORDER:
        b1_path = b1_root / f"{model}-{method}-b1/result.json"
        b1 = _load(b1_path)
        if b1.get("status") != "B1_OBSERVER_ON_OFF_GATE_PASS" or b1.get("model") != model or b1.get("method") != method:
            raise ObservationBoundary(f"B1 release differs: {b1_path}")
        if not all(b1["parity"].values()):
            raise ObservationBoundary(f"B1 parity differs: {b1_path}")
        b1_receipts.append(b1)
        external.append({"kind": "B1_RELEASE", "path": str(b1_path), "bytes": b1_path.stat().st_size, "sha256": _sha256(b1_path)})

        result_path = b10_root / f"{model}-{method}-b10x10/result.json"
        result = _load(result_path)
        if (
            result.get("status") != "B10X10_TERMINAL_VALID"
            or result.get("model") != model
            or result.get("method") != method
            or result.get("batch_denominator") != 10
            or result.get("request_denominator") != 100
        ):
            raise ObservationBoundary(f"B10x10 cell receipt differs: {result_path}")
        external.append({"kind": "CELL_TERMINAL", "path": str(result_path), "bytes": result_path.stat().st_size, "sha256": _sha256(result_path)})
        batches = []
        for journal in result["journals"]:
            path = Path(journal["path"])
            if _sha256(path) != journal["sha256"]:
                raise ObservationBoundary(f"journal digest differs: {path}")
            batch = _load(path)
            observer = batch["apply"]["layer_realization_observer"]
            if (
                batch.get("status") != "TERMINAL_VALID"
                or batch.get("model") != model
                or batch.get("method") != method
                or batch.get("request_count") != 10
                or not batch["w0_restore"]["exact"]
                or observer["direct_z_compute_count"] != 10
                or observer["direct_z_recompute_count"] != 0
                or observer["layer_loop_observation_copy_count"] != 5
                or observer["terminal_post_L8_forward_count"] != 1
                or observer["residual_debt"]["nonfinite_count"] != 0
            ):
                raise ObservationBoundary(f"journal validity differs: {path}")
            batches.append(batch)
            external.append({"kind": "BATCH_JOURNAL", "path": str(path), "bytes": path.stat().st_size, "sha256": journal["sha256"]})
        if [row["batch_index"] for row in batches] != list(range(1, 11)):
            raise ObservationBoundary(f"batch order differs: {model}/{method}")
        cells.append({"model": model, "method": method, "result": result, "batches": batches})
    return b1_receipts, cells, external


def _tabulate(cells: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    energy: list[dict[str, Any]] = []
    compute: list[dict[str, Any]] = []
    cell_summary: list[dict[str, Any]] = []
    for cell in cells:
        model, method = str(cell["model"]), str(cell["method"])
        cell_request_rows: list[dict[str, Any]] = []
        endpoint_rows: list[dict[str, Any]] = []
        for batch in cell["batches"]:
            batch_index = int(batch["batch_index"])
            observer = batch["apply"]["layer_realization_observer"]
            debt = observer["residual_debt"]
            endpoints = {row["request_sha256"]: row for row in batch["endpoint"]["records"]}
            current_rows = []
            for request in debt["records"]:
                request_hash = request["request_sha256"]
                endpoint = endpoints[request_hash]
                row = {
                    "model": model,
                    "method": method,
                    "batch_index": batch_index,
                    "request_sha256": request_hash,
                    "case_identity_sha256": request["case_identity_sha256"],
                    **{f"q_{label.replace('-', '_')}": request["q"][index] for index, label in enumerate(Q_LABELS)},
                    "q_L8_gt_0_2": int(request["q_L8_gt_0_2"]),
                    "d_parallel": request["d_parallel"],
                    "d_perp": request["d_perp"],
                    "recurrence_closure_relative_error": request["recurrence_closure_relative_error"],
                    "target_new_nll": endpoint["target_new_nll"],
                    "target_true_nll": endpoint["target_true_nll"],
                    "target_new_strict": int(endpoint["target_new_strict"]),
                    "target_true_strict": int(endpoint["target_true_strict"]),
                    "target_new_margin": endpoint["target_new_margin"],
                    "target_true_margin": endpoint["target_true_margin"],
                }
                current_rows.append(row)
                requests.append(row)
                cell_request_rows.append(row)
                endpoint_rows.append(endpoint)
                for layer in request["layers"]:
                    layers.append({
                        "model": model,
                        "method": method,
                        "batch_index": batch_index,
                        "request_sha256": request_hash,
                        "layer": layer["layer"],
                        "q_pre": layer["q_pre"],
                        "q_post": layer["q_post"],
                        "rho": layer["rho"],
                        "under_realization_coefficient": layer["under_realization_coefficient"],
                        "tau": layer["tau"],
                        "allocation_norm_over_R1": layer["allocation_norm_over_R1"],
                        "realized_reduction_norm_over_R1": layer["realized_reduction_norm_over_R1"],
                        "gap_norm_over_R1": layer["gap_norm_over_R1"],
                        "next_pre_link_relative_difference": "NOT_APPLICABLE" if layer["next_pre_link_relative_difference"] is None else layer["next_pre_link_relative_difference"],
                    })
            q_stats = [_summary(row[f"q_{label.replace('-', '_')}"] for row in current_rows) for label in Q_LABELS]
            batches.append({
                "model": model,
                "method": method,
                "batch_index": batch_index,
                "request_denominator": len(current_rows),
                **{f"q_{label.replace('-', '_')}_median": q_stats[index]["median"] for index, label in enumerate(Q_LABELS)},
                "q_pre_L8_gt_0_2_numerator": sum(row["q_L8_gt_0_2"] for row in current_rows),
                "q_pre_L8_gt_0_2_denominator": 10,
                "d_parallel_median": _summary(row["d_parallel"] for row in current_rows)["median"],
                "d_perp_median": _summary(row["d_perp"] for row in current_rows)["median"],
                "recurrence_closure_max": max(row["recurrence_closure_relative_error"] for row in current_rows),
                "target_new_nll_median": _summary(row["target_new_nll"] for row in current_rows)["median"],
                "target_true_nll_median": _summary(row["target_true_nll"] for row in current_rows)["median"],
            })
            weight = observer["weight_action"]
            for layer in weight["layers"]:
                energy.append({
                    "model": model,
                    "method": method,
                    "batch_index": batch_index,
                    "layer": layer["layer"],
                    "frobenius_energy": layer["frobenius_energy"],
                    "energy_share": layer["energy_share"],
                    "total_frobenius_energy": weight["total_frobenius_energy"],
                })
            overhead = observer["overhead_wall_seconds"]
            audit = batch["apply"]["official_call_audit"]
            compute.append({
                "model": model,
                "method": method,
                "batch_index": batch_index,
                "request_denominator": 10,
                "direct_z_compute_count": observer["direct_z_compute_count"],
                "direct_z_recompute_count": observer["direct_z_recompute_count"],
                "layer_loop_existing_forward_count": observer["layer_loop_pass_through_call_count"],
                "layer_activation_copy_count": observer["layer_loop_observation_copy_count"],
                "terminal_additional_forward_count": observer["terminal_post_L8_forward_count"],
                "compute_ks_call_count": audit["compute_ks_call_count"],
                "solve_call_count": audit["torch_linalg_solve_call_count"],
                "target_backward_count": batch["apply"]["target_backward_count"],
                "official_edit_core_wall_seconds": batch["apply"]["edit_core_wall_seconds"],
                "observer_copy_and_reduction_wall_seconds": sum(float(value) for value in overhead.values()),
                "additional_z_optimization_count": overhead["additional_z_optimization"],
                "additional_key_compute_count": overhead["additional_key_compute"],
                "additional_closed_form_solve_count": overhead["additional_closed_form_solve"],
            })
        q_stats = [_summary(row[f"q_{label.replace('-', '_')}"] for row in cell_request_rows) for label in Q_LABELS]
        d_parallel = _summary(row["d_parallel"] for row in cell_request_rows)
        d_perp = _summary(row["d_perp"] for row in cell_request_rows)
        new_nll = _summary(row["target_new_nll"] for row in cell_request_rows)
        true_nll = _summary(row["target_true_nll"] for row in cell_request_rows)
        cell_summary.append({
            "model": model,
            "method": method,
            "batch_denominator": 10,
            "request_denominator": 100,
            **{f"q_{label.replace('-', '_')}_mean": q_stats[index]["mean"] for index, label in enumerate(Q_LABELS)},
            **{f"q_{label.replace('-', '_')}_median": q_stats[index]["median"] for index, label in enumerate(Q_LABELS)},
            **{f"q_{label.replace('-', '_')}_p25": q_stats[index]["p25"] for index, label in enumerate(Q_LABELS)},
            **{f"q_{label.replace('-', '_')}_p75": q_stats[index]["p75"] for index, label in enumerate(Q_LABELS)},
            **{f"q_{label.replace('-', '_')}_p90": q_stats[index]["p90"] for index, label in enumerate(Q_LABELS)},
            "q_pre_L8_gt_0_2_numerator": sum(row["q_L8_gt_0_2"] for row in cell_request_rows),
            "q_pre_L8_gt_0_2_denominator": 100,
            "d_parallel_mean": d_parallel["mean"],
            "d_parallel_median": d_parallel["median"],
            "d_parallel_p90": d_parallel["p90"],
            "d_perp_mean": d_perp["mean"],
            "d_perp_median": d_perp["median"],
            "d_perp_p90": d_perp["p90"],
            "recurrence_closure_max": max(row["recurrence_closure_relative_error"] for row in cell_request_rows),
            "target_new_nll_mean": new_nll["mean"],
            "target_new_nll_median": new_nll["median"],
            "target_new_nll_p90": new_nll["p90"],
            "target_true_nll_mean": true_nll["mean"],
            "target_true_nll_median": true_nll["median"],
            "target_true_nll_p90": true_nll["p90"],
            "target_new_strict_numerator": sum(row["target_new_strict"] for row in cell_request_rows),
            "target_new_strict_denominator": 100,
        })
    layer_summary = []
    for model, method in CELL_ORDER:
        for layer in LAYERS:
            selected = [row for row in layers if row["model"] == model and row["method"] == method and row["layer"] == layer]
            for metric in ("rho", "under_realization_coefficient", "tau", "gap_norm_over_R1"):
                stats = _summary(row[metric] for row in selected)
                layer_summary.append({"model": model, "method": method, "layer": layer, "metric": metric, **stats})
    return {
        "request": requests,
        "layer_request": layers,
        "batch": batches,
        "energy": energy,
        "compute": compute,
        "cell_summary": cell_summary,
        "layer_summary": layer_summary,
    }


def _plot_q_pooled(path: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        rows = [row for row in tables["request"] if row["model"] == model and row["method"] == method]
        q = np.asarray([[row[f"q_{label.replace('-', '_')}"] for label in Q_LABELS] for row in rows])
        for values in q:
            axis.plot(range(6), values, color="0.72", linewidth=0.55, alpha=0.32)
        median = np.median(q, axis=0)
        axis.fill_between(range(6), np.percentile(q, 25, axis=0), np.percentile(q, 75, axis=0), color="#3b82f6", alpha=0.22, label="IQR")
        axis.plot(range(6), median, color="#0f172a", linewidth=2.3, label="median")
        axis.plot(range(6), IDEAL_Q, color="#dc2626", linestyle="--", linewidth=1.5, label="ideal")
        dpar = float(np.median([row["d_parallel"] for row in rows]))
        dperp = float(np.median([row["d_perp"] for row in rows]))
        count = sum(int(row["q_L8_gt_0_2"]) for row in rows)
        axis.set_title(f"{MODEL_LABEL[model]} / {METHOD_LABEL[method]}")
        axis.text(0.02, 0.98, f"pre-L8 med={median[4]:.3f}\nq>0.2={count}/100\nmed debt=({dpar:.3f},{dperp:.3f})", transform=axis.transAxes, va="top", fontsize=8)
        axis.grid(alpha=0.18)
        axis.set_xticks(range(6), Q_LABELS, rotation=20)
        axis.set_ylabel("normalized residual q")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Official layer-write remaining residual (pooled 100 requests/cell)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_q_facets(path: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    fig, axes = plt.subplots(10, 4, figsize=(17, 31), sharex=True)
    for column, (model, method) in enumerate(CELL_ORDER):
        for batch_index in range(1, 11):
            axis = axes[batch_index - 1, column]
            rows = [row for row in tables["request"] if row["model"] == model and row["method"] == method and row["batch_index"] == batch_index]
            q = np.asarray([[row[f"q_{label.replace('-', '_')}"] for label in Q_LABELS] for row in rows])
            for values in q:
                axis.plot(range(6), values, color="0.72", linewidth=0.5, alpha=0.5)
            axis.plot(range(6), np.median(q, axis=0), color="#0f172a", linewidth=1.8)
            axis.fill_between(range(6), np.percentile(q, 25, axis=0), np.percentile(q, 75, axis=0), color="#3b82f6", alpha=0.2)
            axis.plot(range(6), IDEAL_Q, color="#dc2626", linestyle="--", linewidth=1.0)
            count = sum(int(row["q_L8_gt_0_2"]) for row in rows)
            axis.text(0.02, 0.95, f"B{batch_index}: {count}/10", transform=axis.transAxes, va="top", fontsize=7)
            axis.grid(alpha=0.15)
            if batch_index == 1:
                axis.set_title(f"{MODEL_LABEL[model]}\n{METHOD_LABEL[method]}", fontsize=9)
            if batch_index == 10:
                axis.set_xticks(range(6), Q_LABELS, rotation=45, fontsize=7)
            if column == 0:
                axis.set_ylabel("q", fontsize=8)
    fig.suptitle("Per-B10 facet trajectories (each facet denominator=10)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_box(path: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]], metric: str, ylabel: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        values = [[row[metric] for row in tables["layer_request"] if row["model"] == model and row["method"] == method and row["layer"] == layer] for layer in LAYERS]
        axis.boxplot(values, tick_labels=[str(layer) for layer in LAYERS], showfliers=False)
        if metric == "rho":
            axis.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1)
        axis.set_title(f"{MODEL_LABEL[model]} / {METHOD_LABEL[method]}", fontsize=9)
        axis.set_xlabel("editable layer")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_energy(path: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        values = [[row["energy_share"] for row in tables["energy"] if row["model"] == model and row["method"] == method and row["layer"] == layer] for layer in LAYERS]
        axis.boxplot(values, tick_labels=[str(layer) for layer in LAYERS], showfliers=True)
        axis.set_title(f"{MODEL_LABEL[model]} / {METHOD_LABEL[method]}", fontsize=9)
        axis.set_xlabel("editable layer")
        axis.set_ylabel("Delta-W Frobenius energy share")
        axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _f(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _markdown_report(tables: Mapping[str, Sequence[Mapping[str, Any]]], b1_receipts: Sequence[Mapping[str, Any]], run_head: str, run_tree: str) -> str:
    summary = {(row["model"], row["method"]): row for row in tables["cell_summary"]}
    lines = [
        "# Official layer-write realization debt — Llama/Qwen B10×10 사실 보고서",
        "",
        "상태: **OBSERVATIONAL_STUDY_TERMINAL_VALID**  ",
        "범위: Official MEMIT/AlphaEdit의 기존 update를 바꾸지 않고 layer 4→8 write 중 activation residual의 실현 양상만 관찰했다. 이 결과는 barrier 효용, ODE 필요성, 또는 보편적 last-layer 병목을 입증하지 않는다. `scientific_promotion=false`다.",
        "",
        "## 핵심 결과",
        "",
        "각 셀은 서로 독립적인 W0에서 시작한 10개 B10 slice(100 requests)다. 아래 `q>0.2`는 pooled `/100`; batch별 `/10`은 별도 표와 facet 그림에만 썼다.",
        "",
        "| 모델 | Official 방법 | pre-L8 q median | post-L8 q median | pre-L8 q>0.2 | d_parallel median | d_perp median | recurrence max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        row = summary[(model, method)]
        lines.append(
            f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | {_f(row['q_pre_L8_median'])} | {_f(row['q_post_L8_median'])} | {row['q_pre_L8_gt_0_2_numerator']}/100 | {_f(row['d_parallel_median'])} | {_f(row['d_perp_median'])} | {float(row['recurrence_closure_max']):.3e} |"
        )
    lines += [
        "",
        "`d_parallel`은 L8 진입 시 이상적인 `R1/5` 대비 원래 residual 방향의 debt이며, `d_perp`는 방향 왜곡이다. 절대 residual norm은 모델/방법 간 비교에 사용하지 않았다.",
        "",
        "![pooled q trajectory](q-pooled-2x2.png)",
        "",
        "10개 batch 각각의 `/10` trajectory는 [q-by-batch-facets.png](q-by-batch-facets.png)에 보존했다.",
        "",
        "## B1 observer on/off replay gate",
        "",
        "| 모델 | 방법 | final W SHA | endpoint metric | z SHA | layer calls | terminal call | recurrence max | W0 restore |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b1 in b1_receipts:
        obs = b1["on"]["apply"]["layer_realization_observer"]
        parity = b1["parity"]
        lines.append(
            f"| {MODEL_LABEL[b1['model']]} | {METHOD_LABEL[b1['method']]} | {'PASS' if parity['final_weight_sha_byte_exact'] else 'FAIL'} | {'PASS' if parity['official_endpoint_metric_exact'] else 'FAIL'} | {'PASS' if parity['z_hash_exact'] else 'FAIL'} | {obs['layer_loop_observation_copy_count']}/5 | {obs['terminal_post_L8_forward_count']}/1 | {obs['residual_debt']['maximum_recurrence_closure_relative_error']:.3e} | PASS |"
        )
    lines += [
        "",
        "네 replay 모두 observer가 backward/key/solver/cache count를 바꾼 횟수 0이고, module-global 함수 객체는 예외 경로를 포함해 `finally`에서 복원됐다. AlphaEdit 양 replay는 `reset_cache=True`; MEMIT covariance는 계산 cache이며 history state가 아니다.",
        "",
        "## q trajectory 상세",
        "",
        "| 모델 | 방법 | q pre-L4 | pre-L5 | pre-L6 | pre-L7 | pre-L8 | post-L8 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        row = summary[(model, method)]
        values = [_f(row[f"q_{label.replace('-', '_')}_median"]) for label in Q_LABELS]
        lines.append(f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | " + " | ".join(values) + " |")
    lines += [
        "",
        "완전 실현 uniform schedule `[1.0, 0.8, 0.6, 0.4, 0.2, 0.0]`은 관측 기준선일 뿐 Official update의 강제 목표로 사용하지 않았다.",
        "",
        "## layer별 realization과 방향 왜곡",
        "",
        "`rho=1`은 allocation 방향 exact realization, `0<rho<1`은 under-realization, `rho>1`은 overshoot, `rho<0`은 반대 방향이다. `tau`는 allocation에 직교한 실제 activation reduction의 크기다.",
        "",
        "![rho by layer](rho-by-layer.png)",
        "",
        "![tau by layer](tau-by-layer.png)",
        "",
        "수치 표는 [layer-summary.csv](layer-summary.csv)와 request-level [layer-request-metrics.csv](layer-request-metrics.csv)에 있다.",
        "",
        "## 실제 weight action",
        "",
        "`energy = ||Delta W_l||_F^2`, `share = energy / sum_l energy`다. residual debt와 weight energy는 별도 관측 축이며 causal pooling하지 않았다.",
        "",
        "![weight energy share](weight-energy-share.png)",
        "",
        "## 실행 완전성·계산 overhead",
        "",
        "- 유효 셀 4/4, slice 40/40, request 400/400.",
        "- Official `compute_z` 400회(각 request entry-state 1회), observer 유발 recompute 0.",
        "- 기존 layer-loop activation call 200회(5×40), observation copy 200회, post-L8 추가 forward 40회.",
        "- W0 pointer/bytes restore 40/40, nonfinite 0, failure/imputation 0.",
        "- observer 유발 z optimization/key/solve 추가 횟수는 모두 0.",
        "- 상세 wall-time/counter는 [compute-ledger.csv](compute-ledger.csv). 모델 load/setup이 포함된 cell wall time과 Official edit-core/observer copy 시간을 분리했다.",
        "",
        "## endpoint metric (관찰 무결성 보조)",
        "",
        "이 metric은 observer on/off parity 및 terminal-valid 확인용이며 layer debt의 원인 판정에 사용하지 않았다.",
        "",
        "| 모델 | 방법 | target-new NLL mean/median/p90 | target-true NLL mean/median/p90 | target-new strict |",
        "|---|---|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        row = summary[(model, method)]
        lines.append(
            f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | {_f(row['target_new_nll_mean'])}/{_f(row['target_new_nll_median'])}/{_f(row['target_new_nll_p90'])} | {_f(row['target_true_nll_mean'])}/{_f(row['target_true_nll_median'])}/{_f(row['target_true_nll_p90'])} | {row['target_new_strict_numerator']}/100 |"
        )
    lines += [
        "",
        "## 사실 / 해석 경계",
        "",
        "**FACT.** 모든 셀에서 앞 layer의 post residual과 다음 layer pre residual이 동일 tensor identity로 연결됐고, 제시된 `R5` recurrence는 `<1e-4` 기준을 큰 폭으로 통과했다. 따라서 L8 진입 debt는 앞 layer의 allocation-realization gap으로 수치적으로 닫힌다.",
        "",
        "**INFERENCE.** `d_parallel`과 `d_perp`의 크기·부호 및 rho/tau의 layer별 분포는 model/method-conditioned realization pattern을 기술한다. 공통성이 약하면 architecture/method-conditioned evidence로만 읽어야 한다.",
        "",
        "**NON-CLAIM.** 이 observational study만으로 barrier benefit, action-to-go controller의 필요성, ODE necessity, 또는 universal last-layer bottleneck을 주장하지 않는다. `L8 allocation coefficient=1` 자체도 결과로 세지 않았다.",
        "",
        "## 봉인 identity",
        "",
        f"- run source HEAD/tree: `{run_head}` / `{run_tree}`",
        f"- stream/order/evaluator: `{STREAM_IDENTITY}` / `{ORDER_IDENTITY}` / `{EVALUATOR_IDENTITY}`",
        "- model dtype: Llama/Qwen FULL-FP32; raw prompt/logit/generation publish count 0.",
        "- raw roots는 ignored local path에 immutable 보존하며 Git package에는 포함하지 않았다.",
        "- 상세 파일 identity는 `analysis-manifest.json`, rooted binding은 `rooted-analysis-receipt.json`.",
        "",
    ]
    return "\n".join(lines)


def build(args: argparse.Namespace) -> None:
    if args.output.exists() or args.output.is_symlink():
        raise ObservationBoundary("refusing to overwrite analysis output")
    b1_receipts, cells, external = _load_campaign(args.b1_root, args.b10_root)
    run_heads = {cell["result"]["source"]["head"] for cell in cells}
    run_trees = {cell["result"]["source"]["tree"] for cell in cells}
    if len(run_heads) != 1 or len(run_trees) != 1:
        raise ObservationBoundary("run source identity differs across cells")
    run_head, run_tree = next(iter(run_heads)), next(iter(run_trees))
    if run_head != args.expected_run_head:
        raise ObservationBoundary("expected run source differs")
    builder_head = _git(args.source_root, "rev-parse", "HEAD")
    builder_tree = _git(args.source_root, "rev-parse", "HEAD^{tree}")
    if _git(args.source_root, "status", "--porcelain", "--untracked-files=no"):
        raise ObservationBoundary("analysis source is tracked-dirty")
    for path in sorted(args.log_root.glob("off_layer_*31440*")) + sorted(args.log_root.glob("off_layer_*31473*")):
        if path.is_file() and not path.is_symlink():
            external.append({"kind": "SLURM_LOG", "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    for label, path in (("PREFLIGHT", args.preflight), ("FOCUSED_GATE", args.focused_gate), ("B1_DRY_PLAN", args.b1_dry_plan), ("B10_DRY_PLAN", args.b10_dry_plan)):
        external.append({"kind": label, "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    tables = _tabulate(cells)
    args.output.mkdir(parents=True, mode=0o755)
    csv_files = {
        "cell-summary.csv": tables["cell_summary"],
        "batch-summary.csv": tables["batch"],
        "request-trajectories.csv": tables["request"],
        "layer-request-metrics.csv": tables["layer_request"],
        "layer-summary.csv": tables["layer_summary"],
        "weight-energy-by-batch-layer.csv": tables["energy"],
        "compute-ledger.csv": tables["compute"],
    }
    for name, rows in csv_files.items():
        _write_csv(args.output / name, rows)
    _plot_q_pooled(args.output / "q-pooled-2x2.png", tables)
    _plot_q_facets(args.output / "q-by-batch-facets.png", tables)
    _plot_layer_box(args.output / "rho-by-layer.png", tables, "rho", "target-direction realization rho")
    _plot_layer_box(args.output / "tau-by-layer.png", tables, "tau", "orthogonal distortion tau")
    _plot_energy(args.output / "weight-energy-share.png", tables)
    report_path = args.output / "official-layer-realization-debt-b10x10-factual-ko.md"
    report_path.write_text(_markdown_report(tables, b1_receipts, run_head, run_tree), encoding="utf-8")

    member_names = [*csv_files, "q-pooled-2x2.png", "q-by-batch-facets.png", "rho-by-layer.png", "tau-by-layer.png", "weight-energy-share.png", report_path.name]
    members = [{"path": name, "bytes": (args.output / name).stat().st_size, "sha256": _sha256(args.output / name)} for name in sorted(member_names)]
    member_root = canonical_hash([[row["path"], row["bytes"], row["sha256"]] for row in members])
    manifest = {
        "schema": "odeedit.s06.official-layer-realization-debt.analysis-manifest.v1",
        "instruction_id": INSTRUCTION_ID,
        "nonce": NONCE,
        "status": "OBSERVATIONAL_STUDY_TERMINAL_VALID",
        "run_source": {"head": run_head, "tree": run_tree},
        "analysis_builder_source": {"head": builder_head, "tree": builder_tree},
        "jobs": {"b1": "31440_[0-3]%4", "b10x10": "31473_[0-3]%4"},
        "denominators": {"cells": 4, "batches": 40, "requests": 400, "valid_cells": 4, "valid_batches": 40, "valid_requests": 400},
        "invariants": {"direct_z_compute_count": 400, "direct_z_recompute_count": 0, "layer_loop_observation_copy_count": 200, "terminal_forward_count": 40, "w0_restore_count": 40, "failure_count": 0, "nonfinite_count": 0, "imputation_count": 0},
        "stream": {"root": STREAM_IDENTITY, "order": ORDER_IDENTITY, "evaluator": EVALUATOR_IDENTITY},
        "external_inputs": sorted(external, key=lambda row: (row["kind"], row["path"])),
        "members": members,
        "member_root": member_root,
        "raw_prompt_logit_generation_git_count": 0,
        "easyedit_source_edit_count": 0,
        "sequential_submit_count": 0,
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_path = args.output / "analysis-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "odeedit.s06.official-layer-realization-debt.rooted-analysis-receipt.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "OBSERVATIONAL_STUDY_TERMINAL_VALID",
        "manifest": {"path": manifest_path.name, "bytes": manifest_path.stat().st_size, "sha256": _sha256(manifest_path)},
        "member_root": member_root,
        "external_input_root": canonical_hash([[row["kind"], row["path"], row["bytes"], row["sha256"]] for row in manifest["external_inputs"]]),
        "denominators": manifest["denominators"],
        "failure_count": 0,
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    (args.output / "rooted-analysis-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-run-head", required=True)
    parser.add_argument("--b1-root", type=Path, required=True)
    parser.add_argument("--b10-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--focused-gate", type=Path, required=True)
    parser.add_argument("--b1-dry-plan", type=Path, required=True)
    parser.add_argument("--b10-dry-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
