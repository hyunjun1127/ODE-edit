"""Analysis-only builder for cumulative B1-to-B10 observer artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project.run_scripts.ode_bf.contracts import canonical_hash

from .analysis import (
    CELL_ORDER,
    METHOD_LABEL,
    MODEL_LABEL,
    Q_LABELS,
    _load,
    _plot_layer_box,
    _plot_q_facets,
    _plot_q_pooled,
    _sha256,
    _summary,
    _tabulate,
    _write_csv,
)
from .contracts import (
    EVALUATOR_IDENTITY,
    LAYERS,
    ORDER_IDENTITY,
    ObservationBoundary,
    STREAM_IDENTITY,
)
from .sequential_contracts import INSTRUCTION_ID, NONCE


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _load_campaign(result_roots: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    for model, method in CELL_ORDER:
        candidates = [
            root / f"{model}-{method}-sequential-b10x10/result.json"
            for root in result_roots
            if (root / f"{model}-{method}-sequential-b10x10/result.json").is_file()
        ]
        if len(candidates) != 1:
            raise ObservationBoundary(f"sequential cell result root is ambiguous/absent: {model}/{method}")
        result_path = candidates[0]
        result = _load(result_path)
        if (
            result.get("status") != "SEQUENTIAL_B1_TO_B10_TERMINAL_VALID"
            or result.get("model") != model
            or result.get("method") != method
            or result.get("batch_denominator") != 10
            or result.get("request_denominator") != 100
            or not result.get("terminal_w0_restore", {}).get("exact")
            or not result.get("terminal_cache_restore", {}).get("exact")
            or not all(bool(value) for key, value in result.get("totals", {}).items() if isinstance(value, bool))
        ):
            raise ObservationBoundary(f"sequential cell terminal differs: {result_path}")
        external.append({"kind": "CELL_TERMINAL", "path": str(result_path), "bytes": result_path.stat().st_size, "sha256": _sha256(result_path)})
        batches = []
        previous_commit: Mapping[str, str] | None = None
        prior_cache_exit: str | None = None
        for journal in result["journals"]:
            path = Path(journal["path"])
            if _sha256(path) != journal["sha256"]:
                raise ObservationBoundary(f"sequential batch digest differs: {path}")
            batch = _load(path)
            observer = batch["apply"]["layer_realization_observer"]
            weight = batch["weight_continuity"]
            cache = batch["cache_continuity"]
            index = int(batch["batch_index"])
            if (
                batch.get("status") != "SEQUENTIAL_BATCH_TERMINAL_VALID"
                or batch.get("model") != model
                or batch.get("method") != method
                or batch.get("request_count") != 10
                or observer["direct_z_compute_count"] != 10
                or observer["direct_z_recompute_count"] != 0
                or observer["layer_loop_observation_copy_count"] != 5
                or observer["terminal_post_L8_forward_count"] != 1
                or observer["residual_debt"]["nonfinite_count"] != 0
                or (previous_commit is not None and dict(weight["entry_sha256"]) != dict(previous_commit))
                or (prior_cache_exit is not None and str(cache["entry_sha256"]) != prior_cache_exit)
            ):
                raise ObservationBoundary(f"sequential batch validity/continuity differs: {path}")
            if index == 1 and batch.get("first_valid_gate", {}).get("status") != "FIRST_B1_TRANSACTION_GATE_PASS":
                raise ObservationBoundary(f"first-valid transaction receipt differs: {path}")
            previous_commit = dict(weight["commit_sha256"])
            prior_cache_exit = str(cache["exit_sha256"])
            batches.append(batch)
            external.append({"kind": "BATCH_JOURNAL", "path": str(path), "bytes": path.stat().st_size, "sha256": journal["sha256"]})
        if [int(row["batch_index"]) for row in batches] != list(range(1, 11)):
            raise ObservationBoundary(f"sequential batch order differs: {model}/{method}")
        cells.append({"model": model, "method": method, "result": result, "batches": batches})
    return cells, external


def _continuity_table(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        batches = cell["batches"]
        cache = [row["cache_continuity"] for row in batches]
        rows.append({
            "model": cell["model"],
            "method": cell["method"],
            "batch_denominator": 10,
            "request_denominator": 100,
            "weight_commit_to_next_entry_numerator": 9,
            "weight_commit_to_next_entry_denominator": 9,
            "cache_kind": cache[0]["kind"],
            "cache_entry_width_B1": cache[0]["entry_width"],
            "cache_exit_width_B10": cache[-1]["exit_width"],
            "cache_entry_exit_links_numerator": 9,
            "cache_entry_exit_links_denominator": 9,
            "direct_z_count": cell["result"]["totals"]["direct_z"],
            "direct_z_recompute_count": cell["result"]["totals"]["recompute"],
            "layer_observation_count": cell["result"]["totals"]["layer_observation"],
            "terminal_forward_count": cell["result"]["totals"]["terminal_forward"],
            "terminal_w0_restore_exact": int(cell["result"]["terminal_w0_restore"]["exact"]),
            "terminal_cache_restore_exact": int(cell["result"]["terminal_cache_restore"]["exact"]),
        })
    return rows


def _independent_comparison(
    tables: Mapping[str, Sequence[Mapping[str, Any]]], independent_package: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = independent_package / "cell-summary.csv"
    manifest_path = independent_package / "analysis-manifest.json"
    receipt_path = independent_package / "rooted-analysis-receipt.json"
    manifest = _load(manifest_path)
    receipt = _load(receipt_path)
    if manifest.get("status") != "OBSERVATIONAL_STUDY_TERMINAL_VALID" or receipt.get("member_root") != manifest.get("member_root"):
        raise ObservationBoundary("independent comparison package differs")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        independent = {(row["model"], row["method"]): row for row in csv.DictReader(handle)}
    sequential = {(row["model"], row["method"]): row for row in tables["cell_summary"]}
    rows = []
    for model, method in CELL_ORDER:
        seq = sequential[(model, method)]
        ind = independent[(model, method)]
        row = {"model": model, "method": method}
        for metric in ("q_pre_L8_median", "q_post_L8_median", "d_parallel_median", "d_perp_median"):
            seq_value = float(seq[metric])
            ind_value = float(ind[metric])
            row[f"sequential_{metric}"] = seq_value
            row[f"independent_{metric}"] = ind_value
            row[f"sequential_minus_independent_{metric}"] = seq_value - ind_value
        rows.append(row)
    external = [
        {"kind": "INDEPENDENT_CELL_SUMMARY", "path": str(summary_path), "bytes": summary_path.stat().st_size, "sha256": _sha256(summary_path)},
        {"kind": "INDEPENDENT_MANIFEST", "path": str(manifest_path), "bytes": manifest_path.stat().st_size, "sha256": _sha256(manifest_path)},
        {"kind": "INDEPENDENT_ROOTED_RECEIPT", "path": str(receipt_path), "bytes": receipt_path.stat().st_size, "sha256": _sha256(receipt_path)},
    ]
    return rows, external


def _plot_update_magnitude(path: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (model, method) in zip(axes.flat, CELL_ORDER, strict=True):
        medians = []
        lower = []
        upper = []
        for layer in LAYERS:
            magnitudes = np.sqrt(np.asarray([
                float(row["frobenius_energy"])
                for row in tables["energy"]
                if row["model"] == model and row["method"] == method and int(row["layer"]) == layer
            ], dtype=np.float64))
            med = float(np.median(magnitudes))
            medians.append(med)
            lower.append(med - float(np.percentile(magnitudes, 25)))
            upper.append(float(np.percentile(magnitudes, 75)) - med)
        axis.bar([str(layer) for layer in LAYERS], medians, color="#2563eb", alpha=0.82)
        axis.errorbar(range(len(LAYERS)), medians, yerr=[lower, upper], fmt="none", ecolor="#0f172a", capsize=3)
        axis.set_title(f"{MODEL_LABEL[model]} / {METHOD_LABEL[method]}", fontsize=9)
        axis.set_xlabel("editable layer")
        axis.set_ylabel("median ||Delta W_l||_F")
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle("Layer-wise Update Magnitude")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _f(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _report(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    continuity: Sequence[Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    run_head: str,
    run_tree: str,
    job_ids: Sequence[str],
) -> str:
    summary = {(row["model"], row["method"]): row for row in tables["cell_summary"]}
    compare = {(row["model"], row["method"]): row for row in comparison}
    lines = [
        "# Official layer-write realization debt — B1→B10 cumulative sequential 사실 보고서",
        "",
        "상태: **SEQUENTIAL_OBSERVATIONAL_STUDY_TERMINAL_VALID**",
        "",
        "범위: **B1→B10 cumulative W/cache sequential**. 각 셀은 W0에서 한 번 시작해 10개 B10 batch를 순서대로 누적했으며 batch별 W0 reset은 0이다. AlphaEdit은 B1 cold entry 뒤 dynamic `cache_c`를 B10까지 연속 소비·append했다. MEMIT은 Official static covariance computation cache만 재사용했다.",
        "",
        "이 연구는 Official update를 변경하지 않은 descriptive observation이다. barrier benefit, ODE necessity, universal last-layer bottleneck 또는 독립↔순차 인과효과를 주장하지 않는다. `scientific_promotion=false`다.",
        "",
        "## 핵심 sequential 결과",
        "",
        "| 모델 | Official 방법 | pre-L8 q median | post-L8 q median | pre-L8 q>0.2 | d_parallel median | d_perp median | recurrence max |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        row = summary[(model, method)]
        lines.append(f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | {_f(row['q_pre_L8_median'])} | {_f(row['q_post_L8_median'])} | {row['q_pre_L8_gt_0_2_numerator']}/100 | {_f(row['d_parallel_median'])} | {_f(row['d_perp_median'])} | {float(row['recurrence_closure_max']):.3e} |")
    lines += [
        "",
        "![sequential activation residual/debt](q-pooled-2x2.png)",
        "",
        "Batch별 `/10` trajectory는 [q-by-batch-facets.png](q-by-batch-facets.png)에 보존했다. Pooled 표의 분모는 `/100`이다.",
        "",
        "## B1→B10 continuity 및 계측 완전성",
        "",
        "| 모델 | 방법 | W commit→next entry | cache B1 entry→B10 exit | cache links | direct-z | recompute | layer obs | terminal fwd | terminal W0/cache restore |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in continuity:
        lines.append(f"| {MODEL_LABEL[row['model']]} | {METHOD_LABEL[row['method']]} | {row['weight_commit_to_next_entry_numerator']}/{row['weight_commit_to_next_entry_denominator']} | {row['cache_entry_width_B1']}→{row['cache_exit_width_B10']} | {row['cache_entry_exit_links_numerator']}/{row['cache_entry_exit_links_denominator']} | {row['direct_z_count']}/100 | {row['direct_z_recompute_count']} | {row['layer_observation_count']}/50 | {row['terminal_forward_count']}/10 | PASS/PASS |")
    lines += [
        "",
        "AlphaEdit cache 폭은 `0→100`; MEMIT의 `0→0`은 request-history cache가 없고 static covariance cache identity만 9/9 연결됐다는 뜻이다. 첫 실제 B1 transaction은 네 셀 모두 observer 5/5, terminal forward 1, recurrence `<1e-4`, finite, W/cache entry contract와 추가 decision/update count 0을 통과했다.",
        "",
        "## q trajectory 및 layer realization",
        "",
        "| 모델 | 방법 | pre-L4 | pre-L5 | pre-L6 | pre-L7 | pre-L8 | post-L8 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        row = summary[(model, method)]
        values = [_f(row[f"q_{label.replace('-', '_')}_median"]) for label in Q_LABELS]
        lines.append(f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | " + " | ".join(values) + " |")
    lines += [
        "",
        "![rho by layer](rho-by-layer.png)",
        "",
        "![tau by layer](tau-by-layer.png)",
        "",
        "`rho`와 `tau`, allocation-realization gap `E`, inherited debt의 request-level 값은 [layer-request-metrics.csv](layer-request-metrics.csv)에 있다.",
        "",
        "## Layer-wise weight update",
        "",
        "`energy=||Delta W_l||_F^2`, `share=energy/sum_l energy`로 별도 저장했다. 아래 막대는 각 셀의 10 sequential batches에서 관측한 `||Delta W_l||_F` median이고 error bar는 IQR이다.",
        "",
        "![Layer-wise Update Magnitude](layer-wise-update-magnitude.png)",
        "",
        "## Independent W0/cold-state 관찰과 descriptive 비교",
        "",
        "아래 차이는 `sequential - independent`이며 execution semantics가 달라 causal effect로 해석하지 않는다.",
        "",
        "| 모델 | 방법 | Δ pre-L8 q median | Δ post-L8 q median | Δ d_parallel median | Δ d_perp median |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        row = compare[(model, method)]
        lines.append(f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | {_f(row['sequential_minus_independent_q_pre_L8_median'])} | {_f(row['sequential_minus_independent_q_post_L8_median'])} | {_f(row['sequential_minus_independent_d_parallel_median'])} | {_f(row['sequential_minus_independent_d_perp_median'])} |")
    lines += [
        "",
        "## Endpoint metric (계측 무결성 보조)",
        "",
        "각 값은 해당 batch 직후 현재 10 requests의 metric이며 controller/selection에는 사용되지 않았다.",
        "",
        "| 모델 | 방법 | target-new NLL mean/median/p90 | target-true NLL mean/median/p90 | target-new strict |",
        "|---|---|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        row = summary[(model, method)]
        lines.append(f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | {_f(row['target_new_nll_mean'])}/{_f(row['target_new_nll_median'])}/{_f(row['target_new_nll_p90'])} | {_f(row['target_true_nll_mean'])}/{_f(row['target_true_nll_median'])}/{_f(row['target_true_nll_p90'])} | {row['target_new_strict_numerator']}/100 |")
    lines += [
        "",
        "## 실행·identity",
        "",
        "- valid cells 4/4, batches 40/40, requests 400/400; failure/nonfinite/imputation 0.",
        "- direct-z 400, recompute 0, layer observations 200, terminal forwards 40, W links 36/36, terminal W0/cache restore 4/4.",
        "- raw prompt/logit/generation publish 0; EasyEdit source/update equation change 0.",
        f"- Slurm valid lineages: `{', '.join(job_ids)}`.",
        f"- run source HEAD/tree: `{run_head}` / `{run_tree}`.",
        f"- stream/order/evaluator: `{STREAM_IDENTITY}` / `{ORDER_IDENTITY}` / `{EVALUATOR_IDENTITY}`.",
        "- raw roots는 ignored local path에 immutable 보존하며 Git package에는 포함하지 않았다.",
        "",
    ]
    return "\n".join(lines)


def build(args: argparse.Namespace) -> None:
    if args.output.exists() or args.output.is_symlink():
        raise ObservationBoundary("refusing to overwrite sequential analysis output")
    cells, external = _load_campaign(args.result_root)
    heads = {cell["result"]["source"]["head"] for cell in cells}
    trees = {cell["result"]["source"]["tree"] for cell in cells}
    if heads != set(args.expected_run_head) or len(trees) != len(heads):
        raise ObservationBoundary("sequential run source identity differs")
    run_head = args.expected_run_head[-1]
    run_tree = next(
        cell["result"]["source"]["tree"]
        for cell in cells
        if cell["result"]["source"]["head"] == run_head
    )
    builder_head = _git(args.source_root, "rev-parse", "HEAD")
    builder_tree = _git(args.source_root, "rev-parse", "HEAD^{tree}")
    if _git(args.source_root, "status", "--porcelain", "--untracked-files=no"):
        raise ObservationBoundary("sequential analysis source is tracked-dirty")
    for job_id in args.job_id:
        for path in sorted(args.log_root.glob(f"*{job_id}*")):
            if path.is_file() and not path.is_symlink():
                external.append({"kind": "SLURM_LOG", "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    for root in args.result_root:
        for path in sorted(root.glob("*/failure.json")):
            if path.is_file() and not path.is_symlink():
                external.append({"kind": "PURE_TECHNICAL_EXCLUSION", "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    for label, path in (("PREFLIGHT", args.preflight), ("FOCUSED_GATE", args.focused_gate), ("DRY_PLAN", args.dry_plan)):
        external.append({"kind": label, "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    tables = _tabulate(cells)
    continuity = _continuity_table(cells)
    comparison, comparison_inputs = _independent_comparison(tables, args.independent_package)
    external.extend(comparison_inputs)

    args.output.mkdir(parents=True, mode=0o755)
    csv_files = {
        "cell-summary.csv": tables["cell_summary"],
        "batch-summary.csv": tables["batch"],
        "request-trajectories.csv": tables["request"],
        "layer-request-metrics.csv": tables["layer_request"],
        "layer-summary.csv": tables["layer_summary"],
        "weight-energy-by-batch-layer.csv": tables["energy"],
        "compute-ledger.csv": tables["compute"],
        "continuity-summary.csv": continuity,
        "independent-descriptive-comparison.csv": comparison,
    }
    for name, rows in csv_files.items():
        _write_csv(args.output / name, rows)
    _plot_q_pooled(args.output / "q-pooled-2x2.png", tables)
    _plot_q_facets(args.output / "q-by-batch-facets.png", tables)
    _plot_layer_box(args.output / "rho-by-layer.png", tables, "rho", "target-direction realization rho")
    _plot_layer_box(args.output / "tau-by-layer.png", tables, "tau", "orthogonal distortion tau")
    _plot_update_magnitude(args.output / "layer-wise-update-magnitude.png", tables)
    report_path = args.output / "official-layer-realization-debt-sequential-b10x10-factual-ko.md"
    report_path.write_text(_report(tables, continuity, comparison, run_head, run_tree, args.job_id), encoding="utf-8")

    member_names = [*csv_files, "q-pooled-2x2.png", "q-by-batch-facets.png", "rho-by-layer.png", "tau-by-layer.png", "layer-wise-update-magnitude.png", report_path.name]
    members = [{"path": name, "bytes": (args.output / name).stat().st_size, "sha256": _sha256(args.output / name)} for name in sorted(member_names)]
    member_root = canonical_hash([[row["path"], row["bytes"], row["sha256"]] for row in members])
    manifest = {
        "schema": "odeedit.s06.official-layer-realization-debt.sequential-analysis-manifest.v1",
        "instruction_id": INSTRUCTION_ID,
        "nonce": NONCE,
        "status": "SEQUENTIAL_OBSERVATIONAL_STUDY_TERMINAL_VALID",
        "run_source": {"head": run_head, "tree": run_tree},
        "run_sources": sorted(
            {
                (cell["result"]["source"]["head"], cell["result"]["source"]["tree"])
                for cell in cells
            }
        ),
        "analysis_builder_source": {"head": builder_head, "tree": builder_tree},
        "jobs": list(args.job_id),
        "technical_exclusions": list(args.technical_exclusion),
        "denominators": {"cells": 4, "batches": 40, "requests": 400, "valid_cells": 4, "valid_batches": 40, "valid_requests": 400},
        "invariants": {"direct_z_compute_count": 400, "direct_z_recompute_count": 0, "layer_loop_observation_copy_count": 200, "terminal_forward_count": 40, "weight_continuity_links": 36, "terminal_w0_restore_count": 4, "terminal_cache_restore_count": 4, "failure_count": 0, "nonfinite_count": 0, "imputation_count": 0},
        "execution_semantics": "B1_TO_B10_CUMULATIVE_W_CACHE_SEQUENTIAL",
        "stream": {"root": STREAM_IDENTITY, "order": ORDER_IDENTITY, "evaluator": EVALUATOR_IDENTITY},
        "external_inputs": sorted(external, key=lambda row: (row["kind"], row["path"])),
        "members": members,
        "member_root": member_root,
        "raw_prompt_logit_generation_git_count": 0,
        "easyedit_source_edit_count": 0,
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_path = args.output / "analysis-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "odeedit.s06.official-layer-realization-debt.sequential-rooted-analysis-receipt.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "SEQUENTIAL_OBSERVATIONAL_STUDY_TERMINAL_VALID",
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
    parser.add_argument("--expected-run-head", action="append", required=True)
    parser.add_argument("--result-root", type=Path, action="append", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--focused-gate", type=Path, required=True)
    parser.add_argument("--dry-plan", type=Path, required=True)
    parser.add_argument("--independent-package", type=Path, required=True)
    parser.add_argument("--job-id", action="append", required=True)
    parser.add_argument("--technical-exclusion", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
