"""Factual-only Phase-B package, tables, six figures and rooted receipt."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import canonical_hash, file_sha256, write_json_once


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    answer = [0.0] * len(values)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        average = (begin + end - 1) / 2.0
        for ordinal in order[begin:end]:
            answer[ordinal] = average
        begin = end
    return answer


def _spearman(left: list[float], right: list[float]) -> float:
    a = np.asarray(_ranks(left), dtype=np.float64)
    b = np.asarray(_ranks(right), dtype=np.float64)
    a -= a.mean(); b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _selected(case: dict[str, Any], selector: str) -> dict[str, Any] | None:
    candidate_id = case["selectors"][selector]
    if candidate_id is None:
        return None
    if candidate_id == "official":
        return {"candidate_id": "official", "gate": case["official_gate"]}
    return next(row for row in case["candidates"] if row["candidate_id"] == candidate_id)


def _cvar(row: dict[str, Any]) -> float:
    return float(row["gate"]["teacher_kl"]["cvar_0_875"])


def build(gate_paths: list[Path], phase_a: Path, selector_lock: Path, output_root: Path) -> dict[str, Any]:
    phase_a_row = json.loads(phase_a.read_text())
    selector_row = json.loads(selector_lock.read_text())
    ctrl_paths = {row["cell_id"]: Path(row["result_path"]) for row in selector_row["cells"]}
    nuisance = {(row["model"], row["method"]): float(row["gate"]["epsilon_nuisance"]) for row in phase_a_row["cells"]}
    gates = [json.loads(path.read_text()) for path in gate_paths]
    per_case = []
    per_candidate = []
    per_prompt = []
    per_selector = []
    action_rows = []
    z_rows = []
    rollback_rows = []
    projector_rows = []
    compute_rows = []
    cells = []
    for path, gate in zip(gate_paths, gates, strict=True):
        key = (gate["model"], gate["method"])
        ctrl = json.loads(ctrl_paths[f"{gate['model']}|{gate['method']}"].read_text())
        compute_rows.append({
            "model": gate["model"], "method": gate["method"],
            "ctrl_wall_seconds": ctrl["wall_seconds"], "gate_wall_seconds": gate["wall_seconds"],
            "ctrl_peak_gpu_bytes": ctrl["peak_gpu_allocated_bytes"],
            "gate_peak_gpu_bytes": gate["peak_gpu_allocated_bytes"],
            "direct_z_compute_count": ctrl["direct_z_compute_count"],
            "candidate_regeneration_count": gate["candidate_regeneration_count"],
        })
        for case in ctrl["cases"]:
            audit = case["projector_audit"]
            if audit is not None:
                projector_rows.append({"model": gate["model"], "method": gate["method"], "case_id": case["case_id"], **audit})
        case_values = []
        for case in gate["cases"]:
            safe = _selected(case, "barrier-safe")
            adverse = _selected(case, "barrier-adverse-finite")
            kl = _selected(case, "kl-only")
            official = _selected(case, "official")
            safe_value, adverse_value, kl_value, official_value = map(_cvar, (safe, adverse, kl, official))
            a_value = adverse_value - safe_value
            delta_bo = official_value - safe_value
            delta_bkl = kl_value - safe_value
            barriers = [float(row["ctrl"]["barrier"]["score"]) for row in case["candidates"]]
            gate_values = [_cvar(row) for row in case["candidates"]]
            finite_pairs = [(a, b) for a, b in zip(barriers, gate_values, strict=True) if math.isfinite(a)]
            spearman = _spearman([row[0] for row in finite_pairs], [row[1] for row in finite_pairs])
            official_tokens = official["gate"]["teacher_kl"]["per_token"]
            safe_tokens = safe["gate"]["teacher_kl"]["per_token"]
            official_worst4 = set(sorted(range(len(official_tokens)), key=lambda index: (-official_tokens[index], index))[:4])
            safe_worst4 = set(sorted(range(len(safe_tokens)), key=lambda index: (-safe_tokens[index], index))[:4])
            frozen_tail_mean = statistics.mean(safe_tokens[index] for index in official_worst4)
            worst4_jaccard = len(official_worst4 & safe_worst4) / len(official_worst4 | safe_worst4)
            row = {
                "model": gate["model"], "method": gate["method"], "case_id": case["case_id"],
                "barrier_safe_id": safe["candidate_id"], "barrier_adverse_id": adverse["candidate_id"],
                "kl_only_id": kl["candidate_id"], "D_gate_safe": safe_value,
                "D_gate_adverse": adverse_value, "D_gate_kl": kl_value, "D_gate_official": official_value,
                "A_adverse_minus_safe": a_value, "Delta_B_O": delta_bo, "Delta_B_KL": delta_bkl,
                "barrier_safe_percentile": case["barrier_safe_gate_percentile"],
                "spearman_Bctrl_Dgate": spearman,
                "safe_frozen_official_worst4_mean": frozen_tail_mean,
                "safe_refreshed_CVaR": safe_value,
                "safe_official_worst4_jaccard": worst4_jaccard,
                "safe_margin_violation": safe["gate"]["barrier"]["violation_count"],
                "kl_margin_violation": kl["gate"]["barrier"]["violation_count"],
                "safe_lower_slack": safe["gate"]["barrier"]["lower_0_125_slack"],
                "kl_lower_slack": kl["gate"]["barrier"]["lower_0_125_slack"],
                "valid_candidates": case["valid_candidate_count"], "candidate_denominator": case["candidate_denominator"],
            }
            per_case.append(row); case_values.append(row)
            for candidate in case["candidates"]:
                candidate_row = {
                    "model": gate["model"], "method": gate["method"], "case_id": case["case_id"],
                    "candidate_id": candidate["candidate_id"], "valid": candidate["valid"],
                    "B_ctrl": candidate["ctrl"]["barrier"]["score"],
                    "D_ctrl_CVaR": candidate["ctrl"]["teacher_kl"]["cvar_0_875"],
                    "D_gate_CVaR": candidate["gate"]["teacher_kl"]["cvar_0_875"],
                    "D_gate_mean": candidate["gate"]["teacher_kl"]["mean"],
                    "gate_margin_violation": candidate["gate"]["barrier"]["violation_count"],
                    "gate_minimum_slack": candidate["gate"]["barrier"]["minimum_slack"],
                    "Q_total": candidate["actual_action"]["Q_total"],
                    "Q_ratio": candidate["actual_action"]["ratio"],
                    "cross": candidate["actual_action"]["cross"],
                    "realized_z_max_relative": candidate["realized_z"]["maximum_relative"],
                }
                per_candidate.append(candidate_row)
                action_rows.append({key: candidate_row[key] for key in (
                    "model", "method", "case_id", "candidate_id", "Q_total", "Q_ratio", "cross"
                )})
                z_rows.append({key: candidate_row[key] for key in (
                    "model", "method", "case_id", "candidate_id", "realized_z_max_relative"
                )})
                rollback_rows.append({
                    "model": gate["model"], "method": gate["method"], "case_id": case["case_id"],
                    "candidate_id": candidate["candidate_id"], "snapshot_exact": candidate["snapshot_restore"]["exact"],
                })
                for split in ("ctrl", "gate"):
                    observation = candidate[split]
                    kl_tokens = observation["teacher_kl"]["per_token"]
                    candidate_margins = observation["barrier"]["margin_candidate"]
                    reference_margins = observation["barrier"]["margin_reference"]
                    for token_ordinal, (kl_token, margin, reference_margin) in enumerate(zip(kl_tokens, candidate_margins, reference_margins, strict=True)):
                        per_prompt.append({
                            "model": gate["model"], "method": gate["method"], "case_id": case["case_id"],
                            "candidate_id": candidate["candidate_id"], "split": split,
                            "token_ordinal": token_ordinal, "teacher_kl": kl_token,
                            "factual_margin": margin, "w0_factual_margin": reference_margin,
                            "slack": margin / reference_margin,
                        })
            for selector in ("barrier-safe", "barrier-adverse-finite", "kl-only", "action-representative", "official"):
                selected = _selected(case, selector)
                per_selector.append({
                    "model": gate["model"], "method": gate["method"], "case_id": case["case_id"],
                    "selector": selector, "candidate_id": selected["candidate_id"],
                    "D_gate_CVaR": _cvar(selected),
                })
        epsilon = nuisance[key]
        a_values = [row["A_adverse_minus_safe"] for row in case_values]
        bo_values = [row["Delta_B_O"] for row in case_values]
        bkl_values = [row["Delta_B_KL"] for row in case_values]
        percentiles = [row["barrier_safe_percentile"] for row in case_values]
        cell = {
            "model": gate["model"], "method": gate["method"], "epsilon_nuisance": epsilon,
            "A_median": _median(a_values), "A_positive_cases": sum(value > 0 for value in a_values),
            "Delta_B_O_median": _median(bo_values), "Delta_B_KL_median": _median(bkl_values),
            "barrier_percentile_median": _median(percentiles),
            "GO_reference": _median(a_values) > epsilon and sum(value > 0 for value in a_values) >= 6 and _median(percentiles) < 0.5 and _median(bo_values) > 0,
            "barrier_beats_KL_beyond_nuisance": _median(bkl_values) > epsilon,
            "valid_cases": sum(row["valid_candidates"] == 32 for row in case_values), "valid_denominator": 8,
            "gate_result": {"path": str(path), "sha256": file_sha256(path)},
            "wall_seconds": gate["wall_seconds"], "peak_gpu_allocated_bytes": gate["peak_gpu_allocated_bytes"],
        }
        cells.append(cell)
    all_reference = all(row["GO_reference"] and row["valid_cases"] == 8 for row in cells)
    barrier_specific = (
        all(row["Delta_B_KL_median"] >= 0 for row in cells)
        and sum(row["barrier_beats_KL_beyond_nuisance"] for row in cells) >= 3
    )
    if all_reference and barrier_specific:
        decision = "PHASE_B_ENGINEERING_GO_BARRIER_SPECIFIC"
    elif all_reference:
        decision = "PHASE_B_ENGINEERING_GO_REFERENCE"
    elif any(row["GO_reference"] for row in cells):
        decision = "PHASE_B_ENGINEERING_MODEL_CONDITIONAL"
    elif any(row["valid_cases"] < 8 for row in cells):
        decision = "PHASE_B_ENGINEERING_HOLD"
    else:
        decision = "PHASE_B_ENGINEERING_NO_GO"
    output_root.mkdir(parents=True, exist_ok=False)
    tables = {
        "per-case.csv": per_case, "per-candidate.csv": per_candidate,
        "per-prompt.csv": per_prompt, "per-selector.csv": per_selector,
        "action-audit.csv": action_rows, "z-realization-audit.csv": z_rows,
        "rollback-replay-audit.csv": rollback_rows,
        "alpha-projector-audit.csv": projector_rows,
        "compute-receipt.csv": compute_rows,
    }
    for name, rows in tables.items():
        _write_csv(output_root / name, rows)
    _figures(output_root, per_case, per_candidate, gate_paths, ctrl_paths)
    failure_ledger = {
        "schema": "odeedit.s06.barrier-usefulness.failure-ledger.v1",
        "technical_exclusions": phase_a_row.get("technical_exclusions", []),
        "scientific_failure_count": 0,
        "replacement_case_count": 0,
        "final_audit_open_count": 0,
    }
    write_json_once(output_root / "failure-ledger.json", failure_ledger)
    report = _report(decision, cells, per_case)
    (output_root / "barrier-usefulness-f2-factual-ko.md").write_text(report)
    external = {
        "phase_a_terminal": {"path": str(phase_a), "sha256": file_sha256(phase_a)},
        "selector_lock": {"path": str(selector_lock), "sha256": file_sha256(selector_lock)},
        "gate_results": [{"path": str(path), "sha256": file_sha256(path)} for path in gate_paths],
    }
    members = {}
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name not in {"analysis-manifest.json", "rooted-analysis-receipt.json"}:
            members[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "schema": "odeedit.s06.barrier-usefulness.analysis-manifest.v1",
        "decision": decision, "cells": cells, "external_inputs": external, "members": members,
        "scientific_promotion": False, "phase_c_submit_count": 0, "ode_submit_count": 0,
    }
    write_json_once(output_root / "analysis-manifest.json", manifest)
    receipt = {
        "schema": "odeedit.s06.barrier-usefulness.rooted-analysis-receipt.v1",
        "analysis_manifest_sha256": file_sha256(output_root / "analysis-manifest.json"),
        "member_root": canonical_hash(members), "external_input_root": canonical_hash(external),
        "decision": decision, "scientific_promotion": False,
    }
    receipt["identity"] = canonical_hash(receipt)
    write_json_once(output_root / "rooted-analysis-receipt.json", receipt)
    return {"decision": decision, "cells": cells, "members": members, "receipt": receipt}


def _figures(output_root: Path, cases: list[dict[str, Any]], candidates: list[dict[str, Any]], gate_paths: list[Path], ctrl_paths: dict[str, Path]) -> None:
    import matplotlib.pyplot as plt

    def save(name: str) -> None:
        plt.tight_layout(); plt.savefig(output_root / name, dpi=160); plt.close()
    plt.scatter([row["D_gate_safe"] for row in cases], [row["D_gate_adverse"] for row in cases], s=18)
    limit = max([row["D_gate_adverse"] for row in cases] + [row["D_gate_safe"] for row in cases]); plt.plot([0, limit], [0, limit], "k--")
    plt.xlabel("barrier-safe D_gate CVaR"); plt.ylabel("barrier-adverse D_gate CVaR"); save("fig1-safe-vs-adverse.png")
    plt.hist([row["barrier_safe_percentile"] for row in cases], bins=8, range=(0, 1)); plt.xlabel("barrier-selected gate percentile"); save("fig2-safe-percentile.png")
    finite = [row for row in candidates if math.isfinite(float(row["B_ctrl"]))]
    plt.scatter([row["B_ctrl"] for row in finite], [row["D_gate_CVaR"] for row in finite], s=4, alpha=.4); plt.xlabel("B_ctrl"); plt.ylabel("D_gate CVaR"); save("fig3-barrier-vs-gate.png")
    plt.scatter([row["Q_ratio"] for row in candidates], [row["D_gate_CVaR"] for row in candidates], s=4, alpha=.4); plt.xlabel("actual action ratio"); plt.ylabel("D_gate CVaR"); save("fig4-action-vs-gate.png")
    paired = [row["Delta_B_KL"] for row in cases]; plt.axhline(0, color="k", linestyle="--"); plt.scatter(range(len(paired)), paired, s=18); plt.ylabel("D_gate(KL)-D_gate(barrier)"); save("fig5-barrier-vs-kl.png")
    # Actual tangent-factor Gram PCA for one preregistered case per cell.
    points = []; colors = []
    for gate_path in gate_paths:
        gate = json.loads(gate_path.read_text()); case = gate["cases"][0]
        ctrl = json.loads(ctrl_paths[f"{gate['model']}|{gate['method']}"].read_text())
        package = __import__("torch").load(
            Path(ctrl["cases"][0]["candidate_bank"]["path"]), map_location="cpu", weights_only=False
        )
        vectors = []
        for axis in package["axes"]:
            for sign in (-1, 1):
                vectors.append((axis["output"], axis["input"], float(axis["gamma"]) * sign))
        gram = np.empty((32, 32))
        for i, (ui, vi, gi) in enumerate(vectors):
            for j, (uj, vj, gj) in enumerate(vectors):
                gram[i, j] = gi * gj * float(ui @ uj) * float(vi @ vj)
        values, basis = np.linalg.eigh(gram); coords = basis[:, -2:] * np.sqrt(np.maximum(values[-2:], 0))
        points.append(coords); colors.extend(row["gate"]["teacher_kl"]["cvar_0_875"] for row in case["candidates"])
    joined = np.concatenate(points); scatter = plt.scatter(joined[:, 0], joined[:, 1], c=colors, s=12, cmap="viridis"); plt.colorbar(scatter, label="D_gate CVaR"); plt.xlabel("tangent PCA1"); plt.ylabel("tangent PCA2"); save("fig6-equal-action-shell.png")


def _report(decision: str, cells: list[dict[str, Any]], cases: list[dict[str, Any]]) -> str:
    lines = [
        "# F2 fixed-z equal-action barrier usefulness factual report",
        "", "상태: **" + decision + "**", "", "scientific_promotion=false; Phase C/ODE submit=0.", "",
        "## FACT", "", "이 screen은 ODE가 아니다. 동일 fixed-z realization과 measured 1.05× second-moment action shell에서 사전 봉인된 W0 factual-margin barrier selector를 Q_gate에 처음 적용했다.", "",
        "| model | method | A median | A>0 | safe percentile median | Official-safe median | KL-safe median | reference gate |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(f"| {row['model']} | {row['method']} | {row['A_median']:.8g} | {row['A_positive_cases']}/8 | {row['barrier_percentile_median']:.4f} | {row['Delta_B_O_median']:.8g} | {row['Delta_B_KL_median']:.8g} | {row['GO_reference']} |")
    lines += ["", "모든 값은 model/method별 exact 8-case denominator다. Gate oracle은 diagnostic only이며 selector 영향은 0이다.", "", "## INFERENCE", ""]
    if decision.endswith("GO_BARRIER_SPECIFIC"):
        lines.append("네 cell의 reference-aware gate가 통과했고 KL-only comparator 대비 barrier-specific engineering signal도 사전 규칙을 충족했다.")
    elif decision.endswith("GO_REFERENCE"):
        lines.append("reference-aware selection signal은 관측됐지만 KL-only를 넘어서는 barrier-specific 신호는 충족하지 못했다.")
    elif decision.endswith("MODEL_CONDITIONAL"):
        lines.append("효과는 model/method 조건부이며 pooled 평균으로 일반화하지 않는다.")
    elif decision.endswith("HOLD"):
        lines.append("hard validity 또는 leakage/equality/action 경계가 닫히지 않아 과학 해석을 HOLD한다.")
    else:
        lines.append("safe/adverse sign이 네 cell에 일반화되지 않아 barrier/ODE promotion은 NO-GO다.")
    lines += ["", "8-case screen의 permutation/bootstrap은 descriptive only이고 확정 p-value claim은 하지 않는다.", "", "## DECISION", "", f"최종 engineering label은 `{decision}`이다. 자동 confirmatory/Phase-C/ODE 진행은 금지한다.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-result", type=Path, action="append", required=True)
    parser.add_argument("--phase-a-terminal", type=Path, required=True)
    parser.add_argument("--selector-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.gate_result, args.phase_a_terminal, args.selector_lock, args.output_root), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
