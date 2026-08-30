"""Build the factual HOLD package when the preregistered W0 anchor gate fails."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from .hashing import canonical_hash, file_sha256, write_json_once


ANCHOR_PATTERN = re.compile(
    r"W0 admissible (?P<split>ctrl|gate) anchors absent for case "
    r"(?P<case_id>\d+): (?P<admissible>\d+)/(?P<required>\d+)"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _anchor_failure(model: str, path: Path) -> dict[str, Any]:
    text = path.read_text()
    matches = list(ANCHOR_PATTERN.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one typed anchor failure in {path}")
    match = matches[0]
    return {
        "model": model,
        "job": "29315_0" if model == "llama3-8b-inst" else "29315_1",
        "case_id": int(match.group("case_id")),
        "split": match.group("split"),
        "admissible": int(match.group("admissible")),
        "required": int(match.group("required")),
        "deficit": int(match.group("required")) - int(match.group("admissible")),
        "result": "SCIENTIFIC_ANCHOR_ADMISSIBILITY_HOLD",
        "stderr_path": str(path),
        "stderr_sha256": file_sha256(path),
        "stderr_bytes": path.stat().st_size,
        "candidate_generation_count": 0,
        "selector_lock_count": 0,
        "q_gate_edited_access_count": 0,
    }


def _placeholder_svg(path: Path, title: str) -> None:
    text = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="280" '
        'viewBox="0 0 960 280">\n'
        '<rect width="960" height="280" fill="#fafafa"/>\n'
        f'<text x="480" y="105" text-anchor="middle" font-family="sans-serif" '
        f'font-size="25">{title}</text>\n'
        '<text x="480" y="155" text-anchor="middle" font-family="sans-serif" '
        'font-size="20">NOT_GENERATED — Q_gate never opened</text>\n'
        '<text x="480" y="195" text-anchor="middle" font-family="sans-serif" '
        'font-size="16">Preregistered W0 anchor admissibility gate failed before candidate creation.</text>\n'
        '</svg>\n'
    )
    path.write_text(text)


def build(
    phase_a_path: Path,
    screen_paths: list[Path],
    llama_stderr: Path,
    qwen_stderr: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"create-once output exists: {output_root}")
    output_root.mkdir(parents=True)
    phase_a = json.loads(phase_a_path.read_text())
    if phase_a["status"] != "PHASE_A_PASS":
        raise RuntimeError("HOLD builder requires the sealed Phase-A PASS receipt")
    screens = [json.loads(path.read_text()) for path in screen_paths]
    if len(screens) != 4 or any(row["status"] != "TERMINAL_VALID" for row in screens):
        raise RuntimeError("expected four terminal-valid corrected Phase-A screen cells")

    phase_a_cells = []
    phase_a_cases = []
    phase_a_candidates = []
    compute_rows = []
    for summary_cell in phase_a["cells"]:
        gate = summary_cell["gate"]
        phase_a_cells.append({
            "model": summary_cell["model"],
            "method": summary_cell["method"],
            "valid_cases": gate["valid_cases"],
            "valid_denominator": gate["valid_denominator"],
            "spread_gt_3x_nuisance_cases": gate["spread_gt_3x_nuisance_cases"],
            "spread_denominator": gate["spread_denominator"],
            "median_spread": gate["median_spread"],
            "epsilon_nuisance": gate["epsilon_nuisance"],
            "phase_a_pass": gate["phase_a_pass"],
        })
    for source_path, screen in zip(screen_paths, screens, strict=True):
        candidate_count = 0
        direct_z_count = 0
        for case in screen["cases"]:
            direct_z_count += int(case["direct_z_compute_count"])
            phase_a_cases.append({
                "stage": "PHASE_A_CORRECTED_F1B",
                "model": screen["model"], "method": screen["method"],
                "case_id": case["case_id"],
                "candidate_denominator": case["candidate_denominator"],
                "valid_candidates": case["valid_candidate_count"],
                "functional_spread": case["functional_spread"],
                "epsilon_nuisance": case["epsilon_nuisance"],
                "spread_gt_3x_nuisance": case["spread_gt_3x_nuisance"],
                "direct_z_compute_count": case["direct_z_compute_count"],
                "direct_z_recompute_count": case["direct_z_recompute_count"],
                "native_context_count": case["native_context_count"],
                "w0_restore": case["w0_pointer_bytes_restore"],
            })
            for candidate in case["candidates"]:
                candidate_count += 1
                phase_a_candidates.append({
                    "stage": "PHASE_A_CORRECTED_F1B",
                    "model": screen["model"], "method": screen["method"],
                    "case_id": case["case_id"], "candidate_id": candidate["candidate_id"],
                    "valid": candidate["valid"],
                    "functional_cvar": candidate["observation"]["functional"]["cvar_0_875"],
                    "action_ratio": candidate["actual_action"]["total_ratio"],
                    "action_cross": candidate["actual_action"]["cross_delta_N"],
                    "equality_relative": candidate["equality_NA_star_relative"],
                    "realized_z_max_relative": candidate["realized_z"]["maximum_relative"],
                    "snapshot_restore_exact": candidate["snapshot_restore"]["exact"],
                })
        compute_rows.append({
            "stage": "PHASE_A_CORRECTED_F1B", "model": screen["model"],
            "method": screen["method"], "case_count": len(screen["cases"]),
            "candidate_count": candidate_count, "direct_z_compute_count": direct_z_count,
            "wall_seconds": screen["wall_seconds"],
            "peak_gpu_allocated_bytes": screen["peak_gpu_allocated_bytes"],
            "source_path": str(source_path), "source_sha256": file_sha256(source_path),
        })

    anchor_rows = [
        _anchor_failure("llama3-8b-inst", llama_stderr),
        _anchor_failure("qwen2.5-7b-inst", qwen_stderr),
    ]
    if {row["case_id"] for row in anchor_rows} != {5685}:
        raise RuntimeError("unexpected failed fresh-case identity")
    _write_csv(output_root / "phase-a-cells.csv", phase_a_cells)
    _write_csv(output_root / "per-case.csv", phase_a_cases)
    _write_csv(output_root / "per-candidate.csv", phase_a_candidates)
    _write_csv(output_root / "anchor-admissibility.csv", anchor_rows)
    _write_csv(output_root / "compute-receipt.csv", compute_rows)
    unavailable = [{
        "status": "NOT_GENERATED_Q_GATE_NEVER_OPENED",
        "reason": "preregistered W0 factual-anchor admissibility gate failed",
    }]
    for name in (
        "per-prompt.csv", "per-selector.csv", "action-audit.csv",
        "z-realization-audit.csv", "rollback-replay-audit.csv", "alpha-projector-audit.csv",
    ):
        _write_csv(output_root / name, unavailable)
    figure_names = [
        ("fig1-safe-vs-adverse.svg", "Barrier-safe vs barrier-adverse gate CVaR"),
        ("fig2-safe-percentile.svg", "Barrier-selected gate percentile"),
        ("fig3-barrier-vs-gate.svg", "B_ctrl vs D_gate"),
        ("fig4-action-vs-gate.svg", "Actual action vs gate CVaR"),
        ("fig5-barrier-vs-kl.svg", "Barrier vs KL-only"),
        ("fig6-equal-action-shell.svg", "Equal-action shell colored by gate risk"),
    ]
    for name, title in figure_names:
        _placeholder_svg(output_root / name, title)

    failure_ledger = {
        "schema": "odeedit.s06.barrier-usefulness.failure-ledger.v1",
        "decision": "PHASE_B_ENGINEERING_HOLD",
        "phase_a_status": "PASS",
        "anchor_failures": anchor_rows,
        "technical_exclusions": phase_a.get("technical_exclusions", []),
        "replacement_case_count": 0,
        "pool_expansion_count": 0,
        "candidate_generation_count": 0,
        "selector_lock_count": 0,
        "q_gate_edited_access_count": 0,
        "phase_c_submit_count": 0,
        "ode_submit_count": 0,
    }
    write_json_once(output_root / "failure-ledger.json", failure_ledger)
    report = _report(phase_a_cells, anchor_rows)
    (output_root / "barrier-usefulness-f2-factual-ko.md").write_text(report)

    external = {
        "phase_a_terminal": {"path": str(phase_a_path), "sha256": file_sha256(phase_a_path)},
        "phase_a_screen_results": [
            {"path": str(path), "sha256": file_sha256(path)} for path in screen_paths
        ],
        "anchor_failure_logs": [
            {"path": str(path), "sha256": file_sha256(path)} for path in (llama_stderr, qwen_stderr)
        ],
    }
    members = {}
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name not in {"analysis-manifest.json", "rooted-analysis-receipt.json"}:
            members[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "schema": "odeedit.s06.barrier-usefulness.analysis-manifest.v1",
        "decision": "PHASE_B_ENGINEERING_HOLD",
        "hold_reason": "PREREGISTERED_W0_ANCHOR_ADMISSIBILITY_INSUFFICIENT",
        "phase_a": {"status": "PASS", "cells": phase_a_cells},
        "phase_b": {
            "candidate_generation_count": 0, "selector_lock_count": 0,
            "q_gate_edited_access_count": 0, "gate_figures_are_status_placeholders": True,
        },
        "external_inputs": external, "members": members,
        "scientific_promotion": False, "phase_c_submit_count": 0, "ode_submit_count": 0,
    }
    write_json_once(output_root / "analysis-manifest.json", manifest)
    receipt = {
        "schema": "odeedit.s06.barrier-usefulness.rooted-analysis-receipt.v1",
        "analysis_manifest_sha256": file_sha256(output_root / "analysis-manifest.json"),
        "member_root": canonical_hash(members), "external_input_root": canonical_hash(external),
        "decision": "PHASE_B_ENGINEERING_HOLD", "scientific_promotion": False,
    }
    receipt["identity"] = canonical_hash(receipt)
    write_json_once(output_root / "rooted-analysis-receipt.json", receipt)
    return {"manifest": manifest, "receipt": receipt}


def _report(cells: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> str:
    lines = [
        "# F2 fixed-z equal-action barrier usefulness 사실 보고서",
        "", "상태: **PHASE_B_ENGINEERING_HOLD**", "",
        "핵심 중단 사유는 사전 등록된 W0 factual anchor admissibility 부족이다. "
        "Q_gate edited-candidate 평가는 한 번도 열리지 않았으므로 barrier usefulness 결과는 없다.",
        "", "## FACT — Phase A corrected F1b", "",
        "| model | method | valid cases | spread > 3×noise | median spread | epsilon nuisance | gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in cells:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['valid_cases']}/{row['valid_denominator']} | "
            f"{row['spread_gt_3x_nuisance_cases']}/{row['spread_denominator']} | "
            f"{row['median_spread']:.9g} | {row['epsilon_nuisance']:.9g} | PASS |"
        )
    lines += [
        "", "네 cell 모두 direct-z 1회/case, candidate 8/8, exact reset/action/all-context realized-z, "
        "left-padding, FULL-FP32 경계를 통과했다.",
        "", "## FACT — Phase B entry anchor gate", "",
        "| model | failed case | split | admissible | required | deficit |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in anchors:
        lines.append(
            f"| {row['model']} | {row['case_id']} | {row['split']} | "
            f"{row['admissible']} | {row['required']} | {row['deficit']} |"
        )
    lines += [
        "", "사전 proposal은 각 fresh case/split에서 W0가 target_true의 모든 token을 strict하게 지지하고 "
        "locked numerical floor보다 큰 anchor를 정확히 32개 요구한다. 부족하면 typed HOLD이며 "
        "pool 확장, case 교체, margin floor 변경은 0이다.",
        "", "## INFERENCE", "",
        "이번 중단은 Phase A의 fixed-z non-uniqueness 소실이 아니라 Phase B reference-anchor 구성 실패다. "
        "Barrier-safe/adverse selector, KL comparator, candidate bank 및 held-out Q_gate가 생성·평가되지 않아 "
        "barrier 유용성의 방향이나 크기는 판단할 수 없다.",
        "", "## DECISION", "",
        "`PHASE_B_ENGINEERING_HOLD`. candidate generation=0, selector lock=0, Q_gate edited access=0, "
        "case replacement=0, Phase C/ODE submit=0, scientific_promotion=false.",
        "", "요구된 gate 기반 표와 6개 figure는 결과가 없는 상태를 수치처럼 보이지 않도록 "
        "`NOT_GENERATED_Q_GATE_NEVER_OPENED` 상태 표/placeholder로만 남겼다.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, action="append", required=True)
    parser.add_argument("--llama-stderr", type=Path, required=True)
    parser.add_argument("--qwen-stderr", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = build(
        args.phase_a, args.screen_result, args.llama_stderr, args.qwen_stderr, args.output_root
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
