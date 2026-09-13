"""Build the canonical factual package for an early FzCB-Edit boundary.

The builder is analysis-only.  It reads immutable result JSON files and never
imports the controller or mutates model/result state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _metric(aggregate: dict[str, Any], name: str, field: str) -> Any:
    return aggregate.get(name, {}).get(field, "NOT_RECORDED")


def _arm_row(arm: str, status: str, aggregate: dict[str, Any], mechanism: dict[str, Any]) -> dict[str, Any]:
    terminal = mechanism.get("terminal") or {}
    waypoints = mechanism.get("waypoints") or []
    barriers = [((item.get("barrier") or {}).get("rectification") or {}) for item in waypoints]
    return {
        "arm": arm,
        "status": status,
        "rewrite_new_nll_mean": _metric(aggregate, "rewrite_target_new", "nll_mean"),
        "rewrite_new_nll_median": _metric(aggregate, "rewrite_target_new", "nll_median"),
        "rewrite_new_nll_p90": _metric(aggregate, "rewrite_target_new", "nll_p90"),
        "rewrite_new_strict_num": _metric(aggregate, "rewrite_target_new", "strict_numerator"),
        "rewrite_new_strict_den": _metric(aggregate, "rewrite_target_new", "strict_denominator"),
        "rewrite_true_nll_mean": _metric(aggregate, "rewrite_target_true", "nll_mean"),
        "rephrase_new_nll_mean": _metric(aggregate, "rephrase_target_new", "nll_mean"),
        "rephrase_new_nll_median": _metric(aggregate, "rephrase_target_new", "nll_median"),
        "rephrase_new_nll_p90": _metric(aggregate, "rephrase_target_new", "nll_p90"),
        "rephrase_new_strict_num": _metric(aggregate, "rephrase_target_new", "strict_numerator"),
        "rephrase_new_strict_den": _metric(aggregate, "rephrase_target_new", "strict_denominator"),
        "rephrase_true_nll_mean": _metric(aggregate, "rephrase_target_true", "nll_mean"),
        "locality_true_nll_mean": _metric(aggregate, "locality_target_true", "nll_mean"),
        "locality_true_nll_median": _metric(aggregate, "locality_target_true", "nll_median"),
        "locality_true_nll_p90": _metric(aggregate, "locality_target_true", "nll_p90"),
        "waypoint_count": len(waypoints) if arm != "PRE_EDIT" else 0,
        "barrier_active_count": sum(bool(item.get("barrier_active")) for item in barriers),
        "terminal_absolute_closure": terminal.get("absolute_closure", "NOT_APPLICABLE"),
        "terminal_budget": terminal.get("budget", "NOT_APPLICABLE"),
        "terminal_spent_action": terminal.get("spent_action", "NOT_APPLICABLE"),
        "terminal_budget_error": terminal.get("budget_error", "NOT_APPLICABLE"),
        "terminal_weight_root": terminal.get("weight_root", "NOT_APPLICABLE"),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-result", type=Path, required=True)
    parser.add_argument("--qwen-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--science-head", required=True)
    parser.add_argument("--science-tree", required=True)
    parser.add_argument("--llama-job", required=True)
    parser.add_argument("--qwen-job", required=True)
    args = parser.parse_args()

    llama = json.loads(args.llama_result.read_text())
    qwen = json.loads(args.qwen_result.read_text())
    if llama.get("status") != "TERMINAL_VALID":
        raise SystemExit("Llama input is not TERMINAL_VALID")
    if qwen.get("status") != "SCIENTIFIC_HOLD" or qwen.get("failure_type") != "ScientificBoundary":
        raise SystemExit("Qwen input is not the sealed ScientificBoundary")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    rows = [_arm_row("PRE_EDIT", "OBSERVATION_ONLY", llama["pre_edit"]["aggregate"], {})]
    for arm in llama["arms"]:
        rows.append(_arm_row(arm["arm"], arm["status"], arm["endpoint"]["aggregate"], arm["mechanism"]))

    table = out / "llama-b1-arm-summary.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    failures = {
        "schema": "odeedit.s06.fzcb-edit-main-method.failure-ledger.v1",
        "entries": [
            {
                "job": args.qwen_job,
                "model": qwen["model"],
                "batch_size": qwen["batch_size"],
                "classification": qwen["failure_type"],
                "status": qwen["status"],
                "evidence": qwen["failure"],
                "denominator": 0,
                "repair_or_relaxation": 0,
            }
        ],
    }
    failure_path = out / "failure-ledger.json"
    failure_path.write_bytes(json.dumps(failures, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")

    by_name = {row["arm"]: row for row in rows}
    refreshed = by_name["REFRESHED_EQUALITY_ONLY"]
    fzcb = by_name["FZCB"]
    same_endpoint = (
        refreshed["terminal_weight_root"] == fzcb["terminal_weight_root"]
        and refreshed["rewrite_new_nll_mean"] == fzcb["rewrite_new_nll_mean"]
        and refreshed["rephrase_new_nll_mean"] == fzcb["rephrase_new_nll_mean"]
    )

    def line(row: dict[str, Any]) -> str:
        return "| " + " | ".join(
            _fmt(row[key])
            for key in (
                "arm", "rewrite_new_nll_mean", "rewrite_true_nll_mean",
                "rephrase_new_nll_mean", "rephrase_true_nll_mean",
                "locality_true_nll_mean", "rewrite_new_strict_num", "rewrite_new_strict_den",
            )
        ) + " |"

    report_lines = [
        "# FzCB-Edit main method initial factual report — B1 scientific hold",
        "",
        "상태: **HOLD_AFTER_QWEN_B1_LOCAL_INFEASIBILITY**. Llama B1은 terminal-valid였지만 ",
        "Qwen B1에서 계약상 typed local barrier infeasibility가 발생했으므로 tolerance, ridge, ",
        "barrier, target 또는 sample을 완화하지 않았고 B10은 제출하지 않았다. promotion=false.",
        "",
        "## 1. 수식과 코드 경로",
        "",
        "수식→자산 상세 mapping은 `plans/updates/server4/fzcb-edit-main-method-equation-mapping.md`에 있다. ",
        "stock MEMIT `compute_z`는 request당 정확히 한 번 호출되어 다섯 arm이 같은 z*를 공유했다. ",
        "새 package는 target, basis/metric, matrix-free control map, equality/suffix KKT, scalar barrier, ",
        "predictor-corrector, transaction, telemetry를 분리했다. F2 q-KL/reference controller와 K0 promotion ",
        "logic은 import하지 않았다.",
        "",
        "## 2. 구현 경계",
        "",
        "- Exact: stock target 1회, MEMIT factor orientation, frozen M0 covariance metric, gauge removal, ",
        "  matrix-free JVP/VJP equality map, equality-only minimum-H control, one-barrier scalar rectification, ",
        "  net corrected-action accounting, s=1 terminal-only atomic commit/rollback.",
        "- Engineering approximation: suffix-value sensitivity는 reduced equality-null 2-axis finite difference ",
        "  correctness prototype다. exact production HVP로 해석하지 않는다.",
        "- Endpoint output NLL/locality는 observation-only이며 controller influence count=0이다.",
        "",
        "## 3. Focused gates",
        "",
        "G0 CPU/toy 7/7 PASS: factorized/dense parity, gauge/H, C/C^T adjoint, equality/suffix KKT, ",
        "frozen identity, scalar rectifier/local infeasibility, torch.func operator, rollback/atomic commit, ",
        "forbidden-import. py_compile, bash syntax, access gate도 PASS했다.",
        "",
        "Llama runtime: padding safety PASS, FULL-FP32, direct-z compute/recompute=1/0, five-arm shared target, ",
        "Official bypass parity=true, cache entry/terminal identity exact, W0 pointer/bytes restore=true/true.",
        "",
        "## 4. Llama B1 initial arm comparison",
        "",
        "분모는 sealed B1의 deterministic first request 1개다. rewrite prompt=1, rephrase prompts=2, ",
        "locality prompts=10이므로 일반 성능 결론이 아니라 technical/engineering smoke다.",
        "",
        "| arm | rewrite new NLL mean | rewrite true NLL mean | rephrase new NLL mean | rephrase true NLL mean | locality true NLL mean | rewrite strict num | den |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[line(row) for row in rows],
        "",
        "## 5. Range, suffix/barrier, corrector/action",
        "",
        "| arm | waypoints | barrier active | terminal closure | budget | spent action | budget error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[2:]:
        report_lines.append("| " + " | ".join(_fmt(row[key]) for key in (
            "arm", "waypoint_count", "barrier_active_count", "terminal_absolute_closure",
            "terminal_budget", "terminal_spent_action", "terminal_budget_error",
        )) + " |")
    report_lines += [
        "",
        f"Llama FZCB barrier active count는 {fzcb['barrier_active_count']}/{fzcb['waypoint_count']}이다. ",
        f"FZCB와 REFRESHED_EQUALITY_ONLY endpoint identity/핵심 NLL exact equality={str(same_endpoint).lower()}. ",
        "따라서 이 B1에서는 barrier contribution claim을 제거한다. STATIC_PATH가 rephrase-new NLL에서 ",
        "가장 낮았지만 request 1개 관측이므로 우월성이나 ODE necessity를 주장하지 않는다.",
        "",
        "## 6. Qwen typed boundary",
        "",
        f"job {args.qwen_job}은 `{qwen['failure']}`로 종료됐다. 이는 equality-null subspace에서 ",
        "현재 상태의 barrier가 불가능하다는 typed scientific/mathematical boundary다. 공식 target/writer ",
        "계산 후 검출됐으며 fallback, backtracking에 의한 재분류, tolerance/ridge/kappa 변경은 0이다. ",
        "Qwen endpoint arm denominator=0이며 성능 수치는 산출하지 않는다.",
        "",
        "## 7. 실행과 다음 권고",
        "",
        f"- Llama job {args.llama_job}: COMPLETED, terminal-valid, wall={_fmt(llama['wall_seconds'])}s, ",
        f"  peak allocated={llama['peak_gpu_allocated_bytes']} bytes.",
        f"- Qwen job {args.qwen_job}: FAILED exit2 by intentional typed boundary; immutable evidence, denominator0.",
        "- B10 jobs: 0. Qwen confirmation이 통과하지 않아 자동 흐름을 중단했다.",
        "- 다음 실행은 barrier feasibility/authority semantics에 대한 사용자/GH 판단 이후에만 권고한다. ",
        "  현재 numerical lock을 사후 완화한 재실행은 권고하지 않는다.",
    ]
    report = out / "fzcb-edit-main-method-initial-factual-ko.md"
    report.write_text("\n".join(item.rstrip() for item in report_lines) + "\n")

    raw_inputs = [
        {"path": str(args.llama_result), "sha256": _sha(args.llama_result), "bytes": args.llama_result.stat().st_size},
        {"path": str(args.qwen_result), "sha256": _sha(args.qwen_result), "bytes": args.qwen_result.stat().st_size},
    ]
    members = []
    for path in (report, table, failure_path):
        members.append({"path": path.name, "sha256": _sha(path), "bytes": path.stat().st_size})
    member_root = hashlib.sha256(_canonical(members)).hexdigest()
    manifest = {
        "schema": "odeedit.s06.fzcb-edit-main-method.analysis-manifest.v1",
        "status": "HOLD_AFTER_QWEN_B1_LOCAL_INFEASIBILITY",
        "science_source": {"head": args.science_head, "tree": args.science_tree},
        "jobs": {"llama_b1": args.llama_job, "qwen_b1": args.qwen_job, "b10": []},
        "denominators": {"llama_cases": 1, "qwen_terminal_cases": 0, "b10_cases": 0},
        "raw_inputs": raw_inputs,
        "members": members,
        "member_root": member_root,
        "scientific_promotion": False,
    }
    manifest_path = out / "analysis-manifest.json"
    manifest_path.write_bytes(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")
    receipt = {
        "schema": "odeedit.s06.fzcb-edit-main-method.rooted-receipt.v1",
        "manifest_sha256": _sha(manifest_path),
        "member_root": member_root,
        "raw_input_root": hashlib.sha256(_canonical(raw_inputs)).hexdigest(),
        "identity": hashlib.sha256(_canonical({
            "manifest_sha256": _sha(manifest_path), "member_root": member_root,
            "raw_input_root": hashlib.sha256(_canonical(raw_inputs)).hexdigest(),
        })).hexdigest(),
    }
    receipt_path = out / "rooted-receipt.json"
    receipt_path.write_bytes(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")


if __name__ == "__main__":
    main()
