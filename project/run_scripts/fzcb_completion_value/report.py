"""Raw-free K0/K1 factual package builder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

from .hashing import canonical_hash, canonical_json, file_sha256


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value == "+Infinity":
        return math.inf
    if value == "-Infinity":
        return -math.inf
    return math.nan


def _summary(result: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for case in result["cases"]:
        for candidate in case["candidates"]:
            rows.append({
                "case_id": int(case["case_id"]), "candidate_id": candidate["candidate_id"],
                "predicted_suffix_value": candidate["predicted_suffix_value"],
                "actual_suffix_action": candidate["actual_suffix_action"],
                "total_realized_cost": candidate["total_realized_cost"],
                "terminal_closure_relative": candidate["terminal_closure_relative"],
                "transition_closure_relative": candidate["transition"]["closure_relative"],
                "rollout_failure": candidate["rollout_failure"],
            })
    nulls = [direction for case in result["cases"] for direction in case["null_directions"]]
    failed = [row for row in rows if row["rollout_failure"] is not None]
    finite_closure = [_number(row["terminal_closure_relative"]) for row in rows if math.isfinite(_number(row["terminal_closure_relative"]))]
    summary = {
        "schema": "odeedit.s06.fzcb-completion-value.k0-summary.v1",
        "decision": "KILL_FZCB_COMPLETION_VALUE",
        "decision_reason": "K0_ACTUAL_SUFFIX_RANGE_FAILURE",
        "k1_submitted_count": 0,
        "case_count": len(result["cases"]), "candidate_count": len(rows),
        "valid_null_direction_count": sum(int(case["valid_null_direction_count"]) for case in result["cases"]),
        "null_direction_denominator": len(nulls),
        "max_adjoint_relative_error": max(float(case["actual_full_model_adjoint_gate"]["relative_error"]) for case in result["cases"]),
        "max_scaled_null_residual": max(float(direction["scaled_null_residual"]) for direction in nulls),
        "max_finite_terminal_closure_relative": max(finite_closure),
        "rollout_success_count": len(rows) - len(failed), "rollout_denominator": len(rows),
        "rollout_failure_count": len(failed),
        "rollout_failures": [
            {"case_id": row["case_id"], "candidate_id": row["candidate_id"], "failure": row["rollout_failure"]}
            for row in failed
        ],
        "case_value_spreads": {
            str(case["case_id"]): float(case["candidate_value_spread"])
            for case in result["cases"]
        },
        "case_repeat_noise": {
            str(case["case_id"]): float(case["candidate_value_repeat_noise"])
            for case in result["cases"]
        },
        "w0_restore_pass_count": sum(bool(case["w0_restore"]["bytes_exact"] and case["w0_restore"]["pointer_exact"]) for case in result["cases"]),
        "cache_restore_pass_count": sum(case["cache_entry"]["root"] == case["cache_terminal"]["root"] for case in result["cases"]),
        "direct_z_compute_count": sum(int(case["direct_z_compute_count"]) for case in result["cases"]),
        "direct_z_recompute_count": sum(int(case["direct_z_recompute_count"]) for case in result["cases"]),
        "full_fp32": bool(result["full_fp32"]),
        "peak_gpu_allocated_bytes": int(result["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": int(result["peak_gpu_reserved_bytes"]),
        "wall_seconds": float(result["wall_seconds"]),
        "scientific_promotion": False,
    }
    return summary, rows


def _report(summary: dict[str, Any], result_sha: str, preflight_sha: str, execution_head: str) -> str:
    spreads = summary["case_value_spreads"]
    lines = [
        "# FzCB completion-value fast-kill K0 사실 보고서",
        "",
        "## 결론",
        "",
        "**KILL_FZCB_COMPLETION_VALUE.** K1 Llama/Qwen 실행은 0이다. K0에서 대부분의 기계적 gate는 닫혔지만, "
        "case 3002의 seed-00 actual suffix rollout이 고정 range tolerance를 넘었다. 실패 candidate를 삭제하거나 "
        "tolerance를 바꾸지 않는 계약에 따라 K0에서 중단했다. 이는 output 보존/편집 성능 결론이 아니다.",
        "",
        "| 항목 | 결과 |",
        "|---|---:|",
        f"| K0 cases | {summary['case_count']}/4 |",
        f"| candidates | {summary['candidate_count']}/20 |",
        f"| valid equality-null directions | {summary['valid_null_direction_count']}/{summary['null_direction_denominator']} |",
        f"| actual suffix rollout 성공 | {summary['rollout_success_count']}/{summary['rollout_denominator']} |",
        f"| actual suffix rollout 실패 | {summary['rollout_failure_count']}/{summary['rollout_denominator']} |",
        f"| max full-model adjoint relative error | {summary['max_adjoint_relative_error']:.9g} |",
        f"| max scaled null residual | {summary['max_scaled_null_residual']:.9g} |",
        f"| max finite terminal closure | {summary['max_finite_terminal_closure_relative']:.9g} |",
        f"| W0 exact restore | {summary['w0_restore_pass_count']}/4 |",
        f"| cache exact restore | {summary['cache_restore_pass_count']}/4 |",
        f"| direct-z compute/recompute | {summary['direct_z_compute_count']}/{summary['direct_z_recompute_count']} |",
        "",
        "## Scientific stop 근거",
        "",
        "case 3002 seed-00의 actual suffix equality solve는 상대 range residual `0.000489372702`였고, "
        "고정 tolerance는 `0.00048828125`였다. 초과량은 약 `1.09145e-6`이다. 이 candidate의 "
        "actual suffix action과 terminal closure는 `+Infinity`로 보존했다. 나머지 19개 candidate는 finite terminal이었다.",
        "",
        "## K0 value spread 및 repeat floor",
        "",
        "| case | candidate V_suf spread | repeated-solve noise | 3×noise 초과 |",
        "|---:|---:|---:|:---:|",
    ]
    for case_id, spread in spreads.items():
        noise = summary["case_repeat_noise"][case_id]
        lines.append(f"| {case_id} | {spread:.12g} | {noise:.12g} | {'PASS' if spread > 3 * noise else 'FAIL'} |")
    lines += [
        "",
        "spread는 모든 case에서 exact repeat noise 0보다 컸지만, K0 actual rollout range gate 실패가 우선한다. "
        "따라서 predictive ordering 가설을 K1 denominator에서 검정하지 않았다.",
        "",
        "## 구현·불변식",
        "",
        "- Llama3-8B-Instruct × Official MEMIT direct-z, FULL-FP32.",
        "- explicit Kronecker 0, dense inverse 0, value-gradient/CBF/full-QCQP/AlphaEdit 0.",
        "- actual full-model JVP/VJP adjoint, fixed-z corrector, exact-copy candidate reset, W0/cache rollback을 사용했다.",
        "- K0 IDs: 14148, 16872, 4164, 3002. K1 sealed IDs는 소비하지 않았다.",
        "- technical lineages 30449/30460/30472/30486은 denominator 0이며 immutable 보존했다.",
        "- terminal science job: 30543, COMPLETED, elapsed 01:40:03.",
        "",
        "## Identity",
        "",
        f"- execution source: `{execution_head}`",
        f"- terminal result SHA256: `{result_sha}`",
        f"- pre-GPU receipt SHA256: `{preflight_sha}`",
        "- promotion: false",
        "- origin/main integration/push: 0 (GH handoff only)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--attempt", action="append", default=[])
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise SystemExit("refusing to overwrite report package")
    args.output_root.mkdir(parents=True)
    result = json.loads(args.result.read_text())
    preflight = json.loads(args.preflight.read_text())
    summary, rows = _summary(result)
    summary["identity"] = canonical_hash(summary)
    summary_path = args.output_root / "k0-summary.json"
    summary_path.write_text(canonical_json(summary) + "\n")
    candidates_path = args.output_root / "k0-candidates.csv"
    with candidates_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report_path = args.output_root / "fzcb-completion-value-fast-kill-k0-factual-ko.md"
    report_path.write_text(_report(summary, file_sha256(args.result), file_sha256(args.preflight), args.execution_head))
    attempts = []
    for value in args.attempt:
        job, source, result_path = value.split("|", 2)
        path = Path(result_path)
        attempts.append({"job": job, "source": source, "result_path": result_path, "sha256": file_sha256(path)})
    manifest = {
        "schema": "odeedit.s06.fzcb-completion-value.analysis-manifest.v1",
        "decision": summary["decision"],
        "analysis_source": {"head": _git(args.source_root, "rev-parse", "HEAD"), "tree": _git(args.source_root, "rev-parse", "HEAD^{tree}")},
        "execution_source_head": args.execution_head,
        "contract": {"path": str(preflight["contract"]["path"]), "sha256": preflight["contract"]["sha256"]},
        "preflight": {"path": str(args.preflight), "sha256": file_sha256(args.preflight), "identity": preflight["identity"]},
        "terminal_result": {"path": str(args.result), "sha256": file_sha256(args.result), "job": "30543"},
        "excluded_technical_attempts": attempts,
        "k1_submission_count": 0,
        "members": [], "scientific_promotion": False,
    }
    for path in (report_path, summary_path, candidates_path):
        manifest["members"].append({"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    manifest["member_root"] = canonical_hash(manifest["members"])
    manifest_path = args.output_root / "analysis-manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n")
    receipt = {
        "schema": "odeedit.s06.fzcb-completion-value.rooted-receipt.v1",
        "manifest_sha256": file_sha256(manifest_path), "member_root": manifest["member_root"],
        "decision": summary["decision"], "execution_result_sha256": file_sha256(args.result),
        "identity": canonical_hash({"manifest_sha256": file_sha256(manifest_path), "member_root": manifest["member_root"], "decision": summary["decision"]}),
    }
    receipt_path = args.output_root / "rooted-receipt.json"
    receipt_path.write_text(canonical_json(receipt) + "\n")
    for path in args.output_root.iterdir():
        os.chmod(path, 0o600)


if __name__ == "__main__":
    main()
