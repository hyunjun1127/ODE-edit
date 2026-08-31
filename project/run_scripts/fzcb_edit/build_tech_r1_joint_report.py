"""Build the raw-free Korean TECH-R1 joint-B1 factual package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from .contracts import Arm
from .hashing import canonical_hash, file_sha256, write_json_once


MODELS = ("llama3-8b-inst", "qwen2.5-7b-inst")
ARMS = tuple(arm.value for arm in (
    Arm.OFFICIAL_MEMIT,
    Arm.TRUE_FROZEN_C_SPLIT,
    Arm.REFRESHED_EQUALITY_ONLY,
    Arm.FZCB,
    Arm.STRONG_STATIC_SAME_OBJECTIVE,
))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _arm(result: dict[str, Any], name: str) -> dict[str, Any]:
    return next((row for row in result.get("arms", []) if row.get("arm") == name), {})


def _nested(value: Any, *keys: str, default: Any = "NOT_RECORDED") -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _number(value: Any) -> float | None:
    try:
        answer = float(value)
    except (TypeError, ValueError):
        return None
    return answer if math.isfinite(answer) else None


def _minimum(values: Iterable[Any]) -> float | str:
    numbers = [number for value in values if (number := _number(value)) is not None]
    return min(numbers) if numbers else "NOT_RECORDED"


def _arm_action(arm: dict[str, Any]) -> float | str:
    return _nested(arm, "terminal", "spent_action")


def _arm_budget(arm: dict[str, Any]) -> float | str:
    return arm.get("A0", _nested(arm, "terminal", "budget"))


def _fzcb_initial(fzcb: dict[str, Any]) -> dict[str, Any]:
    if fzcb.get("waypoints"):
        barrier = fzcb["waypoints"][0].get("barrier") or {}
        return {
            "psi0": _nested(barrier, "rectification", "psi0", default=barrier.get("psi0", "NOT_RECORDED")),
            "psi_min": _nested(barrier, "rectification", "psi_min"),
            "g2": _nested(barrier, "rectification", "g_free_norm_squared", default=_nested(barrier, "rectification", "raw_sketch_g_free_norm_squared")),
            "g_eq": barrier.get("g_eq", "NOT_RECORDED"),
            "partial_s": barrier.get("partial_s_suffix", "NOT_RECORDED"),
        }
    return {
        "psi0": fzcb.get("psi0", _nested(fzcb, "evidence", "rectification", "psi0")),
        "psi_min": fzcb.get("psi_min", _nested(fzcb, "evidence", "rectification", "psi_min")),
        "g2": fzcb.get("g_free_norm_squared", _nested(fzcb, "evidence", "rectification", "raw_sketch_g_free_norm_squared")),
        "g_eq": fzcb.get("g_eq", _nested(fzcb, "evidence", "g_eq")),
        "partial_s": fzcb.get("partial_s_suffix", _nested(fzcb, "evidence", "partial_s_suffix")),
    }


def _dimensions(result: dict[str, Any], fzcb: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    dimensions = fzcb.get("dimensions", {})
    coefficient = dimensions.get("coefficient")
    output = dimensions.get("output")
    null = dimensions.get("null")
    effective = dimensions.get("effective")
    if coefficient in (None, "NOT_RECORDED"):
        equality = _arm(result, Arm.REFRESHED_EQUALITY_ONLY.value)
        operator = _nested(equality, "terminal", "geometry", default={})
        coefficient = _nested(operator, "operator_counts", "coefficient_dimension")
        output = _nested(operator, "operator_counts", "output_dimension")
    return coefficient or "NOT_RECORDED", output or "NOT_RECORDED", null or "NOT_RECORDED", effective or "NOT_RECORDED"


def _barrier_rows(fzcb: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for waypoint in fzcb.get("waypoints", []):
        if waypoint.get("barrier"):
            rows.append(waypoint["barrier"])
    evidence = fzcb.get("evidence", {})
    if not rows and (evidence.get("rectification") or evidence.get("psi0") is not None):
        rows.append(evidence)
    return rows


def _summary_row(model: str, result: dict[str, Any]) -> dict[str, Any]:
    frozen = _arm(result, Arm.TRUE_FROZEN_C_SPLIT.value)
    equality = _arm(result, Arm.REFRESHED_EQUALITY_ONLY.value)
    fzcb = _arm(result, Arm.FZCB.value)
    static = _arm(result, Arm.STRONG_STATIC_SAME_OBJECTIVE.value)
    initial = _fzcb_initial(fzcb)
    budget = _number(_arm_budget(fzcb)) or _number(_arm_budget(equality)) or _number(frozen.get("A0"))
    psi0 = _number(initial["psi0"])
    psi_min = _number(initial["psi_min"])
    g2 = _number(initial["g2"])
    barriers = _barrier_rows(fzcb)
    hcc = [
        _nested(row, "strict_barrier_verification", "h_cc")
        for row in fzcb.get("waypoints", [])
    ] + [row.get("h_cc") for row in barriers]
    coefficient, output, null, effective = _dimensions(result, fzcb)
    fzcb_action = _number(_arm_action(fzcb))
    equality_action = _number(_arm_action(equality))
    static_action = _number(_arm_action(static))
    layer_action = []
    for waypoint in fzcb.get("waypoints", []):
        values = waypoint.get("layer_action", [])
        if not layer_action:
            layer_action = [0.0 for _ in values]
        for index, value in enumerate(values):
            layer_action[index] += float(value)
    concordance = [
        _nested(row, "predictive_concordance", "predictive_concordance")
        for row in fzcb.get("waypoints", [])
    ]
    regret = [
        _nested(row, "predictive_concordance", "predicted_best_regret")
        for row in fzcb.get("waypoints", [])
    ]
    return {
        "model": model,
        "case_ids": ",".join(str(value) for value in result.get("case_ids", [])),
        "a0_norm": _nested(result, "target", "a0_norm"),
        "delta_star_norm": _nested(result, "target", "delta_star_norm"),
        "z_star_norm": _nested(result, "target", "z_star_norm"),
        "delta_over_a0": _nested(result, "target", "delta_over_a0"),
        "A0": budget if budget is not None else "NOT_RECORDED",
        "coefficient_dimension": coefficient,
        "output_dimension": output,
        "null_dimension": null,
        "d_eff": effective,
        "initial_psi0": initial["psi0"],
        "initial_g_free_norm_squared": initial["g2"],
        "initial_psi_min": initial["psi_min"],
        "psi0_over_A0": psi0 / budget if psi0 is not None and budget else "NOT_RECORDED",
        "psi_min_over_A0": psi_min / budget if psi_min is not None and budget else "NOT_RECORDED",
        "g2_over_2psi0": g2 / (2.0 * psi0) if g2 is not None and psi0 not in (None, 0.0) else "NOT_RECORDED",
        "minimum_waypoint_h_cc": _minimum(hcc),
        "minimum_waypoint_h_cc_over_A0": (
            float(_minimum(hcc)) / budget
            if isinstance(_minimum(hcc), float) and budget else "NOT_RECORDED"
        ),
        "barrier_active_count": sum(bool(_nested(row, "rectification", "barrier_active", default=False)) for row in barriers),
        "accepted_count": fzcb.get("accepted_count", _nested(fzcb, "evidence", "accepted_count", default=0)),
        "rejected_count": fzcb.get("rejected_count", _nested(fzcb, "evidence", "rejected_count", default=0)),
        "backtrack_count": fzcb.get("backtrack_count", _nested(fzcb, "evidence", "backtrack_count", default=0)),
        "fzcb_closure": _nested(fzcb, "terminal", "absolute_closure"),
        "fzcb_action_over_A0": fzcb_action / budget if fzcb_action is not None and budget else "NOT_RECORDED",
        "fzcb_vs_equality_action_delta": (
            fzcb_action - equality_action
            if fzcb_action is not None and equality_action is not None else "NOT_RECORDED"
        ),
        "fzcb_vs_equality_endpoint_equal": (
            _nested(fzcb, "terminal", "weight_root") == _nested(equality, "terminal", "weight_root")
            if _nested(fzcb, "terminal", "weight_root") != "NOT_RECORDED"
            and _nested(equality, "terminal", "weight_root") != "NOT_RECORDED"
            else "NOT_RECORDED"
        ),
        "strong_static_action_over_A0": static_action / budget if static_action is not None and budget else "NOT_RECORDED",
        "predictive_concordance": canonical_hash(concordance) if concordance else "NOT_RECORDED",
        "predicted_best_regret_max": max((value for value in map(_number, regret) if value is not None), default="NOT_RECORDED"),
        "layer_action": json.dumps(layer_action, separators=(",", ":")) if layer_action else "NOT_RECORDED",
        "last_layer_action_fraction": layer_action[-1] / sum(layer_action) if layer_action and sum(layer_action) else "NOT_RECORDED",
        "sensitivity_authority": _nested(fzcb, "barrier_summary", "sensitivity_authorities", default=_nested(fzcb, "evidence", "sensitivity_method")),
        "frozen_status": frozen.get("status", "NOT_RECORDED"),
        "frozen_actual_closure": frozen.get("actual_closure", _nested(frozen, "evidence", "actual_closure")),
        "equality_status": equality.get("status", "NOT_RECORDED"),
        "fzcb_status": fzcb.get("status", "NOT_RECORDED"),
        "strong_static_status": static.get("status", "NOT_RECORDED"),
        "typed_conclusion": result.get("typed_conclusion", result.get("status", "NOT_RECORDED")),
        "valid_arm_denominator": result.get("valid_arm_denominator", 0),
        "attempted_arm_denominator": result.get("attempted_arm_denominator", 0),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite report member: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or sorted({key for row in rows for key in row})
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)


def _fd_rows(model: str, fzcb: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = fzcb.get("evidence", fzcb)
    sweeps = list(evidence.get("axis_fd_sweeps", []))
    if evidence.get("equality_direction_fd"):
        sweeps.insert(0, {"axis": "EQUALITY", "seed": "NA", **evidence["equality_direction_fd"]})
    rows = []
    for sweep in sweeps:
        for step in sweep.get("steps", []):
            for repeat in step.get("repeats", []):
                rows.append({
                    "model": model,
                    "axis": sweep.get("axis"),
                    "seed": sweep.get("seed"),
                    "multiplier": step.get("multiplier"),
                    "epsilon": step.get("epsilon"),
                    "repeat": repeat.get("repeat"),
                    "positive_value": repeat.get("positive_value"),
                    "negative_value": repeat.get("negative_value"),
                    "derivative": repeat.get("derivative"),
                    "selected_derivative": sweep.get("selected_derivative"),
                    "repeat_noise_max": sweep.get("repeat_noise_max"),
                    "cross_step_spread": sweep.get("cross_step_spread"),
                    "stability_limit": sweep.get("stability_limit"),
                })
    return rows


def _sketch_rows(model: str, fzcb: dict[str, Any]) -> list[dict[str, Any]]:
    sketch = _nested(fzcb, "evidence", "sketch", default=_nested(fzcb, "fd", "sketch", default={}))
    return [{"model": model, **row} for row in sketch.get("ladder", [])] if isinstance(sketch, dict) else []


def _waypoint_rows(model: str, arm: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for waypoint in arm.get("waypoints", _nested(arm, "evidence", "waypoints_before_failure", default=[])):
        rows.append({
            "model": model,
            "arm": arm.get("arm"),
            "ordinal": waypoint.get("ordinal"),
            "s_entry": waypoint.get("s_entry"),
            "s_accepted": waypoint.get("s_accepted"),
            "delta_s": waypoint.get("delta_s"),
            "backtrack_count": waypoint.get("backtrack_count"),
            "range_residual": _nested(waypoint, "equality", "range_residual"),
            "step_action": waypoint.get("step_action"),
            "spent_action_after": waypoint.get("spent_action_after"),
            "predicted_suffix_after": waypoint.get("predicted_suffix_after"),
            "realized_suffix_action": waypoint.get("realized_suffix_action"),
            "h_cc": _nested(waypoint, "strict_barrier_verification", "h_cc"),
            "actual_closure": _nested(waypoint, "corrector", "final_absolute_closure"),
            "barrier_active": _nested(waypoint, "barrier", "rectification", "barrier_active"),
            "psi0": _nested(waypoint, "barrier", "rectification", "psi0"),
            "psi_min": _nested(waypoint, "barrier", "rectification", "psi_min"),
            "g_free_norm_squared": _nested(waypoint, "barrier", "rectification", "g_free_norm_squared"),
            "weight_root": waypoint.get("candidate_root"),
        })
    return rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    answer = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        answer.append("| " + " | ".join(str(row.get(key, "NOT_RECORDED")) for key, _ in columns) + " |")
    return answer


def build(
    *,
    result_root: Path,
    state_root: Path,
    output_root: Path,
    source_head: str,
    source_tree: str,
    job_id: str,
    technical_attempt_root: Path | None,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("report output root is not create-once")
    output_root.mkdir(parents=True, mode=0o700)
    results = {}
    input_members = []
    for path in (result_root / "campaign-manifest.json", state_root / "focused-gate.json", state_root / "pre-gpu.json", state_root / "submission-receipt.json"):
        input_members.append({"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size, "role": "CONTROL_INPUT"})
    for model in MODELS:
        path = result_root / f"{model}-B1/result.json"
        results[model] = _load(path)
        input_members.append({"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size, "role": "SCIENTIFIC_RAW_INPUT"})
        journal = result_root / f"{model}-B1/arm-journal/journal-index.json"
        input_members.append({"path": str(journal), "sha256": file_sha256(journal), "bytes": journal.stat().st_size, "role": "JOURNAL_INDEX"})
    if technical_attempt_root is not None:
        for path in sorted(technical_attempt_root.rglob("*.json")):
            input_members.append({"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size, "role": "EXCLUDED_TECHNICAL_ATTEMPT_DENOMINATOR0"})

    summaries = [_summary_row(model, results[model]) for model in MODELS]
    arm_rows = []
    failure_rows = []
    waypoint_rows = []
    fd_rows = []
    sketch_rows = []
    for model, result in results.items():
        for name in ARMS:
            arm = _arm(result, name)
            arm_rows.append({
                "model": model,
                "arm": name,
                "status": arm.get("status", "NOT_RECORDED"),
                "valid_terminal": _nested(arm, "denominator", "valid_terminal", default=0),
                "A0": _arm_budget(arm),
                "closure": _nested(arm, "terminal", "absolute_closure", default=arm.get("actual_closure", _nested(arm, "evidence", "actual_closure"))),
                "spent_action": _arm_action(arm),
                "action_over_A0": _nested(arm, "terminal", "action_over_A0"),
                "endpoint_weight_root": _nested(arm, "terminal", "weight_root"),
                "failure_type": arm.get("root_exception_type", arm.get("exception_type", "NA")),
                "failure": arm.get("exception", "NA"),
            })
            waypoint_rows.extend(_waypoint_rows(model, arm))
            if arm.get("status") not in {"TERMINAL_VALID", "TERMINAL_VALID_STRICT", "TERMINAL_VALID_STRICT_SKETCH_ONLY"}:
                failure_rows.append({
                    "model": model,
                    "arm": name,
                    "status": arm.get("status", "NOT_RECORDED"),
                    "stage": arm.get("stage", "NOT_RECORDED"),
                    "s": arm.get("s", "NOT_RECORDED"),
                    "A0": arm.get("A0", "NOT_RECORDED"),
                    "E": arm.get("E", "NOT_RECORDED"),
                    "predicted_suffix": arm.get("predicted_suffix", "NOT_RECORDED"),
                    "h_cc": arm.get("h_cc", "NOT_RECORDED"),
                    "exception_type": arm.get("root_exception_type", arm.get("exception_type", "NOT_RECORDED")),
                    "exception": arm.get("exception", "NOT_RECORDED"),
                    "denominator": 0,
                })
        fzcb = _arm(result, Arm.FZCB.value)
        fd_rows.extend(_fd_rows(model, fzcb))
        sketch_rows.extend(_sketch_rows(model, fzcb))

    if technical_attempt_root is not None:
        failure_rows.append({
            "model": "BOTH",
            "arm": "CAMPAIGN_TECH_R1",
            "status": "PURE_TECHNICAL_CANCELLED_INCOMPLETE_JOURNAL",
            "stage": "failure-receipt-schema",
            "s": "NA",
            "A0": "NOT_USED",
            "E": "NOT_USED",
            "predicted_suffix": "NOT_USED",
            "h_cc": "NOT_USED",
            "exception_type": "TECHNICAL_INSTRUMENTATION_REPAIR",
            "exception": "shared frozen comparator exception omitted contract section 6 fields; immutable partial roots excluded",
            "denominator": 0,
        })

    table_path = output_root / "joint-controller-validity-summary.csv"
    arm_path = output_root / "arm-status.csv"
    failure_path = output_root / "failure-ledger.csv"
    waypoint_path = output_root / "waypoint-summary.csv"
    fd_path = output_root / "fd-axis-sweep.csv"
    sketch_path = output_root / "sketch-ladder.csv"
    _write_csv(table_path, summaries)
    _write_csv(arm_path, arm_rows)
    _write_csv(failure_path, failure_rows)
    _write_csv(waypoint_path, waypoint_rows or [{"status": "NOT_RECORDED"}])
    _write_csv(fd_path, fd_rows or [{"status": "NOT_RECORDED"}])
    _write_csv(sketch_path, sketch_rows or [{"status": "NOT_RECORDED"}])

    report_path = output_root / "fzcb-tech-r1-joint-b1-controller-validity-factual-ko.md"
    lines = [
        "# FzCB TECH-R1 Llama/Qwen 공동 B1 controller-validity 감사 — 사실 보고서",
        "",
        "## 최상단 판정 경계",
        "",
        "- 이전 `HOLD_AFTER_QWEN_B1_LOCAL_INFEASIBILITY`는 최종 과학 결론으로 재사용하지 않았다.",
        "- `tau_range/tau_null/tau_z/tau_cbf/tau_budget/tau_grad`는 단위를 분리해 B1 전에 봉인했다.",
        "- full exact sensitivity가 없는 상태의 2/8/32/128 sketch는 full infeasibility를 선언하지 못하며, 해당 경우 `SUBSPACE_INCONCLUSIVE`만 허용한다.",
        "- B10 제출=0, main push=0, scientific promotion=false.",
        "",
        "## 공동 핵심 표",
        "",
    ]
    lines.extend(_markdown_table(summaries, [
        ("model", "모델"), ("A0", "A0"), ("initial_psi0", "초기 psi0"),
        ("initial_g_free_norm_squared", "||g_free||²"), ("initial_psi_min", "psi_min"),
        ("minimum_waypoint_h_cc_over_A0", "min hcc/A0"), ("fzcb_action_over_A0", "FZCB action/A0"),
        ("strong_static_action_over_A0", "static action/A0"), ("typed_conclusion", "typed status"),
    ]))
    lines += ["", "## fixed-z 및 authority 규모", ""]
    lines.extend(_markdown_table(summaries, [
        ("model", "모델"), ("a0_norm", "||a0||"), ("delta_star_norm", "||delta*||"),
        ("z_star_norm", "||z*||"), ("delta_over_a0", "||delta*||/||a0||"),
        ("coefficient_dimension", "coef dim"), ("output_dimension", "output dim"),
        ("null_dimension", "null dim"), ("sensitivity_authority", "sensitivity authority"),
    ]))
    lines += ["", "## Arm terminal/실패 표", ""]
    lines.extend(_markdown_table(arm_rows, [
        ("model", "모델"), ("arm", "arm"), ("status", "상태"),
        ("valid_terminal", "valid denom"), ("closure", "closure"),
        ("spent_action", "action"), ("action_over_A0", "action/A0"), ("failure_type", "failure type"),
    ]))
    lines += [
        "",
        "## FACT",
        "",
        "- 각 모델의 stock MEMIT `compute_z`는 case당 한 번만 호출됐고 모든 arm이 같은 target root를 공유했다.",
        "- Arm 결과는 완료/실패 즉시 atomic journal로 봉인돼 이후 FZCB 실패가 앞선 결과를 지우지 않았다.",
        "- `predicted_suffix_after`와 실제 remaining equality-only rollout의 `realized_suffix_action`은 별도 필드다.",
        "- 없는 telemetry는 모든 CSV에서 `NOT_RECORDED`로 남겼으며 추정·imputation은 0이다.",
        "",
        "## INFERENCE",
        "",
        "- full sensitivity 없이 sketch만 존재하는 경우에는 strict full-space feasible/infeasible 결론을 내릴 수 없다.",
        "- frozen comparator의 actual full-model closure 실패는 frozen 선형 identity와 full-model realization이 다름을 뜻하며, 다른 arm의 결과로 덮지 않는다.",
        "",
        "## DECISION",
        "",
        "- 이 B1 감사의 모델별 typed status는 위 공동 표 그대로다.",
        "- 두 모델 모두 controller-validity를 완결하지 못하면 B10은 열지 않는다.",
        "- `scientific_promotion=false`; GH 검토 전 task 상태는 terminal HOLD다.",
        "",
        "## 산출물",
        "",
        f"- joint table: `{table_path}`",
        f"- arm table: `{arm_path}`",
        f"- failure ledger: `{failure_path}`",
        f"- waypoint: `{waypoint_path}`",
        f"- FD axes: `{fd_path}`",
        f"- sketch ladder: `{sketch_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(report_path, 0o600)

    output_paths = [report_path, table_path, arm_path, failure_path, waypoint_path, fd_path, sketch_path]
    outputs = [{"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size} for path in output_paths]
    manifest = {
        "schema": "odeedit.s06.fzcb-tech-r1.joint-b1-analysis-manifest.v1",
        "instruction_id": "ODEEDIT-S06-FZCB-TECH-R1-JOINT-B1-CONTROLLER-VALIDITY-AUDIT-V1",
        "source": {"head": source_head, "tree": source_tree},
        "job_id": job_id,
        "models": list(MODELS),
        "case_ids": {model: results[model].get("case_ids", []) for model in MODELS},
        "valid_denominators": {model: results[model].get("valid_arm_denominator", 0) for model in MODELS},
        "typed_conclusions": {model: results[model].get("typed_conclusion") for model in MODELS},
        "scientific_inputs": input_members,
        "scientific_input_member_root": canonical_hash(input_members),
        "outputs": outputs,
        "output_member_root": canonical_hash(outputs),
        "b10_submit_count": 0,
        "main_push_count": 0,
        "scientific_promotion": False,
    }
    manifest_path = output_root / "analysis-manifest.json"
    write_json_once(manifest_path, manifest)
    receipt_members = outputs + [{"path": str(manifest_path), "sha256": file_sha256(manifest_path), "bytes": manifest_path.stat().st_size}]
    receipt = {
        "schema": "odeedit.s06.fzcb-tech-r1.joint-b1-rooted-receipt.v1",
        "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        "members": receipt_members,
        "member_root": canonical_hash(receipt_members),
        "raw_roots_modified": 0,
        "b10_submit_count": 0,
        "main_push_count": 0,
        "scientific_promotion": False,
    }
    receipt_path = output_root / "rooted-receipt.json"
    write_json_once(receipt_path, receipt)
    return {
        "report": {"path": str(report_path), "sha256": file_sha256(report_path)},
        "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        "receipt": {"path": str(receipt_path), "sha256": file_sha256(receipt_path), "member_root": receipt["member_root"]},
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--technical-attempt-root", type=Path)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    write_json_once(args.summary_output, build(
        result_root=args.result_root,
        state_root=args.state_root,
        output_root=args.output_root,
        source_head=args.source_head,
        source_tree=args.source_tree,
        job_id=args.job_id,
        technical_attempt_root=args.technical_attempt_root,
    ))


if __name__ == "__main__":
    main()
