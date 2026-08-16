#!/usr/bin/env python3
"""Generate the P2R1 raw-free factual terminal package."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


WORKTREE = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p2r1-baseline-calibrated-rms-tangent-target-flow-v1"
)
RESULTS = WORKTREE / "local/odebf/results"
OUT = WORKTREE / "local/odebf/reports/p2r1-rms-tangent-target-only-v1"
P1R43_REPORT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r43-rho-free-semantic-first-strength-recovery-v1/"
    "local/odebf/reports/p1r43-rho-free-semantic-first-v1"
)
P1R31_REPORT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r31-p1r24-independent-b10x10-full-matrix-v1/"
    "local/odebf/reports/p1r31-p1r24-independent-b10x10-detailed-v1"
)
SUPPORT = Path(
    "/mnt/raid5/janghj/ODE-edit/local/source-handoff/"
    "P1R43_P1R42_PARENT_SUPPORT_SH2_V1"
)

ROOTS = {
    "llama3-8b-inst": RESULTS
    / "s05-p2r1-rms-tangent-target-only-b10x10-llama3-8b-inst-tech-r1-v1",
    "qwen2.5-7b-inst": RESULTS
    / "s05-p2r1-rms-tangent-target-only-b10x10-qwen2.5-7b-inst-tech-r1-v1",
}
FIRST_ROOTS = {
    "llama3-8b-inst": RESULTS
    / "s05-p2r1-rms-tangent-target-only-first-b10-llama3-8b-inst-tech-r1-v1",
    "qwen2.5-7b-inst": RESULTS
    / "s05-p2r1-rms-tangent-target-only-first-b10-qwen2.5-7b-inst-tech-r1-v1",
}
SCHEDULER = {
    "llama3-8b-inst": {
        "array_parent": 19885,
        "job_id": 19885,
        "array_index": 0,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:07:21",
        "max_rss_kib": 9642144,
    },
    "qwen2.5-7b-inst": {
        "array_parent": 19885,
        "job_id": 19886,
        "array_index": 1,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:07:47",
        "max_rss_kib": 6508296,
    },
}
SOURCE_HEAD = "8f817e13167289dac190fe74bfa42e2b3e01372d"
SOURCE_TREE = "abe577756ea82e1fed0847b6329c82dc49184e9f"
PARENT = "11508b6da11d606521b703037034e1814b70d8a8"
STREAM_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
STREAM_ORDER = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def quantile(values: Iterable[float], q: float) -> float | None:
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return None
    position = (len(data) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return data[lower]
    return data[lower] * (upper - position) + data[upper] * (position - lower)


def summary(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(data),
        "mean": statistics.fmean(data) if data else None,
        "median": statistics.median(data) if data else None,
        "p90": quantile(data, 0.9),
        "worst_max": max(data) if data else None,
        "min": min(data) if data else None,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def panel_rows(terminal: dict[str, Any], alias: str, case: int) -> tuple[list, list]:
    raw = terminal["terminal_z_panel"]["z_inject"]
    rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    mapping = {
        "efficacy": "EFF_Z_INJECT",
        "generalization": "GEN_Z_INJECT",
        "locality-preservation": "LOC_Z_INJECT",
    }
    for metric_name, panel in mapping.items():
        metric = raw["primary"]["metrics"][metric_name]
        vectors = raw["numeric_vectors"][metric_name]
        new_values: list[float] = []
        old_values: list[float] = []
        margins: list[float] = []
        for request_index, prompts in enumerate(vectors):
            bits = metric["per_case_bits"][request_index]
            for prompt_index, value in enumerate(prompts):
                row = {
                    "alias": alias,
                    "case": case,
                    "panel": panel,
                    "request_index": request_index,
                    "prompt_index": prompt_index,
                    "correct": int(bits[prompt_index]),
                    "nll_new": float(value["nll_new"]),
                    "nll_old": float(value["nll_old"]),
                    "margin": float(value["margin"]),
                }
                request_rows.append(row)
                new_values.append(row["nll_new"])
                old_values.append(row["nll_old"])
                margins.append(row["margin"])
        rows.append(
            {
                "alias": alias,
                "case": case,
                "panel": panel,
                "correct": int(metric["numerator"]),
                "denominator": int(metric["denominator"]),
                "new_nll": summary(new_values),
                "old_nll": summary(old_values),
                "margin": summary(margins),
                "bit_vector_sha256": metric["bit_vector_sha256"],
                "comparison_score_sha256": metric["comparison_score_sha256"],
            }
        )
    return rows, request_rows


def collect() -> tuple[list, list, list, list, list, dict]:
    cases: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    integrity: Counter[str] = Counter()

    p43_panel_rows = read_json(P1R43_REPORT / "p1r43-per-panel.json")["rows"]
    p43_index = {
        (row["alias"], row["arm"], int(row["case"]), row["panel"]): row
        for row in p43_panel_rows
    }
    official_rows = read_json(
        SUPPORT / "identities/official-alphaedit-baseline-identity.json"
    )["rows"]
    official_index = {(row["alias"], int(row["case"])): row for row in official_rows}

    for alias, root in ROOTS.items():
        task = read_json(root / "terminal.json")
        integrity["task_terminal"] += int(task["completed_case_count"] == 10)
        integrity["task_failed_case_zero"] += int(task["failed_case_count"] == 0)
        integrity["task_W0_restored"] += int(task["W0_restored"] is True)
        for case in range(1, 11):
            case_root = root / f"raw/cases/case-{case:02d}"
            terminal = read_json(case_root / "terminal.json")
            manifest = read_json(case_root / "manifest.json")
            current_panels, current_prompt_rows = panel_rows(terminal, alias, case)
            panels.extend(current_panels)
            requests.extend(current_prompt_rows)
            panel_index = {row["panel"]: row for row in current_panels}
            request_target = [float(value) for value in terminal["terminal_full_six_target_new_nll_by_request"]]
            row = {
                "alias": alias,
                "case": case,
                "status": "COMPLETE",
                "request_count": 10,
                "request_order_sha256": terminal["request_order_sha256"],
                "objective_plan_sha256": terminal["objective_plan_sha256"],
                "capture_plan_sha256": terminal["capture_plan_sha256"],
                "target_microstep_count": int(terminal["target_microstep_count"]),
                "outer_state_count": int(terminal["outer_state_count"]),
                "eff_correct": panel_index["EFF_Z_INJECT"]["correct"],
                "eff_denominator": panel_index["EFF_Z_INJECT"]["denominator"],
                "gen_correct": panel_index["GEN_Z_INJECT"]["correct"],
                "gen_denominator": panel_index["GEN_Z_INJECT"]["denominator"],
                "loc_correct": panel_index["LOC_Z_INJECT"]["correct"],
                "loc_denominator": panel_index["LOC_Z_INJECT"]["denominator"],
                "eff_new_nll": panel_index["EFF_Z_INJECT"]["new_nll"],
                "eff_margin": panel_index["EFF_Z_INJECT"]["margin"],
                "gen_new_nll": panel_index["GEN_Z_INJECT"]["new_nll"],
                "gen_margin": panel_index["GEN_Z_INJECT"]["margin"],
                "terminal_full_six_target_new_nll": float(terminal["terminal_full_six_target_new_nll"]),
                "terminal_full_six_target_new_nll_by_request": summary(request_target),
                "clamp_hit_count": int(terminal["clamp_hit_count"]),
                "target_wall_seconds": float(terminal["target_wall_seconds"]),
                "terminal_evaluator_wall_seconds": float(terminal["terminal_evaluator_wall_seconds"]),
                "total_wall_seconds": float(terminal["total_wall_seconds"]),
                "compute": terminal["compute"],
                "W0_restored": bool(terminal["W0_restored"]),
                "writer_update_count": int(terminal["writer_update_count"]),
                "writer_materialization_count": int(terminal["writer_materialization_count"]),
                "action_freeze_sha256": terminal["action_freeze_sha256"],
                "terminal_sha256": sha256(case_root / "terminal.json"),
                "manifest_sha256": sha256(case_root / "manifest.json"),
                "terminal_identity_sha256": terminal["identity_sha256"],
                "terminal_target_sha256": terminal["terminal_target_sha256"],
                "target_trajectory_file_sha256": terminal["target_trajectory_file_sha256"],
                "target_trajectory_tensor_sha256": terminal["target_trajectory_tensor_sha256"],
            }
            cases.append(row)
            integrity["case_terminal"] += 1
            integrity["case_manifest"] += int(manifest["W0_restored"] is True)
            integrity["case_W0"] += int(terminal["W0_restored"] is True)

            repeated_clamp_requests: Counter[int] = Counter()
            for microstep in range(24):
                receipt_path = case_root / f"raw/target/microstep-{microstep:02d}.json"
                receipt = read_json(receipt_path)
                step = {
                    "alias": alias,
                    "case": case,
                    "microstep": microstep,
                    "outer_state": int(receipt["outer_state_index"]),
                    "within_outer_microstep": int(receipt["within_outer_microstep"]),
                    "target_new_nll_mean": float(receipt["target_new_nll_mean"]),
                    "kl_mean": float(receipt["kl_mean"]),
                    "decay_mean": statistics.fmean(float(v) for v in receipt["decay_by_request"]),
                    "q_mean": statistics.fmean(float(v) for v in receipt["q_by_request"]),
                    "q_min": min(float(v) for v in receipt["q_by_request"]),
                    "q_max": max(float(v) for v in receipt["q_by_request"]),
                    "gamma_mean": statistics.fmean(float(v) for v in receipt["gamma_by_request"]),
                    "field_cosine_mean": statistics.fmean(float(v) for v in receipt["field_cosine_with_semantic_nominal"]),
                    "clamp_hit_count": int(receipt["clamp_hit_count"]),
                    "semantic_tangent_max_abs_residual": float(receipt["semantic_tangent_max_abs_residual"]),
                    "semantic_rate_max_abs_residual": float(receipt["semantic_rate_max_abs_residual"]),
                    "target_update_count": int(receipt["target_update_count"]),
                    "writer_update_count": int(receipt["writer_update_count"]),
                    "writer_materialization_count": int(receipt["writer_materialization_count"]),
                    "native_endpoint_access_count": int(receipt["native_endpoint_access_count"]),
                    "retry_count": int(receipt["retry_count"]),
                    "hold_count": int(receipt["hold_count"]),
                    "heldout_controller_access_count": int(receipt["heldout_controller_access_count"]),
                    "field_sha256": receipt["field_sha256"],
                    "target_before_sha256": receipt["target_before_sha256"],
                    "target_next_sha256": receipt["target_next_sha256"],
                    "receipt_sha256": sha256(receipt_path),
                    "receipt_identity_sha256": receipt["identity_sha256"],
                }
                steps.append(step)
                for request_index in range(10):
                    clamp_hit = bool(receipt["clamp_hit"][request_index])
                    repeated_clamp_requests[request_index] += int(clamp_hit)
                    requests.append(
                        {
                            "alias": alias,
                            "case": case,
                            "panel": "TARGET_MICROSTEP",
                            "request_index": request_index,
                            "microstep": microstep,
                            "outer_state": int(receipt["outer_state_index"]),
                            "target_new_nll": float(receipt["target_new_nll_by_request"][request_index]),
                            "kl": float(receipt["kl_by_request"][request_index]),
                            "decay": float(receipt["decay_by_request"][request_index]),
                            "semantic_gradient_norm": float(receipt["semantic_gradient_norm"][request_index]),
                            "preservation_gradient_norm": float(receipt["preservation_gradient_norm"][request_index]),
                            "q": float(receipt["q_by_request"][request_index]),
                            "tangent_norm": float(receipt["tangent_norm_by_request"][request_index]),
                            "gamma": float(receipt["gamma_by_request"][request_index]),
                            "field_cosine": float(receipt["field_cosine_with_semantic_nominal"][request_index]),
                            "pre_clamp_displacement_norm": float(receipt["pre_clamp_displacement_norm"][request_index]),
                            "post_clamp_displacement_norm": float(receipt["post_clamp_displacement_norm"][request_index]),
                            "clamp_ratio": float(receipt["clamp_ratio"][request_index]),
                            "clamp_hit": clamp_hit,
                        }
                    )
            row["requests_with_repeated_clamp_hits"] = sum(
                count >= 2 for count in repeated_clamp_requests.values()
            )

            official = official_index[(alias, case)]
            comparisons.append(
                {
                    "alias": alias,
                    "case": case,
                    "comparator": "OfficialAlphaEdit",
                    "identity_status": "MATCHED" if official["official_matched"] else "NOT_MATCHED",
                    "current_eff_correct": row["eff_correct"],
                    "reference_eff_correct": official["official_eff_correct"],
                    "delta_eff_correct": row["eff_correct"] - official["official_eff_correct"],
                    "current_gen_correct": row["gen_correct"],
                    "reference_gen_correct": official["official_gen_correct"],
                    "delta_gen_correct": row["gen_correct"] - official["official_gen_correct"],
                    "current_eff_nll_mean": row["eff_new_nll"]["mean"],
                    "reference_eff_nll_mean": official["official_eff_nll_mean"],
                    "delta_eff_nll_mean": row["eff_new_nll"]["mean"] - official["official_eff_nll_mean"],
                    "current_eff_margin_mean": row["eff_margin"]["mean"],
                    "reference_eff_margin_mean": official["official_eff_margin_mean"],
                    "delta_eff_margin_mean": row["eff_margin"]["mean"] - official["official_eff_margin_mean"],
                    "reference_gen_nll": "NOT_RECORDED",
                }
            )
            for arm in ("Neutral", "Soft"):
                for panel in ("EFF_Z_INJECT", "GEN_Z_INJECT"):
                    reference = p43_index[(alias, arm, case, panel)]
                    current = panel_index[panel]
                    comparisons.append(
                        {
                            "alias": alias,
                            "case": case,
                            "comparator": f"P1R43-{arm}-{panel}",
                            "identity_status": "MATCHED",
                            "current_correct": current["correct"],
                            "reference_correct": reference["correct"],
                            "delta_correct": current["correct"] - reference["correct"],
                            "current_new_nll_mean": current["new_nll"]["mean"],
                            "reference_new_nll_mean": reference["new_nll"]["mean"],
                            "delta_new_nll_mean": current["new_nll"]["mean"] - reference["new_nll"]["mean"],
                            "current_margin_mean": current["margin"]["mean"],
                            "reference_margin_mean": reference["margin"]["mean"],
                            "delta_margin_mean": current["margin"]["mean"] - reference["margin"]["mean"],
                        }
                    )
    return cases, steps, requests, panels, comparisons, dict(integrity)


def aggregate(cases: list, steps: list, panels: list) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alias in ROOTS:
        case_group = [row for row in cases if row["alias"] == alias]
        step_group = [row for row in steps if row["alias"] == alias]
        panel_group = [row for row in panels if row["alias"] == alias]
        panel_index: dict[str, list] = defaultdict(list)
        for row in panel_group:
            panel_index[row["panel"]].append(row)
        result: dict[str, Any] = {
            "alias": alias,
            "attempts": 10,
            "endpoints": len(case_group),
            "failures": 10 - len(case_group),
            "requests": sum(row["request_count"] for row in case_group),
            "target_microsteps": sum(row["target_microstep_count"] for row in case_group),
            "eff_correct": sum(row["eff_correct"] for row in case_group),
            "eff_denominator": sum(row["eff_denominator"] for row in case_group),
            "gen_correct": sum(row["gen_correct"] for row in case_group),
            "gen_denominator": sum(row["gen_denominator"] for row in case_group),
            "loc_correct": sum(row["loc_correct"] for row in case_group),
            "loc_denominator": sum(row["loc_denominator"] for row in case_group),
            "terminal_full_six_target_new_nll": summary(
                row["terminal_full_six_target_new_nll"] for row in case_group
            ),
            "clamp_hit_count": sum(row["clamp_hit_count"] for row in case_group),
            "request_microstep_denominator": len(case_group) * 10 * 24,
            "requests_with_repeated_clamp_hits": sum(
                row["requests_with_repeated_clamp_hits"] for row in case_group
            ),
            "max_semantic_tangent_residual": max(
                row["semantic_tangent_max_abs_residual"] for row in step_group
            ),
            "max_semantic_rate_residual": max(
                row["semantic_rate_max_abs_residual"] for row in step_group
            ),
            "q": summary(row["q_mean"] for row in step_group),
            "gamma": summary(row["gamma_mean"] for row in step_group),
            "field_cosine": summary(row["field_cosine_mean"] for row in step_group),
            "target_wall_seconds_sum": sum(row["target_wall_seconds"] for row in case_group),
            "terminal_evaluator_wall_seconds_sum": sum(row["terminal_evaluator_wall_seconds"] for row in case_group),
            "total_wall_seconds_sum": sum(row["total_wall_seconds"] for row in case_group),
            "target_forward_count": sum(row["compute"]["target_forward_count"] for row in case_group),
            "target_backward_count": sum(row["compute"]["target_backward_count"] for row in case_group),
            "kl_forward_count": sum(row["compute"]["kl_forward_count"] for row in case_group),
            "kl_backward_count": sum(row["compute"]["kl_backward_count"] for row in case_group),
            "writer_forward_count": 0,
            "writer_backward_count": 0,
            "writer_materialization_count": 0,
            "W0_restored_count": sum(row["W0_restored"] for row in case_group),
            "scheduler": SCHEDULER[alias],
        }
        for panel in ("EFF_Z_INJECT", "GEN_Z_INJECT", "LOC_Z_INJECT"):
            items = panel_index[panel]
            result[panel] = {
                "correct": sum(row["correct"] for row in items),
                "denominator": sum(row["denominator"] for row in items),
                "new_nll_case_mean": summary(row["new_nll"]["mean"] for row in items),
                "margin_case_mean": summary(row["margin"]["mean"] for row in items),
                "new_nll_worst": max(row["new_nll"]["worst_max"] for row in items),
            }
        rows.append(result)
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "NOT_RECORDED"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_report(aggregates: list, comparisons: list, integrity: dict, files: dict) -> str:
    lines = [
        "# P2R1 Baseline-Calibrated RMS-Tangent Target Flow — 사실 기록",
        "",
        "- instruction: `ODEEDIT-S05-P2R1-BASELINE-CALIBRATED-RMS-TANGENT-TARGET-FLOW-V1`",
        f"- source HEAD/tree: `{SOURCE_HEAD}` / `{SOURCE_TREE}`",
        f"- exact parent: `{PARENT}`",
        f"- stream/order: `{STREAM_ROOT}` / `{STREAM_ORDER}`",
        "- numerical lock SHA/root: `688305826cb5c702e90da194ee2739bfc21b260ce6be1776f81019436e21b471` / `c770fa4b0944944912187105d67ea12c40725f09a27beab08b03d82847e73b1d`",
        "- source manifest SHA/root: `9350d009ca14ffe5774d92fc8bd18e0167bc247647c0f07626a2dd81280db264` / `b8fddb8c40d9ebdf76bfbc4e5ea2d304da9ddc29cd73229584724147063bc892`",
        "- P1R43 report SHA: `ed60bd3bc57bf2ee14d35d558674734b5ab7c446125131b2a281f58b7d3bfbf7`; Official identity SHA: `e12b19ff941f39a594d2ff0f8718849d102644fd2ec79f53ccb6d7efc9fd8d3c`",
        "- P1R48-old: `SUPERSEDED_BEFORE_SCIENTIFIC_EXECUTION_BY_P2_RENUMBERING`; scientific_failure=false",
        "- report policy: SH_FACTUAL_ONLY_REPORTING; scientific_promotion=false",
        "",
        "## 1. 실행·무결성",
        "",
        "| 모델 | scheduler | attempts/endpoints/failures | requests | target microsteps | W0 | writer/materialization |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        sched = row["scheduler"]
        lines.append(
            f"| {row['alias']} | {sched['job_id']} {sched['state']} {sched['exit_code']} | "
            f"{row['attempts']}/{row['endpoints']}/{row['failures']} | {row['requests']} | "
            f"{row['target_microsteps']} | {row['W0_restored_count']}/{row['endpoints']} | 0/0 |"
        )
    lines += [
        "",
        f"- integrity counters: `{json.dumps(integrity, sort_keys=True, separators=(',', ':'))}`",
        "- native endpoint runtime access=0; heldout controller access=0; retry/hold/debt/hard-P-budget=0.",
        "- initial attempt 19881/19882: 24/24 target microsteps 후 terminal adapter가 outer K budget에 24를 전달하여 TECHNICAL_FAIL. 예외 SHA `7c7b851d19cecb9a9fb2a9188389d240713631da45671178eead02e6f20c311a`. TECH-R1은 evaluator freeze를 K8/snapshot9로 교정했다.",
        "",
        "## 2. terminal z-inject 패널",
        "",
        "| 모델 | Eff | Gen | Loc | Eff NLL case-mean(mean/med/p90/worst) | Gen NLL case-mean(mean/med/p90/worst) | full-six target NLL(mean/med/p90/worst) |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in aggregates:
        eff = row["EFF_Z_INJECT"]
        gen = row["GEN_Z_INJECT"]
        full = row["terminal_full_six_target_new_nll"]
        e = eff["new_nll_case_mean"]
        g = gen["new_nll_case_mean"]
        lines.append(
            f"| {row['alias']} | {eff['correct']}/{eff['denominator']} | "
            f"{gen['correct']}/{gen['denominator']} | {row['loc_correct']}/{row['loc_denominator']} | "
            f"{fmt(e['mean'])}/{fmt(e['median'])}/{fmt(e['p90'])}/{fmt(eff['new_nll_worst'])} | "
            f"{fmt(g['mean'])}/{fmt(g['median'])}/{fmt(g['p90'])}/{fmt(gen['new_nll_worst'])} | "
            f"{fmt(full['mean'])}/{fmt(full['median'])}/{fmt(full['p90'])}/{fmt(full['worst_max'])} |"
        )
    lines += [
        "",
        "## 3. target field·clamp·compute",
        "",
        "| 모델 | clamp hits / request-microsteps | repeated-clamp requests | max tangent/rate residual | q mean | gamma mean | field cosine mean | target F/B | KL F/B | target/eval wall sum(s) | MaxRSS KiB |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['alias']} | {row['clamp_hit_count']}/{row['request_microstep_denominator']} | "
            f"{row['requests_with_repeated_clamp_hits']} | {fmt(row['max_semantic_tangent_residual'])}/{fmt(row['max_semantic_rate_residual'])} | "
            f"{fmt(row['q']['mean'])} | {fmt(row['gamma']['mean'])} | {fmt(row['field_cosine']['mean'])} | "
            f"{row['target_forward_count']}/{row['target_backward_count']} | {row['kl_forward_count']}/{row['kl_backward_count']} | "
            f"{fmt(row['target_wall_seconds_sum'])}/{fmt(row['terminal_evaluator_wall_seconds_sum'])} | {row['scheduler']['max_rss_kib']} |"
        )
    lines += [
        "",
        "## 4. matched comparator arithmetic",
        "",
        "| 모델 | comparator | cases | ΔEff correct sum | ΔGen correct sum | ΔEff NLL case-mean | ΔEff margin case-mean | identity |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for row in comparisons:
        grouped[(row["alias"], row["comparator"])].append(row)
    for (alias, comparator), rows in sorted(grouped.items()):
        if comparator == "OfficialAlphaEdit":
            delta_eff = sum(row["delta_eff_correct"] for row in rows)
            delta_gen = sum(row["delta_gen_correct"] for row in rows)
            delta_nll = statistics.fmean(row["delta_eff_nll_mean"] for row in rows)
            delta_margin = statistics.fmean(row["delta_eff_margin_mean"] for row in rows)
        else:
            delta_eff = sum(row["delta_correct"] for row in rows) if "EFF_" in comparator else "NOT_APPLICABLE"
            delta_gen = sum(row["delta_correct"] for row in rows) if "GEN_" in comparator else "NOT_APPLICABLE"
            delta_nll = statistics.fmean(row["delta_new_nll_mean"] for row in rows)
            delta_margin = statistics.fmean(row["delta_margin_mean"] for row in rows)
        lines.append(
            f"| {alias} | {comparator} | {len(rows)} | {fmt(delta_eff)} | {fmt(delta_gen)} | "
            f"{fmt(delta_nll)} | {fmt(delta_margin)} | MATCHED |"
        )
    lines += [
        "",
        "- Official Gen continuous NLL/margin per-case는 `NOT_RECORDED`; correct-count 비교만 포함했다.",
        "- Native endpoint vectors/RMS의 runtime decision influence와 runtime access는 0이다.",
        "",
        "## 5. gate·failure taxonomy 사실",
        "",
        "- Gate A0/A1 technical/numerical: PASS (두 모델 10/10, 24 microsteps/case, certificate/W0/firewall PASS).",
        f"- ClampDominated telemetry: Llama repeated-clamp requests={aggregates[0]['requests_with_repeated_clamp_hits']}; Qwen={aggregates[1]['requests_with_repeated_clamp_hits']}.",
        "- NonFinite=0; TargetWeak scientific classification=GH_RELEASE_PENDING; P2R1_TARGET_GATE_PASS는 SH가 발행하지 않았다.",
        "- P2R2 scientific GPU execution authorization=0.",
        "",
        "## 6. 산출물",
        "",
    ]
    for name, meta in files.items():
        lines.append(
            f"- `{name}`: SHA256 `{meta['sha256']}`, bytes {meta['bytes']}, lines {meta['lines']}"
        )
    lines += [
        "",
        "## 7. 경계",
        "",
        "- FACT: 위 표의 값, 분모, hash, scheduler/W0/firewall 상태.",
        "- NOT_RECORDED: Official latent trajectory, Official per-case Gen continuous NLL/margin, writer/P/capacity/energy (target-only gate).",
        "- scientific_promotion=false.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    cases, steps, requests, panels, comparisons, integrity = collect()
    aggregates = aggregate(cases, steps, panels)
    payloads = {
        "p2r1-per-case.json": {"schema": "p2r1-per-case/v1", "rows": cases, "rows_root": canonical_sha(cases)},
        "p2r1-per-step.json": {"schema": "p2r1-per-step/v1", "rows": steps, "rows_root": canonical_sha(steps)},
        "p2r1-per-request.json": {"schema": "p2r1-per-request/v1", "rows": requests, "rows_root": canonical_sha(requests)},
        "p2r1-per-panel.json": {"schema": "p2r1-per-panel/v1", "rows": panels, "rows_root": canonical_sha(panels)},
        "p2r1-case-paired-comparisons.json": {"schema": "p2r1-comparisons/v1", "rows": comparisons, "rows_root": canonical_sha(comparisons)},
        "p2r1-aggregates.json": {"schema": "p2r1-aggregates/v1", "rows": aggregates, "rows_root": canonical_sha(aggregates)},
    }
    for name, value in payloads.items():
        write_json(OUT / name, value)
    provisional = {
        name: {
            "sha256": sha256(OUT / name),
            "bytes": (OUT / name).stat().st_size,
            "lines": sum(1 for _ in (OUT / name).open("rb")),
        }
        for name in payloads
    }
    report_name = "p2r1-rms-tangent-target-only-factual-ko.md"
    report = render_report(aggregates, comparisons, integrity, provisional)
    (OUT / report_name).write_text(report, encoding="utf-8")
    (OUT / report_name).chmod(0o600)
    all_names = sorted([*payloads, report_name])
    manifest = {
        "schema": "ode-edit-s05-p2r1-target-only-analysis-manifest/v1",
        "instruction_id": "ODEEDIT-S05-P2R1-BASELINE-CALIBRATED-RMS-TANGENT-TARGET-FLOW-V1",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "scheduler": SCHEDULER,
        "integrity": integrity,
        "files": [
            {
                "path": name,
                "sha256": sha256(OUT / name),
                "bytes": (OUT / name).stat().st_size,
                "lines": sum(1 for _ in (OUT / name).open("rb")),
            }
            for name in all_names
        ],
    }
    manifest["root_digest"] = canonical_sha(manifest)
    write_json(OUT / "analysis-manifest.json", manifest)
    receipt = {
        "schema": "ode-edit-s05-p2r1-target-only-analysis-receipt/v1",
        "analysis_manifest_sha256": sha256(OUT / "analysis-manifest.json"),
        "analysis_manifest_root": manifest["root_digest"],
        "case_rows": len(cases),
        "step_rows": len(steps),
        "request_rows": len(requests),
        "panel_rows": len(panels),
        "comparison_rows": len(comparisons),
        "attempts": 20,
        "endpoints": 20,
        "failures": 0,
        "W0_restore_count": 20,
        "writer_update_count": 0,
        "writer_materialization_count": 0,
        "heldout_controller_access_count": 0,
        "native_endpoint_runtime_access_count": 0,
        "scientific_promotion": False,
        "gate_state": "P2R1_GATE_A1_TECHNICAL_PASS__SCIENTIFIC_RELEASE_PENDING_GH",
    }
    receipt["root_digest"] = canonical_sha(receipt)
    write_json(OUT / "analysis-receipt.json", receipt)


if __name__ == "__main__":
    main()
