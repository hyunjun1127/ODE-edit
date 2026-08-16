#!/usr/bin/env python3
"""Build the bounded raw-free P1R43-T3 terminal analysis package."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


WORKTREE = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r43-t3-fixed-w0-target-horizon-ablation-v1")
OUT = WORKTREE / "local/odebf/reports/p1r43-t3-fixed-w0-target-horizon-ablation-v1"
RESULTS = WORKTREE / "local/odebf/results"
P2 = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p2r1-baseline-calibrated-rms-tangent-target-flow-v1/local/odebf/reports/p2r1-rms-tangent-target-only-v1")
P43 = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r43-rho-free-semantic-first-strength-recovery-v1/local/odebf/reports/p1r43-rho-free-semantic-first-v1")
ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
ENDPOINTS = ("P1R43_T8", "P1R43_T24")
PANELS = (("eff_z_inject", "efficacy"), ("gen_z_inject", "generalization"))
STREAM_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
STREAM_ORDER = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
EVALUATOR = "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145"


def load(path: Path):
    return json.loads(path.read_text())


def canon(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canon(value)).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values, p: float):
    vals = sorted(float(x) for x in values)
    if not vals:
        return None
    index = (len(vals) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return vals[low]
    return vals[low] * (high - index) + vals[high] * (index - low)


def summary(values):
    vals = [float(x) for x in values]
    if not vals:
        return None
    return {
        "count": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "p90": percentile(vals, 0.9),
        "worst_max": max(vals),
        "min": min(vals),
    }


def write_json(name: str, value) -> Path:
    path = OUT / name
    path.write_bytes(canon(value) + b"\n")
    return path


def panel_numeric(panel, endpoint: str, key: str):
    vectors = panel["numeric_vectors"] if endpoint == "P1R43_T8" else panel["z_inject"]["numeric_vectors"]
    return [item for request in vectors[key] for item in request]


def f(value):
    if value is None:
        return "NOT_RECORDED"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def stats_text(row):
    if row is None:
        return "NOT_RECORDED"
    return "/".join(f(row[key]) for key in ("mean", "median", "p90", "worst_max"))


cases = {}
case_rows = []
endpoint_request_rows = []
update_request_rows = []
integrity = Counter()

for alias in ALIASES:
    root = RESULTS / f"s05-p1r43-t3-fixed-w0-target-horizon-b10x10-{alias}-tech-r1-v1"
    terminal_files = sorted(root.glob("raw/cases/case-*/terminal.json"))
    cases[alias] = [load(path) for path in terminal_files]
    assert len(cases[alias]) == 10
    for case in cases[alias]:
        case_id = int(case["case_index"])
        integrity["case_terminal"] += 1
        integrity["W0_pointer"] += int(case["W0_restore"]["pointer_restored_exact"])
        integrity["W0_bytes"] += int(case["W0_restore"]["byte_restored_exact"])
        integrity["calibration_one"] += int(case["entry_gradient_norm_calibration_count"] == 1)
        integrity["reset_8_16_zero"] += int(case["controller_reset_at_8_count"] == 0 and case["controller_reset_at_16_count"] == 0)
        forbidden = (
            case["writer_update_count"] + case["writer_materialization_count"]
            + case["virtual_W_update_count"] + case["native_endpoint_runtime_access_count"]
            + case["heldout_controller_access_count"] + case["retry_count"]
            + case["persistent_freeze_count"] + case["P2R1_operator_influence_count"]
            + case["target_KL_decay_RMS_tangent_gamma_clamp_influence_count"]
        )
        integrity["forbidden_zero"] += int(forbidden == 0)
        row = {
            "alias": alias,
            "case": case_id,
            "status": "COMPLETE",
            "request_count": case["request_count"],
            "target_update_count": case["target_update_count"],
            "h": case["h"],
            "W0_pointer_restored": case["W0_restore"]["pointer_restored_exact"],
            "W0_bytes_restored": case["W0_restore"]["byte_restored_exact"],
            "entry_calibration_count": case["entry_gradient_norm_calibration_count"],
            "reset_at_8_count": case["controller_reset_at_8_count"],
            "reset_at_16_count": case["controller_reset_at_16_count"],
            "forbidden_influence_sum": forbidden,
            "selection_0_7": case["selection_counts_updates_0_7"],
            "selection_8_23": case["selection_counts_updates_8_23"],
            "post_t8_path_length_mean": statistics.mean(case["post_t8_path_length_by_request"]),
            "z24_minus_z8_norm_mean": statistics.mean(case["z24_minus_z8_norm_by_request"]),
            "path_net_ratio": statistics.mean(case["z24_minus_z8_norm_by_request"]) / max(statistics.mean(case["post_t8_path_length_by_request"]), 1e-300),
            "compute": case["compute"],
            "target_wall_seconds": case["target_wall_seconds"],
            "terminal_evaluator_wall_seconds": case["terminal_evaluator_wall_seconds"],
            "total_wall_seconds": case["total_wall_seconds"],
            "identity_sha256": case["identity_sha256"],
            "action_freeze_sha256": case["action_freeze_sha256"],
        }
        for endpoint in ENDPOINTS:
            prefix = "t8" if endpoint.endswith("T8") else "t24"
            row[f"{prefix}_full6_nll"] = case["endpoint_full_six_target_new_nll"][endpoint]
            req_full = case["endpoint_full_six_target_new_nll_by_request"][endpoint]
            row[f"{prefix}_full6_request_summary"] = summary(req_full)
            row[f"{prefix}_full6_ge3_count"] = sum(float(x) >= 3.0 for x in req_full)
            row[f"{prefix}_full6_lt005_count"] = sum(float(x) < 0.05 for x in req_full)
            panel = case["endpoint_z_panels"][endpoint]
            for panel_name, vector_key in PANELS:
                metric = panel[panel_name]
                short = "eff" if panel_name.startswith("eff") else "gen"
                observations = panel_numeric(panel, endpoint, vector_key)
                row[f"{prefix}_{short}_correct"] = metric["numerator"]
                row[f"{prefix}_{short}_denominator"] = metric["denominator"]
                row[f"{prefix}_{short}_new_nll"] = summary(x["nll_new"] for x in observations)
                row[f"{prefix}_{short}_old_nll"] = summary(x["nll_old"] for x in observations)
                row[f"{prefix}_{short}_margin"] = summary(x["margin"] for x in observations)
                for obs in observations:
                    endpoint_request_rows.append({
                        "alias": alias, "case": case_id, "endpoint": endpoint,
                        "panel": vector_key.upper(), "request_index": obs["request_index"],
                        "prompt_index": obs["prompt_index"], "nll_new": obs["nll_new"],
                        "nll_old": obs["nll_old"], "margin": obs["margin"],
                    })
            for request_index, value in enumerate(req_full):
                endpoint_request_rows.append({
                    "alias": alias, "case": case_id, "endpoint": endpoint,
                    "panel": "FULL6_TARGET_NEW", "request_index": request_index,
                    "prompt_index": None, "nll_new": value, "nll_old": None, "margin": None,
                })
        row["delta_t24_minus_t8_full6"] = row["t24_full6_nll"] - row["t8_full6_nll"]
        row["delta_t24_minus_t8_eff_correct"] = row["t24_eff_correct"] - row["t8_eff_correct"]
        row["delta_t24_minus_t8_gen_correct"] = row["t24_gen_correct"] - row["t8_gen_correct"]
        case_rows.append(row)

        case_root = root / "raw/cases" / f"case-{case_id:02d}" / "raw/target"
        for k in range(24):
            step = load(case_root / f"microstep-{k:02d}.json")
            for request_index in range(10):
                update_request_rows.append({
                    "alias": alias, "case": case_id, "k": k,
                    "request_index": request_index,
                    "current_nll": step["current_full_six_target_nll_by_request"][request_index],
                    "gradient_norm": step["current_semantic_gradient_norm_by_request"][request_index],
                    "entry_gradient_norm": step["entry_semantic_gradient_norm_by_request"][request_index],
                    "current_entry_gradient_norm_ratio": step["current_entry_gradient_norm_ratio_by_request"][request_index],
                    "primary_displacement_norm": step["primary_displacement_norm_by_request"][request_index],
                    "primary_nll": step["primary_endpoint_nll_by_request"][request_index],
                    "rescue_eligible": step["rescue_plan"]["rescue_eligible_mask"][request_index],
                    "rescue_alpha": step["rescue_plan"]["alpha_rescue_by_request"][request_index],
                    "rescue_curvature": step["rescue_plan"]["curvature_by_request"][request_index],
                    "rescue_nll": None if step["rescue_endpoint_nll_by_request"] is None else step["rescue_endpoint_nll_by_request"][request_index],
                    "selection": step["selection_by_request"][request_index],
                    "selected_nll": step["selected_endpoint_nll_by_request"][request_index],
                    "selected_displacement_norm": step["selected_displacement_norm_by_request"][request_index],
                    "current_target_sha256": step["semantic_gradient_sha256"],
                    "primary_target_sha256": step["primary_delta_sha256"],
                    "selected_target_sha256": step["target_next_sha256"],
                    "entry_calibration_count": step["entry_norm_calibration_count"],
                    "reset_at_8_count": step["controller_state_reset_at_8_count"],
                    "reset_at_16_count": step["controller_state_reset_at_16_count"],
                    "h": step["h"],
                })


# Aggregate the primary same-trajectory causal comparison.
aggregate_rows = []
for alias in ALIASES:
    rows = [x for x in case_rows if x["alias"] == alias]
    agg = {"alias": alias, "attempts": 10, "endpoints": 10, "typed_failures": 0}
    for endpoint, prefix in (("P1R43_T8", "t8"), ("P1R43_T24", "t24")):
        req_full = [x["nll_new"] for x in endpoint_request_rows if x["alias"] == alias and x["endpoint"] == endpoint and x["panel"] == "FULL6_TARGET_NEW"]
        agg[f"{prefix}_full6_case_mean"] = summary(x[f"{prefix}_full6_nll"] for x in rows)
        agg[f"{prefix}_full6_request_distribution"] = summary(req_full)
        agg[f"{prefix}_full6_ge3_count"] = sum(x >= 3.0 for x in req_full)
        agg[f"{prefix}_full6_lt005_count"] = sum(x < 0.05 for x in req_full)
        for short, panel in (("eff", "EFFICACY"), ("gen", "GENERALIZATION")):
            vals = [x for x in endpoint_request_rows if x["alias"] == alias and x["endpoint"] == endpoint and x["panel"] == panel]
            agg[f"{prefix}_{short}_correct"] = sum(x[f"{prefix}_{short}_correct"] for x in rows)
            agg[f"{prefix}_{short}_denominator"] = sum(x[f"{prefix}_{short}_denominator"] for x in rows)
            agg[f"{prefix}_{short}_new_nll"] = summary(x["nll_new"] for x in vals)
            agg[f"{prefix}_{short}_margin"] = summary(x["margin"] for x in vals)
    deltas = [x["delta_t24_minus_t8_full6"] for x in rows]
    req8 = {(x["case"], x["request_index"]): x["nll_new"] for x in endpoint_request_rows if x["alias"] == alias and x["endpoint"] == "P1R43_T8" and x["panel"] == "FULL6_TARGET_NEW"}
    req24 = {(x["case"], x["request_index"]): x["nll_new"] for x in endpoint_request_rows if x["alias"] == alias and x["endpoint"] == "P1R43_T24" and x["panel"] == "FULL6_TARGET_NEW"}
    agg["t24_minus_t8_full6_case_delta"] = summary(deltas)
    agg["t24_minus_t8_full6_request_delta"] = summary(req24[key] - req8[key] for key in req8)
    agg["case_full6_improved_count"] = sum(x < 0 for x in deltas)
    agg["selection_0_7"] = {name: sum(x["selection_0_7"].get(name, 0) for x in rows) for name in ("PRIMARY", "RESCUE", "CURRENT")}
    agg["selection_8_23"] = {name: sum(x["selection_8_23"].get(name, 0) for x in rows) for name in ("PRIMARY", "RESCUE", "CURRENT")}
    agg["post_t8_path_length"] = summary(x for case in cases[alias] for x in case["post_t8_path_length_by_request"])
    agg["z24_minus_z8_norm"] = summary(x for case in cases[alias] for x in case["z24_minus_z8_norm_by_request"])
    agg["net_path_mean_ratio"] = agg["z24_minus_z8_norm"]["mean"] / agg["post_t8_path_length"]["mean"]
    agg["compute"] = {key: sum(x["compute"][key] for x in rows) for key in rows[0]["compute"] if isinstance(rows[0]["compute"][key], int)}
    agg["target_wall_seconds"] = sum(x["target_wall_seconds"] for x in rows)
    agg["terminal_evaluator_wall_seconds"] = sum(x["terminal_evaluator_wall_seconds"] for x in rows)
    agg["total_wall_seconds"] = sum(x["total_wall_seconds"] for x in rows)
    aggregate_rows.append(agg)


# P2R1 matched method comparison.
p2_case_doc = load(P2 / "p2r1-per-case.json")
p2_cases = {(x["alias"], int(x["case"])): x for x in p2_case_doc["rows"]}
p2_comparisons = []
for row in case_rows:
    comp = p2_cases[(row["alias"], row["case"])]
    p2_comparisons.append({
        "comparison_role": "SECONDARY_METHOD_MATCHED_COMPARISON",
        "alias": row["alias"], "case": row["case"],
        "identity_match": True,
        "shared_parent": "11508b6da11d606521b703037034e1814b70d8a8",
        "stream_root": STREAM_ROOT, "stream_order": STREAM_ORDER, "evaluator_sha256": EVALUATOR,
        "target_operator_hparams": "DIFFER_BY_COMPARISON_DESIGN",
        "t3_t24_full6_nll": row["t24_full6_nll"],
        "p2r1_z24_full6_nll": comp["terminal_full_six_target_new_nll"],
        "delta_t3_minus_p2r1_full6_nll": row["t24_full6_nll"] - comp["terminal_full_six_target_new_nll"],
        "t3_eff_correct": row["t24_eff_correct"], "p2r1_eff_correct": comp["eff_correct"],
        "t3_gen_correct": row["t24_gen_correct"], "p2r1_gen_correct": comp["gen_correct"],
        "t3_eff_nll": row["t24_eff_new_nll"]["mean"], "p2r1_eff_nll": comp["eff_new_nll"]["mean"],
        "t3_gen_nll": row["t24_gen_new_nll"]["mean"], "p2r1_gen_nll": comp["gen_new_nll"]["mean"],
    })


# Original P1R43 historical matched reference, never a direct causal comparator.
p43_case_doc = load(P43 / "p1r43-per-case.json")
p43_panel_doc = load(P43 / "p1r43-per-panel.json")
p43_cases = {(x["alias"], x["arm"], int(x["case"])): x for x in p43_case_doc["rows"]}
p43_panels = {(x["alias"], x["arm"], int(x["case"]), x["panel"]): x for x in p43_panel_doc["rows"]}
p43_comparisons = []
for row in case_rows:
    for arm in ("Neutral", "Soft"):
        hist = p43_cases[(row["alias"], arm, row["case"])]
        ep = p43_panels[(row["alias"], arm, row["case"], "EFF_Z_INJECT")]
        gp = p43_panels[(row["alias"], arm, row["case"], "GEN_Z_INJECT")]
        p43_comparisons.append({
            "comparison_role": "HISTORICAL_MATCHED_REFERENCE",
            "direct_causal_comparator": False,
            "boundary": "P1R43 changes W at each step; P1R43-T3 fixes W0",
            "alias": row["alias"], "arm": arm, "case": row["case"], "identity_match": True,
            "stream_root": STREAM_ROOT, "stream_order": STREAM_ORDER, "evaluator_sha256": EVALUATOR,
            "p1r43_z8_full6_nll_mean": hist["z8_full6_nll"],
            "p1r43_z8_full6_request_median_p90_worst": "NOT_RECORDED",
            "p1r43_eff_correct": ep["correct"], "p1r43_eff_denominator": ep["denominator"],
            "p1r43_eff_nll": ep["new_nll"], "p1r43_eff_margin": ep["margin"],
            "p1r43_gen_correct": gp["correct"], "p1r43_gen_denominator": gp["denominator"],
            "p1r43_gen_nll": gp["new_nll"], "p1r43_gen_margin": gp["margin"],
            "t8_minus_p1r43_full6": row["t8_full6_nll"] - hist["z8_full6_nll"],
            "t24_minus_p1r43_full6": row["t24_full6_nll"] - hist["z8_full6_nll"],
            "t8_minus_p1r43_eff_correct": row["t8_eff_correct"] - ep["correct"],
            "t24_minus_p1r43_eff_correct": row["t24_eff_correct"] - ep["correct"],
            "t8_minus_p1r43_gen_correct": row["t8_gen_correct"] - gp["correct"],
            "t24_minus_p1r43_gen_correct": row["t24_gen_correct"] - gp["correct"],
            "t8_minus_p1r43_eff_nll_mean": row["t8_eff_new_nll"]["mean"] - ep["new_nll"]["mean"],
            "t24_minus_p1r43_eff_nll_mean": row["t24_eff_new_nll"]["mean"] - ep["new_nll"]["mean"],
            "t8_minus_p1r43_eff_margin_mean": row["t8_eff_margin"]["mean"] - ep["margin"]["mean"],
            "t24_minus_p1r43_eff_margin_mean": row["t24_eff_margin"]["mean"] - ep["margin"]["mean"],
            "t8_minus_p1r43_gen_nll_mean": row["t8_gen_new_nll"]["mean"] - gp["new_nll"]["mean"],
            "t24_minus_p1r43_gen_nll_mean": row["t24_gen_new_nll"]["mean"] - gp["new_nll"]["mean"],
            "t8_minus_p1r43_gen_margin_mean": row["t8_gen_margin"]["mean"] - gp["margin"]["mean"],
            "t24_minus_p1r43_gen_margin_mean": row["t24_gen_margin"]["mean"] - gp["margin"]["mean"],
        })


case_path = write_json("p1r43-t3-per-case.json", {"schema": "p1r43-t3-per-case/v1", "rows": case_rows, "rows_root": digest(case_rows)})
update_path = write_json("p1r43-t3-per-request-update.json", {"schema": "p1r43-t3-per-request-update/v1", "rows": update_request_rows, "rows_root": digest(update_request_rows)})
endpoint_path = write_json("p1r43-t3-per-endpoint-request.json", {"schema": "p1r43-t3-per-endpoint-request/v1", "rows": endpoint_request_rows, "rows_root": digest(endpoint_request_rows)})
aggregate_path = write_json("p1r43-t3-aggregates.json", {"schema": "p1r43-t3-aggregates/v1", "rows": aggregate_rows, "rows_root": digest(aggregate_rows)})
p2_path = write_json("p1r43-t3-vs-p2r1-per-case.json", {"schema": "p1r43-t3-vs-p2r1-per-case/v1", "rows": p2_comparisons, "rows_root": digest(p2_comparisons)})
p43_path = write_json("p1r43-t3-vs-p1r43-historical-per-case.json", {"schema": "p1r43-t3-vs-p1r43-historical-per-case/v1", "rows": p43_comparisons, "rows_root": digest(p43_comparisons)})


def aggregate_p2(alias):
    rows = [x for x in p2_comparisons if x["alias"] == alias]
    return {
        "full6_delta": statistics.mean(x["delta_t3_minus_p2r1_full6_nll"] for x in rows),
        "t3_eff": sum(x["t3_eff_correct"] for x in rows), "p2_eff": sum(x["p2r1_eff_correct"] for x in rows),
        "t3_gen": sum(x["t3_gen_correct"] for x in rows), "p2_gen": sum(x["p2r1_gen_correct"] for x in rows),
        "t3_eff_nll": statistics.mean(x["t3_eff_nll"] for x in rows), "p2_eff_nll": statistics.mean(x["p2r1_eff_nll"] for x in rows),
        "t3_gen_nll": statistics.mean(x["t3_gen_nll"] for x in rows), "p2_gen_nll": statistics.mean(x["p2r1_gen_nll"] for x in rows),
    }


def aggregate_p43(alias, arm):
    rows = [x for x in p43_comparisons if x["alias"] == alias and x["arm"] == arm]
    return {
        "p1_full6": statistics.mean(x["p1r43_z8_full6_nll_mean"] for x in rows),
        "t8_delta": statistics.mean(x["t8_minus_p1r43_full6"] for x in rows),
        "t24_delta": statistics.mean(x["t24_minus_p1r43_full6"] for x in rows),
        "p1_eff": sum(x["p1r43_eff_correct"] for x in rows),
        "p1_gen": sum(x["p1r43_gen_correct"] for x in rows),
        "p1_eff_nll": statistics.mean(x["p1r43_eff_nll"]["mean"] for x in rows),
        "p1_eff_margin": statistics.mean(x["p1r43_eff_margin"]["mean"] for x in rows),
        "p1_gen_nll": statistics.mean(x["p1r43_gen_nll"]["mean"] for x in rows),
        "p1_gen_margin": statistics.mean(x["p1r43_gen_margin"]["mean"] for x in rows),
        "t8_eff_nll_delta": statistics.mean(x["t8_minus_p1r43_eff_nll_mean"] for x in rows),
        "t24_eff_nll_delta": statistics.mean(x["t24_minus_p1r43_eff_nll_mean"] for x in rows),
        "t8_eff_margin_delta": statistics.mean(x["t8_minus_p1r43_eff_margin_mean"] for x in rows),
        "t24_eff_margin_delta": statistics.mean(x["t24_minus_p1r43_eff_margin_mean"] for x in rows),
        "t8_gen_nll_delta": statistics.mean(x["t8_minus_p1r43_gen_nll_mean"] for x in rows),
        "t24_gen_nll_delta": statistics.mean(x["t24_minus_p1r43_gen_nll_mean"] for x in rows),
        "t8_gen_margin_delta": statistics.mean(x["t8_minus_p1r43_gen_margin_mean"] for x in rows),
        "t24_gen_margin_delta": statistics.mean(x["t24_minus_p1r43_gen_margin_mean"] for x in rows),
    }


llama = next(x for x in aggregate_rows if x["alias"] == "llama3-8b-inst")
qwen = next(x for x in aggregate_rows if x["alias"] == "qwen2.5-7b-inst")
lines = []
lines += [
    "# P1R43-T3 Fixed-W0 Target-Horizon Causal Ablation 종결 분석",
    "",
    f"- 생성 시각(Asia/Seoul): `{datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}`",
    "- checkpoint: `d646ddf74e7cc9eb85fd5c9c668ef729d00f3464`",
    "- exact P1R43 parent: `11508b6da11d606521b703037034e1814b70d8a8`",
    f"- stream/order: `{STREAM_ROOT}` / `{STREAM_ORDER}`",
    "- 시도/endpoint/typed failure: `20 / 20 / 0`",
    "- scientific_promotion=false",
    "",
    "## 한 페이지 5문항 요약",
    "",
    "1. **T8→T24가 target weakness를 해결했는가?** 두 모델 모두 10/10 case에서 full-six case-mean NLL이 감소하여 계약의 horizon-positive 방향 조건을 충족했다. Llama는 full-six mean `0.260137→0.174681`, Eff `99/100→99/100`, Gen `195/200→196/200`; Qwen은 `1.008113→0.888438`, Eff `93/100→94/100`, Gen `166/200→168/200`이다. 다만 Qwen T24의 full-six mean `0.888438`과 Gen `168/200`은 target weakness가 완전히 해소된 endpoint는 아니다.",
    "2. **두 모델 공통인가?** full-six 10/10 case 개선과 Eff/Gen 비감소는 공통이다. 개선 절대량과 최종 강도는 다르며, Llama T24가 Qwen T24보다 낮은 full-six NLL과 높은 Eff/Gen을 기록했다.",
    "3. **P2R1과 충분히 가까운가?** Llama T24−P2R1 full-six mean delta는 `+0.081329`, Eff `99−100`, Gen `196−197`; Qwen은 `+0.887794`, Eff `94−100`, Gen `168−193`이다. Qwen은 P2R1에 가깝지 않고, Llama도 exact equality는 아니다. 따라서 두 모델 공통으로 단순 P1R43-T3로 복귀할 수 있다는 조건은 충족되지 않았다.",
    "4. **남은 병목은 무엇인가?** 후반 CURRENT 비율은 Llama `140/1600`, Qwen `340/1600`; path/net mean ratio는 `0.9764`, `0.9225`라 큰 진동 지표는 없다. full-six와 Gen이 함께 개선되어 TRAIN_HELDOUT_GAP/LONG_HORIZON_OVERFIT에는 해당하지 않는다. Qwen은 추가 이동 후에도 높은 full-six NLL이 남아 `NORMALIZED_FLOW_UNDERSTRENGTH`; Llama는 소수 hard-tail이 남는 혼합 상태다. `FINITE_SELECTION_STALL`은 Qwen 후반 CURRENT 증가의 동반 telemetry이나 net motion이 작지 않아 단독 최종 분류로 쓰지 않았다.",
    "5. **다음 writer gate target operator는?** 계약 행렬 기준으로 P1R43-T3는 두 모델 공통 target gate를 닫지 못했고 P2R1과의 차이가 Qwen에서 크다. 따라서 writer gate에 넘길 target operator로 단순 P1R43-T3를 선택하는 조건은 충족되지 않았다. 이 문장은 계약 5문항의 gate 답변이며 추가 실험 추천은 아니다.",
    "",
    "## 무결성 및 scientific operator",
    "",
    "- exact P1R43 requestwise primary/rescue/current selector, entry norm 1회, FP64 field/update, FP32 selected target를 사용했다.",
    "- 동일 trajectory의 z8과 z24를 저장했고 h=`0.125`, target updates=`24`, reset@8/16=`0/0`이다.",
    "- evaluator adapter는 T8/T24 모두 snapshot_index=8, accepted_snapshot_count=9, fixed_budget_slots_completed=8이다.",
    "- P1R43 decision-operator replay이며 full executable-trace replay가 아니다. KL/decay/RMS/tangent/gamma/clamp decision influence=0이다.",
    f"- case terminal/W0 pointer/W0 bytes/calibration1/reset8·16 zero/forbidden zero: `{integrity['case_terminal']}/{integrity['W0_pointer']}/{integrity['W0_bytes']}/{integrity['calibration_one']}/{integrity['reset_8_16_zero']}/{integrity['forbidden_zero']}`.",
    "",
    "## Primary causal: 동일 trajectory T8↔T24",
    "",
    "Primary statistical unit은 10개 B10 case-mean paired delta이고, request 분포는 hard-tail 보조 통계다.",
    "",
    "| Model | full6 case mean T8→T24 | 개선 cases | pooled request med T8→T24 | >=3 | <.05 | Eff | Gen | Eff NLL | Gen NLL |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for agg in aggregate_rows:
    lines.append(
        f"| {agg['alias']} | {f(agg['t8_full6_case_mean']['mean'])}→{f(agg['t24_full6_case_mean']['mean'])} "
        f"| {agg['case_full6_improved_count']}/10 | {f(agg['t8_full6_request_distribution']['median'])}→{f(agg['t24_full6_request_distribution']['median'])} "
        f"| {agg['t8_full6_ge3_count']}→{agg['t24_full6_ge3_count']} | {agg['t8_full6_lt005_count']}→{agg['t24_full6_lt005_count']} "
        f"| {agg['t8_eff_correct']}/{agg['t8_eff_denominator']}→{agg['t24_eff_correct']}/{agg['t24_eff_denominator']} "
        f"| {agg['t8_gen_correct']}/{agg['t8_gen_denominator']}→{agg['t24_gen_correct']}/{agg['t24_gen_denominator']} "
        f"| {f(agg['t8_eff_new_nll']['mean'])}→{f(agg['t24_eff_new_nll']['mean'])} "
        f"| {f(agg['t8_gen_new_nll']['mean'])}→{f(agg['t24_gen_new_nll']['mean'])} |"
    )
lines += [
    "",
    "| Model | case Δfull6 mean/median/p90/worst | request Δfull6 mean/median/p90/worst | early P/R/C | late P/R/C | path mean | net mean | net/path |",
    "|---|---|---|---:|---:|---:|---:|---:|",
]
for agg in aggregate_rows:
    e, l = agg["selection_0_7"], agg["selection_8_23"]
    lines.append(f"| {agg['alias']} | {stats_text(agg['t24_minus_t8_full6_case_delta'])} | {stats_text(agg['t24_minus_t8_full6_request_delta'])} | {e['PRIMARY']}/{e['RESCUE']}/{e['CURRENT']} | {l['PRIMARY']}/{l['RESCUE']}/{l['CURRENT']} | {f(agg['post_t8_path_length']['mean'])} | {f(agg['z24_minus_z8_norm']['mean'])} | {f(agg['net_path_mean_ratio'])} |")

lines += [
    "",
    "## Secondary method: P2R1 target-only z24",
    "",
    "`SECONDARY_METHOD_MATCHED_COMPARISON`: model alias, tokenizer/model artifacts, frozen stream/order/case, full-six context plan, evaluator 및 exact P1R43 parent가 일치한다. Target operator/hparams는 비교의 의도된 방법 차이이며 동일하다고 주장하지 않는다.",
    "",
    "| Model | T3/P2 full6 mean | ΔT3−P2 | Eff T3/P2 | Gen T3/P2 | Eff NLL T3/P2 | Gen NLL T3/P2 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for alias in ALIASES:
    agg = next(x for x in aggregate_rows if x["alias"] == alias)
    p = aggregate_p2(alias)
    p2full = agg["t24_full6_case_mean"]["mean"] - p["full6_delta"]
    lines.append(f"| {alias} | {f(agg['t24_full6_case_mean']['mean'])}/{f(p2full)} | {f(p['full6_delta'])} | {p['t3_eff']}/100 / {p['p2_eff']}/100 | {p['t3_gen']}/200 / {p['p2_gen']}/200 | {f(p['t3_eff_nll'])}/{f(p['p2_eff_nll'])} | {f(p['t3_gen_nll'])}/{f(p['p2_gen_nll'])} |")

lines += [
    "",
    "## Mandatory historical reference: original P1R43 Neutral/Soft",
    "",
    "이 절의 비교 역할은 `HISTORICAL_MATCHED_REFERENCE`이며 `DIRECT_CAUSAL_COMPARATOR`가 아니다. Original P1R43은 각 step에서 W를 변경하지만 P1R43-T3는 전체 trajectory에서 W0를 고정한다. 인과 주장은 하지 않는다.",
    "",
    "| Model | Arm | P1R43 z8 full6 mean | T8−P1R43 | T24−P1R43 | P1R43 Eff | P1R43 Gen | P1R43 Eff NLL/margin | P1R43 Gen NLL/margin |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for alias in ALIASES:
    for arm in ("Neutral", "Soft"):
        x = aggregate_p43(alias, arm)
        lines.append(f"| {alias} | {arm} | {f(x['p1_full6'])} | {f(x['t8_delta'])} | {f(x['t24_delta'])} | {x['p1_eff']}/100 | {x['p1_gen']}/200 | {f(x['p1_eff_nll'])}/{f(x['p1_eff_margin'])} | {f(x['p1_gen_nll'])}/{f(x['p1_gen_margin'])} |")
lines += [
    "",
    "| Model | Arm | Eff NLL ΔT8/ΔT24 | Eff margin ΔT8/ΔT24 | Gen NLL ΔT8/ΔT24 | Gen margin ΔT8/ΔT24 |",
    "|---|---|---:|---:|---:|---:|",
]
for alias in ALIASES:
    for arm in ("Neutral", "Soft"):
        x = aggregate_p43(alias, arm)
        lines.append(f"| {alias} | {arm} | {f(x['t8_eff_nll_delta'])}/{f(x['t24_eff_nll_delta'])} | {f(x['t8_eff_margin_delta'])}/{f(x['t24_eff_margin_delta'])} | {f(x['t8_gen_nll_delta'])}/{f(x['t24_gen_nll_delta'])} | {f(x['t8_gen_margin_delta'])}/{f(x['t24_gen_margin_delta'])} |")
lines += [
    "",
    "Original P1R43 terminal full-six per-request median/p90/worst와 threshold별 hard-tail count는 원 raw-free P1R43 package에 `NOT_RECORDED`이다. 따라서 해당 필드는 상세 historical table에서도 `NOT_RECORDED`로 유지했다. Eff/Gen NLL·margin의 mean/median/p90/worst는 원 panel table에 기록되어 machine table에 보존된다.",
    "",
    "## Compute ledger",
    "",
    "| Model | semantic F/B | primary F | rescue F | endpoint F | target wall s | evaluator wall s | total case wall s | materialization |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for agg in aggregate_rows:
    c = agg["compute"]
    lines.append(f"| {agg['alias']} | {c['semantic_target_forward_count']}/{c['semantic_target_backward_count']} | {c['primary_endpoint_forward_count']} | {c['rescue_endpoint_forward_count']} | {c['endpoint_measurement_forward_count']} | {f(agg['target_wall_seconds'])} | {f(agg['terminal_evaluator_wall_seconds'])} | {f(agg['total_wall_seconds'])} | {c['materialization_count']} |")

lines += [
    "",
    "## Gate 및 taxonomy",
    "",
    "- `TECHNICAL_FAIL`: 0/20.",
    "- `HORIZON_POSITIVE`: 두 모델 모두 full-six case mean 10/10 감소, pooled median delta<0, Gen correct 비감소, Gen NLL 개선, >=3 hard-tail 비증가를 충족했다.",
    "- Llama: horizon-positive이며 P2R1과의 잔여 차이는 full-six mean +0.081329, Eff -1, Gen -1이다. 단일 request worst가 T8/T24 모두 높아 mixed hard-tail이 남는다.",
    "- Qwen: `NORMALIZED_FLOW_UNDERSTRENGTH`; 추가 path/net motion은 존재하지만 T24 full-six mean 0.888438, Eff 94/100, Gen 168/200이며 P2R1 대비 full-six +0.887794, Gen -25다.",
    "- `TRAIN_HELDOUT_GAP`, `LONG_HORIZON_OVERFIT`: 해당 없음. 두 모델 모두 full-six와 Gen NLL이 함께 개선됐다.",
    "- `OSCILLATORY_LONG_HORIZON`: 해당 없음. net/path mean ratio는 Llama 0.9764, Qwen 0.9225다.",
    "- `FINITE_SELECTION_STALL`: Qwen late CURRENT 340/1600이 기록됐으나 post-T8 net motion이 작지 않아 단독 최종 분류로 확정하지 않았다.",
    "",
    "## FACT / INFERENCE / NOT_RECORDED",
    "",
    "### FACT",
    "",
    "- attempts/endpoints/failures=20/20/0, W0 pointer+bytes 20/20, forbidden influence0 20/20.",
    "- T8와 T24는 동일한 continuous z0→z24 trajectory의 immutable endpoints다.",
    "- case/request exact paired raw values와 arithmetic deltas는 machine tables에 있다.",
    "",
    "### INFERENCE",
    "",
    "- 장기 horizon은 두 모델에서 target objective와 Gen 지표를 개선했지만, P2R1과의 잔여 차이는 특히 Qwen에서 크다.",
    "- fixed-W0 causal contrast로 horizon 부족은 부분 원인이지만 단독으로 전체 target weakness를 설명하지 못한다.",
    "",
    "### NOT_RECORDED / NOT_APPLICABLE",
    "",
    "- stepwise heldout Eff/Gen/Loc: `NOT_RECORDED` (endpoint-only evaluator).",
    "- W-only, physical locality, Structural-P, capacity, update norm/energy: `NOT_APPLICABLE` (target-only, W=W0).",
    "- Original P1R43 terminal full-six per-request median/p90/worst 및 hard-tail count: `NOT_RECORDED`.",
    "- task-level scheduler elapsed/MaxRSS: scientific case receipt에 `NOT_RECORDED`; job id는 submission receipt에 기록됨.",
    "",
    "## Artifact identities",
    "",
    f"- contract SHA: `d9245782bf3339bfabe48296a9e673ba33763d1bca43be60982a182084ea67e7`",
    f"- P2R1 report SHA: `{sha(P2 / 'p2r1-rms-tangent-target-only-factual-ko.md')}`",
    f"- original P1R43 report SHA: `{sha(P43 / 'p1r43-rho-free-semantic-first-b10x10-factual-ko.md')}`",
    "- detailed exact identities and all file hashes are in `analysis-manifest.json`.",
]

report_path = OUT / "p1r43-t3-fixed-w0-target-horizon-terminal-analysis-ko.md"
report_path.write_text("\n".join(lines) + "\n")

artifact_paths = [case_path, update_path, endpoint_path, aggregate_path, p2_path, p43_path, report_path]
manifest_entries = []
for path in artifact_paths:
    manifest_entries.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "lines": sum(1 for _ in path.open("rb"))})
source_identities = {
    "checkpoint": "d646ddf74e7cc9eb85fd5c9c668ef729d00f3464",
    "checkpoint_tree": "d4c57ea14562af5637f8ede7299782f0348370cb",
    "exact_p1r43_parent": "11508b6da11d606521b703037034e1814b70d8a8",
    "contract_sha256": "d9245782bf3339bfabe48296a9e673ba33763d1bca43be60982a182084ea67e7",
    "numerical_lock_sha256": sha(WORKTREE / "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r43_t3_fixed_w0_target_horizon.json"),
    "source_manifest_sha256": sha(WORKTREE / "project/run_scripts/ode_bf/locks/source_manifest_s05_p1r43_t3_fixed_w0_target_horizon.json"),
    "stream_root": STREAM_ROOT, "stream_order": STREAM_ORDER, "evaluator_sha256": EVALUATOR,
}
comparison_identities = {
    "p2r1_report": {"path": str(P2 / "p2r1-rms-tangent-target-only-factual-ko.md"), "sha256": sha(P2 / "p2r1-rms-tangent-target-only-factual-ko.md")},
    "p2r1_per_case_sha256": sha(P2 / "p2r1-per-case.json"),
    "p2r1_analysis_manifest_sha256": sha(P2 / "analysis-manifest.json"),
    "p2r1_numerical_lock_sha256": sha(P2.parents[3] / "project/run_scripts/ode_bf/locks/numerical_lock_s05_p2r1_rms_tangent_target_only.json"),
    "p1r43_report": {"path": str(P43 / "p1r43-rho-free-semantic-first-b10x10-factual-ko.md"), "sha256": sha(P43 / "p1r43-rho-free-semantic-first-b10x10-factual-ko.md")},
    "p1r43_per_case_sha256": sha(P43 / "p1r43-per-case.json"),
    "p1r43_per_panel_sha256": sha(P43 / "p1r43-per-panel.json"),
    "p1r43_analysis_manifest_sha256": sha(P43 / "analysis-manifest.json"),
    "p1r43_frozen_comparison_identities_sha256": sha(Path("/mnt/raid5/janghj/ODE-edit/local/source-handoff/P1R43_P1R42_PARENT_SUPPORT_SH2_V1/identities/frozen-comparison-identities.json")),
}
manifest = {
    "schema": "p1r43-t3-analysis-manifest/v1", "instruction_id": "ODEEDIT-S05-P1R43-T3-FIXED-W0-TARGET-HORIZON-ABLATION-V1",
    "source_identities": source_identities, "comparison_identities": comparison_identities,
    "integrity": dict(integrity), "attempts": 20, "endpoints": 20, "typed_failures": 0,
    "files": manifest_entries, "root_digest": digest(manifest_entries), "scientific_promotion": False,
}
manifest_path = write_json("analysis-manifest.json", manifest)
receipt = {
    "schema": "p1r43-t3-analysis-receipt/v1", "analysis_agent": "separate_bounded_read_only",
    "contract_read_complete": True, "contract_lines": 582, "attempts": 20, "endpoints": 20, "typed_failures": 0,
    "case_rows": len(case_rows), "update_request_rows": len(update_request_rows), "endpoint_request_rows": len(endpoint_request_rows),
    "p2r1_comparison_rows": len(p2_comparisons), "p1r43_historical_comparison_rows": len(p43_comparisons),
    "analysis_manifest_sha256": sha(manifest_path), "analysis_manifest_root": manifest["root_digest"],
    "gate_state": "HORIZON_POSITIVE_BOTH__P1R43_T3_NOT_COMMON_P2R1_EQUIVALENT__QWEN_NORMALIZED_FLOW_UNDERSTRENGTH",
    "report_sha256": sha(report_path), "scientific_promotion": False,
}
receipt["root_digest"] = digest(receipt)
receipt_path = write_json("analysis-receipt.json", receipt)
print(json.dumps({"report": str(report_path), "report_sha256": sha(report_path), "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "receipt": str(receipt_path), "receipt_sha256": sha(receipt_path)}, indent=2))
