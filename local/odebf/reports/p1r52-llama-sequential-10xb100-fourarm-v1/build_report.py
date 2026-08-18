#!/usr/bin/env python3
"""Build the bounded raw-free P1R52 10xB100 four-arm factual package."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "local/odebf/results"
ARMS = {
    "memit": (
        "Official EasyEdit MEMIT Sequential",
        RESULTS / "s05-p1r52-official-memit-sequential-10xb100-tech-r1-v1",
        "b1-b10-pre-post-checkpoints.json",
        "true-entry-post-final-aggregate.json",
        "entry-pre-immediate-post-final-w10-requests.json",
    ),
    "alphaedit": (
        "Official EasyEdit AlphaEdit Sequential (cache_c ON)",
        RESULTS / "s05-p1r52-official-alphaedit-sequential-cache-on-10xb100-tech-r1-v1",
        "b1-b10-pre-post-checkpoints.json",
        "true-entry-post-final-aggregate.json",
        "entry-pre-immediate-post-final-w10-requests.json",
    ),
    "r52_h_on": (
        "P1R52 Repair-R1 Soft Sequential / Structural-H ON",
        RESULTS / "s05-p1r52-llama-soft-sequential-structuralh-on-10xb100-tech-r3-release-r1-v1",
        "b1-b10-post-final-checkpoints.json",
        "post-final-aggregate.json",
        "immediate-post-final-w10-requests.json",
    ),
    "r52_h_off": (
        "P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF",
        RESULTS / "s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb100-tech-r3-release-r1-v1",
        "b1-b10-post-final-checkpoints.json",
        "post-final-aggregate.json",
        "immediate-post-final-w10-requests.json",
    ),
}
METRICS = ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def dump_once(name: str, value) -> Path:
    path = HERE / name
    write_once(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())
    return path


def summary(panel: dict) -> dict:
    out = {"identity_sha256": panel["identity_sha256"]}
    for metric in METRICS:
        x = panel["metrics"][metric]
        out[metric] = {
            "numerator": x["prompt_numerator"],
            "denominator": x["prompt_denominator"],
            "rate": x["prompt_rate"],
            "strict_numerator": x["strict_request_numerator"],
            "strict_denominator": x["strict_request_denominator"],
            "strict_rate": x["strict_request_rate"],
            "bit_vector_sha256": x["prompt_bit_vector_sha256"],
            "strict_bit_vector_sha256": x["strict_request_bit_vector_sha256"],
            "target_new_nll": x["target_new_nll"],
            "target_old_nll": x["target_old_nll"],
            "margin": x["target_old_minus_new_margin"],
        }
    loc = panel["locality"]
    out["locality"] = {
        "numerator": loc["numerator"], "denominator": loc["denominator"],
        "rate": loc["rate"], "bit_vector_sha256": loc["bit_vector_sha256"],
    }
    return out


def count(x: dict, strict: bool = False) -> str:
    p = "strict_" if strict else ""
    return f"{x[p+'numerator']}/{x[p+'denominator']} ({100*x[p+'rate']:.2f}%)"


def loc(x: dict) -> str:
    return f"{x['numerator']}/{x['denominator']} ({100*x['rate']:.2f}%)"


def nll(x: dict) -> str:
    return f"{x['target_new_nll']['mean']:.6f}/{x['target_old_nll']['mean']:.6f}/{x['margin']['mean']:+.6f}"


def request_projection(arm: str, row: dict) -> dict:
    # Entry-pre fields from the two legacy-running Official cells are deliberately omitted.
    return {
        "arm": arm,
        "case_id": row.get("case_id"),
        "request_index": row.get("request_index"),
        "round": row.get("round"),
        "history_width_at_entry": row.get("history_width_at_entry"),
        "immediate_post": row["immediate_post"],
        "final_W10": row["final_W10"],
        "lifetime_forgetting": {
            metric: row["deltas"][metric].get("lifetime_forgetting_final_minus_post_rate")
            for metric in (*METRICS, "locality")
        },
        "batch_entry_metrics_status": "NONCANONICAL_UNUSED_USER_AMENDMENT",
        "identity_sha256": row["identity_sha256"],
    }


def main() -> None:
    terminals, aggregates, checkpoint_rows, request_rows = {}, {}, {}, []
    input_members = []
    for arm, (_, result, checkpoint_name, aggregate_name, request_name) in ARMS.items():
        for name in ("terminal.json", "manifest.json", checkpoint_name, aggregate_name, request_name):
            path = result / name
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"missing/nonregular input: {path}")
            input_members.append({"arm": arm, "path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
        terminals[arm] = load(result / "terminal.json")
        aggregate = load(result / aggregate_name)
        aggregates[arm] = {
            "immediate_post_B100_panels": summary(aggregate["immediate_post_B100_panels"]),
            "final_W10_B1000": summary(aggregate["final_W10_B1000"]),
            "final_minus_immediate": aggregate["final_W10_minus_immediate_post"],
            "batch_entry_metrics_status": "NONCANONICAL_UNUSED_USER_AMENDMENT",
        }
        checkpoints = load(result / checkpoint_name)
        if checkpoints["row_count"] != 10:
            raise RuntimeError(f"checkpoint denominator differs: {arm}")
        checkpoint_rows[arm] = checkpoints["rows"]
        requests = load(result / request_name)
        if requests["row_count"] != 1000:
            raise RuntimeError(f"request denominator differs: {arm}")
        request_rows.extend(request_projection(arm, row) for row in requests["rows"])

    # Single common original-W0 panel: B1 entry is byte/metric identical across Official cells.
    w0_m = checkpoint_rows["memit"][0]["entry_pre_summary"]
    w0_a = checkpoint_rows["alphaedit"][0]["entry_pre_summary"]
    if w0_m != w0_a:
        raise RuntimeError("common W0 B1 panel differs")
    common_w0 = summary(w0_m)

    batch_rows = []
    for arm, rows in checkpoint_rows.items():
        for row in rows:
            batch_rows.append({
                "arm": arm,
                "round": row["round"],
                "history_width_at_entry": row["history_width_at_entry"],
                "immediate_post": summary(row["immediate_post_summary"]),
                "cumulative_to_date": summary(row["cumulative_evaluation"]),
                "entry_pre": "NONCANONICAL_UNUSED_USER_AMENDMENT",
                "commit_weight_sha256": row["commit_weight_sha256"],
                "terminal_sha256": row["terminal_sha256"],
            })

    controller_rows = []
    for arm in ("r52_h_on", "r52_h_off"):
        result = ARMS[arm][1]
        for batch in range(1, 11):
            bt = load(result / f"raw/batches/b{batch:02d}/terminal.json")
            for step in range(1, 9):
                p = result / f"raw/batches/b{batch:02d}/raw/ode/p1r52-rsa-r42safekdc-m1-soft/accepted-k{step}.json"
                x = load(p)
                routing, progress, material = x["routing"], x["progress"], x["materialization"]
                hrows = bt["structural_h_routing"]
                hrow = hrows[step - 1] if isinstance(hrows, list) and len(hrows) == 8 else None
                controller_rows.append({
                    "arm": arm, "round": batch, "step": step,
                    "alpha_req": routing["alpha_req"], "alpha_apply": routing["alpha_apply"],
                    "predicted": progress.get("predicted"), "actual": progress.get("actual"),
                    "realization_ratio": progress.get("realization_ratio"),
                    "negative_actual": progress.get("actual") is not None and progress["actual"] < 0,
                    "structural_p": x["structural_p"],
                    "capacity": routing["selected_capacity"], "energy": routing["selected_energy"],
                    "entropy": routing["simplex_entropy"], "top1": routing["simplex_top1_share"],
                    "fallback": routing["fallback_to_neutral"], "status": routing["status"],
                    "strength_residual": routing["equality_residual"],
                    "bf16_step_energy": sum(material["realized_bf16_step_energy"].values()),
                    "history_width": bt["history_width_at_entry"],
                    "alpha_cache_append": bt["alpha_solve_cache_append_count"],
                    "alpha_cache_consume": bt["alpha_solve_cache_consume_count"],
                    "structural_h_decision_influence": bt["structural_h_decision_influence_count"],
                    "h_status": hrow["status"] if hrow else "STRUCTURAL_H_OFF",
                    "h_selected": hrow["h_selected"] if hrow else None,
                    "h_neutral": hrow["h_neutral"] if hrow else None,
                    "h_delta": hrow["h_delta"] if hrow else None,
                    "accepted_identity_sha256": x["identity_sha256"],
                })

    hard = {}
    for arm in ARMS:
        rows = [r for r in request_rows if r["arm"] == arm]
        hard[arm] = {
            "requests": len(rows),
            "post_eff_fail": sum(r["immediate_post"]["rewrite_success"]["strict_all_prompts_pass"] == 0 for r in rows),
            "final_eff_fail": sum(r["final_W10"]["rewrite_success"]["strict_all_prompts_pass"] == 0 for r in rows),
            "post_gen_strict_fail": sum(r["immediate_post"]["paraphrase_success"]["strict_all_prompts_pass"] == 0 for r in rows),
            "final_gen_strict_fail": sum(r["final_W10"]["paraphrase_success"]["strict_all_prompts_pass"] == 0 for r in rows),
            "eff_success_to_failure": sum(r["immediate_post"]["rewrite_success"]["strict_all_prompts_pass"] == 1 and r["final_W10"]["rewrite_success"]["strict_all_prompts_pass"] == 0 for r in rows),
            "gen_success_to_failure": sum(r["immediate_post"]["paraphrase_success"]["strict_all_prompts_pass"] == 1 and r["final_W10"]["paraphrase_success"]["strict_all_prompts_pass"] == 0 for r in rows),
        }

    integrity = {}
    for arm, terminal in terminals.items():
        integrity[arm] = {
            "source_head": terminal["source_head"], "round_count": terminal["round_count"],
            "request_count": terminal["request_count"], "action_freeze_checkpoints": terminal["action_freeze_checkpoint_count"],
            "interbatch_W0_restore_count": terminal["interbatch_W0_restore_count"],
            "terminal_W0_restore_count": terminal["terminal_W0_restore_count"],
            "terminal_W0_restore": terminal["terminal_W0_restore"],
            "history_widths": terminal["history_widths"],
            "terminal_active_history_count": terminal["terminal_active_history_count"],
            "terminal_lifetime_anchor_count": terminal["terminal_lifetime_anchor_count"],
            "batch_entry_metrics_status": "NONCANONICAL_UNUSED_USER_AMENDMENT",
            "batch_entry_evaluator_count_canonical": 0 if arm.startswith("r52_") else "NONCANONICAL_UNUSED_RAW_RECEIPT_ONLY",
            "accuracy_added_F_B_generation": [terminal["accuracy_amendment_added_model_forward_count"], terminal["accuracy_amendment_added_backward_count"], terminal["accuracy_amendment_added_generation_count"]],
            "job_compute": terminal["job_compute"],
            "total_wall_seconds": terminal["total_wall_seconds"],
        }

    aggregate_path = dump_once("p1r52-b100-four-arm-aggregate.json", {
        "schema": "p1r52-sequential-b100x10/four-arm-aggregate/v1", "common_W0_B1_panel": common_w0,
        "arms": aggregates, "hard_cohorts": hard,
        "batch_entry_aggregate_policy": "REMOVED_NONCANONICAL_UNUSED",
    })
    batch_path = dump_once("p1r52-b100-four-arm-batch-checkpoints.json", {
        "schema": "p1r52-sequential-b100x10/batch-checkpoints/v1", "row_count": len(batch_rows), "rows": batch_rows,
    })
    request_path = dump_once("p1r52-b100-four-arm-request-retention.json", {
        "schema": "p1r52-sequential-b100x10/request-retention/v1", "row_count": len(request_rows), "rows": request_rows,
    })
    controller_path = dump_once("p1r52-b100-r52-controller-routing.json", {
        "schema": "p1r52-sequential-b100x10/r52-controller-routing/v1", "row_count": len(controller_rows), "rows": controller_rows,
    })
    integrity_path = dump_once("p1r52-b100-four-arm-integrity-compute.json", {
        "schema": "p1r52-sequential-b100x10/integrity-compute/v1", "arms": integrity,
        "input_members": input_members, "input_members_root_sha256": root(input_members),
        "scheduler": {
            "Official_MEMIT": {"job": "20453_0/20454", "state": "COMPLETED", "exit_code": "0:0", "elapsed": "01:12:05"},
            "Official_AlphaEdit": {"job": "20453_1/20455", "state": "COMPLETED", "exit_code": "0:0", "elapsed": "01:14:56"},
            "R52_H_ON": {"job": "20467_0/20467", "state": "COMPLETED", "exit_code": "0:0", "elapsed": "01:52:02"},
            "R52_H_OFF": {"job": "20467_1/20468", "state": "COMPLETED", "exit_code": "0:0", "elapsed": "01:33:39"},
        },
    })

    lines = [
        "# P1R52 Llama Sequential 10×B100 — 4팔 최종 사실 보고서", "",
        "> 범위: 10개 순차 B100, 총 1,000 edits/arm. 모델·evaluator 재실행 없이 봉인된 raw-free 결과만 사용했습니다.", "",
        "## 판정 및 경계", "",
        "- 네 팔 모두 10/10 배치, 1,000/1,000 requests, terminal W0 pointer+byte restore를 완료했습니다.",
        "- 사용자 지시에 따라 method-specific W_(b-1) batch-entry aggregate는 `NONCANONICAL_UNUSED`입니다. 아래 표에는 단일 공통 W0(B1 entry)와 immediate-post/final-W10만 포함합니다.",
        "- EFF ≡ rewrite_success, GEN ≡ paraphrase_success(rephrase_success alias). Rewrite/Rephrase Acc는 별도 strict suffix-token accuracy입니다.",
        "- 결과는 이 1,000-edit Llama 패키지에 한정하며 isolated Structural-H causality를 주장하지 않습니다.", "",
        "## 공통 original-W0 B1 baseline", "",
        "|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|", "|---:|---:|---:|---:|---:|---:|---:|",
        f"|{count(common_w0['rewrite_success'])}|{count(common_w0['rewrite_acc'])}|{count(common_w0['paraphrase_success'])}|{count(common_w0['paraphrase_success'], True)}|{count(common_w0['paraphrase_acc'])}|{count(common_w0['paraphrase_acc'], True)}|{loc(common_w0['locality'])}|", "",
        "## 1,000-request 절대값: immediate-post aggregate와 final W10", "",
        "|Arm|Panel|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|", "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, (label, *_rest) in ARMS.items():
        for key, panel_label in (("immediate_post_B100_panels", "10× immediate-post"), ("final_W10_B1000", "final W10/B1000")):
            x = aggregates[arm][key]
            lines.append(f"|{label}|{panel_label}|{count(x['rewrite_success'])}|{count(x['rewrite_acc'])}|{count(x['paraphrase_success'])}|{count(x['paraphrase_success'], True)}|{count(x['paraphrase_acc'])}|{count(x['paraphrase_acc'], True)}|{loc(x['locality'])}|")
    lines += ["", "## final W10 NLL / margin", "", "|Arm|Rewrite new/true/margin|Rephrase new/true/margin|", "|---|---:|---:|"]
    for arm, (label, *_rest) in ARMS.items():
        x = aggregates[arm]["final_W10_B1000"]
        lines.append(f"|{label}|{nll(x['rewrite_success'])}|{nll(x['paraphrase_success'])}|")

    lines += ["", "## B1–B10 immediate-post 절대 EFF / GEN / LOC", "", "|B|Arm|EFF|GEN|GEN strict|LOC|history width|", "|---:|---|---:|---:|---:|---:|---:|"]
    for row in batch_rows:
        x = row["immediate_post"]
        lines.append(f"|B{row['round']}|{ARMS[row['arm']][0]}|{count(x['rewrite_success'])}|{count(x['paraphrase_success'])}|{count(x['paraphrase_success'], True)}|{loc(x['locality'])}|{row['history_width_at_entry']}|")

    lines += ["", "## hard / forgetting cohort", "", "|Arm|requests|post EFF fail|final EFF fail|post GEN-strict fail|final GEN-strict fail|EFF success→fail|GEN success→fail|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, x in hard.items():
        lines.append(f"|{ARMS[arm][0]}|{x['requests']}|{x['post_eff_fail']}|{x['final_eff_fail']}|{x['post_gen_strict_fail']}|{x['final_gen_strict_fail']}|{x['eff_success_to_failure']}|{x['gen_success_to_failure']}|")
    lines += ["", "## R52 routing / history 사실", ""]
    for arm in ("r52_h_on", "r52_h_off"):
        rows = [r for r in controller_rows if r["arm"] == arm]
        status = Counter(r["h_status"] for r in rows)
        lines.append(f"- {ARMS[arm][0]}: 80/80 writes; negative actual `{sum(r['negative_actual'] for r in rows)}`; fallback `{sum(r['fallback'] for r in rows)}`; H status `{dict(status)}`; max |strength residual| `{max(abs(r['strength_residual']) for r in rows):.3e}`; cache entry widths `[0,100,...,900]`.")
    lines += ["", "## Compute / transaction", "", "|Arm|wall s|completed K|model F/B|evaluator F|tokens|materializations|entry evaluator canonical|W0 restore|", "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for arm, x in integrity.items():
        c = x["job_compute"]["counters"]
        lines.append(f"|{ARMS[arm][0]}|{x['total_wall_seconds']:.1f}|{x['job_compute']['completed_k_total']}|{c.get('model_forward','NOT_RECORDED')}/{c.get('backward','NOT_RECORDED')}|{c.get('evaluator_forward','NOT_RECORDED')}|{c.get('processed_tokens','NOT_RECORDED')}|{c.get('materialization_count','NOT_RECORDED')}|{x['batch_entry_evaluator_count_canonical']}|{x['terminal_W0_restore']['pointer_restored_exact']}/{x['terminal_W0_restore']['byte_restored_exact']}|")
    lines += ["", "## Machine artifacts", "", f"- aggregate: `{aggregate_path}`", f"- batch rows: `{batch_path}` (40)", f"- request rows: `{request_path}` (4000)", f"- R52 controller rows: `{controller_path}` (160)", f"- integrity/compute: `{integrity_path}`", "", "scientific_promotion=false"]
    report = HERE / "p1r52-llama-sequential-10xb100-fourarm-factual-ko.md"
    write_once(report, ("\n".join(lines) + "\n").encode())

    outputs = [aggregate_path, batch_path, request_path, controller_path, integrity_path, report]
    members = [{"path": str(p), "bytes": p.stat().st_size, "lines": len(p.read_text().splitlines()), "sha256": sha(p)} for p in outputs]
    manifest = dump_once("manifest.json", {"schema": "p1r52-sequential-b100x10/four-arm-report-manifest/v1", "status": "PASS", "members": members, "members_root_sha256": root(members)})
    review_checks = {
        "four_terminal_roots": len(terminals) == 4,
        "rounds_10_each": all(x["round_count"] == 10 for x in terminals.values()),
        "requests_1000_each": all(x["request_count"] == 1000 for x in terminals.values()),
        "action_freeze_10_each": all(x["action_freeze_checkpoint_count"] == 10 for x in terminals.values()),
        "terminal_W0_pointer_byte": all(x["terminal_W0_restore"]["pointer_restored_exact"] and x["terminal_W0_restore"]["byte_restored_exact"] for x in terminals.values()),
        "batch_rows_40": len(batch_rows) == 40, "request_rows_4000": len(request_rows) == 4000,
        "controller_rows_160": len(controller_rows) == 160,
        "r52_batch_entry_evaluator_zero": all(terminals[a]["batch_entry_pre_evaluator_count"] == 0 for a in ("r52_h_on", "r52_h_off")),
        "entry_fields_excluded": all("entry_pre" not in r for r in request_rows),
    }
    if not all(review_checks.values()):
        raise RuntimeError(f"independent checks failed: {review_checks}")
    review = dump_once("independent-rawfree-rehash-review.json", {"schema": "p1r52-sequential-b100x10/four-arm-independent-review/v1", "status": "PASS", "checks": review_checks, "manifest_sha256": sha(manifest), "manifest_root": load(manifest)["members_root_sha256"], "model_evaluator_gpu_slurm_actions": [0, 0, 0, 0]})
    final_members = members + [{"path": str(manifest), "bytes": manifest.stat().st_size, "lines": len(manifest.read_text().splitlines()), "sha256": sha(manifest)}, {"path": str(review), "bytes": review.stat().st_size, "lines": len(review.read_text().splitlines()), "sha256": sha(review)}]
    receipt = dump_once("rooted-receipt.json", {"schema": "p1r52-sequential-b100x10/four-arm-rooted-receipt/v1", "status": "TERMINAL_FACTUAL_REPORT_COMPLETE", "canonical_report": str(report), "canonical_report_sha256": sha(report), "canonical_report_bytes": report.stat().st_size, "canonical_report_lines": len(report.read_text().splitlines()), "members": final_members, "package_root_sha256": root(final_members), "raw_prompts_targets_generations_tensors_weights_included": False, "analysis_actions_model_evaluator_gpu_slurm": [0, 0, 0, 0]})
    print(json.dumps({"report": str(report), "report_sha256": sha(report), "report_bytes": report.stat().st_size, "report_lines": len(report.read_text().splitlines()), "manifest": str(manifest), "review": str(review), "receipt": str(receipt), "package_root": load(receipt)["package_root_sha256"], "rows": {"batch": len(batch_rows), "request": len(request_rows), "controller": len(controller_rows)}}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
