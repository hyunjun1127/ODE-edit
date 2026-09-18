"""봉인된 공개 집계만 검산한다. 모델·raw checkpoint를 읽거나 실행하지 않는다."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT / "local/reviews/sl-zflow-seq1000-review-2026-09-16/source"
REPORT = SNAPSHOT / "experiment-reports/servers/server2/single-layer-zflow-seq1000-2026-09-16-v1"
EXECUTION_HEAD = "5d149fec254a7a53b6d91790d186880f248676e6"
CHECKS: Counter = Counter()


def check(group, condition):
    assert condition, group
    CHECKS[group] += 1


def read(name):
    with (REPORT / "aggregates" / name).open() as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return float(row[key])


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)


def total(rows, field):
    return sum(number(row, field) for row in rows)


package = json.loads((REPORT / "package-manifest.json").read_text())
for member in package["members"]:
    relative = member["path"].split("single-layer-zflow-seq1000-2026-09-16-v1/", 1)[1]
    data = (REPORT / relative).read_bytes()
    check("package_member_hash_and_size", len(data) == member["bytes"] and
          hashlib.sha256(data).hexdigest() == member["sha256"])

source_hashes = {}
for name in ("runtime.py", "native_binding.py", "llama_adapter.py", "technical.py",
             "durable.py", "flow_core.py", "transaction.py"):
    relative = "project/run_scripts/single_layer_zflow/" + name
    execution = subprocess.check_output(["git", "show", f"{EXECUTION_HEAD}:{relative}"], cwd=ROOT)
    snapshot = (SNAPSHOT / relative).read_bytes()
    check("execution_source_unchanged", execution == snapshot)
    source_hashes[name] = hashlib.sha256(snapshot).hexdigest()

metrics, paired, batches, nodes, compute = map(read, (
    "metrics.csv", "paired.csv", "batch.csv", "node.csv", "compute.csv"))
for row in metrics:
    check("metric_percent", close(number(row, "percent"),
                                  100 * number(row, "numerator") / number(row, "denominator")))
    check("metric_margin", close(number(row, "success_margin_mean"),
                                (number(row, "new_nll_mean") - number(row, "true_nll_mean")) *
                                (1 if row["metric"] == "NS" else -1)))
for row in paired:
    if not row["denominator"] or number(row, "denominator") == 0:
        continue
    check("paired_transition", close(number(row, "left_success"),
                                     number(row, "retained") + number(row, "lost")) and
          close(number(row, "right_success"), number(row, "retained") + number(row, "gained")) and
          close(number(row, "denominator"), sum(number(row, key) for key in
                ("retained", "lost", "gained", "failed_both"))))
    check("paired_percent", close(number(row, "delta_pp"), 100 *
          (number(row, "gained") - number(row, "lost")) / number(row, "denominator")))

final = {x["metric"]: x for x in metrics if x["batch"] == "B010" and x["scope"] == "seen-full"}
baseline = {x["metric"]: x for x in metrics if x["scope"] == "historical-n4"}
quality = {}
for metric in ("RS", "PS", "NS"):
    a, b = baseline[metric], final[metric]
    check("baseline_identity_and_token_denominators", all(a[key] == b[key] for key in
          ("identity_order_sha256", "denominator", "new_token_den", "true_token_den")))
    pair = next(x for x in paired if x["comparison"] == "N4_W10_TO_SL_ZFLOW_W10"
                and x["metric"] == metric and not x["population"])
    for target in ("new", "true"):
        check("paired_mean_matches_endpoints", close(number(pair, target + "_nll_delta_mean"),
              number(b, target + "_nll_mean") - number(a, target + "_nll_mean")))
    quality[metric] = {"n4_percent": number(a, "percent"), "sl_percent": number(b, "percent"),
                       "delta_pp": number(b, "percent") - number(a, "percent"),
                       "lost": int(pair["lost"]), "gained": int(pair["gained"])}

by_batch = defaultdict(list)
for node in nodes:
    by_batch[node["batch"]].append(node)
trajectory = []
for batch in batches:
    rows = by_batch[batch["batch"]]
    accepted = [x for x in rows if x["phase"] == "accepted"]
    rejected = [x for x in rows if x["phase"] == "rejected"]
    check("batch_accounting", len(rows) == 25 == int(batch["oracle_calls"]) and
          len(accepted) == int(batch["accepted"]) and len(rejected) == int(batch["rejected"]))
    previous = rows[0]
    cost_decreases = edit_increases = 0
    for row in accepted:
        check("accepted_F_decreases", number(row, "F") < number(previous, "F"))
        cost_decreases += number(row, "C") < number(previous, "C")
        edit_increases += number(row, "edit_nll") > number(previous, "edit_nll")
        previous = row
    check("terminal_is_last_accepted", close(number(batch, "terminal_F"), number(previous, "F")))
    for row in rejected:
        check("rejected_F_increases", number(row, "F_trial") > number(row, "F_before"))
    initial_rejects = int(accepted[0]["oracle"]) - 2
    # 비음수 loss 하계에 실제 FP32 acceptance rounding과 작은 추가 여유를 적용한다.
    cheap = [x for x in rejected if number(x, "C") > number(x, "F_before") +
             32 * 2**-23 * max(1, abs(number(x, "F_before"))) + 1e-6]
    r0 = number(rows[1], "stationarity_before")
    residual = number(rows[-1], "stationarity_before") if rows[-1]["phase"] == "rejected" else None
    trajectory.append({
        "batch": batch["batch"], "status": batch["solver_status"],
        "initial_rejected": initial_rejects, "first_accepted_eta": number(accepted[0], "step"),
        "first_trial_C_over_initial_F": number(rows[1], "C") / number(rows[0], "F"),
        "first_edit_nll_below_point1_oracle": min(int(x["oracle"]) for x in accepted if number(x, "edit_nll") < .1),
        "terminal_oracle": int(previous["oracle"]),
        "terminal_C_fraction_of_F": number(previous, "C") / number(previous, "F"),
        "terminal_edit_nll": number(previous, "edit_nll"),
        "terminal_essence_kl": number(previous, "essence_kl"),
        "cost_decreasing_accepted": cost_decreases, "edit_nll_increasing_accepted": edit_increases,
        "terminal_C_reduction_from_accepted_peak": 1-number(previous, "C")/max(number(x,"C") for x in accepted),
        "cheap_cost_reject_count": len(cheap), "cheap_cost_reject_seconds": total(cheap, "seconds"),
        "minimum_cheap_bound_gap": min(number(x, "C")-number(x, "F_before") for x in cheap),
        "stationarity_threshold": 1e-9 + 1e-5*r0,
        "terminal_residual_if_reconstructable": residual,
        "terminal_residual_over_threshold_if_reconstructable": None if residual is None else residual/(1e-9+1e-5*r0),
    })

components = defaultdict(list)
for row in compute:
    components[row["component"]].append(row)
component_seconds = {name: sum(float(x["seconds"]) for x in rows if x["seconds"])
                     for name, rows in components.items() if any(x["seconds"] for x in rows)}
oracle_seconds = sum(component_seconds[x] for x in ("oracle_initial", "oracle_accepted", "oracle_rejected"))
check("oracle_seconds_node_vs_compute", close(oracle_seconds, total(nodes, "seconds")))
phase_totals = {name: total(batches, name) for name in (
    "preparation_seconds", "flow_seconds", "commit_seconds", "evaluation_seconds", "total_seconds")}
result = {
    "review_scope": "공개 집계의 독립 검산. raw/model/checkpoint 독립 재평가는 하지 않음.",
    "snapshot_head": "a178bf4c", "execution_head": EXECUTION_HEAD,
    "raw_main_available_on_review_host": (ROOT/"local/single-layer-zflow/20260916-v1/main/attempt-r1").exists(),
    "checks": dict(CHECKS), "source_sha256": source_hashes,
    "quality": quality, "trajectory": trajectory,
    "totals": {
        "oracle_calls": len(nodes), "accepted": sum(int(x["accepted"]) for x in batches),
        "rejected": sum(int(x["rejected"]) for x in batches),
        "initial_backtracking_rejected": sum(x["initial_rejected"] for x in trajectory),
        "cheap_cost_reject_count": sum(x["cheap_cost_reject_count"] for x in trajectory),
        "cheap_cost_reject_seconds": sum(x["cheap_cost_reject_seconds"] for x in trajectory),
        "oracle_seconds": oracle_seconds, "oracle_rejected_seconds": component_seconds["oracle_rejected"],
        "oracle_rejected_backward_seconds": total([x for x in nodes if x["phase"] == "rejected"], "backward_seconds"),
        "cost_decreasing_accepted": sum(x["cost_decreasing_accepted"] for x in trajectory),
        "edit_nll_increasing_accepted": sum(x["edit_nll_increasing_accepted"] for x in trajectory),
        "phase_seconds_overlapping_components_excluded": phase_totals,
        "flow_fraction_of_batch_total": phase_totals["flow_seconds"]/phase_totals["total_seconds"],
        "checkpoint_bytes": total(batches, "checkpoint_bundle_bytes"),
        "process_peak_allocated_GiB": max(number(x, "peak_gpu_allocated") for x in batches)/2**30,
        "process_peak_reserved_GiB": max(number(x, "peak_gpu_reserved") for x in batches)/2**30,
    },
    "component_seconds_do_not_sum_overlapping_parents": component_seconds,
}
output = Path(__file__).with_name("derived-summary.json")
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(output), "checks_passed": sum(CHECKS.values()),
                  "totals": result["totals"]}, ensure_ascii=False, indent=2))
