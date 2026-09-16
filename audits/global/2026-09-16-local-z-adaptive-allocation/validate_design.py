"""설계·selector의 CPU 검산. 모델 실행·성능 검증을 수행하지 않는다."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = "2026-09-16-local-z-adaptive-allocation"
CONTRACT = ROOT / "plans/global" / f"{BASE}-contract-v1.json"
CELLS = ROOT / "plans/global" / f"{BASE}-cells-v1.csv"
DESIGN = ROOT / "plans/global" / f"{BASE}-design-v1.md"


def choose(candidates, contract, has_past):
    """JSON의 mean-loss/strict/fixed-W0-KL selector를 검산한다."""
    cfg = contract["selector"]
    references = [c for c in candidates if c["common_n4"]]
    if len(references) != 1:
        raise ValueError("COMMON_N4_IDENTITY")
    ref = references[0]
    numeric = ["E", "D", "norm"] + (["H"] if has_past else [])
    if not all(math.isfinite(ref[k]) for k in numeric):
        raise ValueError("NATIVE_TECHNICAL_FAILURE")
    admitted = []
    for c in candidates:
        if not all(math.isfinite(c[k]) for k in numeric):
            continue
        if c["E"] > max(ref["E"], cfg["current_plateau"]) + cfg["epsilon_E"]:
            continue
        if not ref["current_strict"].issubset(c["current_strict"]):
            continue
        if has_past:
            if c["H"] > ref["H"] + cfg["epsilon_E"]:
                continue
            if not ref["past_strict"].issubset(c["past_strict"]):
                continue
        admitted.append(c)
    if not admitted:
        raise AssertionError("N4_MUST_BE_FEASIBLE")
    best_d = min(c["D"] for c in admitted)
    ties = [c for c in admitted if c["D"] <= best_d + cfg["epsilon_D"]]
    selected = min(ties, key=lambda c: (
        not c["common_n4"], c["changes_l8"], c["norm"], c["id"]
    ))
    return selected["id"], sorted(c["id"] for c in admitted)


def record(cid, **changes):
    result = dict(id=cid, common_n4=cid == "N4", E=.01, D=.3, H=.1,
                  norm=1., changes_l8=False,
                  current_strict={"r1", "r2"}, past_strict={"p1"})
    result.update(changes)
    return result


def main():
    c = json.loads(CONTRACT.read_text())
    rows = list(csv.DictReader(CELLS.open()))
    checks = {}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] = "PASS"

    exp = c["experiment"]
    check("common_pre_edit_W0", exp["entry"] == "common pre-edit W0")
    check("no_warm_state_import", not c["state"]["warm_capsule_import"])
    check("zero_edit_memory_not_zero_P",
          c["state"]["initial_memory_nonzero_count"] == 0
          and "never zeroed" in exp["native_original_knowledge_projectors"])
    check("all_seven_arms_new", len(c["arms"]) == 7
          and all(a["execution"] == "new_from_common_W0" for a in c["arms"]))
    check("seventy_cells", len(rows) == exp["new_batches"] == 70)
    for arm in c["arms"]:
        subset = [r for r in rows if r["arm"] == arm["id"]]
        check(f"{arm['id']}_cold_request_order", [int(r["batch"]) for r in subset] == list(range(1, 11))
              and [(int(r["ordinal_start"]), int(r["ordinal_end_exclusive"])) for r in subset]
              == [(100 * k, 100 * (k + 1)) for k in range(10)])
        check(f"{arm['id']}_counts_and_memory",
              all(int(r["target_calls"]) == arm["target_calls_per_batch"]
                  and int(r["solves"]) == arm["solves_per_batch"]
                  and int(r["candidate_endpoints_max"]) == arm["candidates_per_batch"]
                  and r["history_layers"] == ";".join(map(str, arm["history_layers"])) for r in subset))
    computed = {
        "target_request_calls": sum(int(r["target_calls"]) for r in rows),
        "writer_solves": sum(int(r["solves"]) for r in rows),
        "candidate_endpoints_max": sum(int(r["candidate_endpoints_max"]) for r in rows),
    }
    computed["max_adam_updates"] = computed["target_request_calls"] * exp["native_max_adam_updates"]
    computed["max_target_loss_evaluations"] = computed["target_request_calls"] * exp["native_max_target_loss_evaluations"]
    check("cost_arithmetic", all(c["cost_totals"][k] == v for k, v in computed.items()))
    arm_map = {a["id"]: a for a in c["arms"]}
    check("same_family_gate_menu", arm_map["LD"]["gates"] == arm_map["TD"]["gates"])
    check("common_N4_not_terminal_gate_equivalence", arm_map["TD"]["candidates_per_batch"] == len(arm_map["TD"]["gates"]) + 1)
    check("B1_has_no_past_inputs", c["reference"]["past"]["B1"] == "empty")
    check("official_metrics_observer_only", not c["reference"]["official_P_N_or_future_requests_in_controller"])
    check("no_inner_history_updates", c["state"]["inner_history_appends"] == 0)

    n4 = record("N4")
    plateau = record("PLATEAU", E=.04, D=.2)
    lost_current = record("LOST_CURRENT", D=.01, current_strict={"r1"})
    lost_past = record("LOST_PAST", D=.02, past_strict=set())
    bad_past_nll = record("BAD_PAST_NLL", D=.03, H=.2)
    bad_current = record("BAD_CURRENT", E=.1, D=.04)
    candidates = [n4, plateau, lost_current, lost_past, bad_past_nll, bad_current]
    chosen, feasible = choose(candidates, c, has_past=True)
    check("quality_and_past_filters", chosen == "PLATEAU" and feasible == ["N4", "PLATEAU"])
    check("current_confidence_plateau_explicit", plateau["E"] > n4["E"] and chosen == "PLATEAU")
    check("empty_past_condition_not_accidentally_applied", choose([n4, lost_past], c, False)[0] == "LOST_PAST")
    check("candidate_order_invariance", all(choose(list(order), c, True) == (chosen, feasible)
                                            for order in itertools.permutations(candidates)))
    close = record("CLOSE", D=n4["D"] - .5 * c["selector"]["epsilon_D"])
    check("numerical_tie_prefers_N4", choose([close, n4], c, True)[0] == "N4")
    check("all_proposals_rejected_returns_N4", choose([n4, lost_current, lost_past], c, True)[0] == "N4")
    n4_bad = record("N4", E=float("nan"))
    try:
        choose([n4_bad, plateau], c, True)
    except ValueError as error:
        check("nonfinite_native_is_technical_failure", str(error) == "NATIVE_TECHNICAL_FAILURE")
    else:
        raise AssertionError("nonfinite native silently fell back")

    # Tiny arithmetic counterexamples, not a neural model simulation.
    z8, h4, h8_entry, h8_partial = 11., 2., 5., 6.5
    check("terminal_residual_uses_current_L8", z8 - h8_entry == 6.
          and z8 - h8_partial == 4.5 and z8 - h4 != z8 - h8_entry)
    # K=[1,1], ridge=1 -> A=[1/3,1/3]. Gate request1=0 still changes its key.
    delivered_request1 = (0. * 2. + 1. * 3.) / 3.
    check("request_gates_do_not_isolate_requests", delivered_request1 == 1.)
    # Shared total loss can hide individual deterioration: mean .02 -> .02.
    check("mean_quality_does_not_imply_per_context_quality",
          sum([.01, .03]) == sum([.03, .01]) and .03 > .01)

    result = {
        "status": "PASS_CPU_DESIGN_ONLY", "model_or_gpu_run": False,
        "scientific_performance_validated": False,
        "checks": checks, "check_count": len(checks), "cost_arithmetic": computed,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [DESIGN, CONTRACT, CELLS, Path(__file__)]},
        "unverified": c["runtime_binding_pending"],
    }
    output = Path(__file__).with_name("checks.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
