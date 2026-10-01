"""CPU design checks only: no model, CUDA, experiment submission, or source mutation.

Reproduce with EasyEdit/.venv/bin/python inspect_design.py. Writes evidence.json
beside this file. Synthetic tolerances are NOT production oracle thresholds.
"""
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np

HERE = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def norm(text):
    return unicodedata.normalize("NFC", " ".join(text.split()))


def claim(record):
    req = record["requested_rewrite"]
    return norm(req["subject"]), req["relation_id"]


def prompt(record):
    req = record["requested_rewrite"]
    return norm(req["prompt"].format(req["subject"]))


def rank(records, seed, batch, group):
    return sorted(records, key=lambda r: (
        digest(compact([seed, batch, group, r["case_id"]])), r["case_id"]))


def select_general(pool, current, cfg, batch):
    relations = {r["requested_rewrite"]["relation_id"] for r in current}
    matched = [r for r in pool if r["requested_rewrite"]["relation_id"] in relations]
    chosen = rank(matched, cfg["seed"], batch, "general_relation")[:cfg["general_relation_matched_quota"]]
    used = {r["case_id"] for r in chosen}
    rest = [r for r in pool if r["case_id"] not in used]
    chosen += rank(rest, cfg["seed"], batch, "general_global")[:cfg["general_count"] - len(chosen)]
    return chosen


def select_replay(past, current, cfg, batch):
    latest = {}
    for pos, record in enumerate(past):
        latest[claim(record)] = (pos, record)
    exclude = {claim(r) for r in current}
    ordered = [r for _, r in sorted(latest.values(), key=lambda pair: pair[0]) if claim(r) not in exclude]
    chosen = []
    for q in range(4):
        part = ordered[len(ordered) * q // 4:len(ordered) * (q + 1) // 4]
        chosen += rank(part, cfg["seed"], batch, f"replay_age_{q}")[:4]
    used = {r["case_id"] for r in chosen}
    rest = [r for r in ordered if r["case_id"] not in used]
    chosen += rank(rest, cfg["seed"], batch, "replay_fill")[:cfg["replay_max_count"] - len(chosen)]
    return chosen


def inspect_references(cfg):
    sources = {}
    datasets = {}
    for kind, prefix in [("full", "general_pool"), ("fixed", "stream")]:
        path = Path(cfg[f"{prefix}_path"])
        raw = path.read_bytes()
        sha = digest(raw)
        assert sha == cfg[f"{prefix}_sha256"], (kind, "source changed")
        datasets[kind] = json.loads(raw)
        sources[kind] = {"path": str(path), "bytes": len(raw), "sha256": sha, "records": len(datasets[kind])}
    full, fixed = datasets["full"], datasets["fixed"]
    ids = {r["case_id"] for r in fixed}
    claims = {claim(r) for r in fixed}
    subjects = {claim(r)[0] for r in fixed}
    eval_prompts = {p for r in fixed for p in [prompt(r), *map(norm, r["paraphrase_prompts"]), *map(norm, r["neighborhood_prompts"])]}
    pool = [r for r in full if r["case_id"] not in ids]
    counts = {"outside_fixed_ids": len(pool)}
    pool = [r for r in pool if claim(r) not in claims]
    counts["after_claim_exclusion"] = len(pool)
    pool = [r for r in pool if prompt(r) not in eval_prompts]
    counts["after_eval_prompt_exclusion"] = len(pool)
    pool = [r for r in pool if claim(r)[0] not in subjects]
    counts["after_subject_exclusion"] = len(pool)
    counts["relations"] = len({r["requested_rewrite"]["relation_id"] for r in pool})
    assert list(counts.values()) == [11919, 11544, 11134, 10715, 34], counts
    samples = []
    for size, batches in [(4, 2), (100, 10)]:
        for batch in range(batches):
            past, current = fixed[:size * batch], fixed[size * batch:size * (batch + 1)]
            general = select_general(pool, current, cfg, batch)
            replay = select_replay(past, current, cfg, batch)
            assert len(general) == 16 and len({r["case_id"] for r in general}) == 16
            assert all(claim(r) not in claims and claim(r)[0] not in subjects and prompt(r) not in eval_prompts for r in general)
            latest = {claim(r): r["case_id"] for r in past}
            assert all(latest[claim(r)] == r["case_id"] for r in replay)
            assert not ({claim(r) for r in replay} & {claim(r) for r in current})
            assert len(replay) <= 16 and len({claim(r) for r in replay}) == len(replay)
            assert general == select_general(pool, current, cfg, batch)
            assert replay == select_replay(past, current, cfg, batch)
            samples.append({"batch_size": size, "batch_index": batch, "general_count": len(general), "replay_count": len(replay), "general_ids_sha256": digest(compact([r["case_id"] for r in general])), "replay_ids_sha256": digest(compact([r["case_id"] for r in replay]))})
    # Synthetic repeated-claim case checks latest-target and current-batch exclusion.
    def rec(case, subject, target):
        return {"case_id": case, "requested_rewrite": {"subject": subject, "relation_id": "R", "target_new": {"str": target}}}
    past = [rec(1, "a", "old"), rec(2, "b", "keep"), rec(3, "a", "latest")]
    assert {r["case_id"] for r in select_replay(past, [], cfg, 1)} == {2, 3}
    assert {r["case_id"] for r in select_replay(past, [rec(4, "a", "next")], cfg, 1)} == {2}
    return {"sources": sources, "pool_filter_counts": counts, "draft_selection_checks": samples, "replay_supersession_and_current_conflict_check": "PASS", "semantic_disjointness": "NOT_CERTIFIED", "model_teacher_or_tokenizer_checks": "NOT_RUN"}


def inspect_math():
    rng = np.random.default_rng(20261002)
    n, batch, hidden = 9, 4, 7
    Z = rng.normal(size=(n, n))
    A = Z @ Z.T + np.eye(n)
    K = rng.normal(size=(n, batch))
    P = np.linalg.solve(A + K @ K.T, K)
    M = P.T @ A @ P
    S = (K.T @ P + P.T @ K) / 2
    small = S - S @ S
    matrix_error = float(np.linalg.norm(M - small) / np.linalg.norm(M))
    assert matrix_error < 1e-12
    R = rng.normal(size=(hidden, batch))
    scale = 2.7
    def cost(X):
        return float(0.5 * np.sum((X @ M) * X) / scale)
    direct = float(0.5 * np.trace((R @ P.T) @ A @ (R @ P.T).T) / scale)
    assert np.isclose(direct, cost(R), rtol=1e-12, atol=1e-12)
    D = rng.normal(size=R.shape)
    eps = 1e-6
    fd = (cost(R + eps * D) - cost(R - eps * D)) / (2 * eps)
    exact = float(np.sum((R @ M / scale) * D))
    fd_error = abs(fd - exact) / max(1.0, abs(exact))
    assert fd_error < 1e-8
    # Convex toy: .5*(a@r-1)^2 + lambda*|r|_1 + eta/2*m@r^2.
    # Shows an all-five-eligible optimum can use exactly one layer in both arms.
    a = np.array([1., .01, .01, .01, .01])
    lam, m, h = .1, .1, np.sqrt(5.)
    toy = []
    for eta in [0., 1.]:
        r = np.array([(1. - lam) / (1. + eta * m), 0., 0., 0., 0.])
        g = a * (a @ r - 1.) + eta * m * r
        assert abs(g[0] + lam) < 1e-12 and np.max(np.abs(g[1:])) < lam
        assert np.max(np.abs(r)) < .75 * h
        toy.append({"eta": eta, "nonzero_layers": int(np.count_nonzero(r)), "KKT": "PASS", "clamp_feasible": True, "toy_edit_strength": float(a @ r)})
    return {"dtype": "synthetic numpy FP64", "dimensions": {"key": n, "batch": batch, "hidden": hidden}, "M_small_vs_direct_relative_error": matrix_error, "energy_direct_vs_factor_absolute_error": abs(direct - cost(R)), "gradient_directional_relative_error": fd_error, "toy_single_layer_KKT": toy, "scope": "algebra and existence of permitted concentrated solutions; no GPU parity, speed, or model quality certification"}


def inspect_contract(c):
    assert c["arms"] == {"A": {"burden_eta": 0.0}, "B": {"burden_eta": 1.0}}
    assert c["common"]["layers"] == [4, 5, 6, 7, 8]
    assert not c["common"]["initial_layer_subset"]
    assert c["common"]["minimum_nonzero_layers"] == 0
    assert c["common"]["maximum_layer_share"] is None
    assert c["common"]["single_layer_solution_allowed"]
    assert not c["policy"]["automatic_layer_exclusion"]
    assert not c["policy"]["automatic_eta_adjustment"]
    assert not c["runtime"]["save_tensor_or_optimizer_checkpoint"]
    for stage in c["stages"][1:]:
        assert stage["max_science_oracles"] == stage["batches_per_arm"] * stage["call_cap_per_batch"] * stage["arms"]
    diagnostics = c["policy"]["scientific_record_only"]
    assert "one_layer_100_percent" in diagnostics and "NS_decline" in diagnostics
    assert c["runtime"]["reserved_final_oracles"] == 1
    assert c["runtime"]["shared_budget_accountant_required"]
    cross = c["cross_diagnostic"]
    small = c["stages"][1]
    assert cross["small_pilot_max_prompt_states"] == len(cross["states"]) * cross["prompts_per_endpoint_max"] * small["batches_per_arm"] * small["arms"]
    assert not cross["used_as_layer_gate"] and not cross["b100_enabled_by_default"]
    return {"only_arm_difference": "burden_eta: 0 versus 1", "all_five_eligible": True, "single_layer_permitted": True, "diagnostic_gates_do_not_mask_layers": True, "budget_arithmetic": "PASS", "production_policy_unchanged": True}


def main():
    path = HERE / "contract-draft.json"
    raw = path.read_bytes()
    c = json.loads(raw)
    evidence = {"status": "CPU_DESIGN_CHECKED_NOT_MODEL_VALIDATED", "contract_sha256": digest(raw), "design_sha256": digest((HERE / "design-ko.md").read_bytes()), "inspector_sha256": digest(Path(__file__).read_bytes()), "checks": inspect_contract(c), "references": inspect_references(c["references"]), "math": inspect_math(), "gpu_calls": 0, "new_jobs_submitted": 0, "live_jobs_modified": False}
    (HERE / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": evidence["status"], "pool": evidence["references"]["pool_filter_counts"], "math": evidence["math"], "gpu_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
