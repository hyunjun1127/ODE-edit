"""Seal/check a CPU-only experiment specification; never launches a model/job.

--write creates missing generated files, refusing to overwrite different bytes.
Default/--check only verifies them and their source dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import unicodedata

HERE = Path(__file__).resolve().parent
METHOD = HERE.parent
ROOT = METHOD.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def require(condition, label):
    if not condition:
        raise ValueError(label)


def member(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": sha(data)}


def build(dataset_override=None):
    method = json.loads((METHOD / "contract.json").read_text())
    inputs = json.loads((METHOD / "inputs/exact-native-inputs.json").read_text())
    old_manifest = json.loads((METHOD / "artifact-manifest.json").read_text())
    for item in old_manifest["files"]:
        require(member(ROOT / item["path"]) == item, "METHOD_ARTIFACT_CHANGED:" + item["path"])
    require(method["scope"]["layers"] == [4, 5, 6, 7, 8], "ALL_LAYERS")
    require(method["optimizer"]["max_candidate_loss_evaluations"] == 25, "BUDGET25")
    require(method["optimizer"]["max_Adam_updates"] == 24, "UPDATES24")
    require(method["arms"] == [{"name": "A", "eta": 0, "meaning": "native joint local-delta control"},
                               {"name": "B", "eta": 1, "meaning": "native core plus normalized history-conditioned ridge value"}], "ARMS")
    dataset = Path(dataset_override) if dataset_override else Path(inputs["dataset"]["local_path"])
    require(sha(dataset.read_bytes()) == inputs["dataset"]["sha256"], "DATASET_SHA")
    rows = json.loads(dataset.read_text())
    cases = rows[:2000]
    require(len(cases) == 2000, "2K_COUNT")
    require(len({r["case_id"] for r in cases}) == 2000, "UNIQUE_CASE_IDS")
    require(digest([r["case_id"] for r in cases]) == inputs["ordered_case_ids_sha256"], "ORDER")
    require(all(len(r["paraphrase_prompts"]) == 2 and len(r["neighborhood_prompts"]) == 10 for r in cases), "PANELS")
    require(len(rows[2000:2004]) == 4, "PILOT_SLICE")

    # Exact native observer rule: a prior occurrence remains superseded after
    # any later different target, even if an even later target changes back.
    versions, active = {}, {r["case_id"]: True for r in cases}
    for row in cases:
        rw = row["requested_rewrite"]
        key = (unicodedata.normalize("NFC", " ".join(rw["subject"].split())), rw["relation_id"])
        target = rw["target_new"].get("id", rw["target_new"]["str"])
        for old_case, old_target in versions.get(key, []):
            if old_target != target:
                active[old_case] = False
        versions.setdefault(key, []).append((row["case_id"], target))
    conflicts = sum(len({target for _, target in group}) > 1 for group in versions.values())
    require((len(versions), conflicts, sum(not x for x in active.values())) == (1983, 13, 16), "CLAIM_COUNTS")

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["stream_index0", "batch1", "slot0", "case_id", "claim_sha256", "target_new_sha256", "active_at_W20"])
    for i, row in enumerate(cases):
        rw = row["requested_rewrite"]
        claim = [unicodedata.normalize("NFC", " ".join(rw["subject"].split())), rw["relation_id"]]
        writer.writerow([i, i // 100 + 1, i % 100, row["case_id"], digest(claim), digest(rw["target_new"]), int(active[row["case_id"]])])
    case_bytes = output.getvalue().encode()

    endpoints = [{"batch": 0, "scope": "W0_full2k", "ordinal_slice": [0, 2000],
                  "requests": 2000, "R": 2000, "P": 4000, "N": 20000,
                  "shared_when_runtime_and_state_identical": True}]
    for batch in range(1, 21):
        low, high = (batch - 1) * 100, batch * 100
        start = 0 if batch in [5, 10, 20] else low
        count = high - start
        endpoints.append({"batch": batch, "scope": "all_seen" if start == 0 and batch in [5, 10, 20] else "current",
                          "ordinal_slice": [start, high], "current_slice": [low, high], "requests": count,
                          "R": count, "P": 2 * count, "N": 10 * count,
                          "current_rows_reused_from_same_endpoint": batch in [5, 10, 20]})
    counts = sum(e["requests"] for e in endpoints[1:])
    require(counts == 5200 and 13 * counts == 67600, "EVAL_COST")
    for batch in range(1, 21):
        e = endpoints[batch]
        require(e["ordinal_slice"][0] <= (batch - 1) * 100 and e["ordinal_slice"][1] == batch * 100, "BIRTH_COVERAGE")
    evaluation = {"schema": "JLZ_V4_2K_EVALUATION_SCHEDULE_V1", "endpoints": endpoints,
                  "case_schedule_sha256": sha(case_bytes), "final_denominators": {"R": 2000, "P": 4000, "N": 20000},
                  "post_edit_request_evaluations_per_chain": counts,
                  "post_edit_prompt_pairs_per_chain": 67600, "W0_prompt_pairs_once": 26000,
                  "A_B_plus_shared_W0_prompt_pairs": 161200,
                  "two_new_chains_plus_shared_W0_prompt_pairs": 161200,
                  "old_two_arm_schedule_post_edit_pairs_per_chain": 115000,
                  "preference": {"R_P": "new_nll < true_nll", "N": "true_nll < new_nll", "ties": "failure"},
                  "strict": "all desired target tokens argmax under teacher forcing; not free generation",
                  "evaluation_feedback_to_solver": False, "nonobserved_all_seen": "NOT_MEASURED"}

    experiment = {
        "schema": "JLZ_NATIVE_JOINT_V4_EXPERIMENT_2K_V2", "date_kst": "2026-10-02",
        "scope_revision": 2, "latest_user_scope": "v4 두 arm만 실험 돌리는 걸로 해",
        "status": "DESIGN_READY_IMPLEMENTATION_AND_GPU_PENDING",
        "task_id": "jlz-native-joint-v4-bs100x20-20261002-v1",
        "user_request": "실험 진행하도록 실험 설계 진행하라. 2000edit을 기준으로 일단 해보자.",
        "method": member(METHOD / "contract.json"),
        "generic_method": member(METHOD / "portability-contract.json"),
        "optimization_revision": "compute-r1",
        "optimization_evidence": [member(METHOD / "compute/native-work-counts.json"), member(METHOD / "math/compute-reuse-check.json")],
        "implementation_generalization": "B_t/eligible layers/dimensions/native row groups/readouts/benchmark metrics from adapters; this 100x20 schedule is an instance",
        "optimized_route_policy": "qualify compute-r1 capabilities on fixed candidates; record same-objective reference fallback; keep common A/B global-step schedule",
        "inputs": member(METHOD / "inputs/exact-native-inputs.json"),
        "source_inputs_unchanged": True, "batch_size": 100, "sequential_commits": 20,
        "requests_per_chain": 2000, "ordinal_slice": [0, 2000], "seed": 20261002,
        "new_chains": [
            {"id": "JLZ_A", "eta": 0, "kind": "native_joint_delta", "lane": 1, "position": 1},
            {"id": "JLZ_B", "eta": 1, "kind": "native_joint_delta", "lane": 2, "position": 1}],
        "baseline_reuse": {"mode": "optional_read_only_historical_quality_reference",
                           "new_main_runs": False, "new_pilot_runs": False, "fallback_runs": False,
                           "required_for_A_B_completion": False,
                           "runtime_mismatched_timing_comparison": False,
                           "BS10_or_1k_HJ_substitutes_BS100_W20": False},
        "runtime": {"model_revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2", "model_dtype": "float32",
                    "ours_geometry_dtype": "float64", "native_history_storage_append_dtype": "float32",
                    "model_eval": True, "autocast": False, "matmul_TF32": False, "cudnn_TF32": False,
                    "attention": "eager", "initial_fit_microbatch": 4, "OOM_microbatch_sequence": [2, 1],
                    "evaluator_microbatch": 2,
                    "package_source_and_asset_SHA": "MUST_BIND_IN_IMPLEMENTED_LAUNCH_LOCK"},
        "state": {"independent_W0_H0": True, "H0": "zero", "history_refresh": False,
                  "H_append": "post_all_layer_native_pooled_keys_once_per_committed_occurrence_per_layer",
                  "pilot_to_main_state_reuse": False, "within_chain_same_process_RAM": True,
                  "save_checkpoints": False, "exact_resume": "NOT_AVAILABLE",
                  "persistent_reconstructible_full_delta_or_factor": False},
        "pilot": {"small": {"ordinal_slice": [2000, 2004],
                             "case_ids": [r["case_id"] for r in rows[2000:2004]],
                             "batch_size": 2, "committed_batches_per_arm": 1,
                             "second_batch": "entry_only_no_fit_no_commit",
                             "arms": ["A", "B"], "candidate_evaluations_per_commit": 25,
                             "max_Adam_updates_per_commit": 24, "total_science_candidates": 50},
                  "timing": {"ordinal_slice": [0, 100], "batch_size": 100,
                             "arms": ["A", "B"], "warmup_per_arm": 1, "measured_candidates_per_arm": 3,
                             "maximum_logical_candidate_evaluations": 8, "write_or_main_carryover": False},
                  "reference_probe_cost": "separate physical forward/backward ledger; not main solver evaluations",
                  "quality_concentration_realization_or_convergence_gate": False,
                  "eta_or_budget_tuning_from_P_N": False,
                  "main_after_technical_readiness": True},
        "numerical_qualification": {
            "status": "PROPOSED_STARTING_TOLERANCES_NOT_GPU_VALIDATED",
            "identity": "case/target/token/mask/lookup/weights exact",
            "per_request_NLL_KL": {"atol": 5e-5, "rtol": 2e-4},
            "delta_gradient": {"relative_L2_max": 2e-3, "near_zero_max_absolute": 1e-6},
            "FP64_solve_scaled_residual_max": 1e-10,
            "FP64_V_gradient": {"atol": 1e-10, "rtol": 1e-8},
            "Q_eigenvalue_roundoff_interval": [-1e-8, 1.00000001],
            "clamp": {"radius_relative_slack": 5e-6, "absolute_slack": 1e-7},
            "commit_and_rollback_tensor_identity": "bitwise",
            "exceedance": "one same-shape reference check; use correct reference path when possible; confirmed semantic/gradient/state corruption is technical failure",
            "virtual_equals_actual_or_D_equals_UK": "NOT_REQUIRED"},
        "main_counts_ours_only": {"chains": 2, "batches": 40, "logical_candidates": 1000,
                                  "max_Adam_updates": 960, "native_input_row_evaluations": 700000,
                                  "timing_speedup_or_ETA": None},
        "claim_statistics": {"unique_claims": len(versions), "conflicting_claims": conflicts,
                             "active_occurrences_W20": sum(active.values()), "superseded_occurrences_W20": 16,
                             "primary_occurrences": 2000, "filter_training_or_history": False},
        "resources": {"owner_server": "server4", "GH_thread": "01a04939-8873-7673-8dca-4c7fc5e31af0",
                      "SH4_thread": "01a04939-b5c7-7a03-ba2d-ef3343d62cfd", "project_gpu_cap_at_design": 2,
                      "per_job_gpu": 1, "host_memory_mib_max": 60416, "cpus": 8,
                      "recheck_current_policy_and_occupancy": True, "mutate_existing_jobs": False},
        "implementation": {"namespace": "project/run_scripts/jlz_native_joint/", "runner_exists": False,
                           "task_local_baseline_2k_wrapper_required": False, "launch_argv": None,
                           "GPU_run": False, "Slurm_submitted": False, "dispatch_receipt": None,
                           "unresolved_launch_fields": ["production_source_commit_and_closure", "actual_model_stat_tokenizer_SHA",
                               "runtime_versions_backend_token_ID_parity",
                               "pilot_technical_and_timing_receipts", "actual_resource_admission_and_argv"]},
        "outputs": ["launch_manifest", "pilot_receipts", "20_commit_history_ledgers_per_chain",
                    "per_case_RPN_raw", "W5_W10_W20_summary", "birth_to_W20_cohort_retention",
                    "first1k_W10_to_W20", "active_superseded_splits", "NS_leakage",
                    "requested_realized_allocation", "component_time_tokens_memory", "optional_historical_baseline_reuse_verdicts"],
        "completion": "both JLZ_A and JLZ_B have 20 commits and complete scheduled raw; historical baseline availability is optional and never blocks completion; one-arm-only is partial"}

    require([r["id"] for r in experiment["new_chains"]] == ["JLZ_A", "JLZ_B"], "USER_TWO_ARMS_ONLY")
    require(not any(experiment["baseline_reuse"][k] for k in ("new_main_runs", "new_pilot_runs", "fallback_runs")), "NO_BASELINE_GPU")
    generated = {"case-schedule.csv": case_bytes, "evaluation-schedule.json": encoded(evaluation),
                 "experiment.json": encoded(experiment)}
    sources = [METHOD / "artifact-manifest.json", METHOD / "contract.json", METHOD / "portability-contract.json", METHOD / "inputs/exact-native-inputs.json",
               ROOT / "control/gpu-concurrency-policy.tsv", ROOT / "servers/slurm-memory-policy.tsv",
               ROOT / "plans/global/2026-09-19-default-no-experiment-checkpoints.md",
               ROOT / "project/run_scripts/jlz_two_arm/observation.py",
               ROOT / "project/run_scripts/jlz_two_arm/collect.py",
               ROOT / "project/run_scripts/jlz_pilot/prompts.py"]
    validation = {"status": "PASS_CPU_PLAN_AND_SOURCE_BINDINGS_ONLY", "method_manifest_members_verified": len(old_manifest["files"]),
                  "main_requests": 2000, "batches": 20, "batch_size": 100, "new_chains": ["JLZ_A", "JLZ_B"],
                  "baseline_new_main_pilot_fallback": False, "complete_birth_and_endpoint_coverage": True,
                  "case_schedule_sha256": sha(case_bytes), "claim_statistics": experiment["claim_statistics"],
                  "post_edit_prompt_pairs_per_chain": 67600, "A_B_candidates": 1000,
                  "actual_GPU_checks": 0, "production_runner_implemented": False, "source_dependencies": [member(p) for p in sources]}
    generated["validation.json"] = encoded(validation)
    fixed_files = [HERE / "experiment-ko.md", HERE / "gh-sh4-handoff-ko.md", Path(__file__).resolve()]
    artifact_members = [member(p) for p in fixed_files]
    artifact_members += [{"path": str((HERE / name).relative_to(ROOT)), "bytes": len(data), "sha256": sha(data)}
                         for name, data in generated.items()]
    generated["artifact-manifest.json"] = encoded({"schema": "JLZ_V4_2K_DESIGN_ARTIFACTS_V1",
        "status": "DESIGN_ONLY_NOT_SUBMITTED", "GPU_run": False, "files": artifact_members,
        "external_sources": [member(p) for p in sources]})
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--dataset", type=Path, help="same-SHA dataset path on the recipient host")
    args = parser.parse_args()
    outputs = build(args.dataset)
    for name, data in outputs.items():
        path = HERE / name
        if args.write and not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
        require(path.is_file() and path.read_bytes() == data, "GENERATED_BINDING_MISMATCH:" + name)
    print(json.dumps({"status": "PASS_EXPERIMENT_2K_DESIGN", "files": len(outputs), "requests": 2000,
                      "batches": 20, "new_chains": 2, "ours_candidates": 1000,
                      "post_edit_prompt_pairs_per_chain": 67600,
                      "production_runner": False, "GPU_run": False, "submitted": False}))


if __name__ == "__main__":
    main()
