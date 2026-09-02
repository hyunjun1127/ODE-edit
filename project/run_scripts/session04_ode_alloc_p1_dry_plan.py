#!/usr/bin/env python3
"""Outcome-free deterministic dry plan for the approved P1 matched pair."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_alloc.contracts import ARM_ORDER, MODEL_ALIASES, canonical_hash, canonical_json
from project.run_scripts.ode_alloc.p0_artifacts import sha256_file
from project.run_scripts.ode_alloc.p1_contracts import (
    P1_CASE_ORDER,
    P1_INSTRUCTION_ID,
    P1_REQUEST_HASHES,
    P1_RUN_TOKEN,
    P1Policy,
    assert_p1_seal,
    expected_p1_result_name,
)
from project.run_scripts.ode_alloc.p1_firewall import (
    assert_no_alias_specific_scientific_branch,
    assert_p1_evaluator_firewall,
    assert_p1_inner_firewall,
    assert_projector_is_only_adaptive_arm_branch,
)
from project.run_scripts.ode_alloc.selection import load_and_verify_seal_candidate


PACKAGE_ROOT = REPO_ROOT / "project" / "run_scripts" / "ode_alloc"
NUMERICAL_LOCK = PACKAGE_ROOT / "numerical_lock_proposal.json"
SEAL = PACKAGE_ROOT / "split_anchor_seal_candidate.json"


def main() -> int:
    lock = json.loads(NUMERICAL_LOCK.read_text(encoding="utf-8"))
    policy = P1Policy.from_lock(lock)
    seal = load_and_verify_seal_candidate(SEAL)
    assert_p1_seal(seal)
    runtime = PACKAGE_ROOT / "p1_runtime.py"
    evaluator = PACKAGE_ROOT / "p1_evaluator.py"
    assert_p1_inner_firewall((runtime, PACKAGE_ROOT / "p1_contracts.py"))
    assert_p1_evaluator_firewall(evaluator)
    assert_no_alias_specific_scientific_branch(runtime)
    assert_projector_is_only_adaptive_arm_branch(runtime)
    numerical_sha = sha256_file(NUMERICAL_LOCK)
    plan = {
        "schema_version": "ode-alloc-s04-p1-dry-plan/v1",
        "status": "VALIDATED_NO_MODEL_NO_SUBMIT",
        "instruction_id": P1_INSTRUCTION_ID,
        "run_token": P1_RUN_TOKEN,
        "model_aliases": MODEL_ALIASES,
        "arms": tuple(arm.value for arm in ARM_ORDER),
        "case_order": P1_CASE_ORDER,
        "request_hashes": P1_REQUEST_HASHES,
        "anchor_count": len(seal["pretrained_anchor"]),
        "seal_root_digest": seal["root_digest"],
        "numerical_lock_sha256": numerical_sha,
        "common_policy_id": policy.policy_id,
        "budget_id": policy.solver_budget.identity(),
        "result_names": {
            alias: expected_p1_result_name(alias, numerical_sha, P1_RUN_TOKEN)
            for alias in MODEL_ALIASES
        },
        "jobs": {
            "llama3-8b-inst": "odealloc_s04_p1_llama",
            "qwen2.5-7b-inst": "odealloc_s04_p1_qwen",
        },
        "resources_per_job": {
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 60416,
            "walltime": "12:00:00",
            "node": "server2",
        },
        "execution": {
            "model_load": False,
            "gpu": 0,
            "scheduler_submit": False,
            "retry_or_resubmit": False,
        },
    }
    plan["plan_id"] = canonical_hash(plan)
    print(canonical_json(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
