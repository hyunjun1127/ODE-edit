#!/usr/bin/env python3
"""CPU-only seal generation and dry-plan validation; never executes a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.run_scripts.ode_alloc.contracts import (
    MODEL_ALIASES,
    SolverBudget,
    assert_common_model_policy,
    canonical_hash,
    canonical_json,
    reject_global_strength_fields,
)
from project.run_scripts.ode_alloc.firewall import assert_dry_launcher_ast
from project.run_scripts.ode_alloc.selection import (
    assert_seal_source_current,
    build_seal_candidate,
    load_and_verify_seal_candidate,
    write_canonical_json,
)


PACKAGE_ROOT = _REPO_ROOT / "project" / "run_scripts" / "ode_alloc"
DEFAULT_LOCK = PACKAGE_ROOT / "numerical_lock_proposal.json"
DEFAULT_SEAL = PACKAGE_ROOT / "split_anchor_seal_candidate.json"
DEFAULT_DATASET = Path("/mnt/raid5/janghj/EasyEdit/data/counterfact/counterfact.json")


def _load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if value.get("status") != "pending-gh-approval-no-scientific-execution":
        raise RuntimeError("numerical proposal is not pending the required approval")
    boundary = value["execution_boundary"]
    if (
        boundary.get("gpu_now") != 0
        or boundary.get("model_load_now") is not False
        or boundary.get("slurm_submit_now") is not False
    ):
        raise RuntimeError("dry lock grants execution authority")
    if tuple(value.get("model_aliases", ())) != MODEL_ALIASES:
        raise RuntimeError("dry lock model aliases differ")
    reject_global_strength_fields(value)
    common_policy = {
        "allocation_gauge": value["allocation_gauge"],
        "rewrite_gate": value["rewrite_gate"],
        "barriers": value["barriers"],
        "solver": value["solver"],
        "transaction": value["transaction"],
    }
    assert_common_model_policy({alias: common_policy for alias in MODEL_ALIASES})
    solver = value["solver"]
    SolverBudget(
        fixed_k=solver["fixed_k"],
        max_trials_per_step=solver["max_trials_per_step"],
        backtracking_factors=tuple(solver["backtracking_factors"]),
        context_count=1,
    )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session04-ode-alloc-dry-plan",
        description="Build or validate outcome-free Session 04 preparation records",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal", allow_abbrev=False)
    seal.add_argument("--dataset", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    dry = subparsers.add_parser("dry", allow_abbrev=False)
    dry.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    dry.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    dry.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    dry.add_argument("--stage", choices=("p0", "p1", "all"), default="all")
    dry.add_argument("--output", type=Path)
    return parser


def _require_local_output(path: Path) -> Path:
    resolved_parent = path.parent.resolve(strict=True)
    allowed = (_REPO_ROOT / "local" / "odealloc").resolve(strict=True)
    try:
        resolved_parent.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("dry output must stay under local/odealloc") from exc
    if path.exists():
        raise FileExistsError("dry output is create-once")
    return path


def _require_seal_output(path: Path) -> Path:
    destination = path.resolve(strict=False)
    if destination == DEFAULT_SEAL.resolve(strict=False):
        return destination
    local_root = (_REPO_ROOT / "local" / "odealloc").resolve(strict=True)
    try:
        destination.relative_to(local_root)
    except ValueError as exc:
        raise ValueError(
            "seal output must be the tracked candidate or stay under local/odealloc"
        ) from exc
    return destination


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    assert_dry_launcher_ast(Path(__file__))
    if args.command == "seal":
        destination = _require_seal_output(args.output)
        if destination.exists():
            raise FileExistsError("seal candidate output is create-once")
        payload = build_seal_candidate(args.dataset, _REPO_ROOT)
        write_canonical_json(destination, payload)
        print(canonical_json({"root_digest": payload["root_digest"], "status": payload["status"]}))
        return 0
    lock = _load_lock(args.lock)
    seal = load_and_verify_seal_candidate(args.seal)
    assert_seal_source_current(seal, args.dataset)
    stages = ("p0", "p1") if args.stage == "all" else (args.stage,)
    plan = {
        "schema_version": "ode-alloc-s04-dry-plan/v1",
        "status": "validated-no-execution-authority",
        "stages": stages,
        "model_aliases": MODEL_ALIASES,
        "arms": ("native", "generic-adaptive", "ode-alloc"),
        "numerical_lock_sha256": canonical_hash(lock),
        "seal_root_digest": seal["root_digest"],
        "selection": {
            "p0_case_ids": [item["case_id"] for item in seal["p0_identity"]],
            "p1_case_ids": [item["case_id"] for item in seal["p1_order"]],
            "anchor_case_ids": [item["case_id"] for item in seal["pretrained_anchor"]],
        },
        "execution": {"gpu": 0, "model_load": False, "scheduler_submit": False},
        "future_resource_forecast": lock["future_resource_forecast"],
    }
    plan["plan_id"] = canonical_hash(plan)
    if args.output is not None:
        destination = _require_local_output(args.output)
        write_canonical_json(destination, plan)
    print(canonical_json(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
