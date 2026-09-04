"""Deterministic dry plan and point-in-time Server1 resource accounting."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from .preflight import (
    CELL_SPECS,
    GPU_PER_TASK,
    INSTRUCTION_ID,
    MEMORY_MIB_PER_TASK,
    SERVER,
    SLURM_NODE,
    PreflightBoundary,
    _assert_no_symlink_components,
    canonical_hash,
    canonical_json,
    match_project_job,
    parse_local_cap,
    run_preflight,
    wave_rounds,
)


GPU_PATTERNS = (
    re.compile(r"gpu(?::[^,=|]+)?:(\d+)"),
    re.compile(r"gpu[^,=|]*=(\d+)"),
)


def parse_squeue_gpu_rows(text: str, pattern_text: str) -> tuple[int, list[dict[str, object]]]:
    total = 0
    records: list[dict[str, object]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = raw.split("|", 3)
        if len(fields) != 4:
            raise PreflightBoundary("unexpected squeue row shape")
        job_id, job_name, state, tres = fields
        if state != "RUNNING" or not match_project_job(job_name, pattern_text):
            continue
        counts = [int(match.group(1)) for pattern in GPU_PATTERNS for match in pattern.finditer(tres)]
        if len(counts) != 1:
            raise PreflightBoundary(f"cannot resolve running GPU count for job {job_id}")
        total += counts[0]
        records.append(
            {"gpus": counts[0], "job_id": job_id, "job_name": job_name, "state": state}
        )
    return total, records


def query_live_resources(
    cap: dict[str, object],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    completed = runner(
        ["squeue", "-h", "-t", "RUNNING", "-w", str(cap["node"]), "-u", "janghj", "-o", "%i|%j|%T|%b"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise PreflightBoundary(f"squeue resource query failed: {completed.stderr.strip()}")
    active, records = parse_squeue_gpu_rows(completed.stdout, str(cap["job_patterns"]))
    maximum = int(cap["max_project_gpus"])
    free = maximum - active
    if free < 1:
        raise PreflightBoundary("server1 has no free project GPU slot; keep submission pending")
    throttle = min(len(CELL_SPECS), free)
    return {
        "active_project_gpus": active,
        "active_records": records,
        "cap": maximum,
        "free_project_gpus": free,
        "new_concurrent_gpus": throttle,
        "post_release_max": active + throttle,
        "status": "POINT_IN_TIME_RESOURCE_PASS",
        "throttle": throttle,
    }


def _require_create_once_target(path: Path) -> str:
    target = path.absolute()
    if target.exists() or target.is_symlink():
        raise PreflightBoundary(f"create-once target already exists: {target}")
    parent = target.parent
    _assert_no_symlink_components(parent)
    if not parent.is_dir() or parent.is_symlink():
        raise PreflightBoundary(f"create-once target parent is not a regular directory: {parent}")
    return str(target)


def build_dry_plan(
    *,
    repo_root: Path,
    authoritative_root: Path,
    easyedit_source_root: Path,
    easyedit_artifact_root: Path,
    hf_hub_cache: Path,
    source_head: str,
    source_tree: str,
    wave: str,
    output_root: Path,
    log_root: Path,
    local_caps: Path,
    live_resources: bool,
    b1_root: Path | None = None,
    round0_root: Path | None = None,
) -> dict[str, object]:
    rounds = wave_rounds(wave)
    cap = parse_local_cap(local_caps)
    # One representative cell closes all shared bindings; per-cell mapping is
    # immutable below and each production task re-runs the full gate itself.
    binding = run_preflight(
        repo_root=repo_root,
        authoritative_root=authoritative_root,
        easyedit_source_root=easyedit_source_root,
        easyedit_artifact_root=easyedit_artifact_root,
        hf_hub_cache=hf_hub_cache,
        source_head=source_head,
        source_tree=source_tree,
        cell_id=0,
        wave=wave,
        deep_artifact_hash=False,
        b1_root=b1_root,
        round0_root=round0_root,
    )
    resources: dict[str, object]
    if live_resources:
        resources = query_live_resources(cap)
    else:
        resources = {
            "status": "POINT_IN_TIME_CHECK_REQUIRED_AT_SUBMIT",
            "throttle_upper_bound": min(4, int(cap["max_project_gpus"])),
        }
    plan: dict[str, object] = {
        "action_counts": {"gpu": 0, "model": 0, "slurm": 0},
        "array_mapping": {
            str(item.cell_id): {
                "model_alias": item.model_alias,
                "writer_family": item.writer_family,
            }
            for item in CELL_SPECS
        },
        "binding_receipt_identity": binding["receipt_identity_sha256"],
        "cold_w0_and_method_state_per_cohort": True,
        "create_once": {
            "log_root": _require_create_once_target(log_root),
            "output_root": _require_create_once_target(output_root),
            "preseal_after_plan": {
                "log_root_mode": "0700",
                "output_root_mode": "0700",
                "task_roots_must_be_absent": [
                    str(output_root.absolute() / f"task-{item.cell_id}")
                    for item in CELL_SPECS
                ],
            },
        },
        "gpu_per_task": GPU_PER_TASK,
        "instruction_id": INSTRUCTION_ID,
        "memory_mib_per_task": MEMORY_MIB_PER_TASK,
        "model_reload_for_this_wave": 1,
        "round0_wave_release": (
            "B1_COMMON_INTEGRITY_PASS_ALREADY_VERIFIED"
            if wave == "round0"
            else (
                "HOLD_UNTIL_FOUR_B1_TERMINAL_RECEIPTS_PASS"
                if wave == "b1"
                else "NOT_THIS_WAVE"
            )
        ),
        "remaining_wave_release": (
            "ROUND0_COMMON_INTEGRITY_PASS_ALREADY_VERIFIED"
            if wave == "remaining"
            else (
                "HOLD_UNTIL_FOUR_ROUND0_TERMINAL_RECEIPTS_PASS"
                if wave == "round0"
                else "BLOCKED_BEHIND_B1_THEN_ROUND0_COMMON_GATES"
            )
        ),
        "resource_cap": cap,
        "resource_snapshot": resources,
        "round_indices": list(rounds),
        "round_request_count": 1 if wave == "b1" else len(rounds) * 100,
        "sbatch": {
            "array": "0-3%<point-in-time-throttle>",
            "cli_overrides": {
                "error": str(log_root.absolute() / "%A_%a.err"),
                "output": str(log_root.absolute() / "%A_%a.out"),
            },
            "export": "NONE",
            "memory": f"{MEMORY_MIB_PER_TASK}M",
            "node": SLURM_NODE,
            "resource_check": (
                f"AGENT_GPU_CAPS_FILE={local_caps} scripts/check-slurm-resource-cap.sh "
                f"{SERVER} <throttle> {MEMORY_MIB_PER_TASK}M"
            ),
            "script": {
                "b1": "project/run_scripts/session06_orbode_server1_b1.sbatch",
                "round0": "project/run_scripts/session06_orbode_server1_round0.sbatch",
                "remaining": "project/run_scripts/session06_orbode_server1_remaining.sbatch",
            }[wave],
        },
        "schema": "ode-edit.orbode.server1-gated-dry-plan.v1",
        "scientific_outcome_changes_plan": False,
        "source_head": source_head,
        "source_tree": source_tree,
        "state_carry_between_waves": False,
        "status": "DRY_PLAN_PASS",
        "wave": wave,
    }
    plan["plan_identity_sha256"] = canonical_hash(plan)
    return plan


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(allow_abbrev=False)
    value.add_argument("--repo-root", type=Path, required=True)
    value.add_argument("--authoritative-root", type=Path, required=True)
    value.add_argument("--easyedit-source-root", type=Path, required=True)
    value.add_argument("--easyedit-artifact-root", type=Path, required=True)
    value.add_argument("--hf-hub-cache", type=Path, required=True)
    value.add_argument("--source-head", required=True)
    value.add_argument("--source-tree", required=True)
    value.add_argument("--wave", choices=("b1", "round0", "remaining"), required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--log-root", type=Path, required=True)
    value.add_argument("--local-caps", type=Path, required=True)
    value.add_argument("--b1-root", type=Path)
    value.add_argument("--round0-root", type=Path)
    value.add_argument("--live-resource-check", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    plan = build_dry_plan(
        repo_root=args.repo_root,
        authoritative_root=args.authoritative_root,
        easyedit_source_root=args.easyedit_source_root,
        easyedit_artifact_root=args.easyedit_artifact_root,
        hf_hub_cache=args.hf_hub_cache,
        source_head=args.source_head,
        source_tree=args.source_tree,
        wave=args.wave,
        output_root=args.output_root,
        log_root=args.log_root,
        local_caps=args.local_caps,
        live_resources=args.live_resource_check,
        b1_root=args.b1_root,
        round0_root=args.round0_root,
    )
    print(canonical_json(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
