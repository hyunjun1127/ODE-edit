#!/usr/bin/env python3
"""Create-once cap-four submitter for the P1R11 actuator panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts import session05_ode_bf_canonical_alpha_posfield_dry_plan as dry
from project.run_scripts.ode_bf.artifacts import ODEBFArtifactGuard, load_rooted_json, sha256_file
from project.run_scripts.ode_bf.canonical_alpha_posfield import (
    AlphaActuator,
    CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID,
)
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p1_canonical_alpha_posfield_panel import (
    CANONICAL_ALPHA_PARENT_HEAD,
    CANONICAL_ALPHA_RESULT_TOKEN,
    CANONICAL_ALPHA_SOURCE_MANIFEST_FILE,
    expected_canonical_alpha_result_name,
)
from project.run_scripts.ode_bf.resource import gpu_count_from_tres


SESSION_ID = "019fe489-c968-75f3-9965-7cfbc26c0a99"
EXECUTION_BRANCH = "codex/odeeditsh1-s05-canonical-alpha-posfield-p1r11-v1"
SERVER1_PROJECT_GPU_CAP = 4
SERVER1_SLURM_NODE = "devbox"
CANONICAL_GPU_CAP_COMMIT = "fd49e3a"
SUBMISSION_NAMESPACE = "s05-canonical-alpha-posfield-p1r11-v1"
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_canonical_alpha_posfield.sbatch"
SOURCE_MANIFEST = (
    REPO_ROOT / "project/run_scripts/ode_bf/locks" / CANONICAL_ALPHA_SOURCE_MANIFEST_FILE
)
R10_ROOT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-fixed-e8-r8-bound-snap-r1/ODE-edit/local/odebf/results"
)
R10_TERMINAL_MANIFEST = {
    "llama3-8b-inst": (
        "168ae6917ba1e35ca510e766c31a68ed5133b168d6d380eedc9355a6fa8ba786",
        "49ff1232255363a872a00105c66c15804ac14ef7a885d60e9ab4a867151ae4f1",
    ),
    "qwen2.5-7b-inst": (
        "4aa110758c70038246b60214bf4153b35a9842b81eff5122cfb0a83760802489",
        "a4ad0b8b78f5194f0fd396c133caa87bc197bbc6b9b253bd4d6bb5e070e1bdcf",
    ),
}


def _run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("P1R11 receipt is create-once")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _source_manifest_gate(source_head: str) -> str:
    value, raw_sha256 = load_rooted_json(
        SOURCE_MANIFEST,
        expected_schema="ode-edit-s05-canonical-alpha-posfield-p1r11-source-manifest/v1",
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID
        or value.get("expected_parent") != CANONICAL_ALPHA_PARENT_HEAD
        or value.get("execution_branch") != EXECUTION_BRANCH
        or value.get("execution_head_policy") != "runtime-git-head"
        or not isinstance(entries, list)
        or not entries
    ):
        raise ODEBFContractError("P1R11 source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ODEBFContractError("P1R11 source manifest entry differs")
        relative = str(entry["path"])
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("size")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ODEBFContractError("P1R11 source manifest content differs")
        observed.append(relative)
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise ODEBFContractError("P1R11 source manifest ordering differs")
    manifest_relative = SOURCE_MANIFEST.relative_to(REPO_ROOT).as_posix()
    changed_paths = sorted(
        line
        for line in _run(
            [
                "git",
                "diff",
                "--name-only",
                CANONICAL_ALPHA_PARENT_HEAD,
                source_head,
                "--",
            ]
        ).stdout.splitlines()
        if line
    )
    if (
        manifest_relative not in changed_paths
        or observed != [path for path in changed_paths if path != manifest_relative]
    ):
        raise ODEBFContractError("P1R11 source manifest path inventory differs")
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if (
        head != source_head
        or parent != CANONICAL_ALPHA_PARENT_HEAD
        or branch != EXECUTION_BRANCH
        or dirty
    ):
        raise ODEBFContractError("P1R11 execution provenance differs")
    return raw_sha256


def _prior_immutability_gate() -> None:
    for alias, (terminal_sha, manifest_sha) in R10_TERMINAL_MANIFEST.items():
        root = R10_ROOT / f"s05-common-coldcoord-fixed-e8-p1r10-r4-{alias}-v1"
        if (
            root.is_symlink()
            or not root.is_dir()
            or sha256_file(root / "terminal.json") != terminal_sha
            or sha256_file(root / "manifest.json") != manifest_sha
        ):
            raise ODEBFContractError("P1R11 immutable R10 reference differs")


def _local_gpu_cap_gate() -> dict[str, Any]:
    path = REPO_ROOT / "servers/local/gpu-caps.tsv"
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("P1R11 local GPU cap registry is absent")
    rows = [
        line.split("\t")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    matches = [row for row in rows if row[0] == "server1"]
    if (
        len(matches) != 1
        or len(matches[0]) != 5
        or matches[0][1] != SERVER1_SLURM_NODE
        or int(matches[0][2]) != SERVER1_PROJECT_GPU_CAP
    ):
        raise ODEBFContractError("P1R11 local GPU cap registry differs")
    _run(["git", "cat-file", "-e", f"{CANONICAL_GPU_CAP_COMMIT}^{{commit}}"])
    return {
        "server": "server1",
        "slurm_node": SERVER1_SLURM_NODE,
        "project_gpu_cap": SERVER1_PROJECT_GPU_CAP,
        "canonical_tracked_record_commit": CANONICAL_GPU_CAP_COMMIT,
    }


def _scheduler_snapshot() -> list[dict[str, Any]]:
    user = os.environ.get("USER")
    if not user:
        raise ODEBFContractError("P1R11 scheduler user is absent")
    output = _run(
        ["squeue", "-h", "-u", user, "-t", "R,PD", "-o", "%i|%j|%T|%b"]
    ).stdout
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ODEBFContractError("P1R11 scheduler row differs")
        job_id, name, state, tres = fields
        gpu = gpu_count_from_tres(tres)
        project_job = name.startswith("odeedit_")
        if project_job and (gpu is None or gpu <= 0):
            raise ODEBFContractError("P1R11 project GPU TRES is ambiguous")
        rows.append(
            {
                "job_id": job_id,
                "job_name": name,
                "state": state,
                "gpu": 0 if gpu is None else gpu,
                "project_job": project_job,
            }
        )
    return rows


def _pre_submit(source_head: str) -> dict[str, Any]:
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    gpu_cap_contract = _local_gpu_cap_gate()
    manifest_sha = _source_manifest_gate(source_head)
    _prior_immutability_gate()
    first = dry.build_plan(source_head)
    second = dry.build_plan(source_head)
    if first != second or len(first["jobs"]) != 4:
        raise ODEBFContractError("P1R11 dry plan is not deterministic")
    if any(not job["forecast"]["fits_envelope"] for job in first["jobs"]):
        raise ODEBFContractError("P1R11 resource forecast exceeds envelope")
    for alias in MODEL_ALIASES:
        guard = ODEBFArtifactGuard(
            REPO_ROOT,
            REPO_ROOT / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
            alias,
            require_held_ode_alloc=False,
        )
        guard.preflight()
        guard.assert_unchanged()
    scheduler = _scheduler_snapshot()
    active_project_gpu = sum(row["gpu"] for row in scheduler if row["project_job"])
    if active_project_gpu + 4 > SERVER1_PROJECT_GPU_CAP:
        raise ODEBFContractError("P1R11 server1 project GPU cap would be exceeded")
    planned_names = {job["job_name"] for job in first["jobs"]}
    if any(row["job_name"] in planned_names for row in scheduler):
        raise ODEBFContractError("P1R11 scheduler name collides")
    result_parent = REPO_ROOT / "local/odebf/results"
    log_parent = REPO_ROOT / "local/odebf/logs"
    state_parent = REPO_ROOT / "local/odebf/state"
    for job in first["jobs"]:
        if (result_parent / job["result_name"]).exists():
            raise ODEBFContractError("P1R11 result root collides")
        if list(log_parent.glob(f"{job['job_name']}-*")):
            raise ODEBFContractError("P1R11 log namespace collides")
    for suffix in ("intent", "submission-receipt"):
        if (state_parent / f"{SUBMISSION_NAMESPACE}.{suffix}.json").exists():
            raise ODEBFContractError("P1R11 state namespace collides")
    mem_total_mib = int(
        next(
            line.split()[1]
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("MemTotal:")
        )
    ) // 1024
    if mem_total_mib < 4 * 65_000:
        raise ODEBFContractError("P1R11 host memory envelope is insufficient")
    return {
        "source_manifest_sha256": manifest_sha,
        "prior_immutability_gate": True,
        "active_project_gpu": active_project_gpu,
        "server1_project_gpu_cap": SERVER1_PROJECT_GPU_CAP,
        "gpu_cap_contract": gpu_cap_contract,
        "host_total_mib": mem_total_mib,
        "plan": first,
    }


def _submit_one(alias: str, actuator: AlphaActuator, source_head: str) -> str:
    root = (
        REPO_ROOT
        / "local/odebf/results"
        / expected_canonical_alpha_result_name(alias, actuator)
    )
    logs = REPO_ROOT / "local/odebf/logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = dry.JOB_NAMES[(alias, actuator.value)]
    result = _run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            f"--job-name={name}",
            f"--output={logs / (name + '-%j.out')}",
            f"--error={logs / (name + '-%j.err')}",
            f"--chdir={REPO_ROOT}",
            str(SBATCH),
            alias,
            actuator.value,
            str(root),
            source_head,
            CANONICAL_ALPHA_RESULT_TOKEN,
        ]
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R11 Slurm job ID differs")
    return job_id


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    state_root = REPO_ROOT / "local/odebf/state"
    preflight = _pre_submit(args.source_head)
    intent_sha = _write_once(
        state_root / f"{SUBMISSION_NAMESPACE}.intent.json",
        {
            "schema": "ode-edit-s05-canonical-alpha-posfield-p1r11-submit-intent/v1",
            "instruction_id": CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID,
            "source_head": args.source_head,
            "preflight": preflight,
            "four_independent_jobs": True,
            "accepted_while_held": True,
            "retry_or_resubmit_authorized": False,
        },
    )
    jobs: dict[str, str] = {}
    try:
        for alias in MODEL_ALIASES:
            for actuator in AlphaActuator:
                key = f"{alias}:{actuator.value}"
                jobs[key] = _submit_one(alias, actuator, args.source_head)
        receipt_sha = _write_once(
            state_root / f"{SUBMISSION_NAMESPACE}.submission-receipt.json",
            {
                "schema": (
                    "ode-edit-s05-canonical-alpha-posfield-p1r11-"
                    "submission-receipt/v1"
                ),
                "instruction_id": CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID,
                "source_head": args.source_head,
                "intent_sha256": intent_sha,
                "jobs": jobs,
                "accepted_while_held": True,
                "retry_or_resubmit_authorized": False,
            },
        )
        _run(["scontrol", "release", *jobs.values()])
    except Exception:
        if jobs:
            _run(["scancel", *jobs.values()], check=False)
        raise
    print(
        json.dumps(
            {
                "status": "FOUR_JOBS_SUBMITTED",
                "instruction_id": CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID,
                "source_head": args.source_head,
                "jobs": jobs,
                "intent_sha256": intent_sha,
                "submission_receipt_sha256": receipt_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
