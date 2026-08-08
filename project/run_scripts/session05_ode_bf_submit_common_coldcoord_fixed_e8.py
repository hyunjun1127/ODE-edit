#!/usr/bin/env python3
"""Create-once submitter for the common-cold-coordinate R10 pair."""

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

from project.run_scripts import (
    session05_ode_bf_common_coldcoord_fixed_e8_dry_plan as dry,
)
from project.run_scripts.ode_bf.artifacts import (
    ODEBFArtifactGuard,
    load_rooted_json,
    sha256_file,
)
from project.run_scripts.ode_bf.common_cold_coordinate import (
    COMMON_COLD_INSTRUCTION_ID,
)
from project.run_scripts.ode_bf.contracts import (
    MODEL_ALIASES,
    ODEBFContractError,
)
from project.run_scripts.ode_bf.p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_PARENT_HEAD,
    COMMON_COLD_RESULT_TOKEN,
    expected_common_cold_result_name,
)
from project.run_scripts.ode_bf.resource import gpu_count_from_tres
from project.run_scripts.session05_ode_bf_submit_fixed_e8_structfunc_soft import (
    _prior_immutability_gate as _legacy_prior_immutability_gate,
)


SESSION_ID = "019fc63e-5217-7250-9c22-c5b2ec4248f0"
EXECUTION_BRANCH = "codex/odeeditsh1-s05-common-coldcoord-fixed-e8-p1r10-v1"
EXECUTION_REPAIR_PARENT_HEAD = "3e479b260f73c5ed520b10f7574a2074fe903c52"
SECOND_REPAIR_PARENT_HEAD = "abaa366c8d32918ecf96d4f433248b799703001b"
FIRST_REPAIR_PARENT_HEAD = "d60ddaf765f79f4f4f73c2dc455f8aef7521084e"
SERVER1_PROJECT_GPU_CAP = 3
APPROVAL_ENV = "ODEEDIT_S05_COMMON_COLD_R10_R3_RUN_APPROVAL"
SUBMISSION_NAMESPACE = "s05-common-coldcoord-fixed-e8-p1r10-r3-v1"
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_common_coldcoord_fixed_e8.sbatch"
SOURCE_MANIFEST = (
    REPO_ROOT
    / "project/run_scripts/ode_bf/locks/"
    "source_manifest_s05_common_coldcoord_fixed_e8.json"
)
R8_R2_ROOTS = {
    "s05-fixed-e8-solver-isolation-cert-r8-r2-llama3-8b-inst-v1": (
        171,
        "946c74e142f23fabfd8236dc77a60f3e44184f86eb29c6da0e81a85da518549e",
    ),
    "s05-fixed-e8-solver-isolation-cert-r8-r2-qwen2.5-7b-inst-v1": (
        102,
        "86fe8a0076d958e3e501bb3a8ab7a43f7726cfaadb12d4b38547ea5735cc5807",
    ),
}
R8_R2_LOGS = {
    "odeedit_s05_e8r8r2_llama-17697.err": (
        784,
        "3c3f68b280f2a42390a7e6cdf5c44383c79a20dc9a551149e09b8b0c4858acca",
    ),
    "odeedit_s05_e8r8r2_llama-17697.out": (
        388,
        "cf5e39e71012bec7edf3f5b0f29f49274a8f4a950151cb617b26546ba8595186",
    ),
    "odeedit_s05_e8r8r2_qwen-17698.err": (
        1174,
        "ed26184c3381faa7a6474e586ea24a3c086000b6cda11ffad341347646a666ee",
    ),
    "odeedit_s05_e8r8r2_qwen-17698.out": (
        113,
        "1b1a59d4bedc52bd3f2e1ad5bda614c8309963de5967f30a04e98c22f6adfaec",
    ),
}
R8_R2_STATE = {
    "s05-fixed-e8-solver-isolation-cert-r8-r2-v1.intent.json": (
        9719,
        "4a706bcea9d4eac3dbc573938e069e82207e9723de5515f954c48730769184e4",
    ),
    "s05-fixed-e8-solver-isolation-cert-r8-r2-v1.submission-receipt.json": (
        495,
        "436b3940333b8f7b390e2e59566862279098d97789516c759e7ea1160a7503a6",
    ),
}
R10_FAILED_ROOTS = {
    "s05-common-coldcoord-fixed-e8-p1r10-llama3-8b-inst-v1": (
        3,
        "315c553a40159415501be9ba3a6c36616f03cf1e440ee89473291a8ea8ffa9bf",
    ),
    "s05-common-coldcoord-fixed-e8-p1r10-qwen2.5-7b-inst-v1": (
        3,
        "315c553a40159415501be9ba3a6c36616f03cf1e440ee89473291a8ea8ffa9bf",
    ),
}
R10_FAILED_LOGS = {
    "odeedit_s05_r10_llama-17744.out": (
        113,
        "1b1a59d4bedc52bd3f2e1ad5bda614c8309963de5967f30a04e98c22f6adfaec",
    ),
    "odeedit_s05_r10_llama-17744.err": (
        391,
        "3c26fef51ce30b9542b1a3998114d3ec18db670077393bce8ede37f7be3256bd",
    ),
    "odeedit_s05_r10_qwen-17745.out": (
        113,
        "1b1a59d4bedc52bd3f2e1ad5bda614c8309963de5967f30a04e98c22f6adfaec",
    ),
    "odeedit_s05_r10_qwen-17745.err": (
        392,
        "781f3d6f14535461f0b5a4d4c2620a9515a47812b7b3d2e9b34b95f88b539e5c",
    ),
}
R10_FAILED_STATE = {
    "s05-common-coldcoord-fixed-e8-p1r10-v1.intent.json": (
        4995,
        "bc1131624dc025ca54f026f07695975da3235c5c82d60707ef3d8f98d48b8eac",
    ),
    "s05-common-coldcoord-fixed-e8-p1r10-v1.submission-receipt.json": (
        422,
        "19a20233eb804b85a68b8f3b162d478286b7fc82c6506139fde7c17d7c32ddf4",
    ),
}
R10_R1_FAILED_ROOTS = {
    "s05-common-coldcoord-fixed-e8-p1r10-r1-llama3-8b-inst-v1": (
        11,
        "0222a249ac5e30a4c62217096871e930d640f111c22f6781145ecdfdeb277c64",
    ),
    "s05-common-coldcoord-fixed-e8-p1r10-r1-qwen2.5-7b-inst-v1": (
        6,
        "5ff7cd1862a8bcaf5adfa0e026395b99eb8237247e2af3fefafd6ea0115ec7f0",
    ),
}
R10_R1_FAILED_LOGS = {
    "odeedit_s05_r10r1_llama-17748.out": (
        113,
        "1b1a59d4bedc52bd3f2e1ad5bda614c8309963de5967f30a04e98c22f6adfaec",
    ),
    "odeedit_s05_r10r1_llama-17748.err": (
        1189,
        "5fd0df151bf20cfe81cd93aff7ae97d9b5b0bd9efce7844b4f1b96bf8db0b033",
    ),
    "odeedit_s05_r10r1_qwen-17749.out": (
        113,
        "1b1a59d4bedc52bd3f2e1ad5bda614c8309963de5967f30a04e98c22f6adfaec",
    ),
    "odeedit_s05_r10r1_qwen-17749.err": (
        1183,
        "d0d640f2cf00700dd87b4680c02d84d2d04fb2d4453f0ef0411ea5225f0a2c82",
    ),
}
R10_R1_FAILED_STATE = {
    "s05-common-coldcoord-fixed-e8-p1r10-r1-v1.intent.json": (
        5097,
        "2edd8ed15afd261893cd05d8623fbfcb00316f7d2be34bb3a900c9f03bf27501",
    ),
    "s05-common-coldcoord-fixed-e8-p1r10-r1-v1.submission-receipt.json": (
        425,
        "eada3cf17fc6f4fb0c2475b2443147d1633ed038bda850c8838f0d9322980124",
    ),
}
R10_R2_FAILED_ROOTS = {
    "s05-common-coldcoord-fixed-e8-p1r10-r2-llama3-8b-inst-v1": (
        12,
        "a2de09dd0ce3943268239fd6667a061aec18278a463100631ca292350eaddaa2",
    ),
    "s05-common-coldcoord-fixed-e8-p1r10-r2-qwen2.5-7b-inst-v1": (
        12,
        "0fec6e49eed43f01388563d4e727156ec33434479780c0783d0f9e24efc2dc88",
    ),
}
R10_R2_FAILED_LOGS = {
    "odeedit_s05_r10r2_llama-17752.out": (
        113,
        "1b1a59d4bedc52bd3f2e1ad5bda614c8309963de5967f30a04e98c22f6adfaec",
    ),
    "odeedit_s05_r10r2_llama-17752.err": (
        1185,
        "aec162d3a1745920f4a710cee3e88dce459580521e78a25251667c32247112a3",
    ),
    "odeedit_s05_r10r2_qwen-17753.out": (
        113,
        "1b1a59d4bedc52bd3f2e1ad5bda614c8309963de5967f30a04e98c22f6adfaec",
    ),
    "odeedit_s05_r10r2_qwen-17753.err": (
        1186,
        "948757ea6d2c6ffd75c34c3b9c35eb78eab00447ee714d2c36e92291125ffeb3",
    ),
}
R10_R2_FAILED_STATE = {
    "s05-common-coldcoord-fixed-e8-p1r10-r2-v1.intent.json": (
        5168,
        "3b1e2597876559d7f4edc933d3bc55e97f5386776c804081fd90638a7cf973b2",
    ),
    "s05-common-coldcoord-fixed-e8-p1r10-r2-v1.submission-receipt.json": (
        425,
        "78982a5a367d015cb168cb3c5b2a045dee0a3898c1e71a65e88373144a56f85d",
    ),
}


def _run(
    args: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _tree_identity(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_dir():
        raise ODEBFContractError("R10 immutable result root differs")
    members = tuple(path.rglob("*"))
    if any(item.is_symlink() for item in members):
        raise ODEBFContractError("R10 immutable result root contains symlink")
    files = sorted(item for item in members if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(f"{sha256_file(item)}  {relative}\n".encode("utf-8"))
    return len(files), digest.hexdigest()


def _exact_file(path: Path, expected: tuple[int, str]) -> None:
    size, digest = expected
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != size
        or sha256_file(path) != digest
    ):
        raise ODEBFContractError("R10 immutable receipt differs")


def _execution_provenance_gate(source_head: str) -> dict[str, Any]:
    approval = os.environ.get(APPROVAL_ENV)
    expected_approval = f"{COMMON_COLD_INSTRUCTION_ID}:{source_head}"
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    second_repair_parent = _run(["git", "rev-parse", "HEAD^^"]).stdout.strip()
    first_repair_parent = _run(["git", "rev-parse", "HEAD^^^"]).stdout.strip()
    scientific_parent = _run(["git", "rev-parse", "HEAD^^^^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ).stdout
    ancestor = _run(
        ["git", "merge-base", "--is-ancestor", COMMON_COLD_PARENT_HEAD, head],
        check=False,
    )
    if approval != expected_approval:
        raise ODEBFContractError("R10 checkpoint-bound approval is absent")
    if (
        head != source_head
        or parent != EXECUTION_REPAIR_PARENT_HEAD
        or second_repair_parent != SECOND_REPAIR_PARENT_HEAD
        or first_repair_parent != FIRST_REPAIR_PARENT_HEAD
        or scientific_parent != COMMON_COLD_PARENT_HEAD
        or branch != EXECUTION_BRANCH
        or ancestor.returncode != 0
        or dirty
    ):
        raise ODEBFContractError("R10 execution provenance differs")
    return {
        "checkpoint_bound_approval": expected_approval,
        "execution_head": head,
        "exact_execution_repair_parent": parent,
        "exact_second_repair_parent": second_repair_parent,
        "exact_first_repair_parent": first_repair_parent,
        "exact_scientific_parent": scientific_parent,
        "scientific_parent_is_ancestor": True,
        "branch": branch,
        "tracked_tree_clean": True,
    }


def _source_manifest_gate(source_head: str) -> str:
    value, raw_sha256 = load_rooted_json(
        SOURCE_MANIFEST,
        expected_schema=(
            "ode-edit-s05-common-coldcoord-fixed-e8-p1r10-source-manifest/v1"
        ),
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != COMMON_COLD_INSTRUCTION_ID
        or value.get("expected_parent") != COMMON_COLD_PARENT_HEAD
        or value.get("execution_repair_parent") != EXECUTION_REPAIR_PARENT_HEAD
        or value.get("second_repair_parent") != SECOND_REPAIR_PARENT_HEAD
        or value.get("first_repair_parent") != FIRST_REPAIR_PARENT_HEAD
        or value.get("execution_branch") != EXECUTION_BRANCH
        or value.get("execution_head_policy") != "runtime-git-head"
        or not isinstance(entries, list)
        or not entries
    ):
        raise ODEBFContractError("R10 source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ODEBFContractError("R10 source manifest entry differs")
        relative = str(entry["path"])
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("size")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ODEBFContractError("R10 source manifest content differs")
        observed.append(relative)
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise ODEBFContractError("R10 source manifest path ordering differs")
    manifest_relative = SOURCE_MANIFEST.relative_to(REPO_ROOT).as_posix()
    tracked = set(
        _run(["git", "ls-tree", "-r", "--name-only", source_head])
        .stdout.splitlines()
    )
    expected = {
        relative
        for relative in tracked
        if (
            relative.startswith("project/run_scripts/ode_bf/")
            or relative.startswith(
                "project/run_scripts/session05_ode_bf_common_coldcoord_fixed_e8"
            )
            or relative
            == "project/run_scripts/session05_ode_bf_submit_common_coldcoord_fixed_e8.py"
        )
        and relative != manifest_relative
    }
    if set(observed) != expected:
        raise ODEBFContractError("R10 source manifest path set differs")
    return raw_sha256


def _prior_immutability_gate() -> None:
    _legacy_prior_immutability_gate()
    base = REPO_ROOT / "local/odebf"
    for name, expected in R8_R2_ROOTS.items():
        if _tree_identity(base / "results" / name) != expected:
            raise ODEBFContractError("R10 immutable R8-R2 root differs")
    for name, expected in R8_R2_LOGS.items():
        _exact_file(base / "logs" / name, expected)
    for name, expected in R8_R2_STATE.items():
        _exact_file(base / "state" / name, expected)
    for name, expected in R10_FAILED_ROOTS.items():
        if _tree_identity(base / "results" / name) != expected:
            raise ODEBFContractError("R10 immutable failed root differs")
    for name, expected in R10_FAILED_LOGS.items():
        _exact_file(base / "logs" / name, expected)
    for name, expected in R10_FAILED_STATE.items():
        _exact_file(base / "state" / name, expected)
    for name, expected in R10_R1_FAILED_ROOTS.items():
        if _tree_identity(base / "results" / name) != expected:
            raise ODEBFContractError("R10 immutable R1 failed root differs")
    for name, expected in R10_R1_FAILED_LOGS.items():
        _exact_file(base / "logs" / name, expected)
    for name, expected in R10_R1_FAILED_STATE.items():
        _exact_file(base / "state" / name, expected)
    for name, expected in R10_R2_FAILED_ROOTS.items():
        if _tree_identity(base / "results" / name) != expected:
            raise ODEBFContractError("R10 immutable R2 failed root differs")
    for name, expected in R10_R2_FAILED_LOGS.items():
        _exact_file(base / "logs" / name, expected)
    for name, expected in R10_R2_FAILED_STATE.items():
        _exact_file(base / "state" / name, expected)


def _scheduler_snapshot() -> list[dict[str, Any]]:
    user = os.environ.get("USER")
    if not user:
        raise ODEBFContractError("R10 scheduler user identity unavailable")
    output = _run(
        ["squeue", "-h", "-u", user, "-t", "R,PD", "-o", "%i|%j|%T|%b"]
    ).stdout
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ODEBFContractError("R10 scheduler row differs")
        job_id, name, state, tres = fields
        gpu = gpu_count_from_tres(tres)
        project_job = name.startswith("odeedit_")
        if project_job and (gpu is None or gpu <= 0):
            raise ODEBFContractError("R10 project GPU TRES is ambiguous")
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


def _host_memory_gate() -> int:
    total = next(
        (
            line
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("MemTotal:")
        ),
        None,
    )
    if total is None:
        raise ODEBFContractError("R10 host memory identity unavailable")
    fields = total.split()
    if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
        raise ODEBFContractError("R10 host memory units differ")
    total_mib = int(fields[1]) // 1024
    if total_mib < 130_000:
        raise ODEBFContractError("R10 pair exceeds host memory capacity")
    return total_mib


def _pre_submit(source_head: str) -> dict[str, Any]:
    provenance = _execution_provenance_gate(source_head)
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    manifest_sha256 = _source_manifest_gate(source_head)
    _prior_immutability_gate()
    plan = dry.build_plan(source_head)
    for job in plan["jobs"]:
        if not bool(job["forecast"]["fits_envelope"]):
            raise ODEBFContractError("R10 resource forecast exceeds allocation")
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
    active_project_gpu = sum(
        row["gpu"] for row in scheduler if row["project_job"]
    )
    if active_project_gpu + 2 > SERVER1_PROJECT_GPU_CAP:
        raise ODEBFContractError("R10 server1 project GPU cap would be exceeded")
    if any(row["job_name"] in dry.JOB_NAMES.values() for row in scheduler):
        raise ODEBFContractError("R10 scheduler name collides")
    result_parent = REPO_ROOT / "local/odebf/results"
    log_parent = REPO_ROOT / "local/odebf/logs"
    state_parent = REPO_ROOT / "local/odebf/state"
    for alias in MODEL_ALIASES:
        if (result_parent / expected_common_cold_result_name(alias)).exists():
            raise ODEBFContractError("R10 result root collides")
        if list(log_parent.glob(f"{dry.JOB_NAMES[alias]}-*")):
            raise ODEBFContractError("R10 log namespace collides")
    for suffix in ("intent", "submission-receipt"):
        if (state_parent / f"{SUBMISSION_NAMESPACE}.{suffix}.json").exists():
            raise ODEBFContractError("R10 state namespace collides")
    return {
        "execution_provenance": provenance,
        "source_manifest_sha256": manifest_sha256,
        "prior_immutability_gate": True,
        "active_project_gpu": active_project_gpu,
        "server1_project_gpu_cap": SERVER1_PROJECT_GPU_CAP,
        "host_total_mib": _host_memory_gate(),
        "plan": plan,
    }


def _submit_one(alias: str, source_head: str) -> str:
    root = REPO_ROOT / "local/odebf/results" / expected_common_cold_result_name(alias)
    logs = REPO_ROOT / "local/odebf/logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = dry.JOB_NAMES[alias]
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
            str(root),
            source_head,
            COMMON_COLD_RESULT_TOKEN,
        ]
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("R10 Slurm job ID differs")
    return job_id


def _cancel_held(jobs: Mapping[str, str]) -> None:
    if jobs:
        _run(["scancel", *jobs.values()], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument(
        "--state-root", type=Path, default=REPO_ROOT / "local/odebf/state"
    )
    args = parser.parse_args()
    preflight = _pre_submit(args.source_head)
    intent_sha256 = _write_once(
        args.state_root / f"{SUBMISSION_NAMESPACE}.intent.json",
        {
            "schema": "ode-edit-s05-common-coldcoord-fixed-e8-p1r10-r3-submit-intent/v1",
            "instruction_id": COMMON_COLD_INSTRUCTION_ID,
            "source_head": args.source_head,
            "preflight": preflight,
            "pair_back_to_back": True,
            "pair_accepted_while_held": True,
            "retry_or_resubmit_authorized": False,
        },
    )
    jobs: dict[str, str] = {}
    try:
        for alias in MODEL_ALIASES:
            jobs[alias] = _submit_one(alias, args.source_head)
        receipt_sha256 = _write_once(
            args.state_root / f"{SUBMISSION_NAMESPACE}.submission-receipt.json",
            {
                "schema": (
                    "ode-edit-s05-common-coldcoord-fixed-e8-p1r10-r3-"
                    "submission-receipt/v1"
                ),
                "instruction_id": COMMON_COLD_INSTRUCTION_ID,
                "source_head": args.source_head,
                "intent_sha256": intent_sha256,
                "jobs": jobs,
                "pair_accepted_while_held": True,
                "retry_or_resubmit_authorized": False,
            },
        )
        _run(["scontrol", "release", *jobs.values()])
    except Exception:
        _cancel_held(jobs)
        raise
    print(
        json.dumps(
            {
                "status": "PAIR_SUBMITTED",
                "instruction_id": COMMON_COLD_INSTRUCTION_ID,
                "source_head": args.source_head,
                "jobs": jobs,
                "intent_sha256": intent_sha256,
                "submission_receipt_sha256": receipt_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
