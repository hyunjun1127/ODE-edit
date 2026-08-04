#!/usr/bin/env python3
"""One-shot, fail-closed submission of the exact Session 04 P0 model pair."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_alloc.contracts import MODEL_ALIASES, ODEAllocContractError, canonical_json
from project.run_scripts.ode_alloc.firewall import assert_p0_runtime_firewall_ast
from project.run_scripts.ode_alloc.p0_artifacts import P0ArtifactGuard, sha256_file
from project.run_scripts.ode_alloc.p0_runtime import (
    DIAGNOSTIC_INSTRUCTION_ID,
    EXPECTED_BASE,
    INSTRUCTION_ID,
    R2_RUN_TOKEN,
    SEAL_ROOT,
    SESSION_ID,
    expected_result_name,
)
from project.run_scripts.ode_alloc.selection import (
    assert_seal_source_current,
    load_and_verify_seal_candidate,
)


PACKAGE_ROOT = REPO_ROOT / "project" / "run_scripts" / "ode_alloc"
NUMERICAL_LOCK = PACKAGE_ROOT / "numerical_lock_proposal.json"
ARTIFACT_LOCK = PACKAGE_ROOT / "p0_artifact_lock_r1.json"
SEAL = PACKAGE_ROOT / "split_anchor_seal_candidate.json"
SBATCH = REPO_ROOT / "project" / "run_scripts" / "session04_ode_alloc_p0_r1.sbatch"
JOB_NAMES = {
    "llama3-8b-inst": "odealloc_s04_p0r2_llama",
    "qwen2.5-7b-inst": "odealloc_s04_p0r2_qwen",
}
GPU_CAP = 3
CANONICAL_NODE = "server2"
R1_SOURCE_CHECKPOINT = "6f9feba21d717a55b350d16ec82b7ac6216872e4"
R2_EXPECTED_PARENT = "b6a8e06ebdc8bf1fe18fa9ba799a14416e24d071"
PAIR_HOST_MEMORY_MIB = 130000
MEMORY_CAP_MIB_PER_GPU = 66017
NUMERICAL_LOCK_SHA256 = "905a3bbccfb71a5b780586cde49503bde88df92bf8af378d8b88a85b4245f37d"
ARTIFACT_LOCK_SHA256 = "6c327c563e0e805eb73a09cf3a577dea3771bc44127748fe64aba70a803292d2"
SEAL_FILE_SHA256 = "48e8c96acee8e95ffc2b82689c314b4b9f84c8a6979431c8bf03f8987e2b549e"
R2_CHANGED_PATHS = frozenset(
    {
        "project/run_scripts/ode_alloc/p0_runtime.py",
        "project/run_scripts/ode_alloc/tests/test_p0_runtime_contracts.py",
        "project/run_scripts/ode_alloc/tests/test_p0_submission_contracts.py",
        "project/run_scripts/session04_ode_alloc_p0.py",
        "project/run_scripts/session04_ode_alloc_p0_r1.sbatch",
        "project/run_scripts/session04_ode_alloc_submit_p0_r1.py",
    }
)
R1_IMMUTABLE_FILES = {
    "local/odealloc/results/s04-p0-native-identity-r1-llama3-8b-inst-905a3bbc/failure.json": "43ea916a2d05dd8fff739f51742d49f76db12226e6bda4cc7e83297da0354680",
    "local/odealloc/results/s04-p0-native-identity-r1-llama3-8b-inst-905a3bbc/raw/context_templates.json": "c75c821553503783ea3dc75407fb6ffce3038357b3d54c22506fc4285e17a3a0",
    "local/odealloc/results/s04-p0-native-identity-r1-qwen2.5-7b-inst-905a3bbc/failure.json": "7481f36daf9336b118f86ef9df8ce94661894d3f6d7cabdd1f397e0a7e0a9384",
    "local/odealloc/results/s04-p0-native-identity-r1-qwen2.5-7b-inst-905a3bbc/raw/context_templates.json": "04b904762e9135931d5b293b7444c5278cc7485c0128c413201db1367d3e4618",
    "local/odealloc/logs/s04-p0-native-identity-r1-llama3-8b-inst-905a3bbc-16333.out": "6c619747114289754d4905a1e9f008979ac7f9a71dfcfc5d8997e0083ba2c2a8",
    "local/odealloc/logs/s04-p0-native-identity-r1-llama3-8b-inst-905a3bbc-16333.err": "cf9e1ee57869bd0a541a0c478f363c409e009f05713bc4a40fe2fa4e5e587e9e",
    "local/odealloc/logs/s04-p0-native-identity-r1-qwen2.5-7b-inst-905a3bbc-16334.out": "4c60a72477a48e8e2b15699d77a2effbcd4ac165b8345d9203b1731035d3396f",
    "local/odealloc/logs/s04-p0-native-identity-r1-qwen2.5-7b-inst-905a3bbc-16334.err": "200794746d9be2606d2e2079de2af733d8804f6326f72c2c141abb91d3564898",
    "local/odealloc/state/s04-p0-native-identity-r1-905a3bbc.submission-intent.json": "a0d983cec1356b88926ac1141ecdae96105e69e7e8d52932e1d6750b8e3075d7",
    "local/odealloc/state/s04-p0-native-identity-r1-905a3bbc.submission-receipt.json": "4eef5a1f07a2a1a8c1b4177ffea5dfedd780a71ff67846ce17f47a8fdf473865",
}
APPROVED_NUMERICAL_DIFF_PATHS = frozenset(
    {
        "allocation_gauge.epsilon_prior",
        "allocation_gauge.exact_zero_policy",
        "allocation_gauge.max_abs_centered_q",
        "allocation_gauge.minimum_active_layers",
        "allocation_gauge.per_step_projection",
        "allocation_gauge.positive_near_zero_policy",
        "allocation_gauge.q_cap_semantics",
        "allocation_gauge.zero_or_near_zero_policy",
        "execution_boundary.dry_plan_gpu",
        "execution_boundary.dry_plan_model_load",
        "execution_boundary.gpu_now",
        "execution_boundary.model_load_now",
        "execution_boundary.p0_pair_gpu_total",
        "execution_boundary.p0_pair_submit",
        "execution_boundary.p1_submit",
        "execution_boundary.retry_or_resubmit",
        "execution_boundary.slurm_submit_now",
        "execution_seed",
        "instruction_id",
        "p0_identity_lock",
        "parent_checkpoint",
        "schema_version",
        "solver.active_subset_count_at_max",
        "solver.aggregated_constraint_labels",
        "solver.cbf_slack_penalty",
        "solver.cbf_slack_penalty_weight",
        "solver.constraint_residual_tolerance",
        "solver.cpu_fixture_max_constraints",
        "solver.linear_solve_residual_tolerance",
        "solver.per_item_constraints",
        "solver.production_max_constraints",
        "solver.production_qp_backend",
        "solver.sign_residual_tolerance",
        "solver.slack_elimination",
        "solver.stationarity_residual_tolerance",
        "status",
    }
)


def _run(
    args: list[str], *, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _write_once(path: Path, value: dict[str, Any]) -> str:
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _source_gate() -> str:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    grandparent = _run(["git", "rev-parse", "HEAD^^"]).stdout.strip()
    great_grandparent = _run(["git", "rev-parse", "HEAD^^^"]).stdout.strip()
    if (
        parent != R2_EXPECTED_PARENT
        or grandparent != R1_SOURCE_CHECKPOINT
        or great_grandparent != EXPECTED_BASE
        or head in {EXPECTED_BASE, R1_SOURCE_CHECKPOINT, R2_EXPECTED_PARENT}
    ):
        raise ODEAllocContractError("R2 diagnostic checkpoint ancestry differs")
    tracked = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if tracked:
        raise ODEAllocContractError("tracked P0 source is not frozen clean")
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"]).stdout.splitlines()
    if any(
        not (path.startswith("agents/server2/") or path.startswith("audits/servers/server2/"))
        for path in untracked
    ):
        raise ODEAllocContractError("foreign untracked path overlaps P0 worktree")
    changed = _run(["git", "diff", "--name-only", f"{EXPECTED_BASE}..{head}"]).stdout.splitlines()
    if not changed or any(
        not (
            path.startswith("project/run_scripts/ode_alloc/")
            or path.startswith("project/run_scripts/session04_ode_alloc_")
        )
        for path in changed
    ):
        raise ODEAllocContractError("R1 checkpoint changed a non-approved path")
    forbidden = ("session03", "session_03", "knowledge-revision")
    if any(any(token in path.casefold() for token in forbidden) for path in changed):
        raise ODEAllocContractError("R1 checkpoint overlaps a foreign namespace")
    diagnostic_paths = _run(
        ["git", "diff", "--name-only", f"{R2_EXPECTED_PARENT}..{head}"]
    ).stdout.splitlines()
    if set(diagnostic_paths) != R2_CHANGED_PATHS:
        raise ODEAllocContractError("R2 checkpoint differs from the exact diagnostic path set")
    _run(["git", "diff", "--check", f"{EXPECTED_BASE}..{head}"])
    return head


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ODEAllocContractError("P0 JSON root is not an object")
    return value


def _changed_json_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if type(left) is not type(right):
        return {prefix or "<root>"}
    if isinstance(left, dict):
        changed: set[str] = set()
        for key in set(left).union(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                changed.add(path)
            else:
                changed.update(_changed_json_paths(left[key], right[key], path))
        return changed
    if left != right:
        return {prefix or "<root>"}
    return set()


def _r1_immutability_gate() -> str:
    result_roots = (
        REPO_ROOT
        / "local/odealloc/results/s04-p0-native-identity-r1-llama3-8b-inst-905a3bbc",
        REPO_ROOT
        / "local/odealloc/results/s04-p0-native-identity-r1-qwen2.5-7b-inst-905a3bbc",
    )
    observed: set[str] = set()
    for root in result_roots:
        if not root.is_dir() or root.is_symlink():
            raise ODEAllocContractError("R1 result root identity differs")
        for path in root.rglob("*"):
            if path.is_file() or path.is_symlink():
                observed.add(path.relative_to(REPO_ROOT).as_posix())
    logs = REPO_ROOT / "local" / "odealloc" / "logs"
    for prefix in (
        "s04-p0-native-identity-r1-llama3-8b-inst-905a3bbc-16333.",
        "s04-p0-native-identity-r1-qwen2.5-7b-inst-905a3bbc-16334.",
    ):
        for path in logs.glob(f"{prefix}*"):
            observed.add(path.relative_to(REPO_ROOT).as_posix())
    state = REPO_ROOT / "local" / "odealloc" / "state"
    for path in state.glob("s04-p0-native-identity-r1-905a3bbc.*.json"):
        observed.add(path.relative_to(REPO_ROOT).as_posix())
    if observed != set(R1_IMMUTABLE_FILES):
        raise ODEAllocContractError("R1 root/log/state file set differs")
    for relative, expected in sorted(R1_IMMUTABLE_FILES.items()):
        path = REPO_ROOT / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ODEAllocContractError("R1 root/log/state digest differs")
    return hashlib.sha256(canonical_json(R1_IMMUTABLE_FILES).encode("utf-8")).hexdigest()


def _cpu_static_gate(source_head: str) -> dict[str, Any]:
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    _run(["scripts/check-agent-access.sh", "--all-changed"])
    assert_p0_runtime_firewall_ast(
        [
            PACKAGE_ROOT / "p0_runtime.py",
            REPO_ROOT / "project" / "run_scripts" / "session04_ode_alloc_p0.py",
        ]
    )
    for path in sorted((REPO_ROOT / "project" / "run_scripts" / "ode_alloc").rglob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
    for path in (
        REPO_ROOT / "project" / "run_scripts" / "session04_ode_alloc_dry_plan.py",
        REPO_ROOT / "project" / "run_scripts" / "session04_ode_alloc_p0.py",
        Path(__file__),
    ):
        compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
    _run(["bash", "-n", str(SBATCH)])
    r1_immutability_sha = _r1_immutability_gate()
    lock = _strict_json(NUMERICAL_LOCK)
    _strict_json(ARTIFACT_LOCK)
    seal = load_and_verify_seal_candidate(SEAL)
    if (
        sha256_file(NUMERICAL_LOCK) != NUMERICAL_LOCK_SHA256
        or sha256_file(ARTIFACT_LOCK) != ARTIFACT_LOCK_SHA256
        or sha256_file(SEAL) != SEAL_FILE_SHA256
    ):
        raise ODEAllocContractError("P0 lock/seal file digest differs")
    if (
        lock.get("instruction_id") != INSTRUCTION_ID
        or lock.get("parent_checkpoint") != EXPECTED_BASE
        or lock.get("execution_seed") != 41
        or lock["allocation_gauge"].get("max_abs_centered_q") != 1.3862943611198906
        or lock["allocation_gauge"].get("basis_energy_epsilon") != 1e-24
        or lock["solver"].get("production_max_constraints") != 3
        or lock["solver"].get("active_subset_count_at_max") != 8
        or lock["solver"].get("per_item_constraints") is not False
        or lock["p0_identity_lock"].get("seal_root_digest") != SEAL_ROOT
        or seal.get("root_digest") != SEAL_ROOT
    ):
        raise ODEAllocContractError("R1 numerical/seal lock differs")
    old_seal = _run(["git", "show", f"{EXPECTED_BASE}:project/run_scripts/ode_alloc/split_anchor_seal_candidate.json"]).stdout
    if old_seal.encode("utf-8") != SEAL.read_bytes():
        raise ODEAllocContractError("approved split/anchor seal changed after review")
    old_lock = json.loads(
        _run(
            [
                "git",
                "show",
                f"{EXPECTED_BASE}:project/run_scripts/ode_alloc/numerical_lock_proposal.json",
            ]
        ).stdout
    )
    changed_lock_paths = _changed_json_paths(old_lock, lock)
    if changed_lock_paths != APPROVED_NUMERICAL_DIFF_PATHS:
        raise ODEAllocContractError("numerical lock diff exceeds the approved R1 fields")

    tests = _run(
        [
            "/mnt/raid5/janghj/EasyEdit/.venv/bin/python",
            "-W", "error", "-m", "unittest", "discover",
            "-s", "project/run_scripts/ode_alloc/tests", "-p", "test_*.py", "-v",
        ]
    )
    count_match = re.search(r"Ran ([0-9]+) tests?", tests.stderr)
    if (
        count_match is None
        or int(count_match.group(1)) < 41
        or "\nOK\n" not in tests.stderr
    ):
        raise ODEAllocContractError("full warnings-as-errors CPU suite did not report OK")
    unit_test_count = int(count_match.group(1))
    dry_command = [
        "/mnt/raid5/janghj/EasyEdit/.venv/bin/python",
        "project/run_scripts/session04_ode_alloc_dry_plan.py",
        "dry", "--stage", "p0",
    ]
    first = _run(dry_command).stdout
    second = _run(dry_command).stdout
    if first != second:
        raise ODEAllocContractError("P0 dry plan is not byte-repeatable")

    receipts = {}
    for alias in MODEL_ALIASES:
        guard = P0ArtifactGuard(ARTIFACT_LOCK, alias)
        receipt = guard.preflight()
        assert_seal_source_current(seal, guard.dataset)
        guard.assert_unchanged()
        receipts[alias] = {
            "revision": receipt.revision,
            "hparams": receipt.hparams,
            "covariance": list(receipt.covariance),
            "source_root": receipt.easyedit_sources_root,
        }
    return {
        "source_head": source_head,
        "unit_test_count": unit_test_count,
        "dry_plan_sha256": hashlib.sha256(first.encode("utf-8")).hexdigest(),
        "numerical_lock_sha256": sha256_file(NUMERICAL_LOCK),
        "artifact_lock_sha256": sha256_file(ARTIFACT_LOCK),
        "r1_immutability_sha256": r1_immutability_sha,
        "artifacts": receipts,
    }


def _gpu_count(tres: str) -> int | None:
    matches = re.findall(r"gpu(?::[^,=():]+)?:([0-9]+)", tres)
    if matches:
        return sum(int(value) for value in matches)
    matches = re.findall(r"gpu[^,=]*=([0-9]+)", tres)
    if matches:
        return sum(int(value) for value in matches)
    return None


def _node_local_gpu_totals(records: list[dict[str, Any]]) -> tuple[int, int]:
    cluster = sum(int(record["gpus"]) for record in records)
    local = sum(
        int(record["gpus"])
        for record in records
        if CANONICAL_NODE in record["nodes"]
        or record.get("pending_unconstrained") is True
    )
    return local, cluster


def _expand_nodes(expression: str) -> tuple[str, ...]:
    if not expression or expression in {"(null)", "N/A"}:
        return ()
    result = _run(["scontrol", "show", "hostnames", expression])
    nodes = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not nodes:
        raise ODEAllocContractError("Slurm node expression did not expand")
    return nodes


def _node_capacity_gate() -> dict[str, Any]:
    if PAIR_HOST_MEMORY_MIB > 2 * MEMORY_CAP_MIB_PER_GPU:
        raise ODEAllocContractError("pair host-memory request exceeds the fixed cap")
    rows = _run(
        ["sinfo", "-N", "-h", "-n", CANONICAL_NODE, "-o", "%N|%T|%G|%m"]
    ).stdout.splitlines()
    if len(rows) != 1:
        raise ODEAllocContractError("canonical Slurm node inventory differs")
    fields = rows[0].split("|", 3)
    if len(fields) != 4 or fields[0] != CANONICAL_NODE:
        raise ODEAllocContractError("canonical Slurm node metadata differs")
    node, state, gres, memory = fields
    capacity_gpus = _gpu_count(gres)
    if (
        state.casefold().split("+", 1)[0] in {"down", "drain", "drained", "fail", "failing"}
        or capacity_gpus is None
        or capacity_gpus < 2
        or not memory.isdigit()
        or int(memory) < PAIR_HOST_MEMORY_MIB
    ):
        raise ODEAllocContractError("canonical Slurm node lacks pair capacity")
    return {
        "node": node,
        "state": state,
        "gres_gpus": capacity_gpus,
        "node_memory_mib": int(memory),
        "pair_memory_mib": PAIR_HOST_MEMORY_MIB,
        "pair_memory_cap_mib": 2 * MEMORY_CAP_MIB_PER_GPU,
    }


def _scheduler_gate() -> dict[str, Any]:
    queue = _run(
        ["squeue", "-h", "-u", os.environ.get("USER", "janghj"), "-t", "RUNNING,PENDING", "-o", "%i|%j|%T|%b|%N"]
    ).stdout.splitlines()
    records: list[dict[str, Any]] = []
    for line in queue:
        fields = line.split("|", 4)
        if len(fields) != 5:
            raise ODEAllocContractError("Slurm queue schema differs")
        job_id, name, state, tres, node_expression = fields
        if name in JOB_NAMES.values():
            raise ODEAllocContractError("exact P0 job name already exists")
        if not (name.startswith("odeedit_") or name.startswith("odealloc_")):
            continue
        count = _gpu_count(tres)
        if count is None:
            raise ODEAllocContractError("project Slurm job GPU request is unparseable")
        nodes = _expand_nodes(node_expression)
        pending_unconstrained = False
        if state.casefold() == "pending" and not nodes:
            job_record = _run(["scontrol", "show", "job", job_id, "--oneliner"]).stdout
            match = re.search(r"(?:^|\s)ReqNodeList=([^\s]+)", job_record)
            if match is None:
                raise ODEAllocContractError("pending project job lacks ReqNodeList metadata")
            requested = match.group(1)
            if requested == "(null)":
                pending_unconstrained = True
            else:
                nodes = _expand_nodes(requested)
        if state.casefold() == "running" and not nodes:
            raise ODEAllocContractError("running project job lacks allocated node metadata")
        records.append(
            {
                "job_id": job_id,
                "name": name,
                "state": state,
                "gpus": count,
                "nodes": nodes,
                "pending_unconstrained": pending_unconstrained,
            }
        )
    node_local, cluster = _node_local_gpu_totals(records)
    if node_local + 2 > GPU_CAP:
        raise ODEAllocContractError("server2-local project GPU cap would be exceeded")
    return {
        "canonical_node": CANONICAL_NODE,
        "server2_active_pending_project_gpus": node_local,
        "cluster_project_gpus_diagnostic_only": cluster,
        "requested_server2_gpus": 2,
        "cap": GPU_CAP,
        "records": records,
        "node_capacity": _node_capacity_gate(),
    }


def _namespace_gate(numerical_sha: str) -> tuple[dict[str, Path], dict[str, tuple[Path, Path]]]:
    local = REPO_ROOT / "local" / "odealloc"
    results = local / "results"
    logs = local / "logs"
    state = local / "state"
    for path in (local, results, logs, state):
        if path.is_symlink():
            raise ODEAllocContractError("P0 local namespace contains a symlink")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if glob.glob(str(logs / "s04-p0-native-identity-r2-*")):
        raise ODEAllocContractError("P0 log namespace already exists")
    roots: dict[str, Path] = {}
    log_paths: dict[str, tuple[Path, Path]] = {}
    for alias in MODEL_ALIASES:
        name = expected_result_name(alias, numerical_sha, R2_RUN_TOKEN)
        root = results / name
        if root.exists() or root.is_symlink():
            raise ODEAllocContractError("P0 result namespace already exists")
        roots[alias] = root
        prefix = logs / name
        log_paths[alias] = (
            Path(str(prefix) + "-%j.out"),
            Path(str(prefix) + "-%j.err"),
        )
    return roots, log_paths


def main() -> int:
    source_head = _source_gate()
    preflight = _cpu_static_gate(source_head)
    numerical_sha = preflight["numerical_lock_sha256"]
    roots, log_paths = _namespace_gate(numerical_sha)
    first_cap = _scheduler_gate()
    second_cap = _scheduler_gate()
    if (
        first_cap["server2_active_pending_project_gpus"]
        != second_cap["server2_active_pending_project_gpus"]
    ):
        raise ODEAllocContractError("server2-local GPU allocation changed during pre-submit gate")

    state = REPO_ROOT / "local" / "odealloc" / "state"
    marker = state / f"s04-p0-native-identity-r2-{numerical_sha[:8]}.submission-intent.json"
    marker_payload = {
        "schema_version": "ode-alloc-s04-p0-submission-intent-r2/v1",
        "instruction_id": INSTRUCTION_ID,
        "diagnostic_instruction_id": DIAGNOSTIC_INSTRUCTION_ID,
        "run_token": R2_RUN_TOKEN,
        "session_id": SESSION_ID,
        "source_head": source_head,
        "numerical_lock_sha256": numerical_sha,
        "models": list(MODEL_ALIASES),
        "job_names": JOB_NAMES,
        "roots": {alias: root.name for alias, root in roots.items()},
        "gpu_gate": second_cap,
        "attempt_policy": "each-exactly-once-no-retry",
        "created_unix": time.time(),
    }
    marker_sha = _write_once(marker, marker_payload)

    attempts: dict[str, dict[str, Any]] = {}
    for alias in MODEL_ALIASES:
        stdout_path, stderr_path = log_paths[alias]
        command = [
            "sbatch", "--parsable", "--chdir", str(REPO_ROOT),
            "--job-name", JOB_NAMES[alias],
            "--output", str(stdout_path), "--error", str(stderr_path),
            str(SBATCH), alias, str(roots[alias]), source_head, R2_RUN_TOKEN,
        ]
        result = _run(command, check=False)
        job_id = None
        if result.returncode == 0:
            candidate = result.stdout.strip().split(";", 1)[0]
            if candidate.isdigit():
                job_id = candidate
        attempts[alias] = {
            "attempted": True,
            "returncode": result.returncode,
            "job_id": job_id,
            "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
        }

    receipt = state / f"s04-p0-native-identity-r2-{numerical_sha[:8]}.submission-receipt.json"
    receipt_payload = {
        "schema_version": "ode-alloc-s04-p0-submission-receipt-r2/v1",
        "instruction_id": INSTRUCTION_ID,
        "diagnostic_instruction_id": DIAGNOSTIC_INSTRUCTION_ID,
        "run_token": R2_RUN_TOKEN,
        "source_head": source_head,
        "intent_sha256": marker_sha,
        "attempts": attempts,
        "completed_unix": time.time(),
    }
    receipt_sha = _write_once(receipt, receipt_payload)
    result = {
        "status": "SUBMITTED_PAIR" if all(item["job_id"] for item in attempts.values()) else "PARTIAL_SUBMISSION_FAIL_CLOSED",
        "source_head": source_head,
        "numerical_lock_sha256": numerical_sha,
        "jobs": {alias: item["job_id"] for alias, item in attempts.items()},
        "roots": {alias: str(root) for alias, root in roots.items()},
        "intent_sha256": marker_sha,
        "receipt_sha256": receipt_sha,
        "preflight_sha256": hashlib.sha256(canonical_json(preflight).encode("utf-8")).hexdigest(),
    }
    print(canonical_json(result), flush=True)
    return 0 if result["status"] == "SUBMITTED_PAIR" else 1


def _entrypoint() -> None:
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            canonical_json(
                {
                    "status": "PRE_SUBMIT_HOLD",
                    "error_type": type(exc).__name__,
                    "error_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    _entrypoint()
