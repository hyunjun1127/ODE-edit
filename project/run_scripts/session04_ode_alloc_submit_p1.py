#!/usr/bin/env python3
"""Fail-closed CPU/static/resource gate and one-shot P1 pair submission."""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_alloc.contracts import (
    MODEL_ALIASES,
    ODEAllocContractError,
    canonical_hash,
    canonical_json,
)
from project.run_scripts.ode_alloc.p0_artifacts import P0ArtifactGuard, sha256_file
from project.run_scripts.ode_alloc.p1_contracts import (
    P1_RUN_TOKEN,
    SESSION_ID,
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
from project.run_scripts.ode_alloc.p1_runtime import (
    ARTIFACT_LOCK_SHA256,
    NUMERICAL_LOCK_SHA256,
    P1_EXECUTION_TOKEN,
    P1_REPAIR_EXPECTED_BASE,
    P1_REPAIR_INSTRUCTION_ID,
    SEAL_FILE_SHA256,
    expected_p1r1_result_name,
)
from project.run_scripts.ode_alloc.selection import (
    assert_seal_source_current,
    load_and_verify_seal_candidate,
)


PACKAGE_ROOT = REPO_ROOT / "project" / "run_scripts" / "ode_alloc"
NUMERICAL_LOCK = PACKAGE_ROOT / "numerical_lock_proposal.json"
ARTIFACT_LOCK = PACKAGE_ROOT / "p0_artifact_lock_r1.json"
SEAL = PACKAGE_ROOT / "split_anchor_seal_candidate.json"
DRY_PLAN = REPO_ROOT / "project" / "run_scripts" / "session04_ode_alloc_p1_dry_plan.py"
ENTRY = REPO_ROOT / "project" / "run_scripts" / "session04_ode_alloc_p1.py"
SBATCH = REPO_ROOT / "project" / "run_scripts" / "session04_ode_alloc_p1.sbatch"
JOB_NAMES = {
    "llama3-8b-inst": "odealloc_s04_p1r1_llama",
    "qwen2.5-7b-inst": "odealloc_s04_p1r1_qwen",
}
CANONICAL_NODE = "server2"
GPU_CAP = 3
MEMORY_CAP_MIB_PER_GPU = 66017
PAIR_HOST_MEMORY_MIB = 130000
P1_CHANGED_PATHS = frozenset(
    {
        "project/run_scripts/ode_alloc/p1_runtime.py",
        "project/run_scripts/ode_alloc/tests/test_p1_runtime_contracts.py",
        "project/run_scripts/ode_alloc/tests/test_p1_submission_contracts.py",
        "project/run_scripts/session04_ode_alloc_p1.py",
        "project/run_scripts/session04_ode_alloc_p1.sbatch",
        "project/run_scripts/session04_ode_alloc_submit_p1.py",
    }
)
P1_FROZEN_SOURCE_FILES = {
    "project/run_scripts/ode_alloc/p1_contracts.py": "61a55e680aa9fc41c058407ad28ccc8d614207526fee79f1bd0df22413c12e06",
    "project/run_scripts/ode_alloc/p1_evaluator.py": "e3f4b8923d6f298b6f0f07a797705757cc14ff443e208371371b549a79aba86f",
    "project/run_scripts/ode_alloc/p1_firewall.py": "9038fc146062ebb9dee07cf9083b00e2938f22ff5a02033b9013248d2f5ea30c",
    "project/run_scripts/ode_alloc/p1_scoring.py": "1ec74ed521301ccf0a1b8627dbe3726949cbd8d82dc2dc2a28297ea33b9c3ca4",
}
P1_R0_RUNTIME_SHA256 = "d6e271946c999d3226568c997be7783f90ce017a7f475b120bbc6b520265fa43"
P1_R1_RUNTIME_SHA256 = "b0a742ea3193ae7ea4ffc5dfaf02aaf3be22c42970c24f58c4deba6c126a3595"
P0_TERMINAL_FILES = {
    "local/odealloc/results/s04-p0-native-identity-r3-llama3-8b-inst-905a3bbc/manifest.json": "049dae5b7c23640a0fa1e6050bcc5574ead5207f459cbe69a6f7e623ccee2139",
    "local/odealloc/results/s04-p0-native-identity-r3-llama3-8b-inst-905a3bbc/summary.json": "3e34201e8ecdd29c2fee0f81553ca1d1ce97494b351599ee8969f948bfc15632",
    "local/odealloc/results/s04-p0-native-identity-r3-llama3-8b-inst-905a3bbc/terminal.json": "a69375e2a5b006275d9f6b5a9d3420c28f82f8c5f5c1c8bbd468eb5037483aeb",
    "local/odealloc/results/s04-p0-native-identity-r3-qwen2.5-7b-inst-905a3bbc/manifest.json": "4012cc844d83241476ae8babcc14bc62368639410ede6ba6d8ef2bf6be97ad0b",
    "local/odealloc/results/s04-p0-native-identity-r3-qwen2.5-7b-inst-905a3bbc/summary.json": "5fea51e35d876cb8f7418881032106b517bf547875242161cd50da17028b3e40",
    "local/odealloc/results/s04-p0-native-identity-r3-qwen2.5-7b-inst-905a3bbc/terminal.json": "b9fdc550ea60a65a3c7365182439fc27216907c4b5521575c822db75a37884ab",
}
P1_R0_IMMUTABLE_FILES = {
    "local/odealloc/results/s04-p1-matched-llama3-8b-inst-905a3bbc/failure.json": "ac9833d1174f9eb88d024c0754656f064502b837cbeca09972747c2a61eff66e",
    "local/odealloc/results/s04-p1-matched-qwen2.5-7b-inst-905a3bbc/failure.json": "ac9833d1174f9eb88d024c0754656f064502b837cbeca09972747c2a61eff66e",
    "local/odealloc/logs/s04-p1-matched-llama3-8b-inst-905a3bbc-16359.out": "2e5292000e3663a5260a3b7f57557ca1da7780de833f6acdc26930241ce3cf9c",
    "local/odealloc/logs/s04-p1-matched-llama3-8b-inst-905a3bbc-16359.err": "60553d126f67a4828e58224ee6fdf757c50514e3ae257bab5b1910fb48ecd860",
    "local/odealloc/logs/s04-p1-matched-qwen2.5-7b-inst-905a3bbc-16360.out": "2e5292000e3663a5260a3b7f57557ca1da7780de833f6acdc26930241ce3cf9c",
    "local/odealloc/logs/s04-p1-matched-qwen2.5-7b-inst-905a3bbc-16360.err": "c8b465de88a32ad6b41d30f892f9b6f2b22eaedd20fd21f7e87b038a2781d213",
    "local/odealloc/state/s04-p1-matched-905a3bbc.submission-intent.json": "c0be4684656e5c750154e5b794149a977175919c70d4d85ebb885deb0bab71bf",
    "local/odealloc/state/s04-p1-matched-905a3bbc.submission-receipt.json": "9577b89d649a61488150a69061c2b156536ab2b09c2bd5cabdf9ec7f59b8f257",
}


def _run(
    command: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise ODEAllocContractError(
            "P1 pre-submit command failed: "
            + hashlib.sha256(result.stderr.encode("utf-8")).hexdigest()
        )
    return result


def _canonical_write_once(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("P1 submission state is create-once")
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise
    return hashlib.sha256(encoded).hexdigest()


def _source_gate() -> str:
    head = _run(("git", "rev-parse", "HEAD")).stdout.strip()
    parent = _run(("git", "rev-parse", "HEAD^")).stdout.strip()
    if parent != P1_REPAIR_EXPECTED_BASE or head == P1_REPAIR_EXPECTED_BASE:
        raise ODEAllocContractError("P1R1 checkpoint ancestry differs")
    if _run(("git", "status", "--porcelain", "--untracked-files=no")).stdout:
        raise ODEAllocContractError("tracked P1 source is not frozen clean")
    untracked = _run(("git", "ls-files", "--others", "--exclude-standard")).stdout.splitlines()
    if any(
        not (path.startswith("agents/server2/") or path.startswith("audits/servers/server2/"))
        for path in untracked
    ):
        raise ODEAllocContractError("foreign untracked path overlaps P1 worktree")
    changed = set(
        _run(("git", "diff", "--name-only", f"{P1_REPAIR_EXPECTED_BASE}..{head}"))
        .stdout.splitlines()
    )
    if changed != P1_CHANGED_PATHS:
        raise ODEAllocContractError("P1R1 checkpoint differs from exact allowed paths")
    if any(
        token in path.casefold()
        for path in changed
        for token in ("session03", "session_03", "knowledge-revision")
    ):
        raise ODEAllocContractError("P1 checkpoint overlaps a foreign namespace")
    _run(("git", "diff", "--check", f"{P1_REPAIR_EXPECTED_BASE}..{head}"))
    for relative, expected in sorted(P1_FROZEN_SOURCE_FILES.items()):
        if sha256_file(REPO_ROOT / relative) != expected:
            raise ODEAllocContractError("P1 scientific source digest differs")
    runtime = PACKAGE_ROOT / "p1_runtime.py"
    if sha256_file(runtime) != P1_R1_RUNTIME_SHA256:
        raise ODEAllocContractError("P1R1 approved runtime plumbing digest differs")
    base_runtime = _run(
        (
            "git",
            "show",
            f"{P1_REPAIR_EXPECTED_BASE}:project/run_scripts/ode_alloc/p1_runtime.py",
        )
    ).stdout
    if hashlib.sha256(base_runtime.encode("utf-8")).hexdigest() != P1_R0_RUNTIME_SHA256:
        raise ODEAllocContractError("P1 pre-repair runtime digest differs")
    return head


def _p0_terminal_immutability_gate() -> str:
    for relative, expected in sorted(P0_TERMINAL_FILES.items()):
        path = REPO_ROOT / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ODEAllocContractError("P0 terminal artifact digest differs")
    return canonical_hash(P0_TERMINAL_FILES)


def _p1_r0_immutability_gate() -> str:
    for relative, expected in sorted(P1_R0_IMMUTABLE_FILES.items()):
        path = REPO_ROOT / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ODEAllocContractError("P1 R0 failure artifact digest differs")
    for alias in MODEL_ALIASES:
        root = (
            REPO_ROOT
            / "local"
            / "odealloc"
            / "results"
            / expected_p1_result_name(alias, NUMERICAL_LOCK_SHA256, P1_RUN_TOKEN)
        )
        observed = tuple(
            sorted(
                str(path.relative_to(root))
                for path in root.rglob("*")
                if path.is_file()
            )
        )
        if observed != ("failure.json",):
            raise ODEAllocContractError("P1 R0 failure root contents differ")
    return canonical_hash(P1_R0_IMMUTABLE_FILES)


def _cpu_static_gate(source_head: str) -> dict[str, Any]:
    _run(("scripts/check-session-boundary.sh", SESSION_ID))
    _run(("scripts/check-agent-access.sh", "--all-changed"))
    assert_p1_inner_firewall(
        (PACKAGE_ROOT / "p1_runtime.py", PACKAGE_ROOT / "p1_contracts.py")
    )
    assert_p1_evaluator_firewall(PACKAGE_ROOT / "p1_evaluator.py")
    assert_no_alias_specific_scientific_branch(PACKAGE_ROOT / "p1_runtime.py")
    assert_projector_is_only_adaptive_arm_branch(PACKAGE_ROOT / "p1_runtime.py")
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
    for path in (DRY_PLAN, ENTRY, Path(__file__)):
        compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
    _run(("bash", "-n", str(SBATCH)))
    if (
        sha256_file(NUMERICAL_LOCK) != NUMERICAL_LOCK_SHA256
        or sha256_file(ARTIFACT_LOCK) != ARTIFACT_LOCK_SHA256
        or sha256_file(SEAL) != SEAL_FILE_SHA256
    ):
        raise ODEAllocContractError("P1 lock/seal digest differs")
    lock = json.loads(NUMERICAL_LOCK.read_text(encoding="utf-8"))
    policy = P1Policy.from_lock(lock)
    seal = load_and_verify_seal_candidate(SEAL)
    assert_p1_seal(seal)
    tests = _run(
        (
            "/mnt/raid5/janghj/EasyEdit/.venv/bin/python",
            "-W",
            "error",
            "-m",
            "unittest",
            "discover",
            "-s",
            "project/run_scripts/ode_alloc/tests",
            "-p",
            "test_*.py",
            "-v",
        )
    )
    match = re.search(r"Ran ([0-9]+) tests?", tests.stderr)
    if match is None or int(match.group(1)) < 78 or "\nOK\n" not in tests.stderr:
        raise ODEAllocContractError("P1 warnings-as-errors CPU suite differs")
    first = _run(
        (
            "/mnt/raid5/janghj/EasyEdit/.venv/bin/python",
            str(DRY_PLAN.relative_to(REPO_ROOT)),
        )
    ).stdout
    second = _run(
        (
            "/mnt/raid5/janghj/EasyEdit/.venv/bin/python",
            str(DRY_PLAN.relative_to(REPO_ROOT)),
        )
    ).stdout
    if first != second:
        raise ODEAllocContractError("P1 dry plan is not byte-repeatable")
    artifacts: dict[str, Any] = {}
    for alias in MODEL_ALIASES:
        guard = P0ArtifactGuard(ARTIFACT_LOCK, alias)
        receipt = guard.preflight()
        assert_seal_source_current(seal, guard.dataset)
        guard.assert_unchanged()
        artifacts[alias] = {
            "revision": receipt.revision,
            "dataset_sha256": receipt.counterfact,
            "hparams_sha256": receipt.hparams,
            "source_root_sha256": receipt.easyedit_sources_root,
        }
    return {
        "source_head": source_head,
        "unit_test_count": int(match.group(1)),
        "dry_plan_sha256": hashlib.sha256(first.encode("utf-8")).hexdigest(),
        "policy_id": policy.policy_id,
        "budget_id": policy.solver_budget.identity(),
        "p0_terminal_immutability_sha256": _p0_terminal_immutability_gate(),
        "p1_r0_immutability_sha256": _p1_r0_immutability_gate(),
        "artifacts": artifacts,
    }


def _gpu_count(tres: str) -> int | None:
    matches = re.findall(r"gpu(?::[^,=():]+)?:([0-9]+)", tres)
    if matches:
        return sum(int(value) for value in matches)
    matches = re.findall(r"gpu[^,=]*=([0-9]+)", tres)
    if matches:
        return sum(int(value) for value in matches)
    return None


def _expand_nodes(expression: str) -> tuple[str, ...]:
    if not expression or expression in {"(null)", "N/A"}:
        return ()
    result = _run(("scontrol", "show", "hostnames", expression))
    nodes = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not nodes:
        raise ODEAllocContractError("P1 Slurm node expression did not expand")
    return nodes


def _node_capacity_gate() -> dict[str, Any]:
    if PAIR_HOST_MEMORY_MIB > 2 * MEMORY_CAP_MIB_PER_GPU:
        raise ODEAllocContractError("P1 pair host-memory exceeds cap")
    rows = _run(
        ("sinfo", "-N", "-h", "-n", CANONICAL_NODE, "-o", "%N|%T|%G|%m")
    ).stdout.splitlines()
    if len(rows) != 1:
        raise ODEAllocContractError("P1 canonical node inventory differs")
    fields = rows[0].split("|", 3)
    if len(fields) != 4 or fields[0] != CANONICAL_NODE:
        raise ODEAllocContractError("P1 canonical node metadata differs")
    node, state, gres, memory = fields
    capacity = _gpu_count(gres)
    if (
        state.casefold().split("+", 1)[0]
        in {"down", "drain", "drained", "fail", "failing"}
        or capacity is None
        or capacity < 2
        or not memory.isdigit()
        or int(memory) < PAIR_HOST_MEMORY_MIB
    ):
        raise ODEAllocContractError("P1 canonical node lacks pair capacity")
    return {
        "node": node,
        "state": state,
        "gres_gpus": capacity,
        "node_memory_mib": int(memory),
    }


def _scheduler_gate() -> dict[str, Any]:
    rows = _run(
        (
            "squeue",
            "-h",
            "-u",
            os.environ.get("USER", "janghj"),
            "-t",
            "RUNNING,PENDING",
            "-o",
            "%i|%j|%T|%b|%N",
        )
    ).stdout.splitlines()
    records: list[dict[str, Any]] = []
    for line in rows:
        fields = line.split("|", 4)
        if len(fields) != 5:
            raise ODEAllocContractError("P1 Slurm queue schema differs")
        job_id, name, state, tres, expression = fields
        if name in JOB_NAMES.values():
            raise ODEAllocContractError("exact P1 job name already exists")
        if not (name.startswith("odeedit_") or name.startswith("odealloc_")):
            continue
        count = _gpu_count(tres)
        if count is None:
            raise ODEAllocContractError("project Slurm GPU request is unparseable")
        nodes = _expand_nodes(expression)
        pending_unconstrained = False
        if state.casefold() == "pending" and not nodes:
            metadata = _run(("scontrol", "show", "job", job_id, "--oneliner")).stdout
            match = re.search(r"(?:^|\s)ReqNodeList=([^\s]+)", metadata)
            if match is None:
                raise ODEAllocContractError("pending project job lacks node metadata")
            requested = match.group(1)
            if requested == "(null)":
                pending_unconstrained = True
            else:
                nodes = _expand_nodes(requested)
        if state.casefold() == "running" and not nodes:
            raise ODEAllocContractError("running project job lacks node metadata")
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
    local = sum(
        int(record["gpus"])
        for record in records
        if CANONICAL_NODE in record["nodes"]
        or record["pending_unconstrained"] is True
    )
    cluster = sum(int(record["gpus"]) for record in records)
    if local + 2 > GPU_CAP:
        raise ODEAllocContractError("server2-local P1 GPU cap would be exceeded")
    return {
        "canonical_node": CANONICAL_NODE,
        "server2_active_pending_project_gpus": local,
        "cluster_project_gpus_diagnostic_only": cluster,
        "requested_server2_gpus": 2,
        "cap": GPU_CAP,
        "records": records,
        "node_capacity": _node_capacity_gate(),
    }


def _namespace_gate() -> tuple[dict[str, Path], dict[str, tuple[Path, Path]]]:
    local = REPO_ROOT / "local" / "odealloc"
    results = local / "results"
    logs = local / "logs"
    state = local / "state"
    for path in (local, results, logs, state):
        if path.is_symlink():
            raise ODEAllocContractError("P1 local namespace contains a symlink")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if glob.glob(str(logs / "s04-p1r1-matched-*")):
        raise ODEAllocContractError("P1R1 log namespace already exists")
    roots: dict[str, Path] = {}
    paths: dict[str, tuple[Path, Path]] = {}
    for alias in MODEL_ALIASES:
        name = expected_p1r1_result_name(
            alias, NUMERICAL_LOCK_SHA256, P1_EXECUTION_TOKEN
        )
        root = results / name
        if root.exists() or root.is_symlink():
            raise ODEAllocContractError("P1 result namespace already exists")
        roots[alias] = root
        prefix = logs / name
        paths[alias] = (Path(str(prefix) + "-%j.out"), Path(str(prefix) + "-%j.err"))
    return roots, paths


def main() -> int:
    source_head = _source_gate()
    preflight = _cpu_static_gate(source_head)
    first_cap = _scheduler_gate()
    roots, log_paths = _namespace_gate()
    second_cap = _scheduler_gate()
    if (
        first_cap["server2_active_pending_project_gpus"]
        != second_cap["server2_active_pending_project_gpus"]
        or second_cap["server2_active_pending_project_gpus"] != 0
    ):
        raise ODEAllocContractError("server2 P1 GPU allocation changed before submit")
    state = REPO_ROOT / "local" / "odealloc" / "state"
    intent = state / f"s04-p1r1-matched-{NUMERICAL_LOCK_SHA256[:8]}.submission-intent.json"
    intent_payload = {
        "schema_version": "ode-alloc-s04-p1r1-submission-intent/v1",
        "instruction_id": P1_REPAIR_INSTRUCTION_ID,
        "run_token": P1_RUN_TOKEN,
        "execution_token": P1_EXECUTION_TOKEN,
        "session_id": SESSION_ID,
        "source_head": source_head,
        "numerical_lock_sha256": NUMERICAL_LOCK_SHA256,
        "preflight": preflight,
        "scheduler": second_cap,
        "roots": {alias: str(path) for alias, path in roots.items()},
        "job_names": JOB_NAMES,
        "retry_allowed": False,
    }
    intent_sha = _canonical_write_once(intent, intent_payload)
    attempts: dict[str, Any] = {}
    jobs: dict[str, str] = {}
    for alias in MODEL_ALIASES:
        stdout_path, stderr_path = log_paths[alias]
        command = (
            "sbatch",
            "--parsable",
            "--chdir",
            str(REPO_ROOT),
            "--job-name",
            JOB_NAMES[alias],
            "--output",
            str(stdout_path),
            "--error",
            str(stderr_path),
            str(SBATCH),
            alias,
            str(roots[alias]),
            source_head,
            P1_RUN_TOKEN,
            P1_EXECUTION_TOKEN,
        )
        result = _run(command, check=False)
        job_id = result.stdout.strip().split(";", 1)[0] if result.returncode == 0 else ""
        if job_id and not job_id.isdigit():
            raise ODEAllocContractError("P1 sbatch job identity is invalid")
        attempts[alias] = {
            "returncode": result.returncode,
            "job_id": job_id or None,
            "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
        }
        if job_id:
            jobs[alias] = job_id
    receipt = state / f"s04-p1r1-matched-{NUMERICAL_LOCK_SHA256[:8]}.submission-receipt.json"
    receipt_payload = {
        "schema_version": "ode-alloc-s04-p1r1-submission-receipt/v1",
        "instruction_id": P1_REPAIR_INSTRUCTION_ID,
        "run_token": P1_RUN_TOKEN,
        "execution_token": P1_EXECUTION_TOKEN,
        "source_head": source_head,
        "intent_sha256": intent_sha,
        "attempts": attempts,
        "job_ids": jobs,
        "retry_allowed": False,
    }
    receipt_sha = _canonical_write_once(receipt, receipt_payload)
    if len(jobs) != len(MODEL_ALIASES):
        raise ODEAllocContractError("P1 exact pair submission was incomplete")
    result = {
        "status": "SUBMITTED_PAIR",
        "source_head": source_head,
        "instruction_id": P1_REPAIR_INSTRUCTION_ID,
        "execution_token": P1_EXECUTION_TOKEN,
        "jobs": jobs,
        "roots": {alias: str(path) for alias, path in roots.items()},
        "numerical_lock_sha256": NUMERICAL_LOCK_SHA256,
        "preflight_sha256": canonical_hash(preflight),
        "intent_sha256": intent_sha,
        "receipt_sha256": receipt_sha,
    }
    print(canonical_json(result))
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except Exception as exc:
        print(
            canonical_json(
                {
                    "status": "PRE_SUBMIT_HOLD",
                    "error_type": type(exc).__name__,
                    "error_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                }
            ),
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
