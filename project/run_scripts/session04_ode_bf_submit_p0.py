#!/usr/bin/env python3
"""One-shot fail-closed submission of the exact ODE-BF technical P0 pair."""

from __future__ import annotations

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

from project.run_scripts.ode_bf.artifacts import (
    ODEBFArtifactGuard,
    load_rooted_json,
    sha256_file,
)
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.firewall import (
    assert_ast_firewall,
    assert_no_alias_specific_controller_branch,
)
from project.run_scripts.ode_bf.p0_runtime import expected_result_name
from project.run_scripts.ode_bf.resource import (
    PROJECT_JOB_PREFIXES,
    SchedulerJob,
    assert_node_local_capacity,
    assert_pair_capacity,
    forecast_p0_b10_memory,
    gpu_count_from_tres,
)
from project.run_scripts.session04_ode_bf_dry_plan import JOB_NAMES
from project.run_scripts.session04_ode_bf_p0 import RUN_TOKEN


SESSION_ID = "019fc5ec-f85b-7770-a73a-1d19be1cd491"
BRANCH = "codex/odeeditsh2-ode-bf-v1"
BASE_HEAD = "4ab7e920c5c71f96780a0cca25246d771972173f"
PACKAGE_ROOT = REPO_ROOT / "project/run_scripts/ode_bf"
LOCK_ROOT = PACKAGE_ROOT / "locks"
SBATCH = REPO_ROOT / "project/run_scripts/session04_ode_bf_p0.sbatch"
DRY_PLAN = REPO_ROOT / "project/run_scripts/session04_ode_bf_dry_plan.py"
SOURCE_MANIFEST = LOCK_ROOT / "source_manifest.json"
ARTIFACT_LOCK = LOCK_ROOT / "p0_artifact_lock.json"
BASE_ARTIFACT_LOCK = REPO_ROOT / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
LOCK_SHA256 = {
    "p0_artifact_lock.json": "623482b05c670b703eebc65493779e863021509f0c0855d5fafb75c0b7b24910",
    "numerical_lock_p0.json": "23fe5621612f715c52ef70f94a10ae7eaba759b4ac209e0e0f3e09c81ea9feef",
    "p0_b10_seal.json": "7f34b2f6206f5f4f113037d22fbd93ed494ce81638d45b6c1b879beeed9447d7",
    "p0_cpu_sampling_seal.json": "7c1eb44bc6ba01cde6794d3c53767abf850862c620494f6a8c9502876b3f3fc9",
}
PRESERVED_REPAIR_SHA256 = {
    "project/run_scripts/ode_alloc/p1_runtime.py": "b0a742ea3193ae7ea4ffc5dfaf02aaf3be22c42970c24f58c4deba6c126a3595",
    "project/run_scripts/ode_alloc/tests/test_p1_runtime_contracts.py": "d8feb6361791c30abe7c2393b2776a8955a1214d10bab50ae89b03b1c261e79b",
    "project/run_scripts/ode_alloc/tests/test_p1_submission_contracts.py": "fdf52d153e33da915c04db6778cdba2cc550cc3d4358ff2416141812671d5c5a",
    "project/run_scripts/session04_ode_alloc_p1.py": "3c3ad7c51f9db7b15156fbec84a3d0933c4d18f64d96182928ae43e7aec5d74d",
    "project/run_scripts/session04_ode_alloc_p1.sbatch": "8e32dc6b1d13b558eba8475e84edac769f6dd1da0519a6c0a68b9a4d886a1dd7",
    "project/run_scripts/session04_ode_alloc_submit_p1.py": "f00f80f0cc2b2aae99fe348df43fbe207de76e939edc24261f772e1bb52c39c4",
}


def _run(
    args: Sequence[str],
    *,
    check: bool = True,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _write_once(path: Path, value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _source_manifest_gate() -> str:
    value, raw_sha = load_rooted_json(
        SOURCE_MANIFEST,
        expected_schema="ode-edit-s04-ode-bf-source-manifest/v1",
    )
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ODEBFContractError("ODE-BF source manifest is empty")
    locked = {
        entry["path"]: (entry["sha256"], int(entry["size"]))
        for entry in entries
    }
    if len(locked) != len(entries):
        raise ODEBFContractError("ODE-BF source manifest repeats a path")
    observed = _run(["git", "ls-tree", "-r", "--name-only", "HEAD"]).stdout.splitlines()
    expected_paths = {
        path
        for path in observed
        if (
            path.startswith("project/run_scripts/ode_bf/")
            or path.startswith("project/run_scripts/session04_ode_bf_")
        )
        and path != "project/run_scripts/ode_bf/locks/source_manifest.json"
    }
    if set(locked) != expected_paths:
        raise ODEBFContractError("ODE-BF source manifest path set differs")
    for relative, (expected, expected_size) in sorted(locked.items()):
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected
        ):
            raise ODEBFContractError("ODE-BF source manifest content differs")
    return raw_sha


def _source_gate() -> str:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    if parent != BASE_HEAD or head == BASE_HEAD or branch != BRANCH:
        raise ODEBFContractError("ODE-BF checkpoint ancestry/branch differs")
    if _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout:
        raise ODEBFContractError("tracked ODE-BF source is not clean")
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"]).stdout.splitlines()
    if any(
        not (
            path.startswith("agents/server2/")
            or path.startswith("audits/servers/server2/")
        )
        for path in untracked
    ):
        raise ODEBFContractError("foreign untracked path overlaps the ODE-BF worktree")
    changed = set(
        _run(["git", "diff", "--name-only", f"{BASE_HEAD}..{head}"]).stdout.splitlines()
    )
    repair_paths = set(PRESERVED_REPAIR_SHA256)
    if not repair_paths.issubset(changed):
        raise ODEBFContractError("preserved six-path CUDA repair is incomplete")
    for path in changed - repair_paths:
        if not (
            path.startswith("project/run_scripts/ode_bf/")
            or path.startswith("project/run_scripts/session04_ode_bf_")
        ):
            raise ODEBFContractError("checkpoint changed a non-approved path")
    if any("session03" in path.casefold() or "knowledge-revision" in path.casefold() for path in changed):
        raise ODEBFContractError("checkpoint overlaps a foreign namespace")
    for relative, expected in PRESERVED_REPAIR_SHA256.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise ODEBFContractError("preserved CUDA repair byte identity differs")
    _run(["git", "diff", "--check", f"{BASE_HEAD}..{head}"])
    _source_manifest_gate()
    return head


def _cpu_static_gate(source_head: str) -> dict[str, Any]:
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    _run(["scripts/check-agent-access.sh", "--all-changed"])
    for name, expected in LOCK_SHA256.items():
        if sha256_file(LOCK_ROOT / name) != expected:
            raise ODEBFContractError("ODE-BF reviewed lock digest differs")
    numerical, numerical_sha = load_rooted_json(
        LOCK_ROOT / "numerical_lock_p0.json",
        expected_schema="ode-edit-s04-ode-bf-p0-numerical-lock/v1",
    )
    if (
        numerical["edit_batch_size"] != 10
        or numerical["rollout"]["k_resolution"] != 8
        or numerical["rollout"]["correction_cycles"] != 1
        or numerical["p0"]["direct_z_initializations"] != 10
    ):
        raise ODEBFContractError("ODE-BF numerical B10/K8 lock differs")

    scientific = [
        path
        for path in PACKAGE_ROOT.glob("*.py")
        if path.name != "__init__.py"
    ]
    assert_ast_firewall(scientific + [REPO_ROOT / "project/run_scripts/session04_ode_bf_p0.py"])
    assert_no_alias_specific_controller_branch(
        [
            PACKAGE_ROOT / name
            for name in (
                "first_hit.py",
                "history.py",
                "sampling.py",
                "routing.py",
                "barriers.py",
            )
        ]
    )
    for path in sorted(PACKAGE_ROOT.rglob("*.py")) + [
        REPO_ROOT / "project/run_scripts/session04_ode_bf_p0.py",
        REPO_ROOT / "project/run_scripts/session04_ode_bf_dry_plan.py",
        Path(__file__),
    ]:
        compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
    _run(["bash", "-n", str(SBATCH)])

    runtime = "/mnt/raid5/janghj/EasyEdit/.venv/bin/python"
    test_env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{REPO_ROOT}:/mnt/raid5/janghj/EasyEdit",
        "MPLCONFIGDIR": str(REPO_ROOT / "local/odebf/matplotlib-cpu-gate"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    tests = _run(
        [
            runtime,
            "-W",
            "error",
            "-m",
            "unittest",
            "discover",
            "-s",
            "project/run_scripts/ode_bf/tests",
            "-p",
            "test_*.py",
        ],
        env=test_env,
    )
    count_match = re.search(r"Ran ([0-9]+) tests?", tests.stderr)
    if count_match is None or "\nOK\n" not in tests.stderr:
        raise ODEBFContractError("full ODE-BF warnings-as-errors CPU suite did not pass")
    repair_tests = _run(
        [
            runtime,
            "-W",
            "error",
            "-m",
            "unittest",
            "project.run_scripts.ode_alloc.tests.test_p1_runtime_contracts",
            "project.run_scripts.ode_alloc.tests.test_p1_submission_contracts",
        ],
        env=test_env,
    )
    if "\nOK\n" not in repair_tests.stderr:
        raise ODEBFContractError("preserved CUDA repair focused tests did not pass")

    dry_args = [runtime, str(DRY_PLAN), "--source-head", source_head]
    dry_first = _run(dry_args, env=test_env).stdout
    dry_second = _run(dry_args, env=test_env).stdout
    if dry_first != dry_second:
        raise ODEBFContractError("ODE-BF P0 dry plan is not byte-repeatable")

    artifact_receipts: dict[str, Any] = {}
    forecasts: dict[str, Any] = {}
    for alias in MODEL_ALIASES:
        guard = ODEBFArtifactGuard(REPO_ROOT, ARTIFACT_LOCK, alias)
        receipt = guard.preflight()
        guard.assert_unchanged()
        forecast = forecast_p0_b10_memory(ARTIFACT_LOCK, BASE_ARTIFACT_LOCK, alias)
        artifact_receipts[alias] = {
            "lock_sha256": receipt.lock_sha256,
            "revision": receipt.base_model_revision,
            "projector_sha256": receipt.projector_sha256,
            "held_ode_alloc_tree_sha256": receipt.held_ode_alloc_tree_sha256,
        }
        forecasts[alias] = {
            "identity": forecast.identity(),
            "gpu_peak_mib": forecast.forecast_gpu_peak_mib,
            "host_peak_mib": forecast.forecast_host_peak_mib,
        }
    return {
        "source_head": source_head,
        "test_count": int(count_match.group(1)),
        "numerical_lock_sha256": numerical_sha,
        "source_manifest_sha256": _source_manifest_gate(),
        "dry_plan_sha256": hashlib.sha256(dry_first.encode("utf-8")).hexdigest(),
        "artifacts": artifact_receipts,
        "memory_forecasts": forecasts,
    }


def _expand_nodes(expression: str) -> tuple[str, ...]:
    if expression in ("", "(null)", "N/A", "None assigned"):
        return ()
    result = _run(["scontrol", "show", "hostnames", expression])
    nodes = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not nodes:
        raise ODEBFContractError("scheduler node expression did not expand")
    return nodes


def _job_field(value: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(\S+)", value)
    return "" if match is None else match.group(1)


def _scheduler_jobs() -> tuple[SchedulerJob, ...]:
    user = os.environ.get("USER")
    if not user:
        raise ODEBFContractError("scheduler user identity is unavailable")
    queue = _run(
        ["squeue", "-h", "-u", user, "-t", "R,PD", "-o", "%i|%j|%T|%b"]
    )
    records: list[SchedulerJob] = []
    for line in queue.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ODEBFContractError("scheduler queue record schema differs")
        job_id, job_name, state, tres = (field.strip() for field in fields)
        if not job_name.startswith(PROJECT_JOB_PREFIXES) and job_name not in JOB_NAMES.values():
            continue
        gpu_count = gpu_count_from_tres(tres)
        if gpu_count is None or gpu_count <= 0:
            raise ODEBFContractError("project scheduler GPU TRES is ambiguous")
        detail = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
        allocated = _expand_nodes(_job_field(detail, "NodeList"))
        requested = _expand_nodes(_job_field(detail, "ReqNodeList"))
        records.append(
            SchedulerJob(
                job_id,
                job_name,
                state,
                gpu_count,
                allocated,
                requested,
            )
        )
    return tuple(records)


def _output_gate() -> dict[str, Path]:
    result_parent = REPO_ROOT / "local/odebf/results"
    log_parent = REPO_ROOT / "local/odebf/logs"
    state_parent = REPO_ROOT / "local/odebf/state"
    for parent in (result_parent, log_parent, state_parent):
        if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
            raise ODEBFContractError("ODE-BF local output parent is not a real directory")
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    roots = {
        alias: result_parent / expected_result_name(alias)
        for alias in MODEL_ALIASES
    }
    if any(path.exists() or path.is_symlink() for path in roots.values()):
        raise ODEBFContractError("ODE-BF P0 result root already exists")
    for job_name in JOB_NAMES.values():
        if list(log_parent.glob(f"{job_name}-*")):
            raise ODEBFContractError("ODE-BF P0 log namespace already exists")
    intent = state_parent / "s04-p0-native-wb-b10-23fe5621.submission-intent.json"
    receipt = state_parent / "s04-p0-native-wb-b10-23fe5621.submission-receipt.json"
    if intent.exists() or intent.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise ODEBFContractError("ODE-BF P0 pair was already attempted")
    roots["__intent__"] = intent
    roots["__receipt__"] = receipt
    roots["__logs__"] = log_parent
    return roots


def _submit_one(
    *,
    alias: str,
    output_root: Path,
    source_head: str,
    log_parent: Path,
) -> str:
    job_name = JOB_NAMES[alias]
    result = _run(
        [
            "sbatch",
            "--parsable",
            "--job-name",
            job_name,
            "--output",
            str(log_parent / f"{job_name}-%j.out"),
            "--error",
            str(log_parent / f"{job_name}-%j.err"),
            str(SBATCH),
            alias,
            str(output_root),
            source_head,
            RUN_TOKEN,
        ]
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("sbatch did not return a numeric job ID")
    return job_id


def main() -> int:
    source_head = _source_gate()
    gate = _cpu_static_gate(source_head)
    paths = _output_gate()
    before = _scheduler_jobs()
    if any(record.job_name in JOB_NAMES.values() for record in before):
        raise ODEBFContractError("ODE-BF P0 scheduler job name already exists")
    local_before, cluster_before = assert_pair_capacity(before)
    intent = {
        "schema": "ode-edit-s04-ode-bf-p0-submission-intent/v1",
        "source_head": source_head,
        "jobs": JOB_NAMES,
        "results": {
            alias: str(paths[alias].relative_to(REPO_ROOT))
            for alias in MODEL_ALIASES
        },
        "node_local_gpu_before": local_before,
        "cluster_project_gpu_diagnostic": cluster_before,
        "new_gpu_count": 2,
        "cpu_static_gate": gate,
        "retry_or_resubmit": False,
    }
    intent_sha = _write_once(paths["__intent__"], intent)

    submitted: dict[str, str] = {}
    failure: dict[str, str] | None = None
    try:
        first = MODEL_ALIASES[0]
        submitted[first] = _submit_one(
            alias=first,
            output_root=paths[first],
            source_head=source_head,
            log_parent=paths["__logs__"],
        )
        after_first = _scheduler_jobs()
        assert_node_local_capacity(after_first, new_gpu_count=1)
        second = MODEL_ALIASES[1]
        submitted[second] = _submit_one(
            alias=second,
            output_root=paths[second],
            source_head=source_head,
            log_parent=paths["__logs__"],
        )
    except Exception as exc:
        failure = {
            "exception_class": type(exc).__name__,
            "exception_message_sha256": hashlib.sha256(
                str(exc).encode("utf-8")
            ).hexdigest(),
        }

    receipt = {
        "schema": "ode-edit-s04-ode-bf-p0-submission-receipt/v1",
        "status": "SUBMITTED_PAIR" if failure is None else "PARTIAL_OR_FAILED_NO_RETRY",
        "source_head": source_head,
        "intent_sha256": intent_sha,
        "submitted": submitted,
        "failure": failure,
        "retry_or_resubmit": False,
    }
    receipt_sha = _write_once(paths["__receipt__"], receipt)
    if failure is not None:
        raise ODEBFContractError("ODE-BF P0 one-shot submission did not complete")
    print(
        json.dumps(
            {
                "status": "SUBMITTED_PAIR",
                "source_head": source_head,
                "jobs": submitted,
                "intent_sha256": intent_sha,
                "receipt_sha256": receipt_sha,
                "node": "server2",
                "retry_or_resubmit": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


def _entrypoint() -> None:
    try:
        code = main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "PRE_SUBMIT_OR_PARTIAL_HOLD",
                    "exception_class": type(exc).__name__,
                    "exception_message_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                    "retry_or_resubmit": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    raise SystemExit(code)


if __name__ == "__main__":
    _entrypoint()
