"""No-model Server4 preflight for ORBODE B100x10 sequential execution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping

from .preflight import (
    CELL_SPECS,
    MODEL_BINDINGS,
    ORDER_ROOT,
    STREAM_ROOT,
    PreflightBoundary,
    _assert_no_symlink_components,
    _regular,
    canonical_hash,
    cell_spec,
    sha256_file,
    validate_easyedit_source,
    validate_model_artifacts,
    validate_stream,
)
from .sequential_contracts import ARM_ORDER, INSTRUCTION_ID, ROUND_INDICES, SequentialRuntimeLock


SERVER = "server4"
SLURM_NODE = "server4"
MEMORY_MIB_PER_TASK = 60_416
GPU_CAP = 2
EXPECTED_SESSION_ID = "01a04939-b5c7-7a03-ba2d-ef3343d62cfd"
EASYEDIT_SOURCE_ROOT = Path("/data/janghj/EasyEdit-stock-14cea824")
EASYEDIT_ARTIFACT_ROOT = Path("/data/janghj/EasyEdit")
HF_HUB_CACHE_ROOT = Path("/data/janghj/.cache/huggingface/hub")
AUTHORITATIVE_DOC_ROOT = Path(
    "/data/janghj/ODE-edit/local/state/orbode-sequential-b100x10-v1/authoritative-docs"
)
EXECUTION_LOCK_RELATIVE = Path(
    "project/run_scripts/ordered_response_barrier_ode/locks/server4-sequential-b100x10-v1.json"
)
AUTHORITATIVE_DOCS = {
    "2026-09-03-ordered-response-barrier-ode-edit-proposal.md": (
        55_900,
        1_359,
        "03a6fc61258e7643fc281fab8ab0d3ca700bb80f09aaa3eb34591ce5827087be",
    ),
    "2026-09-03-ordered-response-barrier-ode-edit-gh-fast-main-prompt.md": (
        13_143,
        287,
        "8548d016fda343f8b8917bcbe43647b58ad99ab6f796ece2fe4ef661faa4c431",
    ),
    "2026-09-03-ordered-response-barrier-ode-edit-fast-main-table.md": (
        18_113,
        451,
        "e134ac708c556482b912d7103e12a12347a78958943e7fa3d3da0a473908aa40",
    ),
}
EXECUTION_SOURCE_PATHS = (
    "project/run_scripts/ordered_response_barrier_ode",
    "project/run_scripts/session06_orbode_sequential_server4.py",
    "project/run_scripts/session06_orbode_sequential_server4.sbatch",
)


def _git(root: Path, *args: str) -> str:
    value = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if value.returncode:
        raise PreflightBoundary(f"git {' '.join(args)} failed: {value.stderr.strip()}")
    return value.stdout.strip()


def validate_authoritative_docs(root: Path = AUTHORITATIVE_DOC_ROOT) -> list[dict[str, Any]]:
    if root.absolute() != AUTHORITATIVE_DOC_ROOT:
        raise PreflightBoundary("GH-mediated authoritative document root differs")
    _assert_no_symlink_components(root)
    if stat.S_IMODE(root.lstat().st_mode) != 0o700:
        raise PreflightBoundary("authoritative document directory mode differs")
    result: list[dict[str, Any]] = []
    for name, (expected_bytes, expected_lines, expected_sha) in AUTHORITATIVE_DOCS.items():
        path = root / name
        observed = _regular(path, expected_bytes=expected_bytes)
        payload = path.read_bytes()
        if (
            stat.S_IMODE(observed.st_mode) != 0o600
            or payload.count(b"\n") != expected_lines
            or hashlib.sha256(payload).hexdigest() != expected_sha
        ):
            raise PreflightBoundary(f"authoritative document identity differs: {name}")
        payload.decode("utf-8")
        result.append(
            {
                "absolute_path": str(path),
                "bytes": expected_bytes,
                "lines": expected_lines,
                "mode": "0600",
                "sha256": expected_sha,
            }
        )
    return result


def validate_source(root: Path, *, head: str, tree: str) -> dict[str, Any]:
    if (
        _git(root, "rev-parse", "HEAD") != head
        or _git(root, "rev-parse", "HEAD^{tree}") != tree
    ):
        raise PreflightBoundary("source HEAD/tree differs")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise PreflightBoundary("source has tracked modifications")
    shadow = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *EXECUTION_SOURCE_PATHS,
    )
    if shadow:
        raise PreflightBoundary("execution source has untracked/dirty shadow files")
    members: list[dict[str, Any]] = []
    for relative in _git(root, "ls-files", *EXECUTION_SOURCE_PATHS).splitlines():
        path = root / relative
        observed = _regular(path)
        members.append(
            {
                "path": relative,
                "bytes": observed.st_size,
                "sha256": sha256_file(path),
            }
        )
    if not members:
        raise PreflightBoundary("execution source inventory is empty")
    return {
        "head": head,
        "tree": tree,
        "tracked_clean": True,
        "members": members,
        "members_root": canonical_hash(members),
    }


def validate_execution_lock(root: Path) -> dict[str, Any]:
    path = root / EXECUTION_LOCK_RELATIVE
    observed = _regular(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightBoundary("sequential execution lock is not an object")
    body = dict(value)
    identity = body.pop("identity_sha256", None)
    if identity != canonical_hash(body):
        raise PreflightBoundary("sequential execution lock identity differs")
    expected_mapping = {
        str(item.cell_id): [item.model_alias, item.writer_family]
        for item in CELL_SPECS
    }
    if (
        value.get("instruction_id") != INSTRUCTION_ID
        or value.get("cell_mapping") != expected_mapping
        or value.get("arms") != list(ARM_ORDER)
        or value.get("round_indices") != list(ROUND_INDICES)
        or value.get("stream_root") != STREAM_ROOT
        or value.get("order_root") != ORDER_ROOT
        or value.get("T") != 1.0
        or value.get("N") != 4
        or value.get("h") != 0.25
        or value.get("full_fp32") is not True
        or value.get("memory_mib_per_task") != MEMORY_MIB_PER_TASK
        or value.get("array") != "0-3%2"
        or value.get("scientific_promotion") is not False
    ):
        raise PreflightBoundary("sequential execution lock contract differs")
    return {
        "path": str(path),
        "bytes": observed.st_size,
        "sha256": sha256_file(path),
        "identity_sha256": identity,
    }


def parse_local_cap(path: Path) -> dict[str, Any]:
    observed = _regular(path)
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) == 5 and fields[0] == SERVER:
            _, node, cap, memory, patterns = fields
            if node != SLURM_NODE or int(cap) != GPU_CAP or int(memory) != MEMORY_MIB_PER_TASK:
                raise PreflightBoundary("server4 local cap/memory row differs")
            return {
                "path": str(path),
                "bytes": observed.st_size,
                "server": SERVER,
                "node": node,
                "max_project_gpus": int(cap),
                "memory_mib_per_gpu": int(memory),
                "job_patterns": patterns,
            }
    raise PreflightBoundary("server4 local cap row is absent")


def require_create_once_parent(path: Path) -> str:
    target = path.absolute()
    parent = target.parent
    _assert_no_symlink_components(parent)
    if target.exists() or target.is_symlink():
        raise PreflightBoundary(f"create-once target exists: {target}")
    if not parent.is_dir() or parent.is_symlink():
        raise PreflightBoundary(f"create-once parent differs: {parent}")
    return str(target)


def run_preflight(
    *,
    repo_root: Path,
    source_head: str,
    source_tree: str,
    easyedit_source_root: Path,
    easyedit_artifact_root: Path,
    hf_hub_cache: Path,
    local_caps: Path,
    output_parent: Path,
    log_root: Path,
    deep_artifact_hash: bool,
) -> dict[str, Any]:
    if easyedit_source_root.absolute() != EASYEDIT_SOURCE_ROOT:
        raise PreflightBoundary("server4 stock EasyEdit root differs")
    if easyedit_artifact_root.absolute() != EASYEDIT_ARTIFACT_ROOT:
        raise PreflightBoundary("server4 EasyEdit artifact root differs")
    if hf_hub_cache.absolute() != HF_HUB_CACHE_ROOT:
        raise PreflightBoundary("server4 HF cache root differs")
    receipt: dict[str, Any] = {
        "schema": "orbode.server4.sequential-preflight.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "PRE_GPU_BINDING_PASS" if deep_artifact_hash else "DRY_PLAN_BINDING_PASS",
        "source": validate_source(repo_root, head=source_head, tree=source_tree),
        "authoritative_inputs": validate_authoritative_docs(),
        "execution_lock": validate_execution_lock(repo_root),
        "easyedit_source": validate_easyedit_source(easyedit_source_root),
        "stream": validate_stream(repo_root, easyedit_artifact_root),
        "model_artifacts": validate_model_artifacts(
            repo_root,
            easyedit_artifact_root,
            hf_hub_cache,
            deep_hash=deep_artifact_hash,
        ),
        "runtime_lock": SequentialRuntimeLock().payload(),
        "resource": {
            **parse_local_cap(local_caps),
            "gpu_per_task": 1,
            "array": "0-3%2",
            "memory_mib_per_task": MEMORY_MIB_PER_TASK,
            "cpus_per_task": 8,
            "export": "NONE",
        },
        "create_once": {
            "output_parent": require_create_once_parent(output_parent),
            "log_root": require_create_once_parent(log_root),
        },
        "session": {
            "expected_id": EXPECTED_SESSION_ID,
            "boundary_check_required_in_launcher": True,
        },
        "science": {
            "arms": list(ARM_ORDER),
            "round_indices": list(ROUND_INDICES),
            "T": 1.0,
            "N": 4,
            "h": 0.25,
            "full_fp32": True,
            "dynamic_z_recompute_count": 0,
            "controller_evaluator_influence_count": 0,
            "scientific_promotion": False,
        },
        "action_counts": {"gpu": 0, "model": 0, "slurm": 0},
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return receipt


__all__ = [
    "AUTHORITATIVE_DOC_ROOT",
    "EASYEDIT_ARTIFACT_ROOT",
    "EASYEDIT_SOURCE_ROOT",
    "EXPECTED_SESSION_ID",
    "GPU_CAP",
    "HF_HUB_CACHE_ROOT",
    "MEMORY_MIB_PER_TASK",
    "SERVER",
    "SLURM_NODE",
    "parse_local_cap",
    "run_preflight",
]
