#!/usr/bin/env python3
"""Held-inspect-release submitter for the eight-task Sequential-NoH array."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts import session05_ode_bf_historical_h0_sequential_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_historical_h0_sequential.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state"
LOG_ROOT = REPO_ROOT / "local/odebf/logs/p1r23-progress-simplex-sequential-noh"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
SOURCE_MANIFEST = (
    REPO_ROOT
    / "project/run_scripts/ode_bf/locks/source_manifest_s05_historical_h0_sequential.json"
)
NAMESPACE = "s05-p1r23-progress-simplex-sequential-noh-sh1-array-v1"
BRANCH = "codex/p1r23-progress-simplex-sequential-noh-v1"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO_ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_once(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _verify_source_manifest() -> tuple[str, str]:
    value = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    observed_root = value.pop("root_digest", None)
    if observed_root != canonical_hash(value):
        raise ODEBFContractError("P1R23 Full-6 Historical manifest root differs")
    entries = value.get("entries")
    if not isinstance(entries, list) or value.get("entry_count") != len(entries):
        raise ODEBFContractError("P1R23 Full-6 Historical manifest inventory differs")
    for item in entries:
        path = REPO_ROOT / str(item["path"])
        status = path.lstat()
        payload = path.read_bytes()
        object_id = _run(["git", "hash-object", str(path)]).stdout.strip()
        if (
            not path.is_file()
            or path.is_symlink()
            or status.st_mode != int(item["mode"])
            or len(payload) != int(item["size"])
            or hashlib.sha256(payload).hexdigest() != item["sha256"]
            or object_id != item["object_id"]
        ):
            raise ODEBFContractError(
                "P1R23 Full-6 Historical manifest member differs"
            )
    return hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(), observed_root


def submit(source_head: str) -> dict[str, object]:
    if _run(["git", "rev-parse", "HEAD"]).stdout.strip() != source_head or _run(["git", "branch", "--show-current"]).stdout.strip() != BRANCH or _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout:
        raise ODEBFContractError("P1R23 Full-6 Historical execution source differs")
    source_manifest_sha, source_manifest_root = _verify_source_manifest()
    plan = dry.build_plan(source_head)
    if any((RESULT_PARENT / str(job["result_name"])).exists() for job in plan["jobs"]):
        raise ODEBFContractError("P1R23 Full-6 Historical result namespace exists")
    intent_path = STATE_ROOT / f"{NAMESPACE}.intent.json"
    receipt_path = STATE_ROOT / f"{NAMESPACE}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R23 Full-6 Historical submission namespace exists")
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent_sha = _write_once(intent_path, {"schema": "ode-edit-s05-p1r23-sequential-noh-intent/v1", "source_head": source_head, "source_manifest_sha256": source_manifest_sha, "source_manifest_root": source_manifest_root, "dry_plan": plan, "held_then_atomic_release": True, "array": "0-7%4"})
    job_id = _run(["sbatch", "--hold", "--parsable", "--chdir", str(REPO_ROOT), "--job-name", "odeedit_s05_p1r23_sequential_noh", "--output", str(LOG_ROOT / "%A_%a.out"), "--error", str(LOG_ROOT / "%A_%a.err"), str(SBATCH), source_head, str(RESULT_PARENT)]).stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R23 Full-6 Historical scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = ("JobState=PENDING", "Reason=JobHeldUser", "ArrayTaskThrottle=4", "ReqNodeList=devbox", "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1")
    if not all(item in observed for item in required):
        _run(["scancel", job_id])
        raise ODEBFContractError("P1R23 Full-6 Historical held scheduler contract differs")
    receipt = {"schema": "ode-edit-s05-p1r23-sequential-noh-submission/v1", "source_head": source_head, "source_manifest_sha256": source_manifest_sha, "source_manifest_root": source_manifest_root, "job_id": job_id, "array": "0-7%4", "trajectory_count": 8, "ode_trajectory_count": 8, "model_level_alphaedit_rerun_count": 0, "max_concurrent_gpu": 4, "intent_sha256": intent_sha, "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(), "held_then_atomic_release": True}
    receipt_sha = _write_once(receipt_path, receipt)
    _run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    print(json.dumps(submit(args.source_head), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
