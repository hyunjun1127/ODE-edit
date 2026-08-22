#!/usr/bin/env python3
"""No-model final preflight for P4 Euler calibration Stage1 R1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1_runtime import _atomic_write_once
from project.run_scripts.ode_bf.p4_euler_integrator import P4_EULER_INSTRUCTION_ID


EXPECTED_SESSION = "codex://threads/01a028a7-9e3c-7541-81ba-efb40555d17d"
LOCK = Path(
    "project/run_scripts/ode_bf/locks/numerical_lock_s05_p4_euler_calibration_r1.json"
)
MANIFEST = Path(
    "project/run_scripts/ode_bf/locks/source_manifest_s05_p4_euler_calibration_r1.json"
)
LOCK_ROOT = "649d4a271b29cceec3f20080db7c763e47c7f143a9a43fafaa7a4db3806bbe0b"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("P4 Euler calibration preflight input differs")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ODEBFContractError("P4 Euler calibration preflight JSON differs")
    return value


def _verify_rooted(path: Path, *, schema: str) -> tuple[dict[str, Any], str]:
    value = _load(path)
    body = dict(value)
    observed = body.pop("root_digest", None)
    if value.get("schema_version", value.get("schema")) != schema or observed != canonical_hash(body):
        raise ODEBFContractError("P4 Euler calibration rooted input differs")
    return value, _sha(path)


def _verify_manifest(source_head: str) -> Mapping[str, Any]:
    manifest, raw_sha = _verify_rooted(
        REPO_ROOT / MANIFEST,
        schema="ode-edit-s05-p4-euler-calibration-r1-source-manifest/v1",
    )
    if (
        manifest.get("instruction_id") != P4_EULER_INSTRUCTION_ID
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(manifest["source_parent"]), source_head],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise ODEBFContractError("P4 Euler calibration source ancestry differs")
    entries = manifest.get("entries")
    paths: list[str] = []
    for row in entries if isinstance(entries, list) else []:
        member = REPO_ROOT / str(row.get("path"))
        if (
            member.is_symlink()
            or not member.is_file()
            or member.stat().st_size != row.get("size")
            or _sha(member) != row.get("sha256")
        ):
            raise ODEBFContractError("P4 Euler calibration source member differs")
        paths.append(str(row["path"]))
    if not paths or paths != sorted(set(paths)):
        raise ODEBFContractError("P4 Euler calibration source ordering differs")
    return {
        "path": str(REPO_ROOT / MANIFEST),
        "sha256": raw_sha,
        "root_digest": manifest["root_digest"],
        "member_count": len(paths),
    }


def build_receipt(
    *, prior_final: Path,
    source_head: str,
    session_id: str,
) -> Mapping[str, Any]:
    if (
        session_id != EXPECTED_SESSION
        or socket.gethostname() != "server4"
        or os.environ.get("PROJECT_GPU_CAP", "2") != "2"
    ):
        raise ODEBFContractError("P4 Euler calibration session/host/cap differs")
    observed_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    observed_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=REPO_ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    role = subprocess.run(
        ["git", "config", "--get", "agent.role"], cwd=REPO_ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    if (
        observed_head != source_head
        or branch != "codex/server4-p4-euler-calibration-r1"
        or role != "server-head"
        or subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT).returncode != 0
    ):
        raise ODEBFContractError("P4 Euler calibration source/worktree differs")
    numerical, numerical_sha = _verify_rooted(
        REPO_ROOT / LOCK,
        schema="ode-edit-s05-p4-euler-calibration-r1-lock/v1",
    )
    if (
        numerical.get("root_digest") != LOCK_ROOT
        or numerical.get("status") != "CALIBRATION_STAGE1_AUTHORIZED"
        or numerical.get("selected_h") is not None
        or numerical.get("selected_target_horizon") is not None
    ):
        raise ODEBFContractError("P4 Euler calibration numerical lock differs")
    source = _verify_manifest(source_head)
    prior = _load(prior_final)
    prior_body = dict(prior)
    prior_identity = prior_body.pop("identity_sha256", None)
    if (
        prior.get("status") != "FINAL_PRE_GPU_PASS"
        or prior.get("model_load_authorized") is not True
        or prior_identity != canonical_hash(prior_body)
        or prior.get("hf_content_verification")
        != "PRIOR_ACCEPTED_EXACT_CONSUMED_CLOSURE_REUSED"
        or prior.get("hf_duplicate_rehash_count") != 0
    ):
        raise ODEBFContractError("P4 Euler prior accepted deployment receipt differs")
    model_binding = {}
    for alias, row in prior["model_input_binding"].items():
        model_binding[alias] = {
            **row,
            "case_index": 1,
            "calibration_label": "PERMANENT_CALIBRATION_ONLY",
            "confirmatory_eligibility": False,
            "future_reentry_allowed": False,
        }
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-calibration-final-pre-gpu/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "status": "FINAL_PRE_GPU_PASS",
        "owner_session": session_id,
        "source_head": source_head,
        "source_tree": observed_tree,
        "source_manifest": source,
        "numerical_lock": {
            "path": str(REPO_ROOT / LOCK),
            "sha256": numerical_sha,
            "root_digest": LOCK_ROOT,
        },
        "transfer_receipt": prior["transfer_receipt"],
        "model_input_binding": model_binding,
        "hf_closure_binding": prior["hf_closure_binding"],
        "hf_content_verification": "PRIOR_ACCEPTED_EXACT_CONSUMED_CLOSURE_REUSED",
        "hf_duplicate_rehash_count": 0,
        "easyedit_seal": prior["easyedit_seal"],
        "execution_binding": {
            "calibration_lock_root": LOCK_ROOT,
            "h_grid": [0.0625, 0.25, 1.0, 4.0],
            "prefix_M": [1, 3, 5, 10],
            "single_trajectory_prefix_reuse": True,
            "executed_microsteps_per_trajectory": 10,
            "separate_prefix_trajectory_count": 0,
            "duplicate_autograd_evaluation_count": 0,
            "full_fp32": True,
            "offline": True,
            "writer_materialization_count": 0,
            "cache_append_count": 0,
            "heldout_access_count": 0,
            "native_access_count": 0,
        },
        "project_gpu_cap": 2,
        "model_load_authorized": True,
        "model_load_count": 0,
        "gpu_action_count": 0,
        "slurm_submit_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prior-final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        prior_final=args.prior_final,
        source_head=args.source_head,
        session_id=args.session_id,
    )
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _atomic_write_once(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
