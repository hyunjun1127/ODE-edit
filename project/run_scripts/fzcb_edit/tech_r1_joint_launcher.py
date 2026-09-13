"""One top-level cap-aware launcher for the exact Llama/Qwen joint B1 matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .contracts import TechnicalBoundary
from .hashing import canonical_hash, file_sha256, write_json_once


MODELS = ("llama3-8b-inst", "qwen2.5-7b-inst")
GPU_CAP = 4


def joint_matrix() -> list[dict[str, Any]]:
    rows = [
        {
            "array_index": index,
            "model": model,
            "batch": "B1",
            "case_ordinal": 0,
            "arm_contract": [
                "OFFICIAL_MEMIT",
                "TRUE_FROZEN_C_SPLIT",
                "REFRESHED_EQUALITY_ONLY",
                "FZCB",
                "STRONG_STATIC_SAME_OBJECTIVE",
            ],
            "same_request_order_seed_contract": True,
            "per_model_stock_memit_hparams": True,
        }
        for index, model in enumerate(MODELS)
    ]
    indices = [row["array_index"] for row in rows]
    models = [row["model"] for row in rows]
    if indices != [0, 1] or models != list(MODELS) or len(set(models)) != 2:
        raise TechnicalBoundary("joint launcher matrix is incomplete or aliased")
    return rows


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _active_gpu_count() -> tuple[int, list[dict[str, Any]]]:
    command = ["squeue", "-h", "-u", os.environ.get("USER", "janghj"), "-w", "server4", "-t", "R,CG", "-o", "%i|%b"]
    output = subprocess.check_output(command, text=True)
    rows = []
    total = 0
    for line in output.splitlines():
        if not line.strip():
            continue
        job_id, gres = line.split("|", 1)
        match = re.search(r"gpu(?::[^:,]+)?:(\d+)(?:\([^)]*\))?$", gres.strip())
        count = int(match.group(1)) if match else 0
        rows.append({"job_id": job_id, "gres": gres.strip(), "gpu_count": count})
        total += count
    return total, rows


def submit(
    *,
    source_root: Path,
    expected_head: str,
    preflight: Path,
    expected_preflight_sha: str,
    result_root: Path,
    log_root: Path,
    campaign_id: str,
    sbatch_script: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if _git(source_root, "rev-parse", "HEAD") != expected_head:
        raise TechnicalBoundary("joint launcher source HEAD mismatch")
    if _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise TechnicalBoundary("joint launcher source has tracked changes")
    if not preflight.is_file() or preflight.is_symlink() or file_sha256(preflight) != expected_preflight_sha:
        raise TechnicalBoundary("joint launcher preflight identity mismatch")
    preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
    if preflight_payload.get("status") != "PRE_GPU_PASS" or preflight_payload.get("source", {}).get("head") != expected_head:
        raise TechnicalBoundary("joint launcher preflight semantic mismatch")
    if result_root.exists() or result_root.is_symlink():
        raise TechnicalBoundary("joint campaign result root is not create-once")
    if log_root.exists() or log_root.is_symlink():
        raise TechnicalBoundary("joint campaign log root is not create-once")
    active, active_rows = _active_gpu_count()
    requested = len(MODELS)
    if active + requested > GPU_CAP:
        raise TechnicalBoundary(f"server4 GPU cap would be exceeded active={active} new={requested} cap={GPU_CAP}")
    result_root.mkdir(parents=True, mode=0o700)
    log_root.mkdir(parents=True, mode=0o700)
    environment = ",".join([
        "ALL",
        f"FZCB_TECH_R1_SOURCE_ROOT={source_root}",
        f"FZCB_TECH_R1_EXPECTED_HEAD={expected_head}",
        f"FZCB_TECH_R1_PREFLIGHT={preflight.resolve()}",
        f"FZCB_TECH_R1_PREFLIGHT_SHA={expected_preflight_sha}",
        f"FZCB_TECH_R1_RESULT_ROOT={result_root.resolve()}",
        f"FZCB_TECH_R1_CAMPAIGN_ID={campaign_id}",
        f"FZCB_TECH_R1_LOG_ROOT={log_root.resolve()}",
    ])
    command = [
        "sbatch", "--parsable", "--hold", "--array=0-1%2",
        f"--export={environment}",
        f"--output={log_root.resolve()}/%x-%A_%a.out",
        f"--error={log_root.resolve()}/%x-%A_%a.err",
        str(sbatch_script.resolve()),
    ]
    job_id = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
    inspection = subprocess.check_output(["scontrol", "show", "job", job_id], text=True)
    node_pinned = "ReqNodeList=server4" in inspection or "NodeList=server4" in inspection
    gres_pinned = "TresPerNode=gres/gpu:rtx_pro_6000:1" in inspection
    if not node_pinned or not gres_pinned:
        subprocess.call(["scancel", job_id])
        raise TechnicalBoundary(f"held joint job mapping differs: {inspection}")
    manifest = {
        "schema": "odeedit.s06.fzcb-tech-r1.joint-b1-campaign.v1",
        "campaign_id": campaign_id,
        "job_id": job_id,
        "array": "0-1%2",
        "matrix": joint_matrix(),
        "matrix_identity": canonical_hash(joint_matrix()),
        "source": {"root": str(source_root), "head": expected_head, "tree": _git(source_root, "rev-parse", "HEAD^{tree}")},
        "preflight": {"path": str(preflight.resolve()), "sha256": expected_preflight_sha},
        "result_root": str(result_root.resolve()),
        "log_root": str(log_root.resolve()),
        "resource": {"node": "server4", "gres_per_cell": "gpu:rtx_pro_6000:1", "cap": GPU_CAP, "active_before": active, "new": requested, "active_rows": active_rows},
        "held_inspection": inspection,
        "one_model_failure_cancels_other": False,
        "b10_submit_count": 0,
    }
    manifest_path = result_root / "campaign-manifest.json"
    write_json_once(manifest_path, manifest)
    subprocess.check_call(["scontrol", "release", job_id])
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "released": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--sbatch-script", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = submit(
        source_root=args.source_root,
        expected_head=args.expected_head,
        preflight=args.preflight,
        expected_preflight_sha=args.expected_preflight_sha,
        result_root=args.result_root,
        log_root=args.log_root,
        campaign_id=args.campaign_id,
        sbatch_script=args.sbatch_script,
    )
    write_json_once(args.receipt, payload)


if __name__ == "__main__":
    main()
