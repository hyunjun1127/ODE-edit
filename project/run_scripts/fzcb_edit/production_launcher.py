"""Cap-aware held-inspect-release launcher for canary/B10 model cells."""

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


def matrix(stage: str) -> list[dict[str, Any]]:
    if stage not in {"canary", "b10"}:
        raise TechnicalBoundary("production stage must be canary or b10")
    count = 1 if stage == "canary" else 10
    return [
        {
            "array_index": index,
            "model": model,
            "stage": stage,
            "request_count": count,
            "methods": ["memit", "alphaedit"],
            "arms_per_method": ["OFFICIAL", "REFRESHED_EQUALITY_ONLY", "FZCB_SKETCH_K_PROTOTYPE"],
            "joint_batch": True,
            "pristine_w0_per_arm": True,
            "sequential": False,
        }
        for index, model in enumerate(MODELS)
    ]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _active_gpu_count() -> tuple[int, list[dict[str, Any]]]:
    output = subprocess.check_output(
        ["squeue", "-h", "-u", os.environ.get("USER", "janghj"), "-w", "server4", "-t", "R,CG", "-o", "%i|%b"],
        text=True,
    )
    rows: list[dict[str, Any]] = []
    total = 0
    for line in output.splitlines():
        if not line.strip():
            continue
        job_id, gres = line.split("|", 1)
        match = re.search(r"gpu(?::[^:,]+)?:(\d+)(?:\([^)]*\))?$", gres.strip())
        count = int(match.group(1)) if match else 0
        total += count
        rows.append({"job_id": job_id, "gres": gres.strip(), "gpu_count": count})
    return total, rows


def _require_canary(root: Path) -> dict[str, Any]:
    rows = []
    for model in MODELS:
        path = root / f"{model}-canary/result.json"
        if not path.is_file() or path.is_symlink():
            raise TechnicalBoundary(f"canary result missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "TERMINAL_VALID" or payload.get("FZCB_terminal_valid_denominator") != 2:
            raise TechnicalBoundary(f"canary production path is not terminal-valid: {model}")
        rows.append({"model": model, "path": str(path), "sha256": file_sha256(path)})
    return {"status": "PASS", "members": rows, "member_root": canonical_hash(rows)}


def submit(
    *,
    stage: str,
    source_root: Path,
    expected_head: str,
    preflight: Path,
    preflight_sha: str,
    result_root: Path,
    log_root: Path,
    campaign_id: str,
    sbatch_script: Path,
    canary_root: Path | None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if _git(source_root, "rev-parse", "HEAD") != expected_head:
        raise TechnicalBoundary("production launcher source HEAD mismatch")
    if _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise TechnicalBoundary("production launcher tracked tree is dirty")
    if not preflight.is_file() or preflight.is_symlink() or file_sha256(preflight) != preflight_sha:
        raise TechnicalBoundary("production preflight identity mismatch")
    payload = json.loads(preflight.read_text(encoding="utf-8"))
    if payload.get("status") != "PRE_GPU_PASS" or payload.get("source", {}).get("head") != expected_head:
        raise TechnicalBoundary("production preflight semantic mismatch")
    canary = None
    if stage == "b10":
        if canary_root is None:
            raise TechnicalBoundary("B10 release requires canary root")
        canary = _require_canary(canary_root)
    if result_root.exists() or result_root.is_symlink() or log_root.exists() or log_root.is_symlink():
        raise TechnicalBoundary("production result/log root is not create-once")
    active, active_rows = _active_gpu_count()
    requested = len(MODELS)
    if active + requested > GPU_CAP:
        raise TechnicalBoundary(f"server4 GPU cap exceeded active={active} new={requested} cap={GPU_CAP}")
    result_root.mkdir(parents=True, mode=0o700)
    log_root.mkdir(parents=True, mode=0o700)
    environment = ",".join([
        "ALL",
        f"FZCB_B10_SOURCE_ROOT={source_root}",
        f"FZCB_B10_EXPECTED_HEAD={expected_head}",
        f"FZCB_B10_PREFLIGHT={preflight.resolve()}",
        f"FZCB_B10_PREFLIGHT_SHA={preflight_sha}",
        f"FZCB_B10_RESULT_ROOT={result_root.resolve()}",
        f"FZCB_B10_CAMPAIGN_ID={campaign_id}",
        f"FZCB_B10_STAGE={stage}",
    ])
    command = [
        "sbatch", "--parsable", "--hold", "--array=0-1%2",
        f"--export={environment}",
        f"--job-name=fzcb_{stage}_prod_s4",
        f"--output={log_root.resolve()}/%x-%A_%a.out",
        f"--error={log_root.resolve()}/%x-%A_%a.err",
        str(sbatch_script.resolve()),
    ]
    job_id = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
    inspection = subprocess.check_output(["scontrol", "show", "job", job_id], text=True)
    if not ("ReqNodeList=server4" in inspection or "NodeList=server4" in inspection):
        subprocess.call(["scancel", job_id])
        raise TechnicalBoundary("held production job is not pinned to server4")
    if "TresPerNode=gres/gpu:rtx_pro_6000:1" not in inspection:
        subprocess.call(["scancel", job_id])
        raise TechnicalBoundary("held production job GRES differs")
    manifest = {
        "schema": "odeedit.s06.fzcb.atomic-b10-production-pilot.campaign.v1",
        "stage": stage,
        "campaign_id": campaign_id,
        "job_id": job_id,
        "array": "0-1%2",
        "matrix": matrix(stage),
        "matrix_identity": canonical_hash(matrix(stage)),
        "source": {"root": str(source_root), "head": expected_head, "tree": _git(source_root, "rev-parse", "HEAD^{tree}")},
        "preflight": {"path": str(preflight.resolve()), "sha256": preflight_sha},
        "canary_gate": canary,
        "result_root": str(result_root.resolve()),
        "log_root": str(log_root.resolve()),
        "resource": {"node": "server4", "gres_per_cell": "gpu:rtx_pro_6000:1", "cap": GPU_CAP, "active_before": active, "new": requested, "active_rows": active_rows},
        "held_inspection": inspection,
        "sequential_submit_count": 0,
    }
    manifest_path = result_root / "campaign-manifest.json"
    write_json_once(manifest_path, manifest)
    subprocess.check_call(["scontrol", "release", job_id])
    return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path), "released": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("canary", "b10"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--sbatch-script", type=Path, required=True)
    parser.add_argument("--canary-root", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = submit(
        stage=args.stage,
        source_root=args.source_root,
        expected_head=args.expected_head,
        preflight=args.preflight,
        preflight_sha=args.preflight_sha,
        result_root=args.result_root,
        log_root=args.log_root,
        campaign_id=args.campaign_id,
        sbatch_script=args.sbatch_script,
        canary_root=args.canary_root,
    )
    write_json_once(args.receipt, receipt)


if __name__ == "__main__":
    main()
