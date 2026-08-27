#!/usr/bin/env python3
"""Fail-closed entrypoint for the BGODE-R2 Stage-B technical array."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
from pathlib import Path
from typing import Any

from project.run_scripts.barrier_guided_ode.r2.stage_b_contract import (
    MODEL_BINDINGS,
    load_sealed_s1_sample,
    model_binding,
)
from project.run_scripts.barrier_guided_ode.r2.stage_b_probe import INSTRUCTION_ID, run_stage_b
from project.run_scripts.barrier_guided_ode.s1_experiment import s1_easyedit_pins
from project.run_scripts.ode_edit_motivation.contracts import canonical_json
from project.run_scripts.ode_edit_motivation.easyedit_bridge import EasyEditBridge
from project.run_scripts.ode_edit_motivation.gpu_runtime import fixed_pretrained_kwargs, offline_environment
from project.run_scripts.ode_edit_motivation.manifests import fixed_model_spec, preflight_fixed_artifacts


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "project/run_scripts/barrier_guided_ode/r2/locks/bgode-r2-stage-b-numerical-lock.json"
MANIFEST_PATH = REPO_ROOT / "project/run_scripts/barrier_guided_ode/r2/locks/bgode-r2-stage-b-source-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"regular mode0600 input required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("lock/manifest must be a JSON object")
    return value


def _source_gate(head: str, tree: str) -> dict[str, Any]:
    observed_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    observed_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
    if observed_head != head or observed_tree != tree:
        raise ValueError("queued Stage-B source differs")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True
    ):
        raise ValueError("queued Stage-B tracked source is dirty")
    manifest = _private_json(MANIFEST_PATH)
    member_records = []
    for member in manifest.get("members", []):
        path = REPO_ROOT / member["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or format(stat.S_IMODE(path.stat().st_mode), "04o") != member["mode"]
            or path.stat().st_size != member["bytes"]
            or _sha256(path) != member["sha256"]
            or subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", member["path"]],
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            != 0
            or subprocess.run(
                ["git", "diff", "--quiet", head, "--", member["path"]],
                cwd=REPO_ROOT,
                check=False,
            ).returncode
            != 0
        ):
            raise ValueError(f"Stage-B source bytes differ: {member['path']}")
        member_records.append(member)
    from project.run_scripts.ode_edit_motivation.contracts import sha256_bytes

    if sha256_bytes(canonical_json(member_records).encode("utf-8")) != manifest.get("members_root"):
        raise ValueError("Stage-B source manifest root differs")
    lock = _private_json(LOCK_PATH)
    lock_body = {key: value for key, value in lock.items() if key != "identity"}
    if sha256_bytes(canonical_json(lock_body).encode("utf-8")) != lock.get("identity"):
        raise ValueError("Stage-B numerical lock identity differs")
    return {
        "source_manifest_sha256": _sha256(MANIFEST_PATH),
        "members_root": manifest.get("members_root"),
        "numerical_lock_sha256": _sha256(LOCK_PATH),
        "numerical_lock_identity": lock.get("identity"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--easyedit-root", type=Path, default=Path("/mnt/raid5/janghj/EasyEdit"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--model-alias", choices=tuple(MODEL_BINDINGS), required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    binding = model_binding(args.model_alias)
    if args.run_token != binding.run_id:
        raise ValueError("Stage-B run token differs")
    release = _source_gate(args.source_head, args.source_tree)
    sample = load_sealed_s1_sample()
    plan = {
        "schema": "ode-edit-bgode-r2-stage-b-dry-plan/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": args.source_head,
        "source_tree": args.source_tree,
        "model_alias": args.model_alias,
        "run_id": args.run_token,
        "sample": sample.receipt(),
        "sample_identity": sample.identity,
        "batch_size": 1,
        "gpu_count": 1,
        "fixed_target_compute_count": 1,
        "fixed_target_recompute_count": 0,
        "termination_token_count": 0,
        "dynamic_writer_action_count": 0,
        "release": release,
        "scientific_promotion": False,
    }
    if args.preflight_only:
        fixed = preflight_fixed_artifacts(args.easyedit_root, model_alias=args.model_alias)
        bridge = EasyEditBridge(
            args.easyedit_root,
            expected_files=s1_easyedit_pins(),
            include_alphaedit_reference=True,
        ).preflight()
        from transformers import AutoTokenizer

        with offline_environment():
            tokenizer = AutoTokenizer.from_pretrained(
                **fixed_pretrained_kwargs(fixed_model_spec(args.model_alias))
            )
        from project.run_scripts.barrier_guided_ode.r2.stage_b_contract import seal_tokenization

        plan.update(
            {
                "tokenization": seal_tokenization(
                    tokenizer, sample, model_alias=args.model_alias
                ).receipt(),
                "fixed_input_manifest_id": fixed.manifest_id,
                "easyedit_bridge_manifest_id": bridge.manifest_id,
                "external_input_preflight": "PASS",
            }
        )
        print(canonical_json(plan))
        return 0
    if args.output_root is None:
        parser.error("--output-root is required")
    result = run_stage_b(
        easyedit_root=args.easyedit_root,
        output_root=args.output_root,
        source_head=args.source_head,
        source_tree=args.source_tree,
        release_receipt=release,
        model_alias=args.model_alias,
        run_id=args.run_token,
    )
    print(canonical_json({"status": result["status"], "run_id": result["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
