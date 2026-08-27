#!/usr/bin/env python3
"""Fail-closed BGODE-R1 S1 single-request pilot entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.barrier_guided_ode.s1_contract import (
    S1_ARM_ORDER,
    load_sealed_s1_sample,
    seal_s1_tokenization,
)
from project.run_scripts.barrier_guided_ode.s1_experiment import (
    INSTRUCTION_ID,
    RUN_ID,
    run_s1,
    s1_easyedit_pins,
)
from project.run_scripts.ode_edit_motivation.contracts import canonical_json, sha256_bytes
from project.run_scripts.ode_edit_motivation.easyedit_bridge import EasyEditBridge
from project.run_scripts.ode_edit_motivation.gpu_runtime import (
    fixed_pretrained_kwargs,
    offline_environment,
)
from project.run_scripts.ode_edit_motivation.manifests import (
    fixed_model_spec,
    preflight_fixed_artifacts,
)


LOCK_PATH = REPO_ROOT / "project/run_scripts/barrier_guided_ode/locks/bgode-r1-s1-numerical-lock.json"
MANIFEST_PATH = REPO_ROOT / "project/run_scripts/barrier_guided_ode/locks/bgode-r1-s1-source-manifest.json"
RUN_TOKEN = "bgode-r1-s1-llama-b1-request000-six-arm-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_private_json(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"regular mode0600 release artifact required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release artifact must be a JSON object")
    return payload, _sha256(path)


def _source_gate(source_head: str, source_tree: str) -> dict[str, Any]:
    observed_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    observed_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True
    ).strip()
    if observed_head != source_head or observed_tree != source_tree:
        raise ValueError("queued BGODE source HEAD/tree differs")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        text=True,
    )
    if dirty:
        raise ValueError("queued BGODE tracked source is dirty")
    manifest, manifest_sha = _read_private_json(MANIFEST_PATH)
    if (
        manifest.get("schema") != "ode-edit-bgode-r1-s1-source-manifest/v1"
        or manifest.get("instruction_id") != INSTRUCTION_ID
        or not isinstance(manifest.get("entries"), list)
        or not manifest["entries"]
    ):
        raise ValueError("BGODE source manifest header differs")
    for entry in manifest["entries"]:
        relative = entry.get("path") if isinstance(entry, Mapping) else None
        if not isinstance(relative, str):
            raise ValueError("BGODE source manifest entry differs")
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("bytes")
            or _sha256(path) != entry.get("sha256")
        ):
            raise ValueError(f"BGODE source manifest bytes differ: {relative}")
        if subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode != 0 or subprocess.run(
            ["git", "diff", "--quiet", source_head, "--", relative],
            cwd=REPO_ROOT,
            check=False,
        ).returncode != 0:
            raise ValueError(f"BGODE source entry is not exact at queued HEAD: {relative}")
    root = sha256_bytes(canonical_json(manifest["entries"]).encode("utf-8"))
    if root != manifest.get("entries_root_sha256"):
        raise ValueError("BGODE source manifest root differs")
    lock, lock_sha = _read_private_json(LOCK_PATH)
    lock_body = {key: value for key, value in lock.items() if key != "lock_root_sha256"}
    lock_root = sha256_bytes(canonical_json(lock_body).encode("utf-8"))
    if (
        lock.get("schema") != "ode-edit-bgode-r1-s1-numerical-lock/v1"
        or lock.get("instruction_id") != INSTRUCTION_ID
        or lock_root != lock.get("lock_root_sha256")
    ):
        raise ValueError("BGODE numerical lock differs")
    return {
        "source_manifest_sha256": manifest_sha,
        "source_manifest_entries_root_sha256": root,
        "numerical_lock_sha256": lock_sha,
        "numerical_lock_root_sha256": lock_root,
    }


def _preflight(
    source_head: str,
    source_tree: str,
    easyedit_root: Path,
    *,
    external_inputs: bool,
) -> dict[str, Any]:
    release = _source_gate(source_head, source_tree)
    sample = load_sealed_s1_sample()
    plan = {
        "schema": "ode-edit-bgode-r1-s1-dry-plan/v1",
        "instruction_id": INSTRUCTION_ID,
        "run_id": RUN_ID,
        "status": "BGODE_R1_S1_PRE_GPU_PASS",
        "source_head": source_head,
        "source_tree": source_tree,
        "sample": sample.receipt(),
        "sample_identity": sample.identity,
        "batch_size": 1,
        "gpu_count": 1,
        "process_count": 1,
        "arm_order": [arm.value for arm in S1_ARM_ORDER],
        "arm_count": 6,
        "model_load_count": 1,
        "full_fp32_required": True,
        "termination_boundary": {"string": "<|eot_id|>", "token_id": 128009},
        "release": release,
        "scientific_promotion": False,
    }
    if external_inputs:
        fixed = preflight_fixed_artifacts(easyedit_root, model_alias="llama3-8b-inst")
        bridge = EasyEditBridge(
            easyedit_root,
            expected_files=s1_easyedit_pins(),
            include_alphaedit_reference=True,
        ).preflight()
        from transformers import AutoTokenizer

        with offline_environment():
            tokenizer = AutoTokenizer.from_pretrained(
                **fixed_pretrained_kwargs(fixed_model_spec("llama3-8b-inst"))
            )
        tokenization = seal_s1_tokenization(tokenizer, sample)
        plan.update(
            {
                "tokenization": tokenization.receipt(),
                "fixed_input_manifest_id": fixed.manifest_id,
                "easyedit_bridge_manifest_id": bridge.manifest_id,
                "external_input_preflight": "PASS",
            }
        )
    else:
        plan["external_input_preflight"] = "DEFERRED_TO_RUN_SAME_PROCESS"
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--easyedit-root", type=Path, default=Path("/mnt/raid5/janghj/EasyEdit"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    plan = _preflight(
        args.source_head,
        args.source_tree,
        args.easyedit_root,
        external_inputs=args.preflight_only,
    )
    if args.preflight_only:
        print(canonical_json(plan))
        return 0
    if args.output_root is None:
        parser.error("--output-root is required unless --preflight-only is used")
    result = run_s1(
        easyedit_root=args.easyedit_root,
        output_root=args.output_root,
        source_head=args.source_head,
        source_tree=args.source_tree,
        release_receipt=plan["release"],
    )
    print(canonical_json({
        "status": result["status"],
        "run_id": result["run_id"],
        "source_head": result["source_head"],
        "source_tree": result["source_tree"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
