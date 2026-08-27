#!/usr/bin/env python3
"""Fail-closed BGODE-R3 G2 convergence entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
from pathlib import Path
from typing import Any

from project.run_scripts.barrier_guided_ode.r3.g2_experiment import (
    ARM_ORDER,
    INSTRUCTION_ID,
    expected_run_id,
    run_g2,
)
from project.run_scripts.barrier_guided_ode.r3.natural import (
    load_natural_manifest,
    load_natural_record,
    natural_case,
    tokenize_record,
)
from project.run_scripts.barrier_guided_ode.s1_experiment import s1_easyedit_pins
from project.run_scripts.ode_edit_motivation.contracts import canonical_json, sha256_bytes
from project.run_scripts.ode_edit_motivation.easyedit_bridge import EasyEditBridge
from project.run_scripts.ode_edit_motivation.gpu_runtime import fixed_pretrained_kwargs, offline_environment
from project.run_scripts.ode_edit_motivation.manifests import fixed_model_spec, preflight_fixed_artifacts


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g2-cauchy-numerical-lock.json"
MANIFEST_PATH = REPO_ROOT / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g2-source-manifest.json"
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"regular mode0600 JSON required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("G2 release JSON must be an object")
    return value


def validate_source_release(
    head: str,
    tree: str,
    *,
    source_manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    observed_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    observed_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
    if observed_head != head or observed_tree != tree:
        raise ValueError("queued G2 source differs")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True
    ):
        raise ValueError("queued G2 tracked source is dirty")
    source_manifest_path = source_manifest_path.expanduser().resolve(strict=True)
    locks = (REPO_ROOT / "project/run_scripts/barrier_guided_ode/r3/locks").resolve(strict=True)
    if source_manifest_path.parent != locks:
        raise ValueError("G2 source manifest is outside the sealed lock directory")
    manifest = _private_json(source_manifest_path)
    records: list[dict[str, Any]] = []
    for member in manifest.get("members", []):
        relative = member["path"]
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or format(stat.S_IMODE(path.stat().st_mode), "04o") != member["mode"]
            or path.stat().st_size != member["bytes"]
            or _sha256(path) != member["sha256"]
            or subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            != 0
        ):
            raise ValueError(f"G2 source member differs: {relative}")
        records.append(member)
    if sha256_bytes(canonical_json(records).encode("utf-8")) != manifest.get("members_root"):
        raise ValueError("G2 source manifest root differs")
    lock = _private_json(LOCK_PATH)
    body = {key: value for key, value in lock.items() if key != "identity"}
    if sha256_bytes(canonical_json(body).encode("utf-8")) != lock.get("identity"):
        raise ValueError("G2 Cauchy numerical lock identity differs")
    natural, natural_sha = load_natural_manifest()
    if natural.get("identity") != lock.get("natural_topology_manifest_identity"):
        raise ValueError("G2 natural topology identity differs")
    return {
        "members_root": manifest["members_root"],
        "natural_topology_manifest_identity": natural["identity"],
        "natural_topology_manifest_sha256": natural_sha,
        "numerical_lock": lock,
        "numerical_lock_sha256": _sha256(LOCK_PATH),
        "source_manifest_sha256": _sha256(source_manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--easyedit-root", type=Path, default=Path("/mnt/raid5/janghj/EasyEdit"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--model-alias", choices=MODEL_ALIASES, required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--source-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.run_token != expected_run_id(args.model_alias):
        raise ValueError("G2 run token differs")
    release = validate_source_release(
        args.source_head,
        args.source_tree,
        source_manifest_path=args.source_manifest,
    )
    case = natural_case(args.model_alias, "unequal-non-prefix")
    raw, _, _ = load_natural_record(case)
    plan: dict[str, Any] = {
        "array_mapping": {"0": "llama3-8b-inst", "1": "qwen2.5-7b-inst"},
        "arms": [arm.value for arm in ARM_ORDER],
        "batch_size": 1,
        "gpu_count": 1,
        "instruction_id": INSTRUCTION_ID,
        "model_alias": args.model_alias,
        "n_grid": [4, 8, 16, 32],
        "natural_case": dict(case),
        "release": release,
        "run_id": args.run_token,
        "schema": "ode-edit-bgode-r3-g2-dry-plan/v1",
        "scientific_promotion": False,
        "source_head": args.source_head,
        "source_tree": args.source_tree,
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
        tokenization = tokenize_record(tokenizer, raw, model_alias=args.model_alias)
        if tokenization.identity != case["tokenization_identity"]:
            raise ValueError("G2 preflight tokenization differs from the natural seal")
        plan.update(
            {
                "easyedit_bridge_manifest_id": bridge.manifest_id,
                "external_input_preflight": "PASS",
                "fixed_input_manifest_id": fixed.manifest_id,
                "tokenization": tokenization.receipt(),
            }
        )
        print(canonical_json(plan))
        return 0
    if args.output_root is None:
        parser.error("--output-root is required")
    result = run_g2(
        easyedit_root=args.easyedit_root,
        output_root=args.output_root,
        source_head=args.source_head,
        source_tree=args.source_tree,
        release_receipt=release,
        model_alias=args.model_alias,
        run_id=args.run_token,
    )
    print(canonical_json({"run_id": result["run_id"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
