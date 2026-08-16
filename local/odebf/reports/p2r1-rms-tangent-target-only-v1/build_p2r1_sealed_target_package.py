#!/usr/bin/env python3
"""Create the immutable P2R1 sealed-target handoff package."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


WORKTREE = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p2r1-baseline-calibrated-rms-tangent-target-flow-v1"
)
REPORT_DIR = WORKTREE / "local/odebf/reports/p2r1-rms-tangent-target-only-v1"
RESULTS = {
    "llama3-8b-inst": WORKTREE
    / "local/odebf/results/"
    "s05-p2r1-rms-tangent-target-only-b10x10-llama3-8b-inst-tech-r1-v1",
    "qwen2.5-7b-inst": WORKTREE
    / "local/odebf/results/"
    "s05-p2r1-rms-tangent-target-only-b10x10-qwen2.5-7b-inst-tech-r1-v1",
}
SUPPORT_ID = "P2R1_P2R2_SEALED_TARGET_ARTIFACT_SH1_V1"
SOURCE_HEAD = "8f817e13167289dac190fe74bfa42e2b3e01372d"
SOURCE_TREE = "abe577756ea82e1fed0847b6329c82dc49184e9f"
PARENT_HEAD = "11508b6da11d606521b703037034e1814b70d8a8"
CONTRACT_SHA256 = "4bfb6c99e8a71ae822afe08e32e2263e8d6c69485d6b54a5cbd12b985884c414"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_root(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    path.chmod(0o600)


def install(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"invalid source member: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copyfile(source, destination)
    destination.chmod(0o600)


def member(path: Path, root: Path) -> dict[str, object]:
    mode = stat.S_IMODE(path.stat().st_mode)
    if not path.is_file() or path.is_symlink() or mode != 0o600:
        raise RuntimeError(f"invalid package member: {path}")
    return {
        "bytes": path.stat().st_size,
        "mode": "0600",
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_p2r1_sealed_target_package.py DESTINATION")
    destination = Path(sys.argv[1])
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"create-once destination exists: {destination}")
    destination.mkdir(parents=True, mode=0o700)
    destination.chmod(0o700)

    report_names = [
        "p2r1-rms-tangent-target-only-factual-ko.md",
        "p2r1-per-case.json",
        "p2r1-per-step.json",
        "p2r1-per-request.json",
        "p2r1-per-panel.json",
        "p2r1-case-paired-comparisons.json",
        "p2r1-aggregates.json",
        "analysis-manifest.json",
        "analysis-receipt.json",
        "independent-review-receipt.json",
    ]
    for name in report_names:
        install(REPORT_DIR / name, destination / "raw-free-analysis" / name)

    source_members = [
        "project/run_scripts/ode_bf/p2r1_rms_tangent_target.py",
        "project/run_scripts/ode_bf/p2r1_target_only_panel.py",
        "project/run_scripts/ode_bf/p2r1_target_only_runtime.py",
        "project/run_scripts/session05_ode_bf_p2r1_target_only.py",
        "project/run_scripts/session05_ode_bf_p2r1_target_only_dry_plan.py",
        "project/run_scripts/session05_ode_bf_submit_p2r1_target_only.py",
        "project/run_scripts/ode_bf/locks/numerical_lock_s05_p2r1_rms_tangent_target_only.json",
        "project/run_scripts/ode_bf/locks/source_manifest_s05_p2r1_rms_tangent_target_only.json",
    ]
    for relative in source_members:
        install(WORKTREE / relative, destination / "interface-source" / relative)

    state_root = WORKTREE / "local/odebf/state/p2r1-rms-tangent-target-only"
    for source in sorted(state_root.glob("*tech-r1-8f817e131672-v1*.json")):
        install(source, destination / "execution-identities" / source.name)

    target_index: list[dict[str, object]] = []
    for alias, result_root in RESULTS.items():
        install(result_root / "manifest.json", destination / "task-identities" / alias / "manifest.json")
        install(result_root / "terminal.json", destination / "task-identities" / alias / "terminal.json")
        task_terminal = json.loads((result_root / "terminal.json").read_text())
        if task_terminal.get("case_count") != 10 or task_terminal.get("failed_case_count") != 0:
            raise RuntimeError(f"nonterminal task inventory: {alias}")
        for case_index in range(1, 11):
            source_case = result_root / "raw/cases" / f"case-{case_index:02d}"
            target = source_case / "private-target/kstate-targets.pt"
            terminal_path = source_case / "terminal.json"
            terminal = json.loads(terminal_path.read_text())
            if sha256(target) != terminal.get("target_trajectory_file_sha256"):
                raise RuntimeError(f"target SHA mismatch: {alias} case {case_index}")
            base = destination / "sealed-targets" / alias / f"case-{case_index:02d}"
            install(target, base / "kstate-targets.pt")
            for name in (
                "terminal.json",
                "manifest.json",
                "action-freeze.json",
            ):
                install(source_case / name, base / name)
            for name in ("capture-plan.json", "objective-plan.json"):
                install(source_case / "raw" / name, base / "receipts" / name)
            for microstep_receipt in sorted((source_case / "raw/target").glob("microstep-*.json")):
                install(
                    microstep_receipt,
                    base / "receipts/target" / microstep_receipt.name,
                )
            target_index.append(
                {
                    "alias": alias,
                    "case_index": case_index,
                    "file_bytes": target.stat().st_size,
                    "file_sha256": sha256(target),
                    "request_count": terminal.get("request_count"),
                    "target_microstep_count": terminal.get("target_microstep_count"),
                    "target_tensor_sha256": terminal.get("target_trajectory_tensor_sha256"),
                    "terminal_identity_sha256": terminal.get("identity_sha256"),
                    "W0_restored": terminal.get("W0_restored"),
                }
            )

    bundle = destination / "source/p2r1-source.bundle"
    bundle.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    subprocess.run(
        ["git", "bundle", "create", str(bundle), "--all"],
        cwd=WORKTREE,
        check=True,
    )
    bundle.chmod(0o600)
    verify = subprocess.run(
        ["git", "bundle", "verify", str(bundle)],
        cwd=WORKTREE,
        check=True,
        capture_output=True,
        text=True,
    )
    if "complete history" not in (verify.stdout + verify.stderr).lower():
        raise RuntimeError("source bundle is not complete history")

    target_index_payload = {
        "contract_sha256": CONTRACT_SHA256,
        "entries": target_index,
        "entry_count": len(target_index),
        "instruction_id": "ODEEDIT-S05-P2R1-BASELINE-CALIBRATED-RMS-TANGENT-TARGET-FLOW-V1",
        "parent_head": PARENT_HEAD,
        "schema": "ode-edit-s05-p2r1-sealed-target-index/v1",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "stream_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "writer_materialization_count": 0,
    }
    target_index_payload["root_digest"] = canonical_root(target_index_payload)
    write_json(destination / "sealed-target-index.json", target_index_payload)

    intent = {
        "destination": "/mnt/raid5/janghj/ODE-edit/local/source-handoff/" + SUPPORT_ID,
        "directory_mode": "0700",
        "file_mode": "0600",
        "overwrite_delete_count": 0,
        "schema": "ode-edit-s05-p2r1-sealed-target-transport-intent/v1",
        "support_id": SUPPORT_ID,
    }
    intent["root_digest"] = canonical_root(intent)
    write_json(destination / "transport-intent.json", intent)

    payload_files = sorted(
        path
        for path in destination.rglob("*")
        if path.is_file() and path.name not in {"package-manifest.json", "handoff-receipt.json"}
    )
    members = [member(path, destination) for path in payload_files]
    members_root = canonical_root(members)
    file_tree_lines = [
        f"{item['mode']}\t{item['bytes']}\t{item['sha256']}\t{item['path']}" for item in members
    ]
    file_tree_sha256 = hashlib.sha256(("\n".join(file_tree_lines) + "\n").encode()).hexdigest()
    package_manifest = {
        "file_tree_sha256": file_tree_sha256,
        "member_count": len(members),
        "members": members,
        "members_root_sha256": members_root,
        "package_root_sha256": canonical_root(
            {
                "file_tree_sha256": file_tree_sha256,
                "members_root_sha256": members_root,
                "support_id": SUPPORT_ID,
            }
        ),
        "schema": "ode-edit-s05-p2r1-sealed-target-package-manifest/v1",
        "support_id": SUPPORT_ID,
    }
    write_json(destination / "package-manifest.json", package_manifest)
    package_manifest_sha256 = sha256(destination / "package-manifest.json")
    handoff = {
        "action_counts": {
            "gpu": 0,
            "model": 0,
            "scientific_source": 0,
            "slurm": 0,
            "writer_materialization": 0,
        },
        "create_once": True,
        "file_tree_sha256": file_tree_sha256,
        "member_count": len(members),
        "overwrite_delete_count": 0,
        "package_manifest_sha256": package_manifest_sha256,
        "package_root_sha256": package_manifest["package_root_sha256"],
        "schema": "ode-edit-s05-p2r1-sealed-target-handoff-receipt/v1",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "support_id": SUPPORT_ID,
        "target_case_count": len(target_index),
        "target_total_bytes": sum(int(item["file_bytes"]) for item in target_index),
        "transport_pending": True,
    }
    handoff["root_digest"] = canonical_root(handoff)
    write_json(destination / "handoff-receipt.json", handoff)

    for directory in [destination, *[p for p in destination.rglob("*") if p.is_dir()]]:
        directory.chmod(0o700)
    for file_path in [p for p in destination.rglob("*") if p.is_file()]:
        file_path.chmod(0o600)
    print(json.dumps(handoff, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
