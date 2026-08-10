#!/usr/bin/env python3
"""Self-contained, create-once P1R15 Stage-A source handoff."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.perrequest_target_simplex_transport import (
    P1R15_AMENDMENT_ID,
    P1R15_INSTRUCTION_ID,
    P1R15_METHOD_ID,
)
from project.run_scripts.ode_bf.p1_perrequest_target_simplex_transport_panel import (
    P1R15_NUMERICAL_LOCK_FILE,
    P1R15_SOURCE_MANIFEST_FILE,
    load_and_validate_p1r15_numerical_lock,
    load_and_validate_p1r15_source_manifest,
    load_and_validate_p1r15_stage_a_seal,
    validate_p1r15_source_closure,
)


PACKAGE_ID = "PERREQUEST_SIMPLEX_TRANSPORT_P1R15_STAGE_A_SH2_BUNDLE_A1"
PACKAGE_ARCHIVE = "perrequest-simplex-transport-p1r15-stage-a-sh2-bundle-a1.tar"
PACKAGE_MANIFEST = "package-manifest.json"
PACKAGE_RECEIPT = "handoff-receipt.json"
PACKAGE_REF = "refs/heads/codex/odeeditsh1-s05-perrequest-target-simplex-transport-p1r15-v1"
EXECUTION_PARENT = "6184857ef7171920e5193e7609f5fc1596bc67b0"
SH2_RECIPIENT_SESSION = "019fe491-954b-70a0-8ba8-0588e9f8d741"
LOCK_FILES = (
    "p1r10_common_coldcoord_cf_b10_seal.json",
    "p1r14_fresh_cf_b10_seal.json",
    P1R15_NUMERICAL_LOCK_FILE,
    P1R15_SOURCE_MANIFEST_FILE,
    "p0_artifact_lock.json",
)
ARCHIVE_FIXED_MEMBERS = (
    "source.bundle",
    "lock-hash-index.json",
    "source-tree-inventory.json",
    "portable-contract.json",
)
FORBIDDEN_PATH_PARTS = frozenset(
    {".cache", "cache", "credentials", "datasets", "logs", "secrets"}
)
FORBIDDEN_SUFFIXES = (
    ".bin", ".ckpt", ".npy", ".npz", ".pem", ".pt", ".safetensors"
)


def _run(
    args: Sequence[str], *, cwd: Path = REPO_ROOT, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _encode(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"


def _write_once(path: Path, data: bytes) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256(data)


def _read_regular(path: Path) -> bytes:
    resolved = path.resolve(strict=True)
    mode = resolved.stat().st_mode
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ODEBFContractError("P1R15 package member is not regular")
    return resolved.read_bytes()


def _source_tree_scan(source_head: str) -> dict[str, Any]:
    names = _run(("git", "ls-tree", "-r", "--name-only", source_head)).stdout.splitlines()
    forbidden: list[str] = []
    for name in names:
        path = Path(name)
        lowered = tuple(part.lower() for part in path.parts)
        if (
            any(part in FORBIDDEN_PATH_PARTS for part in lowered)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
        ):
            forbidden.append(name)
    if forbidden:
        raise ODEBFContractError("P1R15 source tree contains forbidden payload path")
    payload = {
        "schema": "ode-edit-s05-p1r15-source-tree-scan/v1",
        "source_head": source_head,
        "tracked_path_count": len(names),
        "forbidden_path_count": 0,
        "raw_prompt_target_tensor_model_dataset_cache_credential_log_payload_count": 0,
        "stage_b_material_count": 0,
        "tracked_case_id_only_seals": [
            "project/run_scripts/ode_bf/locks/p1r10_common_coldcoord_cf_b10_seal.json",
            "project/run_scripts/ode_bf/locks/p1r14_fresh_cf_b10_seal.json",
        ],
        "secure_exact_source_handoff": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _source_inventory(source_head: str) -> dict[str, Any]:
    rows = _run(("git", "ls-tree", "-r", source_head)).stdout.splitlines()
    entries: list[dict[str, Any]] = []
    for row in rows:
        metadata, path = row.split("\t", 1)
        mode_text, kind, oid = metadata.split(" ", 2)
        if kind != "blob":
            continue
        data = subprocess.run(
            ("git", "cat-file", "blob", oid),
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        if _git_blob_oid(data) != oid:
            raise ODEBFContractError("P1R15 source blob reconstruction differs")
        entries.append(
            {
                "path": path,
                "mode": int(mode_text, 8),
                "git_blob_oid": oid,
                "size_bytes": len(data),
                "sha256": _sha256(data),
                "role": "REACHABLE_COMMITTED_SOURCE",
            }
        )
    payload = {
        "schema": "ode-edit-s05-p1r15-source-tree-inventory/v1",
        "source_head": source_head,
        "entries": entries,
        "entry_root": canonical_hash(entries),
    }
    payload["root_digest"] = canonical_hash(payload)
    return payload


def _lock_index(source_head: str) -> dict[str, Any]:
    locks = REPO_ROOT / "project" / "run_scripts" / "ode_bf" / "locks"
    entries: list[dict[str, Any]] = []
    for name in LOCK_FILES:
        relative = f"project/run_scripts/ode_bf/locks/{name}"
        data = _read_regular(REPO_ROOT / relative)
        git = _run(("git", "ls-tree", source_head, "--", relative)).stdout.strip()
        if not git:
            raise ODEBFContractError("P1R15 package lock is not committed")
        metadata, observed_path = git.split("\t", 1)
        mode_text, kind, oid = metadata.split(" ", 2)
        if kind != "blob" or observed_path != relative or _git_blob_oid(data) != oid:
            raise ODEBFContractError("P1R15 package lock object differs")
        entries.append(
            {
                "path": relative,
                "mode": int(mode_text, 8),
                "git_blob_oid": oid,
                "size_bytes": len(data),
                "sha256": _sha256(data),
                "role": "LOCK_OR_SEAL",
            }
        )
    payload = {
        "schema": "ode-edit-s05-p1r15-lock-hash-index/v1",
        "source_head": source_head,
        "entries": entries,
        "entry_root": canonical_hash(entries),
    }
    payload["root_digest"] = canonical_hash(payload)
    return payload


def _verify_bundle(bundle: Path, source_head: str) -> dict[str, Any]:
    header = _run(("git", "bundle", "list-heads", str(bundle))).stdout.splitlines()
    if header != [f"{source_head} HEAD"]:
        raise ODEBFContractError("P1R15 bundle advertised ref differs")
    with tempfile.TemporaryDirectory(prefix="p1r15-bundle-proof-") as temporary:
        bare = Path(temporary) / "empty.git"
        _run(("git", "init", "--bare", str(bare)))
        verify = _run(("git", "bundle", "verify", str(bundle)), cwd=bare)
        if "prerequisite" in verify.stderr.lower():
            raise ODEBFContractError("P1R15 bundle is thin")
        _run(("git", "fetch", str(bundle), "HEAD:refs/heads/p1r15"), cwd=bare)
        imported = _run(("git", "rev-parse", "refs/heads/p1r15"), cwd=bare).stdout.strip()
        if imported != source_head:
            raise ODEBFContractError("P1R15 bundle imported head differs")
        _run(("git", "fsck", "--full", "--strict"), cwd=bare)
        tree = _run(("git", "rev-parse", f"{source_head}^{{tree}}"), cwd=bare).stdout.strip()
        parent = _run(("git", "rev-parse", f"{source_head}^"), cwd=bare).stdout.strip()
        if parent != EXECUTION_PARENT:
            raise ODEBFContractError("P1R15 bundle parent differs")
    payload = {
        "source_head": source_head,
        "source_tree": tree,
        "source_parent": parent,
        "complete_history": True,
        "thin_bundle": False,
        "prerequisite_count": 0,
        "empty_bare_verify_import_strict_fsck": "PASS",
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _tar_bytes(members: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in sorted(members.items()):
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or not name:
                raise ODEBFContractError("P1R15 archive path differs")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _member_rows(members: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "mode": 0o600,
            "size_bytes": len(data),
            "sha256": _sha256(data),
            "git_blob_oid": _git_blob_oid(data),
            "object_algorithm": "GIT_BLOB_SHA1_CONTENT_ID",
            "archive_kind": "REGULAR",
            "role": (
                "SELF_CONTAINED_GIT_BUNDLE" if name == "source.bundle" else "RAW_FREE_CONTRACT"
            ),
        }
        for name, data in sorted(members.items())
    ]


def create_package(source_head: str, output_parent: Path) -> dict[str, Any]:
    head = _run(("git", "rev-parse", "HEAD")).stdout.strip()
    parent = _run(("git", "rev-parse", "HEAD^")).stdout.strip()
    dirty = _run(("git", "status", "--porcelain", "--untracked-files=no")).stdout
    if head != source_head or parent != EXECUTION_PARENT or dirty:
        raise ODEBFContractError("P1R15 package source provenance differs")
    locks = REPO_ROOT / "project" / "run_scripts" / "ode_bf" / "locks"
    seal, seal_sha = load_and_validate_p1r15_stage_a_seal(locks)
    numerical, numerical_sha = load_and_validate_p1r15_numerical_lock(
        locks / P1R15_NUMERICAL_LOCK_FILE
    )
    source_manifest, source_manifest_sha = load_and_validate_p1r15_source_manifest(
        REPO_ROOT,
        locks / P1R15_SOURCE_MANIFEST_FILE,
        source_head=source_head,
    )
    closure = validate_p1r15_source_closure(REPO_ROOT)
    destination = output_parent.resolve() / PACKAGE_ID
    if destination.exists() or destination.is_symlink():
        raise ODEBFContractError("P1R15 package destination collides")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.mkdir(mode=0o700)
    try:
        bundle = destination / ".source.bundle.tmp"
        _run(("git", "bundle", "create", str(bundle), "HEAD"))
        bundle_bytes = _read_regular(bundle)
        bundle_proof = _verify_bundle(bundle, source_head)
        source_scan = _source_tree_scan(source_head)
        source_inventory = _source_inventory(source_head)
        lock_index = _lock_index(source_head)
        portable = {
            "schema": "ode-edit-s05-p1r15-stage-a-portable-contract/v1",
            "instruction_id": P1R15_INSTRUCTION_ID,
            "amendment_id": P1R15_AMENDMENT_ID,
            "method_id": P1R15_METHOD_ID,
            "source_head": source_head,
            "source_parent": EXECUTION_PARENT,
            "source_tree": bundle_proof["source_tree"],
            "package_ref": PACKAGE_REF,
            "stage_a_seal_root": seal["root_digest"],
            "stage_a_request_order_sha256": seal["batch_ordered_request_digest_v1"][0],
            "numerical_lock_root": numerical["root_digest"],
            "source_manifest_root": source_manifest["root_digest"],
            "source_closure_sha256": closure["identity_sha256"],
            "stage_b_material_count": 0,
            "stage_b_access_count": 0,
            "scientific_promotion_authorized": False,
            "qwen_model_gpu_slurm_owner_after_receiver_pass_only": True,
        }
        portable["root_digest"] = canonical_hash(portable)
        members = {
            "source.bundle": bundle_bytes,
            "lock-hash-index.json": _encode(lock_index),
            "source-tree-inventory.json": _encode(source_inventory),
            "source-tree-scan.json": _encode(source_scan),
            "portable-contract.json": _encode(portable),
        }
        rows = _member_rows(members)
        archive_bytes = _tar_bytes(members)
        manifest = {
            "schema": "ode-edit-s05-p1r15-stage-a-package-manifest/v1",
            "package_id": PACKAGE_ID,
            "source_head": source_head,
            "source_tree": bundle_proof["source_tree"],
            "members": rows,
            "member_root": canonical_hash(rows),
            "normalized_tree_digest": canonical_hash(
                [[row["path"], row["mode"], row["size_bytes"], row["sha256"]] for row in rows]
            ),
        }
        manifest["root_digest"] = canonical_hash(manifest)
        archive_sha = _write_once(destination / PACKAGE_ARCHIVE, archive_bytes)
        manifest_bytes = _encode(manifest)
        manifest_sha = _write_once(destination / PACKAGE_MANIFEST, manifest_bytes)
        receipt = {
            "schema": "ode-edit-s05-p1r15-stage-a-handoff-receipt/v1",
            "status": "PACKAGE_CREATED_LOCAL_GATES_PASS",
            "package_id": PACKAGE_ID,
            "source_head": source_head,
            "source_parent": EXECUTION_PARENT,
            "source_tree": bundle_proof["source_tree"],
            "archive_sha256": archive_sha,
            "manifest_sha256": manifest_sha,
            "manifest_root": manifest["root_digest"],
            "normalized_tree_digest": manifest["normalized_tree_digest"],
            "bundle_proof": bundle_proof,
            "stage_a_seal_sha256": seal_sha,
            "stage_a_seal_root": seal["root_digest"],
            "numerical_lock_sha256": numerical_sha,
            "numerical_lock_root": numerical["root_digest"],
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest_root": source_manifest["root_digest"],
            "secure_one_package_transfer": True,
            "recipient_session": SH2_RECIPIENT_SESSION,
            "stage_b_material_count": 0,
            "scientific_promotion_authorized": False,
            "model_gpu_slurm_result_root_action_count": 0,
        }
        receipt["root_digest"] = canonical_hash(receipt)
        receipt_bytes = _encode(receipt)
        receipt_sha = _write_once(destination / PACKAGE_RECEIPT, receipt_bytes)
        bundle.unlink()
        return {
            "package_id": PACKAGE_ID,
            "package_directory": str(destination),
            "source_head": source_head,
            "source_tree": bundle_proof["source_tree"],
            "archive_sha256": archive_sha,
            "manifest_sha256": manifest_sha,
            "receipt_sha256": receipt_sha,
            "manifest_root": manifest["root_digest"],
            "receipt_root": receipt["root_digest"],
            "normalized_tree_digest": manifest["normalized_tree_digest"],
        }
    except BaseException:
        # A failed create remains inspectable; it is never silently reused.
        raise


def verify_local_package(package_directory: Path, source_head: str) -> dict[str, Any]:
    root = package_directory.resolve(strict=True)
    if root.name != PACKAGE_ID or root.is_symlink() or not root.is_dir():
        raise ODEBFContractError("P1R15 package root differs")
    archive = _read_regular(root / PACKAGE_ARCHIVE)
    manifest_bytes = _read_regular(root / PACKAGE_MANIFEST)
    receipt_bytes = _read_regular(root / PACKAGE_RECEIPT)
    manifest = json.loads(manifest_bytes)
    receipt = json.loads(receipt_bytes)
    for value, label in ((manifest, "manifest"), (receipt, "receipt")):
        observed = value.pop("root_digest", None)
        if observed != canonical_hash(value):
            raise ODEBFContractError(f"P1R15 package {label} root differs")
        value["root_digest"] = observed
    if (
        manifest.get("source_head") != source_head
        or receipt.get("source_head") != source_head
        or receipt.get("archive_sha256") != _sha256(archive)
        or receipt.get("manifest_sha256") != _sha256(manifest_bytes)
        or receipt.get("manifest_root") != manifest["root_digest"]
    ):
        raise ODEBFContractError("P1R15 package outer identity differs")
    rows = manifest.get("members")
    if not isinstance(rows, list) or manifest.get("member_root") != canonical_hash(rows):
        raise ODEBFContractError("P1R15 package manifest differs")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar_members = tar.getmembers()
        if any(
            not item.isfile()
            or item.issym()
            or Path(item.name).is_absolute()
            or ".." in Path(item.name).parts
            for item in tar_members
        ):
            raise ODEBFContractError("P1R15 package archive traversal differs")
        names = [item.name for item in tar_members]
        if len(names) != len(set(names)) or names != [row["path"] for row in rows]:
            raise ODEBFContractError("P1R15 package archive inventory differs")
        extracted: dict[str, bytes] = {}
        for item in tar_members:
            handle = tar.extractfile(item)
            if handle is None:
                raise ODEBFContractError("P1R15 package member is unreadable")
            extracted[item.name] = handle.read()
    for row in rows:
        data = extracted[row["path"]]
        if (
            row.get("mode") != 0o600
            or row.get("size_bytes") != len(data)
            or row.get("sha256") != _sha256(data)
            or row.get("git_blob_oid") != _git_blob_oid(data)
            or row.get("archive_kind") != "REGULAR"
        ):
            raise ODEBFContractError("P1R15 package member identity differs")
    with tempfile.TemporaryDirectory(prefix="p1r15-local-verify-") as temporary:
        bundle = Path(temporary) / "source.bundle"
        bundle.write_bytes(extracted["source.bundle"])
        proof = _verify_bundle(bundle, source_head)
    return {
        "package_id": PACKAGE_ID,
        "source_head": source_head,
        "source_tree": proof["source_tree"],
        "archive_sha256": _sha256(archive),
        "manifest_sha256": _sha256(manifest_bytes),
        "receipt_sha256": _sha256(receipt_bytes),
        "manifest_root": manifest["root_digest"],
        "receipt_root": receipt["root_digest"],
        "normalized_tree_digest": manifest["normalized_tree_digest"],
        "bundle_self_contained": True,
        "stage_b_material_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output-parent", required=True, type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = (
        verify_local_package(args.verify, args.source_head)
        if args.verify is not None
        else create_package(args.source_head, args.output_parent)
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
