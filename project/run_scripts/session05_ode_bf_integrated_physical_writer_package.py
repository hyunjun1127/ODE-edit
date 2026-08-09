#!/usr/bin/env python3
"""Create a self-contained, create-once P1R14 source handoff."""

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
from project.run_scripts import session05_ode_bf_integrated_physical_writer_dry_plan as dry
from project.run_scripts.ode_bf.integrated_physical_writer import (
    INTEGRATED_PHYSICAL_WRITER_AMENDMENT_IDS,
    INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
    INTEGRATED_PHYSICAL_WRITER_METHOD_ID,
)
from project.run_scripts.ode_bf.p1_integrated_physical_writer_panel import (
    P1R14_EXCLUSION_FILE,
    P1R14_FRESH_SEAL_FILE,
    P1R14_NUMERICAL_LOCK_FILE,
    P1R14_SOURCE_MANIFEST_FILE,
    P1R14_SCIENTIFIC_SOURCE_PATHS,
    P1R14_SESSION_SOURCE_PATHS,
    P1R14_TEST_SOURCE_PATHS,
    load_and_validate_p1r14_numerical_lock,
    load_and_validate_p1r14_seals,
    load_and_validate_p1r14_source_manifest,
    validate_p1r14_source_closure,
)


PACKAGE_ID = "INTEGRATED_PHYSICAL_WRITER_P1R14_SH2_BUNDLE_A3_TECH_R3"
PACKAGE_ARCHIVE = "integrated-physical-writer-p1r14-sh2-bundle-a3-tech-r3.tar"
PACKAGE_MANIFEST = "package-manifest.json"
PACKAGE_RECEIPT = "handoff-receipt.json"
PACKAGE_REF = "refs/heads/codex/odeeditsh1-s05-integrated-physical-writer-p1r14-v1"
EXECUTION_PARENT = "186ded6c2f75cb77c8d777b776a423d8f422dd79"
SESSION_ID = "019fe489-c968-75f3-9965-7cfbc26c0a99"
SH2_RECIPIENT_SESSION = "019fe491-954b-70a0-8ba8-0588e9f8d741"
LOCK_FILES = (
    P1R14_EXCLUSION_FILE,
    P1R14_FRESH_SEAL_FILE,
    P1R14_NUMERICAL_LOCK_FILE,
    P1R14_SOURCE_MANIFEST_FILE,
    "p0_artifact_lock.json",
)
ARCHIVE_MEMBER_NAMES = (
    "source.bundle",
    "lock-hash-index.json",
    "source-tree-inventory.json",
    "source-object-scan.json",
    "portable-contract.json",
)
SOURCE_TREE_PATHS = tuple(
    sorted(
        {
            *P1R14_SCIENTIFIC_SOURCE_PATHS,
            *P1R14_SESSION_SOURCE_PATHS,
            *P1R14_TEST_SOURCE_PATHS,
            *(f"project/run_scripts/ode_bf/locks/{name}" for name in LOCK_FILES),
        }
    )
)
FORBIDDEN_JSON_KEYS = frozenset(
    {
        "prompt", "prompts", "subject", "target_new", "target_true",
        "target_old", "token_ids", "input_ids", "requested_rewrite",
        "neighborhood_prompts", "paraphrase_prompts", "rewrite_prompt",
        "raw_text", "password", "credential", "credentials", "private_key",
        "key_path", "host_name", "hostname", "user_name", "username", "port",
    }
)
CASE_ID_FIELD_NAMES = frozenset(
    {"case_id", "case_ids", "prior_case_id", "prior_case_ids"}
)
TARGET_IDENTITY_FIELD_NAMES = frozenset(
    {
        "frozen_target_load_once_per_case", "frozen_target_replay_exact",
        "memit_target_hparams_sha256", "target", "target_anchor_sha256",
        "target_old_source", "target_span_sha256", "target_token_sha256",
    }
)
PUBLIC_HOST_IDENTITY_PATHS = frozenset(
    {"agents/server2/head-server2.json", "workers/server2/status.json"}
)
PUBLIC_SOURCE_PATH_EXCEPTIONS = frozenset(
    {"local/README.md", "servers/templates/session-boundary.env"}
)
FORBIDDEN_BUNDLE_PATH_COMPONENTS = frozenset(
    {".cache", "cache", "credentials", "data", "datasets", "logs", "secrets"}
)
FORBIDDEN_BUNDLE_SUFFIXES = (
    ".bin", ".ckpt", ".npy", ".npz", ".pem", ".pt", ".safetensors",
)


def _run(
    args: Sequence[str], *, cwd: Path = REPO_ROOT, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args), cwd=cwd, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _git_blob_bytes(object_id: str) -> bytes:
    return subprocess.run(
        ("git", "cat-file", "blob", object_id),
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _is_hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(character in "0123456789abcdef" for character in value)
    )


def _case_id_content_scan(value: Any) -> dict[str, Any]:
    """Allow only already-committed numeric IDs plus raw-free identities."""

    field_names: set[str] = set()
    case_value_count = 0
    target_identity_fields: set[str] = set()
    tensor_digest_fields: set[str] = set()

    def walk(item: Any) -> None:
        nonlocal case_value_count
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ODEBFContractError("integrated case-ID key differs")
                field_names.add(key)
                lowered = key.lower()
                if lowered in CASE_ID_FIELD_NAMES:
                    values = child if isinstance(child, list) else [child]
                    if not values or any(
                        isinstance(member, bool)
                        or (
                            isinstance(member, int)
                            and member < 0
                        )
                        or (
                            not isinstance(member, int)
                            and not (
                                isinstance(member, str)
                                and member
                                and member.isdecimal()
                            )
                        )
                        for member in values
                    ):
                        raise ODEBFContractError(
                            "integrated case-ID value differs"
                        )
                    case_value_count += len(values)
                elif lowered in FORBIDDEN_JSON_KEYS:
                    raise ODEBFContractError(
                        "integrated raw content field is forbidden"
                    )
                if "prompt" in lowered or lowered in {
                    "raw_text", "subject", "token_ids", "input_ids",
                }:
                    raise ODEBFContractError(
                        "integrated prompt or token content is forbidden"
                    )
                if "target" in lowered:
                    if key not in TARGET_IDENTITY_FIELD_NAMES:
                        raise ODEBFContractError(
                            "integrated target content field is forbidden"
                        )
                    if isinstance(child, str) and not (
                        _is_hex_digest(child) or key == "target_old_source"
                    ):
                        raise ODEBFContractError(
                            "integrated target content value is forbidden"
                        )
                    if isinstance(child, (list, Mapping)):
                        raise ODEBFContractError(
                            "integrated target payload is forbidden"
                        )
                    target_identity_fields.add(key)
                if "tensor" in lowered:
                    if not _is_hex_digest(child):
                        raise ODEBFContractError(
                            "integrated tensor payload is forbidden"
                        )
                    tensor_digest_fields.add(key)
                if any(
                    marker in lowered
                    for marker in ("credential", "password", "private_key", "secret")
                ):
                    raise ODEBFContractError(
                        "integrated credential content is forbidden"
                    )
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    if case_value_count <= 0:
        raise ODEBFContractError("integrated case-ID exception is empty")
    hash_fields = sorted(
        key
        for key in field_names
        if key.endswith("sha256")
        or key.endswith("digest")
        or key.endswith("root_digest")
    )
    control_fields = sorted(
        field_names
        - CASE_ID_FIELD_NAMES
        - set(hash_fields)
        - target_identity_fields
        - tensor_digest_fields
    )
    return {
        "case_id_field_names": sorted(CASE_ID_FIELD_NAMES & field_names),
        "case_id_value_count": case_value_count,
        "hash_identity_field_names": hash_fields,
        "target_identity_field_names": sorted(target_identity_fields),
        "tensor_digest_field_names": sorted(tensor_digest_fields),
        "control_field_names": control_fields,
        "prompt_or_target_content_match_count": 0,
        "tensor_payload_match_count": 0,
        "credential_or_private_config_match_count": 0,
        "content_scanner_pass": True,
    }


def _owning_commit_identity(
    source_head: str, relative: str, object_id: str
) -> tuple[str, str, int]:
    commits = _run(("git", "log", "--format=%H", source_head, "--", relative)).stdout.splitlines()
    for commit in commits:
        row = _run(("git", "ls-tree", commit, "--", relative)).stdout.rstrip("\n")
        if "\t" not in row:
            continue
        metadata, observed_path = row.split("\t", 1)
        mode, kind, observed_object = metadata.split(" ", 2)
        if observed_path == relative and kind == "blob" and observed_object == object_id:
            tree = _run(("git", "rev-parse", commit + "^{tree}")).stdout.strip()
            return commit, tree, int(mode, 8)
    raise ODEBFContractError("integrated case-ID owning commit is unavailable")


def _source_bundle_content_scan(source_head: str) -> dict[str, Any]:
    """Scan every named reachable blob before creating a portable bundle."""

    lines = _run(("git", "rev-list", "--objects", source_head)).stdout.splitlines()
    exceptions: list[dict[str, Any]] = []
    public_host_identity_paths: set[str] = set()
    json_blob_count = 0
    for line in lines:
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        object_id, relative = parts
        path = Path(relative)
        lowered_parts = {part.lower() for part in path.parts}
        if relative not in PUBLIC_SOURCE_PATH_EXCEPTIONS and (
            lowered_parts & FORBIDDEN_BUNDLE_PATH_COMPONENTS
            or relative.lower().endswith(FORBIDDEN_BUNDLE_SUFFIXES)
            or relative.startswith("local/")
            or relative.startswith("servers/local/")
        ):
            raise ODEBFContractError(
                "integrated source bundle contains a forbidden path class"
            )
        if not relative.endswith(".json"):
            continue
        data = _git_blob_bytes(object_id)
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ODEBFContractError(
                "integrated source bundle JSON differs"
            ) from exc
        json_blob_count += 1
        all_keys: set[str] = set()

        def collect(item: Any) -> None:
            if isinstance(item, Mapping):
                all_keys.update(str(key) for key in item)
                for child in item.values():
                    collect(child)
            elif isinstance(item, list):
                for child in item:
                    collect(child)

        collect(value)
        case_fields = CASE_ID_FIELD_NAMES & {key.lower() for key in all_keys}
        if case_fields:
            scanner = _case_id_content_scan(value)
            commit, tree, mode = _owning_commit_identity(
                source_head, relative, object_id
            )
            exceptions.append(
                {
                    "canonical_path": relative,
                    "owning_commit": commit,
                    "owning_tree": tree,
                    "committed_mode": mode,
                    "blob_object_id": object_id,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                    "field_name_classification": scanner,
                    "necessity_proof": (
                        "REACHABLE_TREE_BLOB_REQUIRED_FOR_EXACT_COMMIT_CHECKOUT"
                    ),
                    "case_ids_already_committed": True,
                }
            )
        forbidden = {
            key.lower() for key in all_keys if key.lower() in FORBIDDEN_JSON_KEYS
        }
        if forbidden:
            if forbidden == {"hostname"} and relative in PUBLIC_HOST_IDENTITY_PATHS:
                public_host_identity_paths.add(relative)
            else:
                raise ODEBFContractError(
                    "integrated source bundle contains raw/private JSON fields"
                )
    exceptions.sort(key=lambda item: (item["canonical_path"], item["blob_object_id"]))
    identities = [
        (str(item["canonical_path"]), str(item["blob_object_id"]))
        for item in exceptions
    ]
    if len(identities) != len(set(identities)) or not exceptions:
        raise ODEBFContractError(
            "integrated case-ID exception inventory differs"
        )
    payload = {
        "schema": "ode-edit-s05-integrated-physical-writer-p1r14-source-object-scan/v1",
        "source_head": source_head,
        "reachable_object_count": len(lines),
        "json_blob_count": json_blob_count,
        "tracked_case_id_only_exception": exceptions,
        "tracked_case_id_only_exception_count": len(exceptions),
        "public_host_role_identity_paths": sorted(public_host_identity_paths),
        "public_nonprivate_template_paths": sorted(PUBLIC_SOURCE_PATH_EXCEPTIONS),
        "forbidden_path_match_count": 0,
        "raw_prompt_or_target_content_match_count": 0,
        "tensor_payload_match_count": 0,
        "credential_private_config_runtime_log_match_count": 0,
        "scientific_promotion_authorized": False,
        "secure_exact_source_handoff_exception_authorized": True,
    }
    payload["root_digest"] = canonical_hash(payload)
    return payload


def _encode(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"


def _write_once(path: Path, data: bytes) -> str:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256(data)


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ODEBFContractError("integrated package source is not regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(data) != before.st_size
    ):
        raise ODEBFContractError("integrated package source changed during read")
    return data


def _provenance(source_head: str) -> dict[str, Any]:
    head = _run(("git", "rev-parse", "HEAD")).stdout.strip()
    parent = _run(("git", "rev-parse", "HEAD^")).stdout.strip()
    branch = _run(("git", "branch", "--show-current")).stdout.strip()
    dirty = _run(("git", "status", "--porcelain", "--untracked-files=no")).stdout
    if (
        head != source_head
        or parent != EXECUTION_PARENT
        or f"refs/heads/{branch}" != PACKAGE_REF
        or dirty
    ):
        raise ODEBFContractError("integrated package provenance differs")
    return {
        "source_head": head,
        "exact_parent": parent,
        "advertised_ref": PACKAGE_REF,
        "tracked_and_index_clean": True,
    }


def _lock_index(source_head: str) -> dict[str, Any]:
    root = REPO_ROOT / "project/run_scripts/ode_bf/locks"
    source_tree = _run(("git", "rev-parse", source_head + "^{tree}")).stdout.strip()
    rows: list[dict[str, Any]] = []
    for name in LOCK_FILES:
        relative = f"project/run_scripts/ode_bf/locks/{name}"
        path = root / name
        data = _read_regular(path)
        tree = _run(("git", "ls-tree", source_head, "--", relative)).stdout.rstrip()
        if "\t" not in tree:
            raise ODEBFContractError("integrated package lock is not committed")
        metadata, observed = tree.split("\t", 1)
        mode, kind, oid = metadata.split(" ", 2)
        if observed != relative or kind != "blob" or oid != _git_blob_oid(data):
            raise ODEBFContractError("integrated package lock tree differs")
        rows.append(
            {
                "path": relative,
                "mode": int(mode, 8),
                "size": len(data),
                "sha256": _sha256(data),
                "git_blob_oid": oid,
                "object_algorithm": "GIT_BLOB_SHA1",
                "committed_source_head": source_head,
                "committed_source_tree": source_tree,
                "role": "LOCK_OR_PROVENANCE",
            }
        )
    payload = {
        "schema": "ode-edit-s05-integrated-physical-writer-p1r14-lock-index/v1",
        "source_head": source_head,
        "entries": rows,
        "entry_root": canonical_hash(rows),
    }
    payload["root_digest"] = canonical_hash(payload)
    return payload


def _source_tree_inventory(source_head: str) -> dict[str, Any]:
    source_tree = _run(("git", "rev-parse", source_head + "^{tree}")).stdout.strip()
    entries: list[dict[str, Any]] = []
    for relative in SOURCE_TREE_PATHS:
        row = _run(("git", "ls-tree", source_head, "--", relative)).stdout.rstrip("\n")
        if "\t" not in row:
            raise ODEBFContractError("integrated source inventory member is absent")
        metadata, observed_path = row.split("\t", 1)
        mode, kind, object_id = metadata.split(" ", 2)
        if observed_path != relative or kind != "blob":
            raise ODEBFContractError("integrated source inventory tree differs")
        object_data = subprocess.run(
            ("git", "cat-file", "blob", object_id),
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        working = _read_regular(REPO_ROOT / relative)
        if object_data != working or object_id != _git_blob_oid(object_data):
            raise ODEBFContractError("integrated source inventory bytes differ")
        entries.append(
            {
                "path": relative,
                "mode": int(mode, 8),
                "size": len(object_data),
                "sha256": _sha256(object_data),
                "git_blob_oid": object_id,
                "object_algorithm": "GIT_BLOB_SHA1",
                "role": (
                    "TRACKED_LOCK_OR_PROVENANCE"
                    if "/locks/" in relative
                    else "TRACKED_RUNTIME_TEST_OR_SESSION_SOURCE"
                ),
                "committed_source_head": source_head,
                "committed_source_tree": source_tree,
            }
        )
    if [item["path"] for item in entries] != list(SOURCE_TREE_PATHS):
        raise ODEBFContractError("integrated source inventory order differs")
    payload = {
        "schema": "ode-edit-s05-integrated-physical-writer-p1r14-source-tree-inventory/v1",
        "source_head": source_head,
        "source_tree": source_tree,
        "object_algorithm": "GIT_BLOB_SHA1",
        "entries": entries,
        "entry_count": len(entries),
        "entry_root": canonical_hash(entries),
    }
    payload["root_digest"] = canonical_hash(payload)
    return payload


def _parse_bundle_header(path: Path) -> dict[str, Any]:
    heads: list[dict[str, str]] = []
    prerequisites: list[str] = []
    with path.open("rb") as handle:
        first = handle.readline().decode("utf-8").rstrip("\n")
        if first not in ("# v2 git bundle", "# v3 git bundle"):
            raise ODEBFContractError("integrated git bundle schema differs")
        for raw in handle:
            line = raw.decode("utf-8").rstrip("\n")
            if not line:
                break
            if line.startswith("-"):
                prerequisites.append(line[1:].split(" ", 1)[0])
                continue
            commit, name = line.split(" ", 1)
            heads.append({"commit": commit, "name": name})
    return {"version": first, "heads": heads, "prerequisites": prerequisites}


def _verify_bundle(
    bundle: Path,
    source_head: str,
    *,
    verify_imported_contract: bool = True,
) -> dict[str, Any]:
    header = _parse_bundle_header(bundle)
    if header["heads"] != [{"commit": source_head, "name": PACKAGE_REF}]:
        raise ODEBFContractError("integrated package advertised ref differs")
    if header["prerequisites"]:
        raise ODEBFContractError("integrated package bundle is thin")
    with tempfile.TemporaryDirectory(prefix="p1r14-bundle-proof-") as temporary:
        root = Path(temporary)
        bare = root / "empty.git"
        checkout = root / "checkout"
        _run(("git", "init", "--bare", str(bare)))
        _run(("git", "bundle", "verify", str(bundle)), cwd=bare)
        _run(
            (
                "git", "fetch", str(bundle),
                f"{PACKAGE_REF}:refs/heads/p1r14-import",
            ),
            cwd=bare,
        )
        _run(("git", "fsck", "--full", "--strict"), cwd=bare)
        imported = _run(
            ("git", "rev-parse", "refs/heads/p1r14-import"), cwd=bare
        ).stdout.strip()
        if imported != source_head:
            raise ODEBFContractError("integrated package imported head differs")
        _run(("git", "clone", "--no-checkout", "--no-local", str(bare), str(checkout)))
        _run(("git", "checkout", "--detach", source_head), cwd=checkout)
        checked = _run(("git", "rev-parse", "HEAD"), cwd=checkout).stdout.strip()
        parent = _run(("git", "rev-parse", "HEAD^"), cwd=checkout).stdout.strip()
        tree = _run(("git", "rev-parse", "HEAD^{tree}"), cwd=checkout).stdout.strip()
        dirty = _run(("git", "status", "--porcelain"), cwd=checkout).stdout
        imported_contract: dict[str, Any] = {}
        if verify_imported_contract:
            locks = checkout / "project/run_scripts/ode_bf/locks"
            exclusion, seal = load_and_validate_p1r14_seals(locks)
            numerical, numerical_sha = load_and_validate_p1r14_numerical_lock(
                locks / P1R14_NUMERICAL_LOCK_FILE,
                exclusion_root=exclusion["root_digest"],
                fresh_seal_root=seal["root_digest"],
                artifact_lock_root=_sha256(
                    _read_regular(locks / "p0_artifact_lock.json")
                ),
            )
            source_manifest, source_manifest_sha = load_and_validate_p1r14_source_manifest(
                checkout,
                locks / P1R14_SOURCE_MANIFEST_FILE,
                source_head=source_head,
            )
            closure = validate_p1r14_source_closure(checkout)
            imported_contract = {
                "imported_fresh_seal_root": seal["root_digest"],
                "imported_historical_exclusion_root": exclusion["root_digest"],
                "imported_numerical_lock_root": numerical["root_digest"],
                "imported_numerical_lock_sha256": numerical_sha,
                "imported_source_manifest_root": source_manifest["root_digest"],
                "imported_source_manifest_sha256": source_manifest_sha,
                "imported_source_closure_sha256": closure["identity_sha256"],
            }
        if checked != source_head or parent != EXECUTION_PARENT or dirty:
            raise ODEBFContractError("integrated package checkout lineage differs")
        return {
            "bundle_header_version": header["version"],
            "advertised_ref": PACKAGE_REF,
            "complete_history": True,
            "prerequisite_count": 0,
            "empty_bare_verify": True,
            "strict_fsck": True,
            "detached_checkout": True,
            "source_head": checked,
            "exact_parent": parent,
            "source_tree": tree,
            "checkout_clean": True,
            "imported_contract_verification": verify_imported_contract,
            **imported_contract,
        }


def _tar_bytes(members: Mapping[str, bytes]) -> bytes:
    if tuple(sorted(members)) != tuple(sorted(ARCHIVE_MEMBER_NAMES)):
        raise ODEBFContractError("integrated package archive inventory differs")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(members):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ODEBFContractError("integrated package path escapes")
            data = members[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def verify_package_manifest(
    manifest: Mapping[str, Any], archive_data: bytes
) -> dict[str, Any]:
    payload = dict(manifest)
    root = payload.pop("root_digest", None)
    if root != canonical_hash(payload):
        raise ODEBFContractError("integrated package manifest root differs")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(ARCHIVE_MEMBER_NAMES):
        raise ODEBFContractError("integrated package manifest inventory differs")
    by_path: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ODEBFContractError("integrated package manifest entry differs")
        required = {
            "path", "mode", "size", "sha256", "git_blob_oid", "role",
            "semantic_role", "archive_kind", "committed_source_member",
            "object_algorithm",
        }
        if set(entry) != required:
            raise ODEBFContractError("integrated package manifest fields differ")
        path = str(entry["path"])
        if (
            path in by_path
            or path not in ARCHIVE_MEMBER_NAMES
            or entry["mode"] != 0o600
            or not isinstance(entry["size"], int)
            or entry["size"] < 0
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or not isinstance(entry["git_blob_oid"], str)
            or len(entry["git_blob_oid"]) != 40
            or entry["object_algorithm"] != "GIT_BLOB_SHA1"
            or entry["archive_kind"] != "REGULAR_FILE"
            or entry["committed_source_member"] is not False
        ):
            raise ODEBFContractError("integrated package manifest value differs")
        by_path[path] = entry
    observed: list[list[Any]] = []
    with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:") as archive:
        members = archive.getmembers()
        if len(members) != len(ARCHIVE_MEMBER_NAMES):
            raise ODEBFContractError("integrated package archive count differs")
        if [member.name for member in members] != sorted(by_path):
            raise ODEBFContractError("integrated package archive order differs")
        for member in members:
            path = member.name
            if (
                path not in by_path
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or Path(path).is_absolute()
                or ".." in Path(path).parts
                or member.mode != 0o600
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
            ):
                raise ODEBFContractError("integrated package archive member differs")
            stream = archive.extractfile(member)
            if stream is None:
                raise ODEBFContractError("integrated package member is unreadable")
            data = stream.read()
            entry = by_path[path]
            if (
                len(data) != entry["size"]
                or _sha256(data) != entry["sha256"]
                or _git_blob_oid(data) != entry["git_blob_oid"]
            ):
                raise ODEBFContractError("integrated package member identity differs")
            observed.append(
                [path, member.mode, len(data), _sha256(data), _git_blob_oid(data)]
            )
    expected_tree = canonical_hash(sorted(observed))
    if (
        payload.get("schema")
        != "ode-edit-s05-integrated-physical-writer-p1r14-package-manifest/v1"
        or payload.get("package_id") != PACKAGE_ID
        or not isinstance(payload.get("source_head"), str)
        or len(str(payload.get("source_head"))) != 40
        or payload.get("exact_parent") != EXECUTION_PARENT
        or payload.get("object_algorithm") != "GIT_BLOB_SHA1"
        or payload.get("source_tree_inventory_root") is None
        or payload.get("source_object_scan_root") is None
        or payload.get("lock_index_root") is None
        or payload.get("bundle_proof_identity") is None
        or payload.get("entry_count") != len(entries)
        or payload.get("archive_member_count") != len(ARCHIVE_MEMBER_NAMES)
        or payload.get("scientific_promotion_authorized") is not False
        or payload.get("model_dataset_cache_credential_payload_count") != 0
        or payload.get("raw_prompt_target_tensor_payload_count") != 0
        or payload.get("case_ids_already_committed") is not True
        or payload.get("secure_create_once_recipient")
        != "SH2_SERVER2_CANONICAL_SESSION"
        or payload.get("recipient_session") != SH2_RECIPIENT_SESSION
        or payload.get("source_head") != payload.get("bundle_source_head")
        or payload.get("source_tree") != payload.get("bundle_source_tree")
        or payload.get("exact_parent") != payload.get("bundle_exact_parent")
        or payload.get("prerequisite_count") != 0
        or payload.get("complete_history") is not True
        or payload.get("empty_bare_verify") is not True
        or payload.get("checkout_clean") is not True
        or payload.get("strict_fsck") is not True
        or payload.get("object_algorithm") != "GIT_BLOB_SHA1"
        or payload.get("archive_sha256") != _sha256(archive_data)
        or payload.get("archive_size") != len(archive_data)
        or payload.get("entry_root") != canonical_hash(entries)
        or payload.get("normalized_tree_digest") != expected_tree
    ):
        raise ODEBFContractError("integrated package archive root differs")
    payload["root_digest"] = root
    return payload


def create_package(source_head: str, output_parent: Path) -> dict[str, Any]:
    provenance = _provenance(source_head)
    _run(("scripts/check-session-boundary.sh", SESSION_ID))
    locks = REPO_ROOT / "project/run_scripts/ode_bf/locks"
    exclusion, seal = load_and_validate_p1r14_seals(locks)
    closure = validate_p1r14_source_closure(REPO_ROOT)
    source_manifest, source_manifest_sha = load_and_validate_p1r14_source_manifest(
        REPO_ROOT,
        locks / P1R14_SOURCE_MANIFEST_FILE,
        source_head=source_head,
    )
    artifact_lock_sha = _sha256(_read_regular(locks / "p0_artifact_lock.json"))
    numerical, numerical_sha = load_and_validate_p1r14_numerical_lock(
        locks / P1R14_NUMERICAL_LOCK_FILE,
        exclusion_root=exclusion["root_digest"],
        fresh_seal_root=seal["root_digest"],
        artifact_lock_root=artifact_lock_sha,
    )
    dry_plan = dry.build_plan(source_head, repository_root=REPO_ROOT)
    if any(
        dry_plan.get(name) is not False
        for name in ("model_load", "gpu_use", "slurm_submit", "result_root_creation")
    ):
        raise ODEBFContractError("integrated package dry action boundary differs")
    parent = output_parent.expanduser().resolve()
    if parent.is_symlink() or not parent.is_dir():
        raise ODEBFContractError("integrated package parent differs")
    destination = parent / PACKAGE_ID
    if destination.exists() or destination.is_symlink():
        raise ODEBFContractError("integrated package create-once child collides")
    with tempfile.TemporaryDirectory(
        prefix=f".{PACKAGE_ID}.building-", dir=parent
    ) as temporary:
        building = Path(temporary)
        os.chmod(building, 0o700)
        bundle = building / ".source.bundle"
        _run(("git", "bundle", "create", str(bundle), PACKAGE_REF))
        bundle_proof = _verify_bundle(bundle, source_head)
        bundle_data = _read_regular(bundle)
        index = _lock_index(source_head)
        source_inventory = _source_tree_inventory(source_head)
        object_scan = _source_bundle_content_scan(source_head)
        contract = {
            "schema": "ode-edit-s05-integrated-physical-writer-p1r14-portable-contract/v1",
            "package_id": PACKAGE_ID,
            "instruction_id": INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
            "amendment_ids": list(INTEGRATED_PHYSICAL_WRITER_AMENDMENT_IDS),
            "method_id": INTEGRATED_PHYSICAL_WRITER_METHOD_ID,
            "source_head": source_head,
            "source_tree": bundle_proof["source_tree"],
            "exact_parent": EXECUTION_PARENT,
            "fresh_seal_root": seal["root_digest"],
            "historical_exclusion_root": exclusion["root_digest"],
            "numerical_lock_sha256": numerical_sha,
            "numerical_lock_root": numerical["root_digest"],
            "artifact_lock_sha256": artifact_lock_sha,
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest_root": source_manifest["root_digest"],
            "source_closure_sha256": closure["identity_sha256"],
            "source_tree_inventory_root": source_inventory["root_digest"],
            "source_object_scan_root": object_scan["root_digest"],
            "lock_index_root": index["root_digest"],
            "artifact_identities": dry_plan["artifacts"],
            "numerical_lock_identities": dry_plan["numerical_locks"],
            "resource_contract": {
                "gpu_per_job": 1,
                "cpu_per_job": 8,
                "host_memory_mib": 65_000,
                "time_limit": "23:59:00",
                "point_in_time_cap_and_capacity_recheck_required": True,
                "package_creation_model_gpu_slurm_action_count": 0,
            },
            "session_id": SESSION_ID,
            "recipient_session": SH2_RECIPIENT_SESSION,
            "one_arm_registry": True,
            "scientific_promotion_authorized": False,
            "model_dataset_cache_credential_payload_count": 0,
            "raw_prompt_target_tensor_payload_count": 0,
            "case_ids_already_committed": True,
            "secure_create_once_recipient": "SH2_SERVER2_CANONICAL_SESSION",
        }
        contract["root_digest"] = canonical_hash(contract)
        members = {
            "source.bundle": bundle_data,
            "lock-hash-index.json": _encode(index),
            "source-tree-inventory.json": _encode(source_inventory),
            "source-object-scan.json": _encode(object_scan),
            "portable-contract.json": _encode(contract),
        }
        archive_data = _tar_bytes(members)
        archive_sha = _write_once(building / PACKAGE_ARCHIVE, archive_data)
        entries = [
            {
                "path": name,
                "mode": 0o600,
                "size": len(data),
                "sha256": _sha256(data),
                "git_blob_oid": _git_blob_oid(data),
                "object_algorithm": "GIT_BLOB_SHA1",
                "role": (
                    "SELF_CONTAINED_GIT_BUNDLE"
                    if name == "source.bundle"
                    else "RAW_FREE_PROVENANCE"
                ),
                "semantic_role": "PORTABLE_SOURCE_OR_LOCK_EVIDENCE",
                "archive_kind": "REGULAR_FILE",
                "committed_source_member": False,
            }
            for name, data in sorted(members.items())
        ]
        manifest = {
            "schema": "ode-edit-s05-integrated-physical-writer-p1r14-package-manifest/v1",
            "package_id": PACKAGE_ID,
            "source_head": source_head,
            "source_tree": bundle_proof["source_tree"],
            "exact_parent": EXECUTION_PARENT,
            "bundle_source_head": bundle_proof["source_head"],
            "bundle_source_tree": bundle_proof["source_tree"],
            "bundle_exact_parent": bundle_proof["exact_parent"],
            "prerequisite_count": bundle_proof["prerequisite_count"],
            "complete_history": bundle_proof["complete_history"],
            "empty_bare_verify": bundle_proof["empty_bare_verify"],
            "strict_fsck": bundle_proof["strict_fsck"],
            "checkout_clean": bundle_proof["checkout_clean"],
            "bundle_proof_identity": canonical_hash(bundle_proof),
            "source_tree_inventory_root": source_inventory["root_digest"],
            "source_object_scan_root": object_scan["root_digest"],
            "lock_index_root": index["root_digest"],
            "object_algorithm": "GIT_BLOB_SHA1",
            "entries": entries,
            "entry_count": len(entries),
            "archive_member_count": len(ARCHIVE_MEMBER_NAMES),
            "entry_root": canonical_hash(entries),
            "archive_sha256": archive_sha,
            "archive_size": len(archive_data),
            "normalized_tree_digest": canonical_hash(
                [
                    [
                        item["path"], item["mode"], item["size"],
                        item["sha256"], item["git_blob_oid"],
                    ]
                    for item in entries
                ]
            ),
            "recipient_session": SH2_RECIPIENT_SESSION,
            "secure_create_once_recipient": "SH2_SERVER2_CANONICAL_SESSION",
            "scientific_promotion_authorized": False,
            "model_dataset_cache_credential_payload_count": 0,
            "raw_prompt_target_tensor_payload_count": 0,
            "case_ids_already_committed": True,
        }
        manifest["root_digest"] = canonical_hash(manifest)
        verify_package_manifest(manifest, archive_data)
        manifest_data = _encode(manifest)
        manifest_sha = _write_once(building / PACKAGE_MANIFEST, manifest_data)
        receipt = {
            "schema": "ode-edit-s05-integrated-physical-writer-p1r14-handoff-receipt/v1",
            "package_id": PACKAGE_ID,
            "source_head": source_head,
            "source_tree": bundle_proof["source_tree"],
            "exact_parent": EXECUTION_PARENT,
            "recipient_session": SH2_RECIPIENT_SESSION,
            "provenance": provenance,
            "bundle_proof": bundle_proof,
            "portable_contract_root": contract["root_digest"],
            "source_tree_inventory_root": source_inventory["root_digest"],
            "source_object_scan_root": object_scan["root_digest"],
            "tracked_case_id_only_exception_count": object_scan[
                "tracked_case_id_only_exception_count"
            ],
            "archive_sha256": archive_sha,
            "manifest_sha256": manifest_sha,
            "manifest_root": manifest["root_digest"],
            "normalized_tree_digest": manifest["normalized_tree_digest"],
            "top_level_file_count": 3,
            "top_level_mode": 0o600,
            "secure_one_package_transfer": True,
            "case_ids_already_committed": True,
            "raw_prompt_target_tensor_payload_included": False,
            "credential_private_config_runtime_log_included": False,
            "transfer_count_authorized": 1,
            "overwrite_delete_update_authorized": False,
            "model_gpu_slurm_result_root_action_count": 0,
            "scientific_promotion_authorized": False,
        }
        receipt["root_digest"] = canonical_hash(receipt)
        receipt_data = _encode(receipt)
        receipt_sha = _write_once(building / PACKAGE_RECEIPT, receipt_data)
        bundle.unlink()
        if destination.exists() or destination.is_symlink():
            raise ODEBFContractError("integrated package create-once child collides")
        os.rename(building, destination)
    return {
        "status": "PACKAGE_CREATED_LOCAL_VERIFY_PASS",
        "package_id": PACKAGE_ID,
        "archive_sha256": archive_sha,
        "manifest_sha256": manifest_sha,
        "receipt_sha256": receipt_sha,
        "manifest_root": manifest["root_digest"],
        "receipt_root": receipt["root_digest"],
        "normalized_tree_digest": manifest["normalized_tree_digest"],
        "source_tree_inventory_root": source_inventory["root_digest"],
        "source_object_scan_root": object_scan["root_digest"],
        "tracked_case_id_only_exception_count": object_scan[
            "tracked_case_id_only_exception_count"
        ],
        "source_head": source_head,
        "model_gpu_slurm_result_root_action_count": 0,
    }


def _decode_rooted_json(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFContractError(f"integrated {label} JSON differs") from exc
    if not isinstance(value, dict):
        raise ODEBFContractError(f"integrated {label} mapping differs")
    payload = dict(value)
    root = payload.pop("root_digest", None)
    if root != canonical_hash(payload):
        raise ODEBFContractError(f"integrated {label} root differs")
    value["root_digest"] = root
    return value


def verify_local_package(package_directory: Path, source_head: str) -> dict[str, Any]:
    """Verify the exact local package bound later by the independent SH2 ACK."""

    root = package_directory.expanduser().resolve(strict=True)
    if root.name != PACKAGE_ID or root.is_symlink() or not root.is_dir():
        raise ODEBFContractError("integrated local package directory differs")
    if stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise ODEBFContractError("integrated local package directory mode differs")
    expected = {PACKAGE_ARCHIVE, PACKAGE_MANIFEST, PACKAGE_RECEIPT}
    if {path.name for path in root.iterdir()} != expected:
        raise ODEBFContractError("integrated local package outer inventory differs")
    files: dict[str, bytes] = {}
    for name in sorted(expected):
        path = root / name
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ODEBFContractError("integrated local package outer mode differs")
        files[name] = _read_regular(path)
    manifest = _decode_rooted_json(files[PACKAGE_MANIFEST], label="manifest")
    verify_package_manifest(manifest, files[PACKAGE_ARCHIVE])
    receipt = _decode_rooted_json(files[PACKAGE_RECEIPT], label="receipt")
    if (
        receipt.get("schema")
        != "ode-edit-s05-integrated-physical-writer-p1r14-handoff-receipt/v1"
        or receipt.get("package_id") != PACKAGE_ID
        or receipt.get("source_head") != source_head
        or receipt.get("source_tree") != manifest.get("source_tree")
        or receipt.get("exact_parent") != EXECUTION_PARENT
        or receipt.get("recipient_session") != SH2_RECIPIENT_SESSION
        or receipt.get("archive_sha256") != _sha256(files[PACKAGE_ARCHIVE])
        or receipt.get("manifest_sha256") != _sha256(files[PACKAGE_MANIFEST])
        or receipt.get("manifest_root") != manifest.get("root_digest")
        or receipt.get("normalized_tree_digest")
        != manifest.get("normalized_tree_digest")
        or receipt.get("top_level_file_count") != 3
        or receipt.get("top_level_mode") != 0o600
        or receipt.get("secure_one_package_transfer") is not True
        or receipt.get("case_ids_already_committed") is not True
        or receipt.get("raw_prompt_target_tensor_payload_included") is not False
        or receipt.get("credential_private_config_runtime_log_included") is not False
        or receipt.get("model_gpu_slurm_result_root_action_count") != 0
        or receipt.get("scientific_promotion_authorized") is not False
    ):
        raise ODEBFContractError("integrated local package receipt differs")
    archived: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(files[PACKAGE_ARCHIVE]), mode="r:") as archive:
        for member in archive.getmembers():
            stream = archive.extractfile(member)
            if stream is None:
                raise ODEBFContractError("integrated local archive member differs")
            archived[member.name] = stream.read()
    contract = _decode_rooted_json(
        archived["portable-contract.json"], label="portable contract"
    )
    inventory = _decode_rooted_json(
        archived["source-tree-inventory.json"], label="source inventory"
    )
    object_scan = _decode_rooted_json(
        archived["source-object-scan.json"], label="source object scan"
    )
    lock_index = _decode_rooted_json(
        archived["lock-hash-index.json"], label="lock index"
    )
    locks = REPO_ROOT / "project/run_scripts/ode_bf/locks"
    exclusion, seal = load_and_validate_p1r14_seals(locks)
    numerical, numerical_sha = load_and_validate_p1r14_numerical_lock(
        locks / P1R14_NUMERICAL_LOCK_FILE,
        exclusion_root=exclusion["root_digest"],
        fresh_seal_root=seal["root_digest"],
        artifact_lock_root=_sha256(_read_regular(locks / "p0_artifact_lock.json")),
    )
    source_manifest, source_manifest_sha = load_and_validate_p1r14_source_manifest(
        REPO_ROOT,
        locks / P1R14_SOURCE_MANIFEST_FILE,
        source_head=source_head,
    )
    source_closure = validate_p1r14_source_closure(REPO_ROOT)
    inventory_entries = inventory.get("entries")
    lock_entries = lock_index.get("entries")
    if (
        contract.get("package_id") != PACKAGE_ID
        or contract.get("source_head") != source_head
        or contract.get("exact_parent") != EXECUTION_PARENT
        or contract.get("recipient_session") != SH2_RECIPIENT_SESSION
        or contract.get("source_tree_inventory_root") != inventory["root_digest"]
        or contract.get("source_object_scan_root") != object_scan["root_digest"]
        or contract.get("lock_index_root") != lock_index["root_digest"]
        or receipt.get("portable_contract_root") != contract["root_digest"]
        or receipt.get("source_tree_inventory_root") != inventory["root_digest"]
        or receipt.get("source_object_scan_root") != object_scan["root_digest"]
        or contract.get("fresh_seal_root") != seal["root_digest"]
        or contract.get("historical_exclusion_root") != exclusion["root_digest"]
        or contract.get("numerical_lock_root") != numerical["root_digest"]
        or contract.get("numerical_lock_sha256") != numerical_sha
        or contract.get("source_manifest_root") != source_manifest["root_digest"]
        or contract.get("source_manifest_sha256") != source_manifest_sha
        or contract.get("source_closure_sha256")
        != source_closure["identity_sha256"]
        or inventory.get("source_head") != source_head
        or inventory.get("source_tree") != contract.get("source_tree")
        or object_scan.get("source_head") != source_head
        or lock_index.get("source_head") != source_head
        or contract.get("resource_contract")
        != {
            "gpu_per_job": 1,
            "cpu_per_job": 8,
            "host_memory_mib": 65_000,
            "time_limit": "23:59:00",
            "point_in_time_cap_and_capacity_recheck_required": True,
            "package_creation_model_gpu_slurm_action_count": 0,
        }
        or contract.get("case_ids_already_committed") is not True
        or manifest.get("source_tree_inventory_root") != inventory["root_digest"]
        or manifest.get("source_object_scan_root") != object_scan["root_digest"]
        or manifest.get("lock_index_root") != lock_index["root_digest"]
        or object_scan.get("raw_prompt_or_target_content_match_count") != 0
        or object_scan.get("tensor_payload_match_count") != 0
        or object_scan.get("credential_private_config_runtime_log_match_count") != 0
        or not isinstance(inventory_entries, list)
        or inventory.get("entry_count") != len(SOURCE_TREE_PATHS)
        or tuple(item.get("path") for item in inventory_entries) != SOURCE_TREE_PATHS
        or inventory.get("entry_root") != canonical_hash(inventory_entries)
        or any(
            set(item)
            != {
                "path", "mode", "size", "sha256", "git_blob_oid",
                "object_algorithm", "role", "committed_source_head",
                "committed_source_tree",
            }
            or item.get("object_algorithm") != "GIT_BLOB_SHA1"
            or item.get("committed_source_head") != source_head
            or item.get("committed_source_tree") != inventory.get("source_tree")
            for item in inventory_entries
        )
        or not isinstance(lock_entries, list)
        or tuple(item.get("path") for item in lock_entries)
        != tuple(f"project/run_scripts/ode_bf/locks/{name}" for name in LOCK_FILES)
        or lock_index.get("entry_root") != canonical_hash(lock_entries)
    ):
        raise ODEBFContractError("integrated local package semantic contract differs")
    return {
        "package_id": PACKAGE_ID,
        "source_head": source_head,
        "source_tree": manifest["source_tree"],
        "archive_sha256": _sha256(files[PACKAGE_ARCHIVE]),
        "manifest_sha256": _sha256(files[PACKAGE_MANIFEST]),
        "receipt_sha256": _sha256(files[PACKAGE_RECEIPT]),
        "manifest_root": manifest["root_digest"],
        "receipt_root": receipt["root_digest"],
        "normalized_tree_digest": manifest["normalized_tree_digest"],
        "source_tree_inventory_root": inventory["root_digest"],
        "source_object_scan_root": object_scan["root_digest"],
        "tracked_case_id_only_exception_count": object_scan[
            "tracked_case_id_only_exception_count"
        ],
        "fresh_seal_root": seal["root_digest"],
        "historical_exclusion_root": exclusion["root_digest"],
        "numerical_lock_root": numerical["root_digest"],
        "numerical_lock_sha256": numerical_sha,
        "source_manifest_root": source_manifest["root_digest"],
        "source_manifest_sha256": source_manifest_sha,
        "source_closure_sha256": source_closure["identity_sha256"],
        "artifact_identities": contract["artifact_identities"],
        "numerical_lock_identities": contract["numerical_lock_identities"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            create_package(args.source_head, args.output_parent),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
