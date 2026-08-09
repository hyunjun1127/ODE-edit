#!/usr/bin/env python3
"""Create a portable, raw-free SH2 source/reference handoff for R13.

The package is deliberately a checkpoint handoff, not an execution or
transport tool.  It turns the corrected R12 A1-R1 reference package into a
relative, immutable snapshot and couples it to a complete R13 Git bundle.
"""

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
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.ode_bf_observability import (
    ODE_BF_ARM_REGISTRY,
    R13_LIVE_CELL_IDS,
)

# The panel is intentionally imported when its lightweight dependencies are
# available.  The fallback keeps this offline package verifier importable in a
# source-only Python environment where the optional model stack is absent.
try:
    from project.run_scripts.ode_bf.p1_universal_observability_panel import (
        BG_SOFT_REFERENCE_LOCK_FILE,
        UNIVERSAL_OBS_AMENDMENT_ID,
        UNIVERSAL_OBS_BASE_RUNTIME,
        UNIVERSAL_OBS_INSTRUCTION_ID,
        UNIVERSAL_OBS_LOCK_FILE,
        UNIVERSAL_OBS_R10_CASE_ROOT,
        UNIVERSAL_OBS_R12_REFERENCE_SHA256,
        UNIVERSAL_OBS_R12_PORTABLE_ARCHIVE_SHA256,
        UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_ROOT,
        UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_SHA256,
        UNIVERSAL_OBS_R12_PORTABLE_NORMALIZED_TREE,
        UNIVERSAL_OBS_R12_PORTABLE_PACKAGE_ID,
        UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_ROOT,
        UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_SHA256,
        UNIVERSAL_OBS_R12_REFERENCE_CLOSURE_ROOT,
        UNIVERSAL_OBS_SCHEMA_NAMESPACE,
        UNIVERSAL_OBS_SESSION_SOURCE_PATHS,
        UNIVERSAL_OBS_SOURCE_MANIFEST_FILE,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - source-only fallback
    if exc.name != "torch":
        raise
    BG_SOFT_REFERENCE_LOCK_FILE = "p1r12_bg_soft_reference_lock.json"
    UNIVERSAL_OBS_AMENDMENT_ID = "ODEEDIT-S05-ODE-BF-UNIVERSAL-OBS-RS-BG-TARGETHOLD-P1R13-V1-A1"
    UNIVERSAL_OBS_BASE_RUNTIME = "bfd77ddbf046333c59a8e589429d426fd5eee4c4"
    UNIVERSAL_OBS_INSTRUCTION_ID = "ODEEDIT-S05-ODE-BF-UNIVERSAL-OBS-RS-BG-TARGETHOLD-P1R13-V1"
    UNIVERSAL_OBS_LOCK_FILE = "numerical_lock_s05_universal_observability.json"
    UNIVERSAL_OBS_R10_CASE_ROOT = "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
    UNIVERSAL_OBS_R12_REFERENCE_SHA256 = "0f227472b92244861b48d9bd01561a3df282d0332ae03587f1c3897dbf2c5a5c"
    UNIVERSAL_OBS_R12_PORTABLE_PACKAGE_ID = "BGSOFT_R10_FROZEN_BUNDLE_A1_R1"
    UNIVERSAL_OBS_R12_PORTABLE_ARCHIVE_SHA256 = "9defda0d6a21ea05a8bdcef8357249548740ad3eead21fc17709df376ea28295"
    UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_SHA256 = "3e38ea8b48e91dca7f161ba2ee2015ba025c5f58d1c293c15086221a4ffd0b48"
    UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_SHA256 = "84ea98a2dc4122f0e8369d3988487e7b195ecbb66f55dbd455e368371e643c76"
    UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_ROOT = "8b67746e3595f32aaa6975452e8cea795b1b652cbac15054f7b01991a64f9ea6"
    UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_ROOT = "d59024624d9f3c6bcb3e03e77150255aec086c95ec828710c506efd4b3453f4a"
    UNIVERSAL_OBS_R12_PORTABLE_NORMALIZED_TREE = "76e3ad535583d578194ecdcc33a8498c4d65e46302b314aa446fb0da7980d66f"
    UNIVERSAL_OBS_R12_REFERENCE_CLOSURE_ROOT = "127e48c7bd1af5b9b99c3e234bca690e24eb38c426987af8efcbb918eeb9975c"
    UNIVERSAL_OBS_SCHEMA_NAMESPACE = "ode-edit-s05-universal-obs-rs-bg-targethold-p1r13-a1"
    UNIVERSAL_OBS_SESSION_SOURCE_PATHS = (
        "project/run_scripts/session05_ode_bf_submit_universal_observability.py",
        "project/run_scripts/session05_ode_bf_universal_observability.py",
        "project/run_scripts/session05_ode_bf_universal_observability.sbatch",
        "project/run_scripts/session05_ode_bf_universal_observability_dry_plan.py",
        "project/run_scripts/session05_ode_bf_universal_observability_package.py",
    )
    UNIVERSAL_OBS_SOURCE_MANIFEST_FILE = "source_manifest_s05_universal_observability.json"


PACKAGE_ID = "UNIVERSAL_OBSERVABILITY_R13_SH2_BUNDLE_A1"
PACKAGE_TOKEN = "odeedit-s05-universal-obs-r13-a1"
PACKAGE_ARCHIVE = "universal-observability-r13-sh2-bundle-a1.tar"
PACKAGE_MANIFEST = "package-manifest.json"
PACKAGE_RECEIPT = "handoff-receipt.json"
LOCK_HASH_INDEX = "lock-hash-index.json"
R12_PACKAGE_ID = UNIVERSAL_OBS_R12_PORTABLE_PACKAGE_ID
R12_PACKAGE_MANIFEST = "package-manifest.json"
R12_PACKAGE_RECEIPT = "handoff-receipt.json"
ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
REFERENCE_ARMS = tuple(R13_LIVE_CELL_IDS)
FACTORIAL_ARM_IDS = tuple(sorted(ODE_BF_ARM_REGISTRY))
OBSERVABILITY_DIAGNOSTICS = {
    "D1": "EVERY_FIELD_OVERLAY_VS_NOHOOK_BEFORE_GUARD",
    "D2": "POSTFREEZE_W_ONLY_VS_ADDITIVE_Z_ORACLE",
    "D3": "ENTRY_AND_BOUNDARY_SELECTED_V_SINGLE_ALL_WRITE_AUDIT",
    "D4": "TARGET_HOLD_CATCH_UP_KILL_TEST",
}
SOURCE_SESSION_PATHS = UNIVERSAL_OBS_SESSION_SOURCE_PATHS
LOCK_RELATIVES = (
    "project/run_scripts/ode_bf/locks/" + UNIVERSAL_OBS_SOURCE_MANIFEST_FILE,
    "project/run_scripts/ode_bf/locks/" + UNIVERSAL_OBS_LOCK_FILE,
    "project/run_scripts/ode_bf/locks/" + BG_SOFT_REFERENCE_LOCK_FILE,
    "project/run_scripts/ode_bf/locks/source_manifest_s05_bg_soft_missing_cell_a1.json",
    "project/run_scripts/ode_bf/locks/numerical_lock_s05_common_coldcoord_fixed_e8.json",
    "project/run_scripts/ode_bf/locks/p1r10_common_coldcoord_cf_b10_seal.json",
    "project/run_scripts/ode_bf/locks/p1r2_p_population_seal.json",
    "project/run_scripts/ode_bf/locks/p1r2_seqb10_stream_seal.json",
    "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
    "project/run_scripts/ode_bf/locks/source_manifest_s05_common_coldcoord_fixed_e8.json",
)
RAW_FREE_LOCK_RELATIVES = (
    "project/run_scripts/ode_bf/locks/" + UNIVERSAL_OBS_SOURCE_MANIFEST_FILE,
    "project/run_scripts/ode_bf/locks/" + UNIVERSAL_OBS_LOCK_FILE,
    "project/run_scripts/ode_bf/locks/" + BG_SOFT_REFERENCE_LOCK_FILE,
    "project/run_scripts/ode_bf/locks/source_manifest_s05_bg_soft_missing_cell_a1.json",
    "project/run_scripts/ode_bf/locks/numerical_lock_s05_common_coldcoord_fixed_e8.json",
    "project/run_scripts/ode_bf/locks/source_manifest_s05_common_coldcoord_fixed_e8.json",
)
R12_REFERENCE_LINKS = {
    "terminal_sha256": "terminal.json",
    "manifest_sha256": "manifest.json",
    "action_freeze_sha256": "raw/common-cold/action-freeze.json",
    "bootstrap_bg_sha256": "raw/common-cold/bootstrap-bg.json",
    "bg_neutral_field0_sha256": "raw/fixed-e8/BG-NEUTRAL/field-0000.json",
    "n32_native_sha256": "raw/common-cold/N32_NATIVE-postfreeze.json",
    "w0_stepwise_sha256": "raw/stepwise/W0_NO_EDIT.json",
    "native_stepwise_sha256": "raw/stepwise/N32_NATIVE.json",
    "panel_sha256": "raw/stepwise/common-cold-panel.json",
    "evaluator_parity_sha256": "raw/stepwise/reused-warm-evaluator-parity.json",
}
FORBIDDEN_JSON_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "subject",
        "target_new",
        "target_true",
        "target_old",
        "token_ids",
        "input_ids",
        "requested_rewrite",
        "neighborhood_prompts",
        "paraphrase_prompts",
        "rewrite_prompt",
        "raw_text",
        "password",
        "credential",
        "credentials",
        "private_key",
        "key_path",
        "host_name",
        "hostname",
        "user_name",
        "username",
        "port",
    }
)
FORBIDDEN_BUNDLE_COMPONENTS = frozenset(
    {".cache", "cache", "credentials", "data", "datasets", "logs", "secrets"}
)
FORBIDDEN_BUNDLE_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".pem",
    ".pt",
    ".safetensors",
)
# This is a public, empty handoff-boundary template already reachable from the
# immutable base history.  A complete Git bundle necessarily retains it, while
# all other ``local/`` content remains forbidden.
PUBLIC_SOURCE_HISTORY_TEMPLATE_PATHS = frozenset({"local/README.md"})


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rooted_json(
    path: Path, *, expected_schema: str | None = None
) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ODEBFContractError("universal-observability rooted JSON object differs")
    observed = value.pop("root_digest", None)
    expected = canonical_hash(value)
    value["root_digest"] = observed
    if observed != expected:
        raise ODEBFContractError("universal-observability rooted JSON digest differs")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ODEBFContractError("universal-observability rooted JSON schema differs")
    return value, sha256_file(path)


def _git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _require_sha1_repository(*, cwd: Path = REPO_ROOT) -> None:
    if _run(["git", "rev-parse", "--show-object-format"], cwd=cwd).stdout.strip() != "sha1":
        raise ODEBFContractError("universal-observability package object format differs")


def _read_regular_file_nofollow(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ODEBFContractError(
            "universal-observability package source is not a stable regular file"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ODEBFContractError(
                "universal-observability package source is not a stable regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or sum(len(chunk) for chunk in chunks) != before.st_size:
        raise ODEBFContractError("universal-observability package source changed during read")
    return b"".join(chunks), before


def _write_json_once(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(payload)


def _walk_raw_free(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            label = str(key)
            if label.lower() in FORBIDDEN_JSON_KEYS:
                raise ODEBFContractError(
                    "universal-observability package raw/private JSON key is forbidden"
                )
            _walk_raw_free(child)
    elif isinstance(value, list):
        for child in value:
            _walk_raw_free(child)
    elif isinstance(value, str) and value.startswith("/"):
        raise ODEBFContractError(
            "universal-observability package absolute path value is forbidden"
        )


def validate_raw_free_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("universal-observability package JSON source differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFContractError(
            "universal-observability package JSON parse failed"
        ) from exc
    if not isinstance(value, dict):
        raise ODEBFContractError("universal-observability package JSON root differs")
    _walk_raw_free(value)
    return value


def _safe_relative(relative: str) -> Path:
    path = Path(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative
    ):
        raise ODEBFContractError("universal-observability package relative path differs")
    return path


def _tree_entry(source_head: str, relative: str, *, cwd: Path = REPO_ROOT) -> tuple[int, str]:
    row = _run(["git", "ls-tree", source_head, "--", relative], cwd=cwd).stdout.rstrip("\n")
    if "\t" not in row:
        raise ODEBFContractError("universal-observability package Git tree entry differs")
    metadata, observed = row.split("\t", 1)
    mode, kind, object_id = metadata.split(" ", 2)
    if observed != relative or kind != "blob" or int(mode, 8) not in (0o100644, 0o100755):
        raise ODEBFContractError("universal-observability package Git tree entry differs")
    return int(mode, 8), object_id


def _source_manifest_path(repo_root: Path) -> Path:
    return repo_root / "project/run_scripts/ode_bf/locks" / UNIVERSAL_OBS_SOURCE_MANIFEST_FILE


def _expected_source_paths(source_head: str, *, cwd: Path = REPO_ROOT) -> set[str]:
    tracked = set(_run(["git", "ls-tree", "-r", "--name-only", source_head], cwd=cwd).stdout.splitlines())
    manifest_relative = (
        "project/run_scripts/ode_bf/locks/" + UNIVERSAL_OBS_SOURCE_MANIFEST_FILE
    )
    return {
        relative
        for relative in tracked
        if (
            relative.startswith("project/run_scripts/ode_bf/")
            or relative in SOURCE_SESSION_PATHS
        )
        and relative != manifest_relative
    }


def validate_source_manifest(source_head: str, *, repo_root: Path = REPO_ROOT) -> str:
    """Validate the committed R13 source closure, including Git modes/OIDs."""

    path = _source_manifest_path(repo_root)
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("universal-observability source manifest is absent")
    try:
        value, raw_sha256 = load_rooted_json(
            path,
            expected_schema=f"{UNIVERSAL_OBS_SCHEMA_NAMESPACE}-source-manifest/v1",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFContractError("universal-observability source manifest differs") from exc
    entries = value.get("entries")
    base = value.get("base_runtime", value.get("expected_parent"))
    if not isinstance(entries, list) or not entries:
        raise ODEBFContractError("universal-observability source manifest header differs")
    if (
        value.get("instruction_id") != UNIVERSAL_OBS_INSTRUCTION_ID
        or value.get("amendment_id") != UNIVERSAL_OBS_AMENDMENT_ID
        or base != UNIVERSAL_OBS_BASE_RUNTIME
        or value.get("case_seal_root_digest") != UNIVERSAL_OBS_R10_CASE_ROOT
        or value.get("execution_head_policy") != "runtime-git-head"
        or value.get("entry_count") != len(entries)
    ):
        raise ODEBFContractError("universal-observability source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("path"), str)
            or entry.get("mode") not in (0o100644, 0o100755)
            or not isinstance(entry.get("object_id"), str)
            or len(str(entry["object_id"])) != 40
            or not isinstance(entry.get("sha256"), str)
            or len(str(entry["sha256"])) != 64
            or not isinstance(entry.get("size"), int)
            or not isinstance(entry.get("role"), str)
            or not entry["role"]
        ):
            raise ODEBFContractError("universal-observability source manifest entry differs")
        relative = str(entry["path"])
        _safe_relative(relative)
        candidate = repo_root / relative
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != entry["size"]
            or sha256_file(candidate) != entry["sha256"]
        ):
            raise ODEBFContractError("universal-observability source manifest content differs")
        mode, object_id = _tree_entry(source_head, relative, cwd=repo_root)
        if mode != entry["mode"] or object_id != entry["object_id"]:
            raise ODEBFContractError("universal-observability source manifest object differs")
        observed.append(relative)
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise ODEBFContractError("universal-observability source manifest ordering differs")
    if set(observed) != _expected_source_paths(source_head, cwd=repo_root):
        raise ODEBFContractError("universal-observability source manifest path set differs")
    return raw_sha256


def _validate_source_head(source_head: str) -> str:
    if len(source_head) != 40 or any(character not in "0123456789abcdef" for character in source_head):
        raise ODEBFContractError("universal-observability package source head differs")
    _require_sha1_repository()
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    base = _run(
        ["git", "merge-base", "--is-ancestor", UNIVERSAL_OBS_BASE_RUNTIME, source_head],
        check=False,
    )
    if head != source_head or not branch or dirty or base.returncode != 0:
        raise ODEBFContractError("universal-observability package source provenance differs")
    return branch


def _source_bundle_content_scan(source_head: str) -> dict[str, Any]:
    """Reject model/data/private path classes from the bundled history."""

    lines = _run(["git", "rev-list", "--objects", source_head]).stdout.splitlines()
    forbidden = 0
    for line in lines:
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        relative = parts[1]
        if relative in PUBLIC_SOURCE_HISTORY_TEMPLATE_PATHS:
            continue
        lowered = {part.lower() for part in Path(relative).parts}
        if (
            lowered & FORBIDDEN_BUNDLE_COMPONENTS
            or relative.lower().endswith(FORBIDDEN_BUNDLE_SUFFIXES)
            or relative.startswith("local/")
            or relative.startswith("servers/local/")
        ):
            forbidden += 1
    if forbidden:
        raise ODEBFContractError(
            "universal-observability source bundle contains a forbidden path class"
        )
    return {
        "reachable_object_count": len(lines),
        "forbidden_path_match_count": 0,
        "raw_prompt_or_target_content_match_count": 0,
        "tensor_payload_match_count": 0,
        "credential_private_config_runtime_log_match_count": 0,
        "public_nonprivate_template_paths": sorted(
            PUBLIC_SOURCE_HISTORY_TEMPLATE_PATHS
        ),
        "scientific_promotion_authorized": False,
    }


def _parse_bundle_header(path: Path) -> dict[str, Any]:
    heads: list[dict[str, str]] = []
    prerequisites: list[str] = []
    with path.open("rb") as handle:
        first = handle.readline().decode("utf-8").rstrip("\n")
        if first not in ("# v2 git bundle", "# v3 git bundle"):
            raise ODEBFContractError("universal-observability git bundle schema differs")
        for raw in handle:
            line = raw.decode("utf-8").rstrip("\n")
            if not line:
                break
            if line.startswith("-"):
                prerequisites.append(line[1:].split(" ", 1)[0])
            else:
                commit, name = line.split(" ", 1)
                heads.append({"commit": commit, "name": name})
    return {"version": first, "heads": heads, "prerequisites": prerequisites}


def _verify_self_contained_bundle(
    path: Path, source_head: str, advertised_ref: str
) -> dict[str, Any]:
    source_tree = _run(["git", "rev-parse", source_head + "^{tree}"]).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="odeedit-r13-bundle-proof-") as temporary:
        root = Path(temporary)
        bare = root / "empty.git"
        checkout = root / "checkout"
        _run(["git", "init", "--bare", str(bare)])
        _run(["git", "-C", str(bare), "bundle", "verify", str(path)])
        _run(["git", "-C", str(bare), "fetch", str(path), f"{advertised_ref}:{advertised_ref}"])
        _run(["git", "-C", str(bare), "fsck", "--full"])
        ancestor = _run(
            ["git", "-C", str(bare), "merge-base", "--is-ancestor", UNIVERSAL_OBS_BASE_RUNTIME, advertised_ref],
            check=False,
        )
        imported_head = _run(["git", "-C", str(bare), "rev-parse", advertised_ref]).stdout.strip()
        imported_tree = _run(["git", "-C", str(bare), "rev-parse", advertised_ref + "^{tree}"]).stdout.strip()
        _run(["git", "clone", "--no-checkout", str(bare), str(checkout)])
        _run(["git", "-C", str(checkout), "checkout", "--detach", source_head])
        checkout_head = _run(["git", "-C", str(checkout), "rev-parse", "HEAD"]).stdout.strip()
        clean = _run(["git", "-C", str(checkout), "status", "--porcelain"]).stdout
        if (
            ancestor.returncode != 0
            or imported_head != source_head
            or imported_tree != source_tree
            or checkout_head != source_head
            or clean
        ):
            raise ODEBFContractError("universal-observability complete bundle checkout differs")
        # The imported checkout, rather than the source worktree, proves the
        # committed source-manifest object/mode contract.
        validate_source_manifest(source_head, repo_root=checkout)
    receipt = {
        "empty_bare_repository_bundle_verify": True,
        "empty_bare_repository_import": True,
        "empty_bare_repository_fsck_full": True,
        "materialized_checkout": True,
        "materialized_checkout_clean": True,
        "advertised_ref": advertised_ref,
        "source_head": source_head,
        "source_tree": source_tree,
        "base_runtime_is_ancestor": True,
        "external_prerequisite_count": 0,
        "complete_required_lineage_object_closure": True,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return receipt


def _create_complete_bundle(path: Path, source_head: str, branch: str) -> dict[str, Any]:
    advertised_ref = "refs/heads/" + branch
    _run(["git", "bundle", "create", str(path), branch])
    _run(["git", "bundle", "verify", str(path)])
    header = _parse_bundle_header(path)
    if header["heads"] != [{"commit": source_head, "name": advertised_ref}] or header["prerequisites"]:
        raise ODEBFContractError("universal-observability git bundle identity differs")
    proof = _verify_self_contained_bundle(path, source_head, advertised_ref)
    return {
        **header,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "base_runtime_included": UNIVERSAL_OBS_BASE_RUNTIME,
        "complete_history": True,
        "self_contained_verification": proof,
        "source_object_content_scan": _source_bundle_content_scan(source_head),
    }


def _validate_r12_handoff(handoff: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if handoff.is_symlink() or not handoff.is_dir():
        raise ODEBFContractError("universal-observability R12 handoff directory differs")
    manifest_path = handoff / R12_PACKAGE_MANIFEST
    receipt_path = handoff / R12_PACKAGE_RECEIPT
    try:
        manifest, manifest_sha256 = load_rooted_json(manifest_path)
        receipt, receipt_sha256 = load_rooted_json(receipt_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFContractError("universal-observability R12 handoff metadata differs") from exc
    archive = receipt.get("archive")
    manifest_link = receipt.get("manifest")
    if (
        manifest.get("package_id") != R12_PACKAGE_ID
        or manifest_sha256 != UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_SHA256
        or receipt_sha256 != UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_SHA256
        or manifest.get("root_digest") != UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_ROOT
        or receipt.get("root_digest") != UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_ROOT
        or manifest.get("normalized_tree_digest")
        != UNIVERSAL_OBS_R12_PORTABLE_NORMALIZED_TREE
        or manifest.get("frozen_reference_closure", {}).get("root_digest")
        != UNIVERSAL_OBS_R12_REFERENCE_CLOSURE_ROOT
        or manifest.get("runtime_technical_repair_parent") != UNIVERSAL_OBS_BASE_RUNTIME
        or manifest.get("bundle", {}).get("complete_history") is not True
        or manifest.get("bundle", {}).get("prerequisites") != []
        or manifest.get("bundle", {}).get("self_contained_verification", {}).get("empty_bare_repository_import") is not True
        or receipt.get("package_id") != R12_PACKAGE_ID
        or receipt.get("transport_executed_by_packager") is not False
        or not isinstance(archive, Mapping)
        or not isinstance(manifest_link, Mapping)
        or manifest_link.get("sha256") != manifest_sha256
        or receipt.get("manifest", {}).get("root_digest") != manifest.get("root_digest")
        or not isinstance(receipt_sha256, str)
    ):
        raise ODEBFContractError("universal-observability R12 handoff identity differs")
    archive_name = archive.get("name")
    if not isinstance(archive_name, str):
        raise ODEBFContractError("universal-observability R12 archive identity differs")
    archive_path = handoff / archive_name
    if (
        archive_path.is_symlink()
        or not archive_path.is_file()
        or archive.get("sha256") != sha256_file(archive_path)
        or archive.get("sha256") != UNIVERSAL_OBS_R12_PORTABLE_ARCHIVE_SHA256
        or archive.get("size") != archive_path.stat().st_size
    ):
        raise ODEBFContractError("universal-observability R12 archive identity differs")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or manifest.get("entry_count") != len(entries):
        raise ODEBFContractError("universal-observability R12 archive manifest differs")
    return manifest, receipt, archive_path


def _r12_archive_entries(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    expected: dict[str, Mapping[str, Any]] = {}
    prefix = R12_PACKAGE_ID + "/"
    for entry in manifest["entries"]:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ODEBFContractError("universal-observability R12 archive member differs")
        name = str(entry["path"])
        role = entry.get("role")
        if (
            not name.startswith(prefix)
            or entry.get("mode") != 0o600
            or entry.get("archive_kind") != "REGULAR_FILE"
            or entry.get("object_algorithm") != "GIT_BLOB_SHA1"
            or not isinstance(entry.get("object_id"), str)
            or len(str(entry["object_id"])) != 40
            or not isinstance(entry.get("sha256"), str)
            or len(str(entry["sha256"])) != 64
            or not isinstance(role, str)
        ):
            raise ODEBFContractError("universal-observability R12 archive member differs")
        expected[name] = entry
    if len(expected) != len(manifest["entries"]):
        raise ODEBFContractError("universal-observability R12 archive member repeats")
    return expected


def _copy_tar_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    entry: Mapping[str, Any],
    snapshot_root: Path,
) -> Path:
    prefix = R12_PACKAGE_ID + "/"
    if (
        not member.isfile()
        or member.issym()
        or member.islnk()
        or member.mode != 0o600
        or member.mtime != 0
        or member.uid != 0
        or member.gid != 0
        or member.uname != ""
        or member.gname != ""
    ):
        raise ODEBFContractError("universal-observability R12 archive member safety differs")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ODEBFContractError("universal-observability R12 archive member is unreadable")
    data = extracted.read()
    if (
        len(data) != entry["size"]
        or _sha256_bytes(data) != entry["sha256"]
        or _git_blob_oid(data) != entry["object_id"]
        or not member.name.startswith(prefix)
    ):
        raise ODEBFContractError("universal-observability R12 archive member hash differs")
    relative = member.name.removeprefix(prefix)
    _safe_relative(relative)
    target = snapshot_root / relative
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _rehash_reference_links(snapshot_root: Path) -> dict[str, Any]:
    lock_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / BG_SOFT_REFERENCE_LOCK_FILE
    lock, lock_sha256 = load_rooted_json(
        lock_path,
        expected_schema="ode-edit-s05-bg-soft-missing-cell-p1r12-a1-reference-lock/v1",
    )
    if lock_sha256 != UNIVERSAL_OBS_R12_REFERENCE_SHA256:
        raise ODEBFContractError("universal-observability R12 reference lock differs")
    aliases = lock.get("aliases")
    if not isinstance(aliases, Mapping) or set(aliases) != set(ALIASES):
        raise ODEBFContractError("universal-observability R12 aliases differ")
    alias_receipts: dict[str, Any] = {}
    for alias in ALIASES:
        root = snapshot_root / "reference/r10" / alias
        frozen = aliases[alias].get("frozen_r10_reference")
        if not isinstance(frozen, Mapping) or root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("universal-observability R12 reference closure differs")
        link_count = 0
        for key, relative in R12_REFERENCE_LINKS.items():
            path = root / relative
            if path.is_symlink() or not path.is_file() or frozen.get(key) != sha256_file(path):
                raise ODEBFContractError("universal-observability R12 reference link differs")
            link_count += 1
        arms: dict[str, Any] = {}
        for arm in REFERENCE_ARMS:
            fixed = root / "raw/fixed-e8" / arm
            accepted = sorted(fixed.rglob("accepted-*.json")) if fixed.is_dir() else []
            if not accepted:
                raise ODEBFContractError("universal-observability R12 arm closure is absent")
            transition_links = 0
            trial_links = 0
            for accepted_path in accepted:
                accepted_value = validate_raw_free_json(accepted_path)
                ordinal = accepted_path.stem.split("-", 1)[1]
                transition = accepted_path.parent / f"transition-{ordinal}.json"
                if (
                    transition.is_symlink()
                    or not transition.is_file()
                    or accepted_value.get("transition_sha256") != sha256_file(transition)
                ):
                    raise ODEBFContractError("universal-observability R12 transition link differs")
                transition_value = validate_raw_free_json(transition)
                trial = accepted_path.parent / f"trial-{ordinal}.json"
                if (
                    trial.is_symlink()
                    or not trial.is_file()
                    or transition_value.get("trial_receipt_sha256") != sha256_file(trial)
                ):
                    raise ODEBFContractError("universal-observability R12 trial link differs")
                validate_raw_free_json(trial)
                transition_links += 1
                trial_links += 1
            records = [
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(fixed.rglob("*.json"))
                if path.is_file() and not path.is_symlink()
            ]
            arms[arm] = {
                "path_count": len(records),
                "accepted_prefix_count": len(accepted),
                "accepted_to_transition_link_rehash_count": transition_links,
                "transition_to_trial_link_rehash_count": trial_links,
                "tree_digest": canonical_hash(records),
            }
            if not records:
                raise ODEBFContractError("universal-observability R12 arm closure is empty")
        alias_receipts[alias] = {
            "r12_reference_link_rehash_count": link_count,
            "arms": arms,
        }
        alias_receipts[alias]["root_digest"] = canonical_hash(alias_receipts[alias])
    receipt = {
        "schema": f"{UNIVERSAL_OBS_SCHEMA_NAMESPACE}-r12-reference-closure/v1",
        "r12_reference_lock_sha256": lock_sha256,
        "required_arms": list(REFERENCE_ARMS),
        "aliases": alias_receipts,
        "all_required_links_rehashed": True,
        "all_r13_arm_closures_nonempty": True,
        "w0_native_reference_relative_only": True,
        "scientific_promotion_authorized": False,
    }
    receipt["root_digest"] = canonical_hash(receipt)
    return receipt


def snapshot_r12_reference_inventory(
    handoff: Path, snapshot_root: Path
) -> tuple[tuple[tuple[str, Path, str], ...], dict[str, Any]]:
    """Snapshot only the immutable raw-free R12 R1 reference closure."""

    manifest, receipt, archive_path = _validate_r12_handoff(handoff)
    expected = _r12_archive_entries(manifest)
    selected = {
        name: entry
        for name, entry in expected.items()
        if name.startswith(R12_PACKAGE_ID + "/reference/r10/")
        or name.startswith(R12_PACKAGE_ID + "/reference/r10-report/")
    }
    if not selected:
        raise ODEBFContractError("universal-observability R12 reference archive is empty")
    with tarfile.open(archive_path, mode="r:") as archive:
        members = {member.name: member for member in archive.getmembers()}
        if set(members) != set(expected):
            raise ODEBFContractError("universal-observability R12 archive inventory differs")
        rows: list[tuple[str, Path, str]] = []
        for name, entry in sorted(selected.items()):
            target = _copy_tar_member(archive, members[name], entry, snapshot_root)
            relative = target.relative_to(snapshot_root).as_posix()
            if target.suffix == ".json":
                validate_raw_free_json(target)
            role = (
                "IMMUTABLE_RAW_FREE_R10_REPORT"
                if "/reference/r10-report/" in name
                else "IMMUTABLE_RAW_FREE_R12_LINKED_R10_RECEIPT"
            )
            rows.append((relative, target, role))
    for name, value in ((R12_PACKAGE_MANIFEST, manifest), (R12_PACKAGE_RECEIPT, receipt)):
        target = snapshot_root / "reference/r12" / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_json_once(target, value)
        validate_raw_free_json(target)
        rows.append((target.relative_to(snapshot_root).as_posix(), target, "IMMUTABLE_R12_HANDOFF_METADATA"))
    closure = _rehash_reference_links(snapshot_root)
    rows.sort(key=lambda item: item[0])
    paths = [relative for relative, _path, _role in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ODEBFContractError("universal-observability reference inventory ordering differs")
    return tuple(rows), closure


def _create_lock_hash_index(path: Path, source_head: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for relative in LOCK_RELATIVES:
        _safe_relative(relative)
        source = REPO_ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise ODEBFContractError("universal-observability package lock source differs")
        data, metadata = _read_regular_file_nofollow(source)
        mode, object_id = _tree_entry(source_head, relative)
        if object_id != _git_blob_oid(data):
            raise ODEBFContractError("universal-observability package lock object differs")
        entries.append(
            {
                "path": relative,
                "mode": mode,
                "size": metadata.st_size,
                "sha256": _sha256_bytes(data),
                "object_id": object_id,
                "object_algorithm": "GIT_BLOB_SHA1",
                "role": "RAW_FREE_LOCK_CONTENT" if relative in RAW_FREE_LOCK_RELATIVES else "HASH_ONLY_RAW_OR_PRIVATE_LOCK",
                "content_included": relative in RAW_FREE_LOCK_RELATIVES,
            }
        )
    value: dict[str, Any] = {
        "schema": f"{UNIVERSAL_OBS_SCHEMA_NAMESPACE}-lock-hash-index/v1",
        "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
        "amendment_id": UNIVERSAL_OBS_AMENDMENT_ID,
        "entries": entries,
        "entry_count": len(entries),
        "raw_id_or_content_count": 0,
    }
    value["root_digest"] = canonical_hash(value)
    _write_json_once(path, value)
    validate_raw_free_json(path)
    return value


def _source_inventory(bundle: Path, lock_index: Path) -> tuple[tuple[str, Path, str], ...]:
    rows: list[tuple[str, Path, str]] = [
        ("source/source.bundle", bundle, "SELF_CONTAINED_GIT_BUNDLE"),
        ("source/" + LOCK_HASH_INDEX, lock_index, "RAW_FREE_LOCK_HASH_INDEX"),
    ]
    for relative in RAW_FREE_LOCK_RELATIVES:
        source = REPO_ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise ODEBFContractError("universal-observability package lock source differs")
        validate_raw_free_json(source)
        rows.append(("source/" + relative, source, "SOURCE_OR_LOCK_MANIFEST"))
    return tuple(rows)


def _tracked_source_metadata(path: Path, source_head: str, source_tree: str, object_id: str) -> dict[str, Any]:
    try:
        relative = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return {"committed_membership": False}
    result = _run(["git", "ls-tree", source_head, "--", relative], check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {"committed_membership": False}
    metadata, observed = result.stdout.rstrip("\n").split("\t", 1)
    mode, kind, committed_object = metadata.split(" ", 2)
    if observed != relative or kind != "blob" or committed_object != object_id:
        raise ODEBFContractError("universal-observability committed source object differs")
    return {
        "committed_membership": True,
        "committed_path": relative,
        "committed_mode": int(mode, 8),
        "committed_object_id": committed_object,
        "committed_source_head": source_head,
        "committed_source_tree": source_tree,
    }


def _validate_manifest_entries(entries: Sequence[Mapping[str, Any]]) -> None:
    required = {
        "path", "mode", "size", "sha256", "role", "semantic_role", "archive_kind",
        "object_id", "object_algorithm", "committed_membership",
    }
    names = [entry.get("path") for entry in entries]
    if len(names) != len(set(names)):
        raise ODEBFContractError("universal-observability package manifest path repeats")
    prefix = PACKAGE_ID + "/"
    for entry in entries:
        if (
            not required.issubset(entry)
            or not isinstance(entry.get("path"), str)
            or not str(entry["path"]).startswith(prefix)
            or str(entry["path"])[len(prefix):] != _safe_relative(str(entry["path"])[len(prefix):]).as_posix()
            or entry.get("mode") != 0o600
            or not isinstance(entry.get("size"), int)
            or entry["size"] < 0
            or not isinstance(entry.get("sha256"), str)
            or len(str(entry["sha256"])) != 64
            or entry.get("semantic_role") != entry.get("role")
            or entry.get("archive_kind") != "REGULAR_FILE"
            or entry.get("object_algorithm") != "GIT_BLOB_SHA1"
            or not isinstance(entry.get("object_id"), str)
            or len(str(entry["object_id"])) != 40
            or type(entry.get("committed_membership")) is not bool
        ):
            raise ODEBFContractError("universal-observability package manifest entry differs")
        if entry["committed_membership"]:
            if (
                entry.get("committed_mode") not in (0o100644, 0o100755)
                or entry.get("committed_object_id") != entry["object_id"]
                or not isinstance(entry.get("committed_path"), str)
                or not isinstance(entry.get("committed_source_head"), str)
                or not isinstance(entry.get("committed_source_tree"), str)
            ):
                raise ODEBFContractError("universal-observability committed manifest identity differs")
        elif entry.get("uncommitted_content_class") not in {
            "GENERATED_PACKAGE_MEMBER", "EXTERNAL_RAW_FREE_REFERENCE"
        }:
            raise ODEBFContractError("universal-observability uncommitted manifest classification differs")


def _entry_manifest(
    inventory: Iterable[tuple[str, Path, str]], *, source_head: str
) -> tuple[list[dict[str, Any]], str]:
    source_tree = _run(["git", "rev-parse", source_head + "^{tree}"]).stdout.strip()
    entries: list[dict[str, Any]] = []
    for relative, path, role in sorted(inventory):
        _safe_relative(relative)
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError("universal-observability package source file differs")
        data, _metadata = _read_regular_file_nofollow(path)
        object_id = _git_blob_oid(data)
        tracked = _tracked_source_metadata(path, source_head, source_tree, object_id)
        if not tracked["committed_membership"]:
            tracked["uncommitted_content_class"] = (
                "EXTERNAL_RAW_FREE_REFERENCE" if role.startswith("IMMUTABLE_") else "GENERATED_PACKAGE_MEMBER"
            )
        entries.append(
            {
                "path": f"{PACKAGE_ID}/{relative}",
                "mode": 0o600,
                "size": len(data),
                "sha256": _sha256_bytes(data),
                "role": role,
                "semantic_role": role,
                "archive_kind": "REGULAR_FILE",
                "object_id": object_id,
                "object_algorithm": "GIT_BLOB_SHA1",
                "regular_file": True,
                "symlink": False,
                **tracked,
            }
        )
    if [item["path"] for item in entries] != sorted(item["path"] for item in entries):
        raise ODEBFContractError("universal-observability package entry ordering differs")
    _validate_manifest_entries(entries)
    return entries, canonical_hash(entries)


def _write_deterministic_tar(path: Path, inventory: Iterable[tuple[str, Path, str]]) -> None:
    with path.open("xb") as output:
        with tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for relative, source, _role in sorted(inventory):
                data, _metadata = _read_regular_file_nofollow(source)
                info = tarfile.TarInfo(f"{PACKAGE_ID}/{relative}")
                info.size = len(data)
                info.mode = 0o600
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                archive.addfile(info, io.BytesIO(data))


def _verify_deterministic_tar(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    _validate_manifest_entries(entries)
    expected = {str(entry["path"]): dict(entry) for entry in entries}
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != sorted(expected) or len(members) != len(expected):
            raise ODEBFContractError("universal-observability package archive inventory differs")
        for member in members:
            entry = expected[member.name]
            if (
                not member.isfile()
                or member.issym()
                or member.islnk()
                or member.mode != entry["mode"]
                or member.size != entry["size"]
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
            ):
                raise ODEBFContractError("universal-observability package archive entry differs")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ODEBFContractError("universal-observability package archive read failed")
            data = extracted.read()
            if _sha256_bytes(data) != entry["sha256"] or _git_blob_oid(data) != entry["object_id"]:
                raise ODEBFContractError("universal-observability package archive hash differs")


def build_package_plan(source_head: str, r12_handoff_dir: Path) -> dict[str, Any]:
    branch = _validate_source_head(source_head)
    source_manifest_sha256 = validate_source_manifest(source_head)
    manifest, receipt, archive = _validate_r12_handoff(r12_handoff_dir)
    return {
        "schema": f"{UNIVERSAL_OBS_SCHEMA_NAMESPACE}-handoff-dry-plan/v1",
        "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
        "amendment_id": UNIVERSAL_OBS_AMENDMENT_ID,
        "package_id": PACKAGE_ID,
        "source_head": source_head,
        "execution_branch": branch,
        "base_runtime": UNIVERSAL_OBS_BASE_RUNTIME,
        "case_seal_root_digest": UNIVERSAL_OBS_R10_CASE_ROOT,
        "factorial_shape": [2, 2, 2],
        "factorial_arm_registry": list(FACTORIAL_ARM_IDS),
        "factorial_arm_count": len(FACTORIAL_ARM_IDS),
        "r13_live_cells": list(REFERENCE_ARMS),
        "diagnostics": dict(OBSERVABILITY_DIAGNOSTICS),
        "d1_d2_d3_d4_observation_only": True,
        "source_manifest_sha256": source_manifest_sha256,
        "r12_package_id": manifest["package_id"],
        "r12_archive_sha256": sha256_file(archive),
        "r12_handoff_receipt_root_digest": receipt["root_digest"],
        "model_or_dataset_file_count": 0,
        "credential_or_private_config_file_count": 0,
        "transfer_executed": False,
        "create_once": True,
        "scientific_promotion_authorized": False,
    }


def create_package(source_head: str, r12_handoff_dir: Path, output_parent: Path) -> dict[str, Any]:
    plan = build_package_plan(source_head, r12_handoff_dir)
    if output_parent.is_symlink() or (output_parent.exists() and not output_parent.is_dir()):
        raise ODEBFContractError("universal-observability package parent differs")
    output_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    package_name = f"{PACKAGE_TOKEN}-{source_head[:12]}-v1"
    destination = output_parent / package_name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("universal-observability package destination is create-once")
    temporary = Path(tempfile.mkdtemp(prefix=f".{package_name}.building-", dir=output_parent))
    os.chmod(temporary, 0o700)
    try:
        bundle_path = temporary / "source.bundle"
        bundle = _create_complete_bundle(bundle_path, source_head, str(plan["execution_branch"]))
        lock_index_path = temporary / LOCK_HASH_INDEX
        lock_index = _create_lock_hash_index(lock_index_path, source_head)
        with tempfile.TemporaryDirectory(prefix=f".{package_name}.r12-snapshot-", dir=output_parent) as reference_temporary:
            references, closure = snapshot_r12_reference_inventory(
                r12_handoff_dir, Path(reference_temporary)
            )
            inventory = _source_inventory(bundle_path, lock_index_path) + references
            entries, normalized_tree_digest = _entry_manifest(inventory, source_head=source_head)
            archive_path = temporary / PACKAGE_ARCHIVE
            _write_deterministic_tar(archive_path, inventory)
            _verify_deterministic_tar(archive_path, entries)
            manifest: dict[str, Any] = {
                "schema": f"{UNIVERSAL_OBS_SCHEMA_NAMESPACE}-handoff-package-manifest/v1",
                "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
                "amendment_id": UNIVERSAL_OBS_AMENDMENT_ID,
                "package_id": PACKAGE_ID,
                "package_name": package_name,
                "source_head": source_head,
                "base_runtime": UNIVERSAL_OBS_BASE_RUNTIME,
                "case_seal_root_digest": UNIVERSAL_OBS_R10_CASE_ROOT,
                "factorial_shape": [2, 2, 2],
                "factorial_arm_registry": list(FACTORIAL_ARM_IDS),
                "factorial_arm_count": len(FACTORIAL_ARM_IDS),
                "diagnostics": dict(OBSERVABILITY_DIAGNOSTICS),
                "d1_d2_d3_d4_observation_only": True,
                "bundle": bundle,
                "r12_reference_closure": closure,
                "lock_hash_index_root_digest": lock_index["root_digest"],
                "entries": entries,
                "entry_count": len(entries),
                "normalized_tree_digest": normalized_tree_digest,
                "no_symlinks": True,
                "prohibited_raw_content_count": 0,
                "model_or_dataset_file_count": 0,
                "credential_or_private_config_file_count": 0,
                "secure_transfer_boundary": {
                    "route_policy": "CONFIGURED_PRIVATE_CREATE_ONCE_ROUTE_NO_ENDPOINT_DISCLOSURE",
                    "raw_prompt_target_tensor_payload_included": False,
                    "credential_private_config_runtime_log_included": False,
                    "scientific_promotion_authorized": False,
                },
            }
            manifest["root_digest"] = canonical_hash(manifest)
            manifest_sha256 = _write_json_once(temporary / PACKAGE_MANIFEST, manifest)
        archive_sha256 = sha256_file(archive_path)
        receipt: dict[str, Any] = {
            "schema": f"{UNIVERSAL_OBS_SCHEMA_NAMESPACE}-handoff-receipt/v1",
            "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
            "amendment_id": UNIVERSAL_OBS_AMENDMENT_ID,
            "package_id": PACKAGE_ID,
            "package_name": package_name,
            "source_head": source_head,
            "base_runtime": UNIVERSAL_OBS_BASE_RUNTIME,
            "archive": {"name": PACKAGE_ARCHIVE, "sha256": archive_sha256, "size": archive_path.stat().st_size},
            "manifest": {
                "name": PACKAGE_MANIFEST,
                "sha256": manifest_sha256,
                "root_digest": manifest["root_digest"],
                "entry_count": len(entries),
                "normalized_tree_digest": normalized_tree_digest,
            },
            "creation_count": 1,
            "authorized_transfer_count": 1,
            "transport_executed_by_packager": False,
            "publication": "CREATE_ONCE",
            "scientific_promotion_authorized": False,
            "plan": plan,
        }
        receipt["root_digest"] = canonical_hash(receipt)
        receipt_sha256 = _write_json_once(temporary / PACKAGE_RECEIPT, receipt)
        os.rename(temporary, destination)
    except Exception:
        # The create-once destination is never published on failure.  Keep the
        # private staging directory for forensic inspection rather than deleting
        # potentially useful evidence from a failed closure attempt.
        raise
    return {
        "status": "UNIVERSAL_OBSERVABILITY_R13_SH2_BUNDLE_A1_CREATED",
        "package_directory": str(destination),
        "source_head": source_head,
        "archive_sha256": archive_sha256,
        "manifest_sha256": manifest_sha256,
        "receipt_sha256": receipt_sha256,
        "manifest_root_digest": manifest["root_digest"],
        "normalized_tree_digest": normalized_tree_digest,
        "entry_count": len(entries),
        "transfer_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--r12-handoff-dir", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    value = (
        build_package_plan(args.source_head, args.r12_handoff_dir)
        if args.dry_run
        else create_package(args.source_head, args.r12_handoff_dir, args.output_parent)
    )
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
