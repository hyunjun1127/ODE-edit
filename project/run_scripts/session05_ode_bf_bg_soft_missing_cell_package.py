#!/usr/bin/env python3
"""Create the exact source/reference package for the R12 A1 SH2 handoff."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
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
from project.run_scripts.ode_bf.p1_bg_soft_missing_cell_panel import (
    BG_SOFT_AMENDMENT_ID,
    BG_SOFT_EXECUTION_REPAIR_PARENT,
    BG_SOFT_IMPLEMENTATION_PARENT,
    BG_SOFT_INSTRUCTION_ID,
    BG_SOFT_PARENT_HEAD,
    BG_SOFT_REFERENCE_LOCK_FILE,
    BG_SOFT_SCHEMA_NAMESPACE,
    BG_SOFT_SOURCE_MANIFEST_FILE,
)
from project.run_scripts.session05_ode_bf_submit_bg_soft_missing_cell import (
    EXECUTION_BRANCH,
    R10_REFERENCE_REPO,
    R10_REFERENCE_RESULTS,
    _frozen_r10_reference_gate,
    _source_manifest_gate,
)


PACKAGE_ID = "BGSOFT_R10_FROZEN_BUNDLE_A1"
PACKAGE_TOKEN = "odeedit-s05-bgsoft-r12-a1"
PACKAGE_ARCHIVE = "bgsoft-r10-frozen-bundle-a1.tar"
PACKAGE_MANIFEST = "package-manifest.json"
PACKAGE_RECEIPT = "handoff-receipt.json"
R10_REPORT_RELATIVE = Path(
    "experiment-reports/servers/server1/"
    "2026-08-09-s05-common-coldcoord-fixed-e8-p1r10-r4-pair.md"
)
R10_REPORT_SHA256 = (
    "17285a1cf814c701fd9afda35dcf8da78e0a2c713458f88ed98c4540150e91ec"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/mnt/raid5/janghj/ODE-edit/local/source-handoff"
)
ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
LOCK_RELATIVES = (
    "project/run_scripts/ode_bf/locks/" + BG_SOFT_SOURCE_MANIFEST_FILE,
    "project/run_scripts/ode_bf/locks/" + BG_SOFT_REFERENCE_LOCK_FILE,
    "project/run_scripts/ode_bf/locks/numerical_lock_s05_common_coldcoord_fixed_e8.json",
    "project/run_scripts/ode_bf/locks/p1r10_common_coldcoord_cf_b10_seal.json",
    "project/run_scripts/ode_bf/locks/p1r2_p_population_seal.json",
    "project/run_scripts/ode_bf/locks/p1r2_seqb10_stream_seal.json",
    "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
    "project/run_scripts/ode_bf/locks/source_manifest_s05_common_coldcoord_fixed_e8.json",
)
RAW_FREE_LOCK_RELATIVES = (
    "project/run_scripts/ode_bf/locks/" + BG_SOFT_SOURCE_MANIFEST_FILE,
    "project/run_scripts/ode_bf/locks/" + BG_SOFT_REFERENCE_LOCK_FILE,
    "project/run_scripts/ode_bf/locks/numerical_lock_s05_common_coldcoord_fixed_e8.json",
    "project/run_scripts/ode_bf/locks/source_manifest_s05_common_coldcoord_fixed_e8.json",
)
LOCK_HASH_INDEX = "lock-hash-index.json"
REFERENCE_REQUIRED = (
    "terminal.json",
    "manifest.json",
    "raw/common-cold/action-freeze.json",
    "raw/common-cold/bootstrap-bg.json",
    "raw/common-cold/N32_NATIVE-postfreeze.json",
    "raw/common-cold/canonical-terminal-capture-contract.json",
    "raw/common-cold/context-overlay-gate.json",
    "raw/common-cold/reused-warm-context-identity.json",
    "raw/stepwise/W0_NO_EDIT.json",
    "raw/stepwise/N32_NATIVE.json",
    "raw/stepwise/action-freeze.json",
    "raw/stepwise/common-cold-panel.json",
    "raw/stepwise/reused-warm-evaluator-parity.json",
)
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
        "case_id",
        "case_ids",
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


def _run(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _validate_source_head(source_head: str) -> None:
    if len(source_head) != 40 or any(
        character not in "0123456789abcdef" for character in source_head
    ):
        raise ODEBFContractError("BG-Soft package source head differs")
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    implementation_parent = _run(["git", "rev-parse", "HEAD^^"]).stdout.strip()
    scientific_parent = _run(["git", "rev-parse", "HEAD^^^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ).stdout
    if (
        head != source_head
        or parent != BG_SOFT_EXECUTION_REPAIR_PARENT
        or implementation_parent != BG_SOFT_IMPLEMENTATION_PARENT
        or scientific_parent != BG_SOFT_PARENT_HEAD
        or branch != EXECUTION_BRANCH
        or dirty
    ):
        raise ODEBFContractError("BG-Soft package source provenance differs")


def _walk_json(value: Any, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            label = str(key)
            if label.lower() in FORBIDDEN_JSON_KEYS:
                raise ODEBFContractError(
                    "BG-Soft package raw/private JSON key is forbidden"
                )
            _walk_json(item, path=path + (label,))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_json(item, path=path + (str(index),))
        return
    if isinstance(value, str) and value.startswith("/"):
        raise ODEBFContractError(
            "BG-Soft package absolute path value is forbidden"
        )


def validate_raw_free_json(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("BG-Soft package JSON source differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFContractError("BG-Soft package JSON parse failed") from exc
    _walk_json(value)


def _reference_root(alias: str) -> Path:
    if alias not in ALIASES:
        raise ODEBFContractError("BG-Soft package alias differs")
    return R10_REFERENCE_RESULTS / (
        "s05-common-coldcoord-fixed-e8-p1r10-r4-" + alias + "-v1"
    )


def reference_inventory() -> tuple[tuple[str, Path, str], ...]:
    """Return the exact raw-free R10 closure copied into the archive."""

    report = R10_REFERENCE_REPO / R10_REPORT_RELATIVE
    if (
        report.is_symlink()
        or not report.is_file()
        or _sha256_file(report) != R10_REPORT_SHA256
    ):
        raise ODEBFContractError("BG-Soft frozen R10 report differs")
    rows: list[tuple[str, Path, str]] = [
        (
            "reference/r10-report/" + R10_REPORT_RELATIVE.name,
            report,
            "IMMUTABLE_RAW_FREE_R10_REPORT",
        )
    ]
    for alias in ALIASES:
        root = _reference_root(alias)
        if root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("BG-Soft frozen R10 alias root differs")
        selected: set[Path] = {root / item for item in REFERENCE_REQUIRED}
        selected.update((root / "raw/fixed-e8/BG-NEUTRAL").glob("*.json"))
        stepwise = root / "raw/stepwise/BG-NEUTRAL"
        if stepwise.is_dir() and not stepwise.is_symlink():
            selected.update(stepwise.glob("*.json"))
        failure = root / "raw/common-cold/failure-BG-NEUTRAL.json"
        if failure.is_file() and not failure.is_symlink():
            selected.add(failure)
        for path in sorted(selected):
            validate_raw_free_json(path)
            relative = path.relative_to(root).as_posix()
            rows.append(
                (
                    f"reference/r10/{alias}/{relative}",
                    path,
                    "IMMUTABLE_RAW_FREE_R10_RECEIPT",
                )
            )
    archive_paths = [item[0] for item in rows]
    if archive_paths != sorted(archive_paths) or len(archive_paths) != len(
        set(archive_paths)
    ):
        raise ODEBFContractError("BG-Soft reference inventory ordering differs")
    return tuple(rows)


def _create_lock_hash_index(path: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for relative in LOCK_RELATIVES:
        source = REPO_ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise ODEBFContractError("BG-Soft package lock source differs")
        entries.append(
            {
                "path": relative,
                "size": source.stat().st_size,
                "sha256": _sha256_file(source),
                "content_included": relative in RAW_FREE_LOCK_RELATIVES,
            }
        )
    value: dict[str, Any] = {
        "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-lock-hash-index/v1",
        "instruction_id": BG_SOFT_INSTRUCTION_ID,
        "amendment_id": BG_SOFT_AMENDMENT_ID,
        "entries": entries,
        "entry_count": len(entries),
        "raw_id_or_content_count": 0,
    }
    value["root_digest"] = canonical_hash(value)
    _write_json_once(path, value)
    validate_raw_free_json(path)
    return value


def _source_inventory(
    bundle_path: Path, lock_index_path: Path
) -> tuple[tuple[str, Path, str], ...]:
    rows: list[tuple[str, Path, str]] = [
        ("source/source.bundle", bundle_path, "THIN_GIT_BUNDLE"),
        (
            "source/" + LOCK_HASH_INDEX,
            lock_index_path,
            "RAW_FREE_LOCK_HASH_INDEX",
        ),
    ]
    for relative in RAW_FREE_LOCK_RELATIVES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError("BG-Soft package lock source differs")
        validate_raw_free_json(path)
        rows.append(("source/" + relative, path, "SOURCE_OR_LOCK_MANIFEST"))
    return tuple(rows)


def _parse_bundle_header(path: Path) -> dict[str, Any]:
    heads: list[dict[str, str]] = []
    prerequisites: list[str] = []
    with path.open("rb") as handle:
        first = handle.readline().decode("utf-8").rstrip("\n")
        if first not in ("# v2 git bundle", "# v3 git bundle"):
            raise ODEBFContractError("BG-Soft git bundle schema differs")
        for raw in handle:
            line = raw.decode("utf-8").rstrip("\n")
            if not line:
                break
            if line.startswith("-"):
                prerequisites.append(line[1:].split(" ", 1)[0])
                continue
            commit, name = line.split(" ", 1)
            heads.append({"commit": commit, "name": name})
    return {
        "version": first,
        "heads": heads,
        "prerequisites": prerequisites,
    }


def _create_thin_bundle(path: Path, source_head: str) -> dict[str, Any]:
    _run(
        [
            "git",
            "bundle",
            "create",
            str(path),
            "HEAD",
            "^" + BG_SOFT_PARENT_HEAD,
        ]
    )
    _run(["git", "bundle", "verify", str(path)])
    header = _parse_bundle_header(path)
    if header["heads"] != [{"commit": source_head, "name": "HEAD"}] or header[
        "prerequisites"
    ] != [BG_SOFT_PARENT_HEAD]:
        raise ODEBFContractError("BG-Soft git bundle identity differs")
    return {
        **header,
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
        "exact_scientific_parent_prerequisite": BG_SOFT_PARENT_HEAD,
        "execution_repair_parent": BG_SOFT_EXECUTION_REPAIR_PARENT,
        "implementation_parent": BG_SOFT_IMPLEMENTATION_PARENT,
        "complete_history": False,
    }


def _entry_manifest(
    inventory: Iterable[tuple[str, Path, str]],
) -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    for relative, path, role in sorted(inventory):
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ODEBFContractError("BG-Soft package relative path differs")
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError("BG-Soft package source file differs")
        entries.append(
            {
                "path": f"{PACKAGE_ID}/{relative}",
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
                "role": role,
                "regular_file": True,
                "symlink": False,
            }
        )
    paths = [str(item["path"]) for item in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ODEBFContractError("BG-Soft package entry ordering differs")
    return entries, canonical_hash(entries)


def _write_deterministic_tar(
    path: Path,
    inventory: Iterable[tuple[str, Path, str]],
) -> None:
    with path.open("xb") as output:
        with tarfile.open(
            fileobj=output,
            mode="w",
            format=tarfile.GNU_FORMAT,
        ) as archive:
            for relative, source, _ in sorted(inventory):
                data = source.read_bytes()
                info = tarfile.TarInfo(f"{PACKAGE_ID}/{relative}")
                info.size = len(data)
                info.mode = 0o600
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                archive.addfile(info, io.BytesIO(data))


def _verify_deterministic_tar(
    path: Path, entries: Sequence[Mapping[str, Any]]
) -> None:
    expected = {str(item["path"]): dict(item) for item in entries}
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        observed_names = [item.name for item in members]
        if observed_names != sorted(expected) or len(members) != len(expected):
            raise ODEBFContractError("BG-Soft package archive inventory differs")
        for member in members:
            item = expected[member.name]
            if (
                not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != item["size"]
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
            ):
                raise ODEBFContractError("BG-Soft package archive entry differs")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ODEBFContractError("BG-Soft package archive read failed")
            if hashlib.sha256(extracted.read()).hexdigest() != item["sha256"]:
                raise ODEBFContractError("BG-Soft package archive hash differs")


def build_package_plan(source_head: str) -> dict[str, Any]:
    _validate_source_head(source_head)
    source_manifest_sha256 = _source_manifest_gate(source_head)
    frozen_reference = _frozen_r10_reference_gate()
    references = reference_inventory()
    return {
        "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-handoff-dry-plan/v1",
        "instruction_id": BG_SOFT_INSTRUCTION_ID,
        "amendment_id": BG_SOFT_AMENDMENT_ID,
        "package_id": PACKAGE_ID,
        "source_head": source_head,
        "exact_parent": BG_SOFT_PARENT_HEAD,
        "execution_repair_parent": BG_SOFT_EXECUTION_REPAIR_PARENT,
        "implementation_parent": BG_SOFT_IMPLEMENTATION_PARENT,
        "source_manifest_sha256": source_manifest_sha256,
        "frozen_r10_reference_gate": frozen_reference,
        "raw_free_reference_file_count": len(references),
        "archive_file_count_before_bundle_and_locks": len(references),
        "symlink_count": 0,
        "raw_content_count": 0,
        "model_or_dataset_file_count": 0,
        "credential_or_private_config_file_count": 0,
        "transfer_executed": False,
        "create_once": True,
    }


def create_package(source_head: str, output_parent: Path) -> dict[str, Any]:
    plan = build_package_plan(source_head)
    if output_parent.is_symlink() or (
        output_parent.exists() and not output_parent.is_dir()
    ):
        raise ODEBFContractError("BG-Soft package parent differs")
    output_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    package_name = f"{PACKAGE_TOKEN}-{source_head[:12]}-v1"
    destination = output_parent / package_name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("BG-Soft package destination is create-once")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{package_name}.building-", dir=output_parent)
    )
    os.chmod(temporary, 0o700)
    bundle_path = temporary / "source.bundle"
    bundle = _create_thin_bundle(bundle_path, source_head)
    lock_index_path = temporary / LOCK_HASH_INDEX
    lock_index = _create_lock_hash_index(lock_index_path)
    inventory = _source_inventory(bundle_path, lock_index_path) + reference_inventory()
    entries, normalized_tree_digest = _entry_manifest(inventory)
    archive_path = temporary / PACKAGE_ARCHIVE
    _write_deterministic_tar(archive_path, inventory)
    _verify_deterministic_tar(archive_path, entries)
    manifest = {
        "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-handoff-package-manifest/v1",
        "instruction_id": BG_SOFT_INSTRUCTION_ID,
        "amendment_id": BG_SOFT_AMENDMENT_ID,
        "package_id": PACKAGE_ID,
        "package_name": package_name,
        "source_head": source_head,
        "exact_parent": BG_SOFT_PARENT_HEAD,
        "execution_repair_parent": BG_SOFT_EXECUTION_REPAIR_PARENT,
        "implementation_parent": BG_SOFT_IMPLEMENTATION_PARENT,
        "bundle": bundle,
        "lock_hash_index_root_digest": lock_index["root_digest"],
        "entries": entries,
        "entry_count": len(entries),
        "normalized_tree_digest": normalized_tree_digest,
        "no_symlinks": True,
        "raw_content_count": 0,
        "model_or_dataset_file_count": 0,
        "credential_or_private_config_file_count": 0,
    }
    manifest["root_digest"] = canonical_hash(manifest)
    manifest_sha256 = _write_json_once(temporary / PACKAGE_MANIFEST, manifest)
    archive_sha256 = _sha256_file(archive_path)
    receipt = {
        "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-handoff-receipt/v1",
        "instruction_id": BG_SOFT_INSTRUCTION_ID,
        "amendment_id": BG_SOFT_AMENDMENT_ID,
        "package_id": PACKAGE_ID,
        "package_name": package_name,
        "source_head": source_head,
        "exact_parent": BG_SOFT_PARENT_HEAD,
        "execution_repair_parent": BG_SOFT_EXECUTION_REPAIR_PARENT,
        "implementation_parent": BG_SOFT_IMPLEMENTATION_PARENT,
        "archive": {
            "name": PACKAGE_ARCHIVE,
            "sha256": archive_sha256,
            "size": archive_path.stat().st_size,
        },
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
        "plan": plan,
    }
    receipt["root_digest"] = canonical_hash(receipt)
    receipt_sha256 = _write_json_once(temporary / PACKAGE_RECEIPT, receipt)
    os.rename(temporary, destination)
    return {
        "status": "BGSOFT_R10_FROZEN_BUNDLE_A1_CREATED",
        "package_directory": str(destination),
        "source_head": source_head,
        "exact_parent": BG_SOFT_PARENT_HEAD,
        "execution_repair_parent": BG_SOFT_EXECUTION_REPAIR_PARENT,
        "implementation_parent": BG_SOFT_IMPLEMENTATION_PARENT,
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
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    value = (
        build_package_plan(args.source_head)
        if args.dry_run
        else create_package(args.source_head, args.output_parent)
    )
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
