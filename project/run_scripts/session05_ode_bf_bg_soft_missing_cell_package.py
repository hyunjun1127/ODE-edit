#!/usr/bin/env python3
"""Create the exact source/reference package for the R12 A1 SH2 handoff."""

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
from project.run_scripts.ode_bf.p1_bg_soft_missing_cell_panel import (
    BG_SOFT_AMENDMENT_ID,
    BG_SOFT_INSTRUCTION_ID,
    BG_SOFT_PARENT_HEAD,
    BG_SOFT_REFERENCE_LOCK_FILE,
    BG_SOFT_SCHEMA_NAMESPACE,
    BG_SOFT_SOURCE_MANIFEST_FILE,
)
from project.run_scripts.session05_ode_bf_submit_bg_soft_missing_cell import (
    BG_SOFT_EXECUTION_REPAIR_PARENT,
    BG_SOFT_IMPLEMENTATION_PARENT,
    BG_SOFT_PACKAGE_ID,
    BG_SOFT_PACKAGE_REPAIR_INSTRUCTION_ID,
    BG_SOFT_PACKAGE_REPAIR_PARENT,
    BG_SOFT_RUNTIME_TECH_REPAIR_PARENT,
    EXECUTION_BRANCH,
    R10_REFERENCE_REPO,
    R10_REFERENCE_RESULTS,
    _frozen_r10_reference_gate,
    _source_manifest_gate,
)


PACKAGE_REPAIR_INSTRUCTION_ID = BG_SOFT_PACKAGE_REPAIR_INSTRUCTION_ID
PACKAGE_ID = BG_SOFT_PACKAGE_ID
PACKAGE_TOKEN = "odeedit-s05-bgsoft-r12-a1-r1"
PACKAGE_ARCHIVE = "bgsoft-r10-frozen-bundle-a1-r1.tar"
PACKAGE_MANIFEST = "package-manifest.json"
PACKAGE_RECEIPT = "handoff-receipt.json"
SH2_RECIPIENT_SESSION = "019fe491-954b-70a0-8ba8-0588e9f8d741"
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
REFERENCE_ARMS = ("BG-NEUTRAL", "RS-NEUTRAL", "RS-SOFT")
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
    "raw/common-cold/bootstrap-rs.json",
    "raw/common-cold/N32_NATIVE-postfreeze.json",
    "raw/common-cold/canonical-terminal-capture-contract.json",
    "raw/common-cold/context-degeneracy-audit.json",
    "raw/common-cold/context-overlay-gate.json",
    "raw/common-cold/prior-qwen-ordinal6-rca.json",
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
CASE_ID_FIELD_NAMES = frozenset({"case_id", "case_ids"})
TARGET_IDENTITY_FIELD_NAMES = frozenset(
    {
        "frozen_target_load_once_per_case",
        "frozen_target_replay_exact",
        "memit_target_hparams_sha256",
        "target",
        "target_anchor_sha256",
        "target_old_source",
        "target_span_sha256",
        "target_token_sha256",
    }
)
PUBLIC_HOST_IDENTITY_PATHS = frozenset(
    {"agents/server2/head-server2.json", "workers/server2/status.json"}
)
PUBLIC_SOURCE_PATH_EXCEPTIONS = frozenset(
    {"local/README.md", "servers/templates/session-boundary.env"}
)
FORBIDDEN_BUNDLE_PATH_COMPONENTS = frozenset(
    {
        ".cache",
        "cache",
        "credentials",
        "data",
        "datasets",
        "logs",
        "secrets",
    }
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


def _read_regular_file_nofollow(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ODEBFContractError(
            "BG-Soft package source is not a stable regular file"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ODEBFContractError(
                "BG-Soft package source is not a stable regular file"
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
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    data = b"".join(chunks)
    if identity_before != identity_after or len(data) != before.st_size:
        raise ODEBFContractError("BG-Soft package source changed during read")
    return data, before


def _is_hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(character in "0123456789abcdef" for character in value)
    )


def _case_id_content_scan(value: Any) -> dict[str, Any]:
    field_names: set[str] = set()
    case_value_count = 0
    target_identity_fields: set[str] = set()
    tensor_digest_fields: set[str] = set()

    def walk(item: Any) -> None:
        nonlocal case_value_count
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ODEBFContractError(
                        "BG-Soft case-ID exception key differs"
                    )
                field_names.add(key)
                lowered = key.lower()
                if lowered in CASE_ID_FIELD_NAMES:
                    values = child if isinstance(child, list) else [child]
                    if not values or any(
                        (
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
                        )
                        for member in values
                    ):
                        raise ODEBFContractError(
                            "BG-Soft case-ID exception value differs"
                        )
                    case_value_count += len(values)
                elif lowered in FORBIDDEN_JSON_KEYS:
                    raise ODEBFContractError(
                        "BG-Soft raw content field is forbidden"
                    )
                if "prompt" in lowered or lowered in {
                    "raw_text",
                    "subject",
                    "token_ids",
                    "input_ids",
                }:
                    raise ODEBFContractError(
                        "BG-Soft prompt or token content is forbidden"
                    )
                if "target" in lowered:
                    if key not in TARGET_IDENTITY_FIELD_NAMES:
                        raise ODEBFContractError(
                            "BG-Soft target content field is forbidden"
                        )
                    if isinstance(child, str) and not (
                        _is_hex_digest(child) or key == "target_old_source"
                    ):
                        raise ODEBFContractError(
                            "BG-Soft target content value is forbidden"
                        )
                    if isinstance(child, (list, Mapping)):
                        raise ODEBFContractError(
                            "BG-Soft target payload is forbidden"
                        )
                    target_identity_fields.add(key)
                if "tensor" in lowered:
                    if not _is_hex_digest(child):
                        raise ODEBFContractError(
                            "BG-Soft tensor payload is forbidden"
                        )
                    tensor_digest_fields.add(key)
                if any(
                    marker in lowered
                    for marker in (
                        "credential",
                        "password",
                        "private_key",
                        "secret",
                    )
                ):
                    raise ODEBFContractError(
                        "BG-Soft credential content is forbidden"
                    )
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    if case_value_count <= 0:
        raise ODEBFContractError("BG-Soft case-ID exception is empty")
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


def _git_blob_bytes(object_id: str) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", object_id],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _owning_commit_identity(
    source_head: str, relative: str, object_id: str
) -> tuple[str, str, int]:
    commits = _run(
        ["git", "log", "--format=%H", source_head, "--", relative]
    ).stdout.splitlines()
    for commit in commits:
        row = _run(["git", "ls-tree", commit, "--", relative]).stdout.rstrip(
            "\n"
        )
        if "\t" not in row:
            continue
        metadata, observed_path = row.split("\t", 1)
        mode, kind, observed_object = metadata.split(" ", 2)
        if (
            observed_path == relative
            and kind == "blob"
            and observed_object == object_id
        ):
            tree = _run(["git", "rev-parse", commit + "^{tree}"]).stdout.strip()
            return commit, tree, int(mode, 8)
    raise ODEBFContractError("BG-Soft case-ID owning commit is unavailable")


def _source_bundle_content_scan(source_head: str) -> dict[str, Any]:
    lines = _run(["git", "rev-list", "--objects", source_head]).stdout.splitlines()
    exceptions: list[dict[str, Any]] = []
    public_host_identity_paths: set[str] = set()
    json_blob_count = 0
    for line in lines:
        parts = line.split(" ", 1)
        object_id = parts[0]
        if len(parts) != 2:
            continue
        relative = parts[1]
        path = Path(relative)
        lowered_parts = {part.lower() for part in path.parts}
        if relative not in PUBLIC_SOURCE_PATH_EXCEPTIONS and (
            lowered_parts & FORBIDDEN_BUNDLE_PATH_COMPONENTS
            or relative.lower().endswith(FORBIDDEN_BUNDLE_SUFFIXES)
            or relative.startswith("local/")
            or relative.startswith("servers/local/")
        ):
            raise ODEBFContractError(
                "BG-Soft source bundle contains a forbidden path class"
            )
        if not relative.endswith(".json"):
            continue
        data = _git_blob_bytes(object_id)
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ODEBFContractError(
                "BG-Soft source bundle JSON differs"
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
        if forbidden - CASE_ID_FIELD_NAMES:
            if forbidden == {"hostname"} and relative in PUBLIC_HOST_IDENTITY_PATHS:
                public_host_identity_paths.add(relative)
            else:
                raise ODEBFContractError(
                    "BG-Soft source bundle contains raw/private JSON fields"
                )
    exceptions.sort(key=lambda item: (item["canonical_path"], item["blob_object_id"]))
    paths = [item["canonical_path"] for item in exceptions]
    if len(paths) != len(set(paths)):
        raise ODEBFContractError(
            "BG-Soft case-ID exception path repeats across history"
        )
    if not exceptions:
        raise ODEBFContractError("BG-Soft case-ID exception inventory is empty")
    receipt = {
        "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-source-object-scan/v1",
        "source_head": source_head,
        "reachable_object_count": len(lines),
        "json_blob_count": json_blob_count,
        "tracked_case_id_only_exception": exceptions,
        "tracked_case_id_only_exception_count": len(exceptions),
        "public_host_role_identity_paths": sorted(public_host_identity_paths),
        "public_nonprivate_template_paths": sorted(
            PUBLIC_SOURCE_PATH_EXCEPTIONS
        ),
        "forbidden_path_match_count": 0,
        "raw_prompt_or_target_content_match_count": 0,
        "tensor_payload_match_count": 0,
        "credential_private_config_runtime_log_match_count": 0,
        "scientific_promotion_authorized": False,
        "secure_exact_source_handoff_exception_authorized": True,
    }
    receipt["root_digest"] = canonical_hash(receipt)
    return receipt


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
    package_repair_parent = _run(
        ["git", "rev-parse", "HEAD^^"]
    ).stdout.strip()
    execution_repair_parent = _run(
        ["git", "rev-parse", "HEAD^^^"]
    ).stdout.strip()
    implementation_parent = _run(
        ["git", "rev-parse", "HEAD^^^^"]
    ).stdout.strip()
    scientific_parent = _run(
        ["git", "rev-parse", "HEAD^^^^^"]
    ).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ).stdout
    if (
        head != source_head
        or parent != BG_SOFT_RUNTIME_TECH_REPAIR_PARENT
        or package_repair_parent != BG_SOFT_PACKAGE_REPAIR_PARENT
        or execution_repair_parent != BG_SOFT_EXECUTION_REPAIR_PARENT
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


def _rehash_link(source: Mapping[str, Any], key: str, target: Path) -> None:
    expected = source.get(key)
    if not isinstance(expected, str) or expected != _sha256_file(target):
        raise ODEBFContractError("BG-Soft frozen reference link differs")


def _arm_closure_receipt(
    root: Path, arm: str, selected: set[Path]
) -> dict[str, Any]:
    fixed = root / "raw/fixed-e8" / arm
    arm_paths = sorted(
        path
        for path in selected
        if fixed in path.parents
        or (root / "raw/stepwise" / arm) in path.parents
    )
    if not arm_paths:
        raise ODEBFContractError("BG-Soft frozen arm closure is empty")
    accepted = sorted(fixed.rglob("accepted-*.json"))
    if not accepted:
        raise ODEBFContractError("BG-Soft frozen arm accepted prefix is empty")
    transition_link_count = 0
    trial_link_count = 0
    for accepted_path in accepted:
        ordinal = accepted_path.stem.split("-", 1)[1]
        transition = accepted_path.parent / f"transition-{ordinal}.json"
        trial = accepted_path.parent / f"trial-{ordinal}.json"
        if transition not in selected or trial not in selected:
            raise ODEBFContractError("BG-Soft frozen linked receipt is missing")
        accepted_value = json.loads(accepted_path.read_text(encoding="utf-8"))
        transition_value = json.loads(transition.read_text(encoding="utf-8"))
        _rehash_link(accepted_value, "transition_sha256", transition)
        _rehash_link(transition_value, "trial_receipt_sha256", trial)
        transition_link_count += 1
        trial_link_count += 1
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
            "role": f"IMMUTABLE_RAW_FREE_R10_{arm}_RECEIPT",
        }
        for path in arm_paths
    ]
    receipt = {
        "arm": arm,
        "path_count": len(records),
        "accepted_prefix_count": len(accepted),
        "accepted_to_transition_link_rehash_count": transition_link_count,
        "transition_to_trial_link_rehash_count": trial_link_count,
        "tree_digest": canonical_hash(records),
    }
    receipt["root_digest"] = canonical_hash(receipt)
    return receipt


def _build_reference_inventory(
    *,
    reference_roots: Mapping[str, Path] | None = None,
    report_path: Path | None = None,
) -> tuple[tuple[tuple[str, Path, str], ...], dict[str, Any]]:
    """Return exact raw-free R10 closure and rooted per-arm receipts."""

    report = (
        R10_REFERENCE_REPO / R10_REPORT_RELATIVE
        if report_path is None
        else report_path
    )
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
    aliases: dict[str, Any] = {}
    for alias in ALIASES:
        root = (
            _reference_root(alias)
            if reference_roots is None
            else reference_roots[alias]
        )
        if root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("BG-Soft frozen R10 alias root differs")
        selected: set[Path] = {root / "terminal.json", root / "manifest.json"}
        selected.update((root / "raw/common-cold").glob("*.json"))
        selected.update((root / "raw/stepwise").glob("*.json"))
        for arm in REFERENCE_ARMS:
            selected.update((root / "raw/fixed-e8" / arm).rglob("*.json"))
            stepwise = root / "raw/stepwise" / arm
            if stepwise.is_dir() and not stepwise.is_symlink():
                selected.update(stepwise.rglob("*.json"))
        required = {root / item for item in REFERENCE_REQUIRED}
        if not required.issubset(selected) or any(
            path.is_symlink() or not path.is_file() for path in required
        ):
            raise ODEBFContractError("BG-Soft frozen required closure differs")
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
        terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        link_targets = {
            "action_freeze_sha256": root / "raw/common-cold/action-freeze.json",
            "canonical_terminal_capture_contract_sha256": root
            / "raw/common-cold/canonical-terminal-capture-contract.json",
            "context_degeneracy_audit_sha256": root
            / "raw/common-cold/context-degeneracy-audit.json",
            "context_overlay_gate_sha256": root
            / "raw/common-cold/context-overlay-gate.json",
            "n32_postfreeze_sha256": root
            / "raw/common-cold/N32_NATIVE-postfreeze.json",
            "prior_qwen_ordinal6_rca_sha256": root
            / "raw/common-cold/prior-qwen-ordinal6-rca.json",
            "reused_warm_context_identity_sha256": root
            / "raw/common-cold/reused-warm-context-identity.json",
            "reused_warm_evaluator_parity_sha256": root
            / "raw/stepwise/reused-warm-evaluator-parity.json",
            "stepwise_panel_sha256": root
            / "raw/stepwise/common-cold-panel.json",
        }
        for key, target in link_targets.items():
            if target not in selected:
                raise ODEBFContractError("BG-Soft frozen linked receipt is missing")
            _rehash_link(terminal, key, target)
        _rehash_link(manifest, "terminal_sha256", root / "terminal.json")
        _rehash_link(
            manifest,
            "stepwise_panel_sha256",
            root / "raw/stepwise/common-cold-panel.json",
        )
        alias_receipt = {
            "alias": alias,
            "common_manifest_link_rehash_count": len(link_targets) + 2,
            "arms": {
                arm: _arm_closure_receipt(root, arm, selected)
                for arm in REFERENCE_ARMS
            },
        }
        alias_receipt["root_digest"] = canonical_hash(alias_receipt)
        aliases[alias] = alias_receipt
    archive_paths = [item[0] for item in rows]
    if archive_paths != sorted(archive_paths) or len(archive_paths) != len(
        set(archive_paths)
    ):
        raise ODEBFContractError("BG-Soft reference inventory ordering differs")
    closure = {
        "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-frozen-reference-closure/v1",
        "aliases": aliases,
        "required_arms": list(REFERENCE_ARMS),
        "all_arm_path_counts_nonzero": all(
            alias_receipt["arms"][arm]["path_count"] > 0
            for alias_receipt in aliases.values()
            for arm in REFERENCE_ARMS
        ),
        "link_rehash_pass": True,
    }
    closure["root_digest"] = canonical_hash(closure)
    return tuple(rows), closure


def reference_inventory() -> tuple[tuple[str, Path, str], ...]:
    return _build_reference_inventory()[0]


def reference_closure_receipt() -> dict[str, Any]:
    return _build_reference_inventory()[1]


def _snapshot_reference_inventory(
    snapshot_root: Path,
) -> tuple[tuple[tuple[str, Path, str], ...], dict[str, Any]]:
    """Copy and revalidate one immutable private snapshot of all R10 inputs."""

    external_rows, external_closure = _build_reference_inventory()
    external_records: list[tuple[str, str, int, str]] = []
    for relative, source, role in external_rows:
        data, metadata = _read_regular_file_nofollow(source)
        external_records.append(
            (relative, role, metadata.st_size, hashlib.sha256(data).hexdigest())
        )
        target = snapshot_root / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    snapshot_roots = {
        alias: snapshot_root / "reference/r10" / alias for alias in ALIASES
    }
    snapshot_report = (
        snapshot_root / "reference/r10-report" / R10_REPORT_RELATIVE.name
    )
    snapshot_rows, snapshot_closure = _build_reference_inventory(
        reference_roots=snapshot_roots,
        report_path=snapshot_report,
    )
    snapshot_records = [
        (relative, role, path.stat().st_size, _sha256_file(path))
        for relative, path, role in snapshot_rows
    ]
    if (
        external_records != snapshot_records
        or external_closure != snapshot_closure
    ):
        raise ODEBFContractError("BG-Soft reference snapshot closure differs")
    return snapshot_rows, snapshot_closure


def _create_lock_hash_index(path: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for relative in LOCK_RELATIVES:
        source = REPO_ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise ODEBFContractError("BG-Soft package lock source differs")
        index_row = _run(
            ["git", "ls-files", "-s", "--", relative]
        ).stdout.rstrip("\n")
        if "\t" not in index_row:
            raise ODEBFContractError("BG-Soft package lock object differs")
        metadata, observed_path = index_row.split("\t", 1)
        mode, indexed_object, stage = metadata.split(" ", 2)
        data = source.read_bytes()
        object_id = _git_blob_oid(data)
        if observed_path != relative or stage != "0":
            raise ODEBFContractError("BG-Soft package lock object differs")
        if indexed_object != object_id:
            raise ODEBFContractError("BG-Soft package lock object differs")
        entries.append(
            {
                "path": relative,
                "mode": int(mode, 8),
                "size": source.stat().st_size,
                "sha256": _sha256_file(source),
                "object_id": object_id,
                "object_algorithm": "GIT_BLOB_SHA1",
                "role": (
                    "RAW_FREE_LOCK_CONTENT"
                    if relative in RAW_FREE_LOCK_RELATIVES
                    else "HASH_ONLY_RAW_OR_PRIVATE_LOCK"
                ),
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
        ("source/source.bundle", bundle_path, "SELF_CONTAINED_GIT_BUNDLE"),
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


def _verify_self_contained_bundle(
    path: Path, source_head: str, advertised_ref: str
) -> dict[str, Any]:
    source_tree = _run(["git", "rev-parse", source_head + "^{tree}"]).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="odeedit-r12-bundle-proof-") as temporary:
        root = Path(temporary)
        bare = root / "empty.git"
        checkout = root / "checkout"
        _run(["git", "init", "--bare", str(bare)])
        _run(["git", "-C", str(bare), "bundle", "verify", str(path)])
        _run(
            [
                "git",
                "-C",
                str(bare),
                "fetch",
                str(path),
                f"{advertised_ref}:{advertised_ref}",
            ]
        )
        _run(["git", "-C", str(bare), "fsck", "--full"])
        observed_chain = [
            _run(["git", "-C", str(bare), "rev-parse", advertised_ref + suffix])
            .stdout.strip()
            for suffix in ("", "^", "^^", "^^^", "^^^^", "^^^^^")
        ]
        expected_chain = [
            source_head,
            BG_SOFT_RUNTIME_TECH_REPAIR_PARENT,
            BG_SOFT_PACKAGE_REPAIR_PARENT,
            BG_SOFT_EXECUTION_REPAIR_PARENT,
            BG_SOFT_IMPLEMENTATION_PARENT,
            BG_SOFT_PARENT_HEAD,
        ]
        if observed_chain != expected_chain:
            raise ODEBFContractError("BG-Soft complete bundle lineage differs")
        _run(["git", "clone", "--no-checkout", str(bare), str(checkout)])
        _run(["git", "-C", str(checkout), "checkout", "--detach", source_head])
        checkout_head = _run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"]
        ).stdout.strip()
        checkout_tree = _run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"]
        ).stdout.strip()
        checkout_dirty = _run(
            ["git", "-C", str(checkout), "status", "--porcelain"]
        ).stdout
        if (
            checkout_head != source_head
            or checkout_tree != source_tree
            or checkout_dirty
        ):
            raise ODEBFContractError("BG-Soft complete bundle checkout differs")
    receipt = {
        "empty_bare_repository_bundle_verify": True,
        "empty_bare_repository_import": True,
        "empty_bare_repository_fsck_full": True,
        "materialized_checkout": True,
        "materialized_checkout_clean": True,
        "advertised_ref": advertised_ref,
        "source_head": source_head,
        "source_tree": source_tree,
        "exact_lineage": observed_chain,
        "external_prerequisite_count": 0,
        "complete_required_lineage_object_closure": True,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return receipt


def _create_complete_bundle(path: Path, source_head: str) -> dict[str, Any]:
    advertised_ref = "refs/heads/" + EXECUTION_BRANCH
    _run(
        [
            "git",
            "bundle",
            "create",
            str(path),
            EXECUTION_BRANCH,
        ]
    )
    _run(["git", "bundle", "verify", str(path)])
    header = _parse_bundle_header(path)
    if header["heads"] != [
        {"commit": source_head, "name": advertised_ref}
    ] or header["prerequisites"]:
        raise ODEBFContractError("BG-Soft git bundle identity differs")
    proof = _verify_self_contained_bundle(path, source_head, advertised_ref)
    content_scan = _source_bundle_content_scan(source_head)
    return {
        **header,
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
        "exact_scientific_parent_included": BG_SOFT_PARENT_HEAD,
        "runtime_technical_repair_parent": BG_SOFT_RUNTIME_TECH_REPAIR_PARENT,
        "package_repair_parent": BG_SOFT_PACKAGE_REPAIR_PARENT,
        "execution_repair_parent": BG_SOFT_EXECUTION_REPAIR_PARENT,
        "implementation_parent": BG_SOFT_IMPLEMENTATION_PARENT,
        "complete_history": True,
        "self_contained_verification": proof,
        "source_object_content_scan": content_scan,
    }


def _git_blob_oid(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _tracked_source_metadata(
    path: Path, source_head: str, source_tree: str, object_id: str
) -> dict[str, Any]:
    try:
        relative = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return {"committed_membership": False}
    result = _run(
        ["git", "ls-tree", source_head, "--", relative], check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"committed_membership": False}
    metadata, observed_path = result.stdout.rstrip("\n").split("\t", 1)
    mode, kind, committed_object = metadata.split(" ", 2)
    if (
        observed_path != relative
        or kind != "blob"
        or committed_object != object_id
    ):
        raise ODEBFContractError("BG-Soft committed source object differs")
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
        "path",
        "mode",
        "size",
        "sha256",
        "role",
        "semantic_role",
        "archive_kind",
        "object_id",
        "object_algorithm",
        "committed_membership",
    }
    paths = [entry.get("path") for entry in entries]
    if len(paths) != len(set(paths)):
        raise ODEBFContractError("BG-Soft package manifest path repeats")
    package_prefix = PACKAGE_ID + "/"
    for entry in entries:
        committed = entry.get("committed_membership")
        sha256 = entry.get("sha256")
        object_id = entry.get("object_id")
        role = entry.get("role")
        if (
            not required.issubset(entry)
            or not isinstance(entry.get("path"), str)
            or not str(entry["path"]).startswith(package_prefix)
            or ".." in Path(str(entry["path"])).parts
            or Path(str(entry["path"])).as_posix() != str(entry["path"])
            or not isinstance(entry.get("mode"), int)
            or entry.get("mode") != 0o600
            or isinstance(entry.get("size"), bool)
            or not isinstance(entry.get("size"), int)
            or entry.get("size") < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(role, str)
            or not role
            or entry.get("semantic_role") != role
            or entry.get("archive_kind") != "REGULAR_FILE"
            or entry.get("object_algorithm") != "GIT_BLOB_SHA1"
            or not isinstance(object_id, str)
            or len(object_id) != 40
            or any(character not in "0123456789abcdef" for character in object_id)
            or type(committed) is not bool
        ):
            raise ODEBFContractError("BG-Soft package manifest entry differs")
        if committed:
            committed_path = entry.get("committed_path")
            committed_head = entry.get("committed_source_head")
            committed_tree = entry.get("committed_source_tree")
            if (
                not isinstance(committed_path, str)
                or committed_path.startswith("/")
                or ".." in Path(committed_path).parts
                or Path(committed_path).as_posix() != committed_path
                or entry.get("committed_mode") not in (0o100644, 0o100755)
                or entry.get("committed_object_id") != object_id
                or not isinstance(committed_head, str)
                or len(committed_head) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in committed_head
                )
                or not isinstance(committed_tree, str)
                or len(committed_tree) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in committed_tree
                )
                or "uncommitted_content_class" in entry
            ):
                raise ODEBFContractError(
                    "BG-Soft committed manifest identity differs"
                )
        elif entry.get("uncommitted_content_class") not in {
            "GENERATED_PACKAGE_MEMBER",
            "EXTERNAL_RAW_FREE_REFERENCE",
        }:
            raise ODEBFContractError(
                "BG-Soft uncommitted manifest classification differs"
            )


def _entry_manifest(
    inventory: Iterable[tuple[str, Path, str]],
    *,
    source_head: str,
) -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    source_tree = _run(["git", "rev-parse", source_head + "^{tree}"]).stdout.strip()
    for relative, path, role in sorted(inventory):
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ODEBFContractError("BG-Soft package relative path differs")
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError("BG-Soft package source file differs")
        data = path.read_bytes()
        object_id = _git_blob_oid(data)
        tracked = _tracked_source_metadata(
            path, source_head, source_tree, object_id
        )
        if tracked["committed_membership"] is False:
            tracked["uncommitted_content_class"] = (
                "EXTERNAL_RAW_FREE_REFERENCE"
                if role.startswith("IMMUTABLE_RAW_FREE_R10_")
                else "GENERATED_PACKAGE_MEMBER"
            )
        entry = {
            "path": f"{PACKAGE_ID}/{relative}",
            "mode": 0o600,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "role": role,
            "semantic_role": role,
            "archive_kind": "REGULAR_FILE",
            "object_id": object_id,
            "object_algorithm": "GIT_BLOB_SHA1",
            "regular_file": True,
            "symlink": False,
            **tracked,
        }
        entries.append(entry)
    paths = [str(item["path"]) for item in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ODEBFContractError("BG-Soft package entry ordering differs")
    _validate_manifest_entries(entries)
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
    _validate_manifest_entries(entries)
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
                or member.mode != item["mode"]
                or member.size != item["size"]
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
            ):
                raise ODEBFContractError("BG-Soft package archive entry differs")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ODEBFContractError("BG-Soft package archive read failed")
            data = extracted.read()
            if (
                hashlib.sha256(data).hexdigest() != item["sha256"]
                or _git_blob_oid(data) != item["object_id"]
            ):
                raise ODEBFContractError("BG-Soft package archive hash differs")


def build_package_plan(source_head: str) -> dict[str, Any]:
    _validate_source_head(source_head)
    source_manifest_sha256 = _source_manifest_gate(source_head)
    frozen_reference = _frozen_r10_reference_gate()
    references, reference_closure = _build_reference_inventory()
    return {
        "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-handoff-dry-plan/v1",
        "instruction_id": BG_SOFT_INSTRUCTION_ID,
        "amendment_id": BG_SOFT_AMENDMENT_ID,
        "package_id": PACKAGE_ID,
        "package_repair_instruction_id": PACKAGE_REPAIR_INSTRUCTION_ID,
        "source_head": source_head,
        "exact_parent": BG_SOFT_PARENT_HEAD,
        "runtime_technical_repair_parent": BG_SOFT_RUNTIME_TECH_REPAIR_PARENT,
        "package_repair_parent": BG_SOFT_PACKAGE_REPAIR_PARENT,
        "execution_repair_parent": BG_SOFT_EXECUTION_REPAIR_PARENT,
        "implementation_parent": BG_SOFT_IMPLEMENTATION_PARENT,
        "source_manifest_sha256": source_manifest_sha256,
        "frozen_r10_reference_gate": frozen_reference,
        "frozen_reference_closure": reference_closure,
        "raw_free_reference_file_count": len(references),
        "archive_file_count_before_bundle_and_locks": len(references),
        "symlink_count": 0,
        "raw_reference_content_count": 0,
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
    bundle = _create_complete_bundle(bundle_path, source_head)
    lock_index_path = temporary / LOCK_HASH_INDEX
    lock_index = _create_lock_hash_index(lock_index_path)
    with tempfile.TemporaryDirectory(
        prefix=f".{package_name}.reference-snapshot-", dir=output_parent
    ) as reference_temporary:
        references, reference_closure = _snapshot_reference_inventory(
            Path(reference_temporary)
        )
        inventory = _source_inventory(bundle_path, lock_index_path) + references
        entries, normalized_tree_digest = _entry_manifest(
            inventory, source_head=source_head
        )
        archive_path = temporary / PACKAGE_ARCHIVE
        _write_deterministic_tar(archive_path, inventory)
        _verify_deterministic_tar(archive_path, entries)
        manifest = {
            "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-handoff-package-manifest/v1",
            "instruction_id": BG_SOFT_INSTRUCTION_ID,
            "amendment_id": BG_SOFT_AMENDMENT_ID,
            "package_repair_instruction_id": PACKAGE_REPAIR_INSTRUCTION_ID,
            "package_id": PACKAGE_ID,
            "package_name": package_name,
            "source_head": source_head,
            "exact_parent": BG_SOFT_PARENT_HEAD,
            "runtime_technical_repair_parent": BG_SOFT_RUNTIME_TECH_REPAIR_PARENT,
            "package_repair_parent": BG_SOFT_PACKAGE_REPAIR_PARENT,
            "execution_repair_parent": BG_SOFT_EXECUTION_REPAIR_PARENT,
            "implementation_parent": BG_SOFT_IMPLEMENTATION_PARENT,
            "bundle": bundle,
            "secure_transfer_boundary": {
                "recipient_session": SH2_RECIPIENT_SESSION,
                "route_policy": (
                    "CONFIGURED_PRIVATE_CREATE_ONCE_ROUTE_NO_ENDPOINT_DISCLOSURE"
                ),
                "case_ids_already_committed": True,
                "scientific_promotion_authorized": False,
                "raw_prompt_target_tensor_payload_included": False,
                "credential_private_config_runtime_log_included": False,
            },
            "frozen_reference_closure": reference_closure,
            "lock_hash_index_root_digest": lock_index["root_digest"],
            "entries": entries,
            "entry_count": len(entries),
            "normalized_tree_digest": normalized_tree_digest,
            "no_symlinks": True,
            "tracked_case_id_only_exception_count": bundle[
                "source_object_content_scan"
            ]["tracked_case_id_only_exception_count"],
            "prohibited_raw_content_count": 0,
            "model_or_dataset_file_count": 0,
            "credential_or_private_config_file_count": 0,
        }
        manifest["root_digest"] = canonical_hash(manifest)
        manifest_sha256 = _write_json_once(
            temporary / PACKAGE_MANIFEST, manifest
        )
    archive_sha256 = _sha256_file(archive_path)
    receipt = {
        "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-handoff-receipt/v1",
        "instruction_id": BG_SOFT_INSTRUCTION_ID,
        "amendment_id": BG_SOFT_AMENDMENT_ID,
        "package_repair_instruction_id": PACKAGE_REPAIR_INSTRUCTION_ID,
        "package_id": PACKAGE_ID,
        "package_name": package_name,
        "source_head": source_head,
        "exact_parent": BG_SOFT_PARENT_HEAD,
        "runtime_technical_repair_parent": BG_SOFT_RUNTIME_TECH_REPAIR_PARENT,
        "package_repair_parent": BG_SOFT_PACKAGE_REPAIR_PARENT,
        "execution_repair_parent": BG_SOFT_EXECUTION_REPAIR_PARENT,
        "implementation_parent": BG_SOFT_IMPLEMENTATION_PARENT,
        "secure_one_package_transfer": True,
        "recipient_session": SH2_RECIPIENT_SESSION,
        "recipient_path_control": "DISTINCT_ABSENT_CREATE_ONCE_CHILD",
        "case_ids_already_committed": True,
        "scientific_promotion_authorized": False,
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
        "status": "BGSOFT_R10_FROZEN_BUNDLE_A1_R1_CREATED",
        "package_directory": str(destination),
        "source_head": source_head,
        "exact_parent": BG_SOFT_PARENT_HEAD,
        "runtime_technical_repair_parent": BG_SOFT_RUNTIME_TECH_REPAIR_PARENT,
        "package_repair_parent": BG_SOFT_PACKAGE_REPAIR_PARENT,
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
