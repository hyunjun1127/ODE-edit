"""Fail-closed consumer preflight for the externally transferred P4 stream.

The producer/GH must provide the expected manifest SHA, package root, and the
exact scientific binding.  Until those values and the package exist, this
module reports a blocker and never guesses samples, order, seeds, datasets, or
evaluator identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
import stat
import tarfile
from typing import Mapping

from project.run_scripts.alphaedit_runtime_path_seal import canonical_hash


STREAM_SCHEMA = "ode-edit-p4-sealed-independent-b10x10-stream/v1"
EXPECTED_MODEL_ALIASES = ["llama3-8b-inst", "qwen2.5-7b-inst"]

TRANSFER_V2_ARCHIVE_SHA256 = (
    "748d9ff04ac584faf96af362b06fc9c34bc962062d8c1c72879f1c0311139756"
)
TRANSFER_V2_SOURCE_MANIFEST_SHA256 = (
    "3fd6e52fe838e1bff003af9d6bb83b23e661ba26ba5b831bd935bc1530ebf032"
)
TRANSFER_V2_ROOTED_RECEIPT_SHA256 = (
    "ed47bc390722f9411537452e4e70f1e0a760a67db123bca589c612fbf1fd2ce4"
)
TRANSFER_V2_MEMBERS_ROOT = (
    "28189ad33cdcda0c01d269216514508d62b15cd6c16ed053afd4683577169843"
)
TRANSFER_V2_STREAM_ROOT = (
    "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
)
TRANSFER_V2_ORDER_SHA256 = (
    "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
)
TRANSFER_V2_EVALUATOR_IDENTITY = (
    "72b8ecb737157a42d6a055ffd339dc3f907876a0a98165a00cca49ca9bbed07d"
)
TRANSFER_V2_PAYLOAD_FILES = 15
TRANSFER_V2_PAYLOAD_BYTES = 163_211
TRANSFER_V2_PAYLOAD_LINES = 3_287
TRANSFER_V2_ARCHIVE_BYTES = 184_320
TRANSFER_V2_EVALUATORS = {
    "evaluator/eval_utils_counterfact.py": (
        7_941,
        "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145",
    ),
    "evaluator/eval_utils_zsre.py": (
        4_168,
        "4370a3b2c7eff88bfeffd6412be485c0195bae794f7c7a6d39b9dd9e01299d97",
    ),
    "evaluator/summarize.py": (
        7_455,
        "64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0",
    ),
}


class P4SealedStreamError(RuntimeError):
    """Raised when transferred samples or evaluator bindings fail closed."""


@dataclass(frozen=True, slots=True)
class StreamReadiness:
    path: str
    status: str
    present: bool
    full_read: bool
    blocker: str | None


def stream_readiness(root: Path) -> StreamReadiness:
    if not root.exists():
        return StreamReadiness(
            path=str(root),
            status="BLOCKED_SEALED_STREAM_TRANSFER",
            present=False,
            full_read=False,
            blocker="sealed stream destination is absent",
        )
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise P4SealedStreamError("sealed stream root is not a canonical directory")
    return StreamReadiness(
        path=str(root),
        status="PRESENT_UNVERIFIED",
        present=True,
        full_read=False,
        blocker="producer/GH manifest identity is required",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise P4SealedStreamError(f"invalid stream field: {label}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise P4SealedStreamError(f"stream member escapes package: {label}")
    return value


def _read_regular_0600(path: Path, label: str) -> tuple[bytes, tuple[int, ...]]:
    observed = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        raise P4SealedStreamError(f"{label} is not a mode-0600 regular file")
    before = (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )
    raw = path.read_bytes()
    after_stat = path.stat()
    after = (
        after_stat.st_dev,
        after_stat.st_ino,
        after_stat.st_size,
        after_stat.st_mtime_ns,
        after_stat.st_ctime_ns,
    )
    if before != after or len(raw) != observed.st_size:
        raise P4SealedStreamError(f"{label} changed during full read")
    return raw, before


def _json_bytes(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P4SealedStreamError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise P4SealedStreamError(f"{label} is not a JSON object")
    return value


def _preflight_transfer_v2_archive(archive: Path) -> dict[str, object]:
    observed = archive.lstat()
    if (
        archive.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_size != TRANSFER_V2_ARCHIVE_BYTES
        or _sha256_file(archive) != TRANSFER_V2_ARCHIVE_SHA256
    ):
        raise P4SealedStreamError("transferred archive identity differs")
    names: set[str] = set()
    regular_count = 0
    regular_bytes = 0
    directory_count = 0
    with tarfile.open(archive, "r:") as handle:
        for member in handle.getmembers():
            path = PurePosixPath(member.name)
            normalized = path.as_posix().rstrip("/") or "."
            if path.is_absolute() or ".." in path.parts:
                raise P4SealedStreamError("archive member escapes package")
            if normalized in names:
                raise P4SealedStreamError("archive contains a duplicate member")
            names.add(normalized)
            if member.isdir():
                if member.mode != 0o700:
                    raise P4SealedStreamError("archive directory mode differs")
                directory_count += 1
            elif member.isreg():
                if normalized == "." or member.mode != 0o600:
                    raise P4SealedStreamError("archive regular member mode differs")
                regular_count += 1
                regular_bytes += member.size
            else:
                raise P4SealedStreamError(
                    "archive contains a link, device, or unsupported member"
                )
    if (
        regular_count != TRANSFER_V2_PAYLOAD_FILES
        or regular_bytes != TRANSFER_V2_PAYLOAD_BYTES
        or directory_count != 4
    ):
        raise P4SealedStreamError("archive bounded listing totals differ")
    return {
        "archive_path": str(archive),
        "archive_sha256": TRANSFER_V2_ARCHIVE_SHA256,
        "archive_bytes": TRANSFER_V2_ARCHIVE_BYTES,
        "archive_regular_count": regular_count,
        "archive_directory_count": directory_count,
        "path_traversal_count": 0,
        "link_device_unsupported_count": 0,
        "duplicate_member_count": 0,
    }


def preflight_transferred_stream_v2(
    root: Path,
    *,
    archive: Path,
    dataset_path: Path,
) -> dict[str, object]:
    """Full-read the authoritative P4 transfer-v2 package and bindings."""

    archive_receipt = _preflight_transfer_v2_archive(archive)
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise P4SealedStreamError("transfer-v2 extract root is not canonical")
    if stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise P4SealedStreamError("transfer-v2 extract root mode differs")

    manifest_path = root / "source-manifest.json"
    receipt_path = root / "rooted-receipt.json"
    manifest_raw, _ = _read_regular_0600(manifest_path, "source manifest")
    receipt_raw, _ = _read_regular_0600(receipt_path, "rooted receipt")
    if hashlib.sha256(manifest_raw).hexdigest() != TRANSFER_V2_SOURCE_MANIFEST_SHA256:
        raise P4SealedStreamError("transfer-v2 source manifest SHA256 differs")
    if hashlib.sha256(receipt_raw).hexdigest() != TRANSFER_V2_ROOTED_RECEIPT_SHA256:
        raise P4SealedStreamError("transfer-v2 rooted receipt SHA256 differs")
    manifest = _json_bytes(manifest_raw, "source manifest")
    rooted_receipt = _json_bytes(receipt_raw, "rooted receipt")
    members = manifest.get("members")
    if (
        manifest.get("schema")
        != "ode-edit-p4-r52-stream-transfer-source-manifest/v2"
        or manifest.get("status") != "SOURCE_READY"
        or manifest.get("raw_payload_count") != 0
        or not isinstance(members, list)
        or len(members) != 13
        or manifest.get("members_root_sha256") != TRANSFER_V2_MEMBERS_ROOT
        or manifest.get("evaluator_configuration_identity_sha256")
        != TRANSFER_V2_EVALUATOR_IDENTITY
    ):
        raise P4SealedStreamError("transfer-v2 source manifest contract differs")

    identities: list[dict[str, object]] = []
    sealed_names: list[str] = []
    for index, entry in enumerate(members):
        if not isinstance(entry, dict):
            raise P4SealedStreamError("transfer-v2 member record differs")
        relative = _safe_relative(entry.get("path"), f"members:{index}:path")
        raw, _ = _read_regular_0600(root / relative, relative)
        expected_bytes = entry.get("bytes")
        expected_lines = entry.get("lines")
        expected_sha = entry.get("sha256")
        if (
            entry.get("mode") != "0600"
            or entry.get("type") != "regular_non_symlink"
            or not isinstance(expected_bytes, int)
            or not isinstance(expected_lines, int)
            or len(raw) != expected_bytes
            or raw.count(b"\n") != expected_lines
            or not isinstance(expected_sha, str)
            or hashlib.sha256(raw).hexdigest() != expected_sha
        ):
            raise P4SealedStreamError("transfer-v2 member identity differs")
        identities.append(dict(entry))
        sealed_names.append(relative)
    if sealed_names != sorted(sealed_names) or len(sealed_names) != len(set(sealed_names)):
        raise P4SealedStreamError("transfer-v2 member order/uniqueness differs")
    if canonical_hash(identities) != TRANSFER_V2_MEMBERS_ROOT:
        raise P4SealedStreamError("transfer-v2 member root differs")

    actual_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    expected_files = sorted(sealed_names + ["rooted-receipt.json", "source-manifest.json"])
    if actual_files != expected_files:
        raise P4SealedStreamError("transfer-v2 file closure differs")
    all_raw = [_read_regular_0600(root / name, name)[0] for name in actual_files]
    if (
        len(actual_files) != TRANSFER_V2_PAYLOAD_FILES
        or sum(map(len, all_raw)) != TRANSFER_V2_PAYLOAD_BYTES
        or sum(raw.count(b"\n") for raw in all_raw) != TRANSFER_V2_PAYLOAD_LINES
    ):
        raise P4SealedStreamError("transfer-v2 extracted totals differ")
    for path in root.rglob("*"):
        if path.is_dir() and (path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o700):
            raise P4SealedStreamError("transfer-v2 directory identity differs")

    if (
        rooted_receipt.get("schema")
        != "ode-edit-p4-r52-stream-transfer-rooted-receipt/v2"
        or rooted_receipt.get("status") != "SOURCE_READY"
        or rooted_receipt.get("source_manifest_sha256")
        != TRANSFER_V2_SOURCE_MANIFEST_SHA256
        or rooted_receipt.get("members_root_sha256") != TRANSFER_V2_MEMBERS_ROOT
        or rooted_receipt.get("canonical_stream_root") != TRANSFER_V2_STREAM_ROOT
        or rooted_receipt.get("all_request_order_sha256")
        != TRANSFER_V2_ORDER_SHA256
        or rooted_receipt.get("evaluator_configuration_identity_sha256")
        != TRANSFER_V2_EVALUATOR_IDENTITY
    ):
        raise P4SealedStreamError("transfer-v2 rooted receipt contract differs")

    from .benchmark import PINNED_SOURCE_SHA256
    from .p1r24_independent_b10x10_selection import (
        load_historical_h0_batches,
        verify_historical_h0_fresh_seal,
    )
    from .request_digest import ordered_request_digest_v1

    stream_path = root / "canonical/p1r24_independent_b10x10_stream_seal.json"
    stream = _json_bytes(_read_regular_0600(stream_path, "stream seal")[0], "stream seal")
    verify_historical_h0_fresh_seal(stream)
    if (
        stream.get("root_digest") != TRANSFER_V2_STREAM_ROOT
        or stream.get("all_request_order_sha256") != TRANSFER_V2_ORDER_SHA256
    ):
        raise P4SealedStreamError("transfer-v2 stream/order identity differs")
    batches = load_historical_h0_batches(dataset_path, stream)
    batch_digests = [
        ordered_request_digest_v1([str(item["request_sha256"]) for item in batch])
        for batch in batches
    ]

    identity_path = root / "canonical/stream-order-context-identity.json"
    identity = _json_bytes(
        _read_regular_0600(identity_path, "stream/order/context identity")[0],
        "stream/order/context identity",
    )
    cases = identity.get("cases")
    model_binding = identity.get("model_stream_binding")
    if (
        identity.get("schema")
        != "ode-edit-p4-r52-independent-b10x10-stream-identity/v1"
        or identity.get("status") != "SOURCE_READY_READ_ONLY"
        or not isinstance(cases, list)
        or len(cases) != 10
        or not isinstance(model_binding, dict)
        or model_binding.get("llama_alias") != EXPECTED_MODEL_ALIASES[0]
        or model_binding.get("qwen_alias") != EXPECTED_MODEL_ALIASES[1]
        or model_binding.get("llama_batch_ordered_request_digest_v1") != batch_digests
        or model_binding.get("qwen_batch_ordered_request_digest_v1") != batch_digests
        or any(
            case.get("case_index") != index + 1
            or case.get("round_index") != index
            or case.get("request_order_sha256") != batch_digests[index]
            or case.get("members") != [
                {key: item[key] for key in (
                    "case_id", "collision_sha256", "ordinal", "rank_sha256",
                    "relation_sha256", "request_sha256", "round_ordinal",
                )}
                for item in stream["requests"][index * 10 : (index + 1) * 10]
            ]
            for index, case in enumerate(cases)
        )
    ):
        raise P4SealedStreamError("transfer-v2 case/model stream binding differs")

    evaluator_path = root / "canonical/evaluator-context-mapping.json"
    evaluator = _json_bytes(
        _read_regular_0600(evaluator_path, "evaluator mapping")[0],
        "evaluator mapping",
    )
    evaluator_core = dict(evaluator)
    observed_evaluator_identity = evaluator_core.pop(
        "evaluator_configuration_identity_sha256", None
    )
    evaluator_core.pop("configuration_identity_formula", None)
    evaluator_members = evaluator.get("evaluator_members")
    if (
        observed_evaluator_identity != TRANSFER_V2_EVALUATOR_IDENTITY
        or canonical_hash(evaluator_core) != TRANSFER_V2_EVALUATOR_IDENTITY
        or evaluator.get("stream_root") != TRANSFER_V2_STREAM_ROOT
        or evaluator.get("all_request_order_sha256") != TRANSFER_V2_ORDER_SHA256
        or not isinstance(evaluator_members, list)
        or {
            str(item.get("packaged_path")): (item.get("bytes"), item.get("sha256"))
            for item in evaluator_members
        }
        != TRANSFER_V2_EVALUATORS
        or {
            "evaluator/eval_utils_counterfact.py": (
                7_941, PINNED_SOURCE_SHA256["experiments/py/eval_utils_counterfact.py"]
            ),
            "evaluator/eval_utils_zsre.py": (
                4_168, PINNED_SOURCE_SHA256["experiments/py/eval_utils_zsre.py"]
            ),
            "evaluator/summarize.py": (
                7_455, PINNED_SOURCE_SHA256["experiments/summarize.py"]
            ),
        }
        != TRANSFER_V2_EVALUATORS
    ):
        raise P4SealedStreamError("transfer-v2 evaluator identity differs")

    dataset = dataset_path.resolve(strict=True)
    source = stream.get("source")
    if (
        dataset.is_symlink()
        or not dataset.is_file()
        or not isinstance(source, dict)
        or dataset.stat().st_size != source.get("size_bytes")
        or _sha256_file(dataset) != source.get("sha256")
    ):
        raise P4SealedStreamError("transfer-v2 canonical dataset identity differs")

    binding = {
        "model_aliases": EXPECTED_MODEL_ALIASES,
        "independent_slice_count_per_model": 10,
        "entries_per_slice": 10,
        "slice_membership_root": TRANSFER_V2_STREAM_ROOT,
        "order_root": TRANSFER_V2_ORDER_SHA256,
        "seed_identity": canonical_hash(identity.get("selection")),
        "dataset_identity": str(source["sha256"]),
        "evaluator_identity": TRANSFER_V2_EVALUATOR_IDENTITY,
    }
    result: dict[str, object] = {
        "schema": "ode-edit-p4-r52-stream-transfer-consumer-receipt/v2",
        "status": "TRANSFER_FULL_READ_PASS",
        "archive": archive_receipt,
        "extract_root": str(root),
        "manifest_sha256": TRANSFER_V2_SOURCE_MANIFEST_SHA256,
        "rooted_receipt_sha256": TRANSFER_V2_ROOTED_RECEIPT_SHA256,
        "member_count": TRANSFER_V2_PAYLOAD_FILES,
        "member_bytes": TRANSFER_V2_PAYLOAD_BYTES,
        "member_lines": TRANSFER_V2_PAYLOAD_LINES,
        "member_root": TRANSFER_V2_MEMBERS_ROOT,
        "stream_root": TRANSFER_V2_STREAM_ROOT,
        "order_sha256": TRANSFER_V2_ORDER_SHA256,
        "evaluator_identity": TRANSFER_V2_EVALUATOR_IDENTITY,
        "binding": binding,
        "batch_ordered_request_digest_v1": batch_digests,
        "full_read": True,
        "raw_payload_count": 0,
        "model_gpu_slurm_action_count": 0,
    }
    result["identity_sha256"] = canonical_hash(result)
    return result


def preflight_sealed_stream(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_package_root: str,
    expected_binding: Mapping[str, object],
) -> dict[str, object]:
    """Fully read a transferred package without interpreting sample content.

    The manifest must be ``manifest.json`` and is itself excluded from the
    member list.  ``binding`` is compared as canonical JSON, so model aliases,
    B10x10 slice membership/order, seeds, dataset, and evaluator identities are
    all producer/GH-owned values rather than server4 inventions.
    """

    if len(expected_manifest_sha256) != 64 or len(expected_package_root) != 64:
        raise P4SealedStreamError("expected stream identity is absent or malformed")
    ready = stream_readiness(root)
    if not ready.present:
        raise P4SealedStreamError(ready.status)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise P4SealedStreamError("stream manifest is absent or symlinked")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise P4SealedStreamError("stream manifest SHA256 differs")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P4SealedStreamError("stream manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != STREAM_SCHEMA:
        raise P4SealedStreamError("stream manifest schema differs")
    rooted = dict(manifest)
    observed_root = rooted.pop("root_digest", None)
    if observed_root != canonical_hash(rooted):
        raise P4SealedStreamError("stream manifest root digest differs")
    if canonical_hash(manifest.get("binding")) != canonical_hash(expected_binding):
        raise P4SealedStreamError("sample/order/seed/evaluator binding differs")
    binding = manifest.get("binding")
    if not isinstance(binding, dict):
        raise P4SealedStreamError("stream binding is absent")
    if binding.get("model_aliases") != EXPECTED_MODEL_ALIASES:
        raise P4SealedStreamError("stream model aliases/order differ")
    if binding.get("independent_slice_count_per_model") != 10:
        raise P4SealedStreamError("stream independent slice count differs")
    if binding.get("entries_per_slice") != 10:
        raise P4SealedStreamError("stream B10 membership differs")
    for key in (
        "slice_membership_root",
        "order_root",
        "seed_identity",
        "dataset_identity",
        "evaluator_identity",
    ):
        value = binding.get(key)
        if not isinstance(value, str) or not value:
            raise P4SealedStreamError(f"stream binding is incomplete: {key}")

    raw_members = manifest.get("members")
    if not isinstance(raw_members, list):
        raise P4SealedStreamError("stream member manifest is absent")
    identities: list[dict[str, object]] = []
    names: list[str] = []
    for index, entry in enumerate(raw_members):
        if not isinstance(entry, dict):
            raise P4SealedStreamError("stream member entry differs")
        relative = _safe_relative(entry.get("path"), f"members:{index}:path")
        size = entry.get("size")
        sha = entry.get("sha256")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(sha, str)
            or len(sha) != 64
        ):
            raise P4SealedStreamError("stream member identity differs")
        path = root / relative
        observed = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
            raise P4SealedStreamError("stream member is not regular/non-symlink")
        before = (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        )
        if observed.st_size != size or _sha256_file(path) != sha:
            raise P4SealedStreamError("stream member SHA256 or size differs")
        after_stat = path.stat()
        after = (
            after_stat.st_dev,
            after_stat.st_ino,
            after_stat.st_size,
            after_stat.st_mtime_ns,
            after_stat.st_ctime_ns,
        )
        if after != before:
            raise P4SealedStreamError("stream member changed during full read")
        identities.append({"path": relative, "sha256": sha, "size": size})
        names.append(relative)
    if names != sorted(names) or len(names) != len(set(names)):
        raise P4SealedStreamError("stream members are not unique and sorted")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != set(names) | {"manifest.json"}:
        raise P4SealedStreamError("stream package contains an unsealed member")
    allowed_directories = {
        parent.as_posix()
        for name in names
        for parent in Path(name).parents
        if parent != Path(".")
    }
    actual_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if actual_directories != allowed_directories:
        raise P4SealedStreamError("stream package directory closure differs")
    package_root = canonical_hash(identities)
    if package_root != expected_package_root or manifest.get("member_root") != package_root:
        raise P4SealedStreamError("stream package member root differs")
    return {
        "path": str(root),
        "status": "TRANSFER_FULL_READ_PASS",
        "manifest_sha256": expected_manifest_sha256,
        "manifest_root_digest": observed_root,
        "package_root": package_root,
        "member_count": len(identities),
        "binding": binding,
        "full_read": True,
        "decision_influence": "sealed-members-only",
    }
