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
import stat
from typing import Mapping

from project.run_scripts.alphaedit_runtime_path_seal import canonical_hash


STREAM_SCHEMA = "ode-edit-p4-sealed-independent-b10x10-stream/v1"
EXPECTED_MODEL_ALIASES = ["llama3-8b-inst", "qwen2.5-7b-inst"]


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
