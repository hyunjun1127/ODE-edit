"""Immutable contracts used by the ODE-Edit motivation diagnostic.

This module intentionally has no dependency on EasyEdit.  The diagnostic can
therefore validate inputs and freeze a run snapshot before importing the
upstream repository or allocating a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import string
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


class ContractError(ValueError):
    """Raised when data cannot be represented by a diagnostic contract."""


class ProvenanceMismatchError(RuntimeError):
    """Raised when a file no longer matches its frozen provenance."""


def canonical_json(value: Any) -> str:
    """Return deterministic, strict JSON for hashable diagnostic metadata."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_text(name: str, value: Any, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a string")
    if not value or not value.strip():
        raise ContractError(f"{name} must not be empty")
    if len(value) > max_length:
        raise ContractError(f"{name} exceeds {max_length} characters")
    for char in value:
        category = unicodedata.category(char)
        if category.startswith("C") and char not in ("\n", "\t"):
            raise ContractError(f"{name} contains a control character")
    return value


def _single_empty_format_field(value: str, name: str) -> None:
    try:
        parsed = list(string.Formatter().parse(value))
    except ValueError as exc:
        raise ContractError(f"{name} is not a valid format template") from exc
    fields = [(field, spec, conversion) for _, field, spec, conversion in parsed if field is not None]
    if fields != [("", "", None)]:
        raise ContractError(f"{name} must contain exactly one '{{}}' field")


@dataclass(frozen=True, slots=True)
class EditRequest:
    """Allow-listed, canonical edit request accepted by the diagnostic."""

    case_id: str
    prompt: str
    subject: str
    target_new: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EditRequest":
        if not isinstance(raw, Mapping):
            raise ContractError("edit request must be a mapping")
        allowed = {"case_id", "prompt", "subject", "target_new"}
        unknown = set(raw) - allowed
        missing = allowed - set(raw)
        if unknown:
            raise ContractError(f"unexpected edit request fields: {sorted(unknown)}")
        if missing:
            raise ContractError(f"missing edit request fields: {sorted(missing)}")

        raw_case_id = raw["case_id"]
        if isinstance(raw_case_id, bool) or not isinstance(raw_case_id, (str, int)):
            raise ContractError("case_id must be a string or integer")
        case_id = _validate_text("case_id", str(raw_case_id), max_length=128)
        if any(char not in string.ascii_letters + string.digits + "._-" for char in case_id):
            raise ContractError("case_id may contain only ASCII letters, digits, '.', '_' and '-'")
        if case_id in {".", ".."}:
            raise ContractError("case_id must not be a relative-path component")

        prompt = _validate_text("prompt", raw["prompt"], max_length=4096)
        _single_empty_format_field(prompt, "prompt")
        subject = _validate_text("subject", raw["subject"], max_length=512)

        target_raw = raw["target_new"]
        if isinstance(target_raw, Mapping):
            if set(target_raw) != {"str"}:
                raise ContractError("mapping target_new must contain only the 'str' field")
            target_raw = target_raw["str"]
        target = _validate_text("target_new", target_raw, max_length=2048).strip()
        if not target:
            raise ContractError("target_new must contain non-whitespace text")

        return cls(case_id=case_id, prompt=prompt, subject=subject, target_new=target)

    @property
    def request_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()).encode("utf-8"))

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "subject": self.subject,
            "target_new": self.target_new,
        }

    def to_easyedit(self) -> dict[str, str]:
        # EasyEdit's MEMIT path independently adds this prefix.  Adding it here
        # makes direct compute_z calls and execute_memit calls agree.
        target = self.target_new if self.target_new.startswith(" ") else f" {self.target_new}"
        return {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "subject": self.subject,
            "target_new": target,
        }


def sanitize_edit_requests(raw_requests: Iterable[Mapping[str, Any]]) -> tuple[EditRequest, ...]:
    requests = tuple(EditRequest.from_mapping(raw) for raw in raw_requests)
    if not requests:
        raise ContractError("at least one edit request is required")
    case_ids = [request.case_id for request in requests]
    if len(set(case_ids)) != len(case_ids):
        raise ContractError("case_id values must be unique")
    return requests


@dataclass(frozen=True, slots=True)
class FileRecord:
    path: str
    sha256: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True, slots=True)
class ExpectedFileIdentity:
    """Required byte identity for fail-closed artifact preflight."""

    sha256: str
    size: int

    def __post_init__(self) -> None:
        normalized = self.sha256.lower()
        if len(normalized) != 64 or any(char not in string.hexdigits for char in normalized):
            raise ContractError("expected sha256 must be a full 64-character hex digest")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ContractError("expected file size must be a non-negative integer")
        object.__setattr__(self, "sha256", normalized)

    @classmethod
    def from_value(cls, value: Any) -> "ExpectedFileIdentity":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != {"sha256", "size"}:
            raise ContractError("expected file identity requires exactly 'sha256' and 'size'")
        return cls(sha256=value["sha256"], size=value["size"])


@dataclass(frozen=True, slots=True)
class ProvenanceManifest:
    """Frozen file identity checked before an upstream import."""

    label: str
    files: tuple[FileRecord, ...]

    @property
    def manifest_id(self) -> str:
        payload = {"label": self.label, "files": [record.to_dict() for record in self.files]}
        return sha256_bytes(canonical_json(payload).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "manifest_id": self.manifest_id,
            "files": [record.to_dict() for record in self.files],
        }

    def assert_current(self) -> None:
        mismatches: list[str] = []
        for record in self.files:
            path = Path(record.path)
            if not path.is_file():
                mismatches.append(f"{record.path}: missing")
                continue
            actual_size = path.stat().st_size
            if actual_size != record.size:
                mismatches.append(
                    f"{record.path}: expected size {record.size}, got {actual_size}"
                )
                continue
            actual_hash = sha256_file(path)
            if actual_hash != record.sha256:
                mismatches.append(
                    f"{record.path}: expected sha256 {record.sha256}, got {actual_hash}"
                )
        if mismatches:
            raise ProvenanceMismatchError("provenance preflight failed:\n" + "\n".join(mismatches))


def freeze_provenance(
    paths: Iterable[str | Path],
    *,
    label: str,
    expected_hashes: Mapping[str, str] | None = None,
) -> ProvenanceManifest:
    """Hash regular files and optionally pin them to externally supplied hashes."""

    label = _validate_text("provenance label", label, max_length=256)
    expected_hashes = dict(expected_hashes or {})
    for expected_path, expected_hash in expected_hashes.items():
        normalized = str(expected_hash).lower()
        if len(normalized) != 64 or any(char not in string.hexdigits for char in normalized):
            raise ContractError(
                f"expected hash for {expected_path} must be a full 64-character SHA-256"
            )
        expected_hashes[expected_path] = normalized
    records: list[FileRecord] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ContractError(f"provenance path is not a regular file: {path}")
        key = str(path)
        if key in seen:
            raise ContractError(f"duplicate provenance path: {path}")
        seen.add(key)
        actual_hash = sha256_file(path)
        expected = expected_hashes.get(key)
        if expected is not None and actual_hash != expected.lower():
            raise ProvenanceMismatchError(
                f"{path}: expected sha256 {expected.lower()}, got {actual_hash}"
            )
        records.append(FileRecord(path=key, sha256=actual_hash, size=path.stat().st_size))

    if not records:
        raise ContractError("provenance must contain at least one file")
    unused = set(expected_hashes) - seen
    if unused:
        raise ContractError(f"expected hashes contain unknown paths: {sorted(unused)}")
    return ProvenanceManifest(label=label, files=tuple(sorted(records, key=lambda item: item.path)))


def preflight_pinned_files(
    expected_files: Mapping[str, ExpectedFileIdentity | Mapping[str, Any]],
    *,
    label: str,
    base_dir: str | Path | None = None,
) -> ProvenanceManifest:
    """Verify every supplied path against both pinned size and full SHA-256.

    There is no discovery fallback: a caller must explicitly enumerate every
    file it wants in the manifest, and every entry must carry both fields.
    """

    if not expected_files:
        raise ContractError("pinned preflight requires at least one expected file")
    root = Path(base_dir).expanduser().resolve(strict=True) if base_dir is not None else None
    records: list[FileRecord] = []
    seen: set[str] = set()
    errors: list[str] = []
    for raw_path, raw_identity in expected_files.items():
        identity = ExpectedFileIdentity.from_value(raw_identity)
        candidate = Path(raw_path).expanduser()
        if root is not None:
            if candidate.is_absolute():
                raise ContractError("pinned paths must be relative when base_dir is supplied")
            candidate = root / candidate
        try:
            path = candidate.resolve(strict=True)
        except FileNotFoundError:
            errors.append(f"{candidate}: missing")
            continue
        key = str(path)
        if key in seen:
            raise ContractError(f"duplicate pinned path: {path}")
        seen.add(key)
        if not path.is_file():
            errors.append(f"{path}: not a regular file")
            continue
        actual_size = path.stat().st_size
        if actual_size != identity.size:
            errors.append(f"{path}: expected size {identity.size}, got {actual_size}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != identity.sha256:
            errors.append(f"{path}: expected sha256 {identity.sha256}, got {actual_hash}")
            continue
        records.append(FileRecord(path=key, sha256=actual_hash, size=actual_size))
    if errors:
        raise ProvenanceMismatchError("pinned file preflight failed:\n" + "\n".join(errors))
    return ProvenanceManifest(
        label=_validate_text("provenance label", label, max_length=256),
        files=tuple(sorted(records, key=lambda item: item.path)),
    )


@dataclass(frozen=True, slots=True)
class ContextManifest:
    """Deep-frozen context templates used for every proposal in a run."""

    source: str
    templates: tuple[tuple[str, ...], ...]

    @classmethod
    def freeze(
        cls,
        templates: Sequence[Sequence[str]],
        *,
        source: str,
    ) -> "ContextManifest":
        source = _validate_text("context source", source, max_length=256)
        frozen: list[tuple[str, ...]] = []
        if isinstance(templates, (str, bytes)) or not templates:
            raise ContractError("context templates must be a non-empty nested sequence")
        for group_index, group in enumerate(templates):
            if isinstance(group, (str, bytes)) or not group:
                raise ContractError(f"context group {group_index} must be non-empty")
            frozen_group: list[str] = []
            for item_index, template in enumerate(group):
                value = _validate_text(
                    f"context template {group_index}:{item_index}",
                    template,
                    max_length=8192,
                )
                _single_empty_format_field(value, f"context template {group_index}:{item_index}")
                frozen_group.append(value)
            frozen.append(tuple(frozen_group))
        return cls(source=source, templates=tuple(frozen))

    @property
    def manifest_id(self) -> str:
        payload = {"source": self.source, "templates": [list(group) for group in self.templates]}
        return sha256_bytes(canonical_json(payload).encode("utf-8"))

    def to_easyedit(self) -> list[list[str]]:
        return [list(group) for group in self.templates]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "manifest_id": self.manifest_id,
            "templates": [list(group) for group in self.templates],
        }


@dataclass(frozen=True, slots=True)
class ParameterRecord:
    name: str
    sha256: str
    shape: tuple[int, ...]
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Identity of a model/context/request entry state."""

    model_id: str
    context_id: str
    request_ids: tuple[str, ...]
    hparams_sha256: str
    parameters: tuple[ParameterRecord, ...]
    provenance_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ContractError("snapshot model_id must not be empty")
        if not self.request_ids:
            raise ContractError("snapshot must contain request ids")
        names = [record.name for record in self.parameters]
        if not names or len(names) != len(set(names)):
            raise ContractError("snapshot parameter names must be non-empty and unique")

    @property
    def snapshot_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict(include_id=False)).encode("utf-8"))

    @property
    def state_id(self) -> str:
        """Hash model/request/context/parameter state, excluding file provenance."""

        payload = self.to_dict(include_id=False)
        payload.pop("provenance_ids")
        return sha256_bytes(canonical_json(payload).encode("utf-8"))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_id": self.model_id,
            "context_id": self.context_id,
            "request_ids": list(self.request_ids),
            "hparams_sha256": self.hparams_sha256,
            "parameters": [record.to_dict() for record in self.parameters],
            "provenance_ids": list(self.provenance_ids),
        }
        if include_id:
            payload["snapshot_id"] = self.snapshot_id
            payload["state_id"] = self.state_id
        return payload

    def parameter(self, name: str) -> ParameterRecord:
        for record in self.parameters:
            if record.name == name:
                return record
        raise ContractError(f"parameter is not part of snapshot: {name}")


@dataclass(frozen=True, slots=True)
class LowRankFactor:
    """Canonical ``left @ right.T`` factorization in weight orientation."""

    weight_name: str
    left: torch.Tensor
    right: torch.Tensor
    expected_weight_sha256: str
    native_update_transposed: bool = False

    def __post_init__(self) -> None:
        if not self.weight_name:
            raise ContractError("weight_name must not be empty")
        if self.left.ndim != 2 or self.right.ndim != 2:
            raise ContractError("low-rank factors must be matrices")
        if self.left.shape[1] != self.right.shape[1] or self.left.shape[1] == 0:
            raise ContractError("low-rank factors must share a non-zero rank")
        if not self.left.is_floating_point() or not self.right.is_floating_point():
            raise ContractError("low-rank factors must be floating-point tensors")
        if not torch.isfinite(self.left).all() or not torch.isfinite(self.right).all():
            raise ContractError("low-rank factors must be finite")
        if not isinstance(self.native_update_transposed, bool):
            raise ContractError("native_update_transposed must be boolean")
        object.__setattr__(self, "left", self.left.detach().clone())
        object.__setattr__(self, "right", self.right.detach().clone())

    @property
    def rank(self) -> int:
        return self.left.shape[1]

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.left.shape[0], self.right.shape[0])

    def scaled(self, scale: float) -> "LowRankFactor":
        if not math.isfinite(scale):
            raise ContractError("factor scale must be finite")
        return LowRankFactor(
            weight_name=self.weight_name,
            left=self.left * scale,
            right=self.right,
            expected_weight_sha256=self.expected_weight_sha256,
            native_update_transposed=self.native_update_transposed,
        )


class ProposalSemantics(str, Enum):
    """State semantics under which a multi-layer factor set was measured."""

    ORDERED_GAUSS_SEIDEL = "ordered-gauss-seidel"
    SYNCHRONOUS_FROZEN_SNAPSHOT = "synchronous-frozen-snapshot"


@dataclass(frozen=True, slots=True)
class MemitFactorProposal:
    """MEMIT factor set with explicit ordered or synchronous semantics.

    ``snapshot`` is always the guarded entry state.  Ordered EasyEdit factors
    are generated after temporarily applying preceding layer updates and must
    not be described as same-snapshot factors.  Only
    ``SYNCHRONOUS_FROZEN_SNAPSHOT`` guarantees every factor was measured
    without changing any target weight.
    """

    snapshot: SnapshotManifest
    factors: tuple[LowRankFactor, ...]
    semantics: ProposalSemantics
    solver_name: str
    residual_denominator: int | None = None

    def __post_init__(self) -> None:
        if not self.factors:
            raise ContractError("factor proposal must not be empty")
        if not isinstance(self.semantics, ProposalSemantics):
            try:
                object.__setattr__(self, "semantics", ProposalSemantics(self.semantics))
            except ValueError as exc:
                raise ContractError(f"unknown proposal semantics: {self.semantics}") from exc
        if not isinstance(self.solver_name, str) or not self.solver_name:
            raise ContractError("factor proposal solver_name must not be empty")
        if self.semantics is ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT:
            if (
                isinstance(self.residual_denominator, bool)
                or not isinstance(self.residual_denominator, int)
                or self.residual_denominator <= 0
            ):
                raise ContractError(
                    "synchronous proposal requires a positive residual_denominator"
                )
        elif self.residual_denominator is not None:
            raise ContractError(
                "ordered proposal uses layer-dependent denominators and must record None"
            )
        names = [factor.weight_name for factor in self.factors]
        if len(names) != len(set(names)):
            raise ContractError("factor proposal contains duplicate weights")
        for factor in self.factors:
            record = self.snapshot.parameter(factor.weight_name)
            if factor.expected_weight_sha256 != record.sha256:
                raise ContractError(
                    f"factor hash for {factor.weight_name} does not match proposal snapshot"
                )
            if factor.weight_shape != record.shape:
                raise ContractError(
                    f"factor shape {factor.weight_shape} does not match {record.shape}"
                )

    @property
    def snapshot_id(self) -> str:
        return self.snapshot.snapshot_id

    @property
    def entry_snapshot_id(self) -> str:
        """Entry-state identity excluding factor-source file provenance."""

        return self.snapshot.state_id

    @property
    def is_synchronous(self) -> bool:
        return self.semantics is ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT

    def assert_same_entry_snapshot(self, other: "MemitFactorProposal") -> None:
        if self.entry_snapshot_id != other.entry_snapshot_id:
            raise ContractError(
                "proposal entry states differ: "
                f"{self.entry_snapshot_id} != {other.entry_snapshot_id}"
            )

    def assert_same_snapshot(self, other: "MemitFactorProposal") -> None:
        if not self.is_synchronous or not other.is_synchronous:
            raise ContractError(
                "same-snapshot comparison requires two synchronous-frozen-snapshot proposals"
            )
        self.assert_same_entry_snapshot(other)


def orient_easyedit_factor(
    adjusted_keys: torch.Tensor,
    residuals: torch.Tensor,
    *,
    weight_name: str,
    weight_shape: Sequence[int],
    expected_weight_sha256: str,
) -> LowRankFactor:
    """Orient EasyEdit's ``(adj_k, resid)`` pair without materializing ΔW."""

    shape = tuple(int(dim) for dim in weight_shape)
    if len(shape) != 2:
        raise ContractError("MEMIT target weight must be a matrix")
    direct_shape = (adjusted_keys.shape[0], residuals.shape[0])
    if direct_shape == shape:
        left, right = adjusted_keys, residuals
        native_update_transposed = False
    elif direct_shape[::-1] == shape:
        left, right = residuals, adjusted_keys
        native_update_transposed = True
    else:
        raise ContractError(
            f"EasyEdit factors imply {direct_shape} (or transpose), not target shape {shape}"
        )
    return LowRankFactor(
        weight_name=weight_name,
        left=left,
        right=right,
        expected_weight_sha256=expected_weight_sha256,
        native_update_transposed=native_update_transposed,
    )
