"""Fail-closed provenance for replaying frozen MEMIT direct-z targets.

The AlphaEdit follow-up must reuse the exact direct-z tensors produced by the
completed MEMIT diagnostic.  This module deliberately only reads and hashes
those artifacts: it neither deserializes tensors nor creates cache paths, so a
missing or mismatched source artifact can never trigger a recomputation.
"""

from __future__ import annotations

import json
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    ContractError,
    ExpectedFileIdentity,
    ProvenanceManifest,
    canonical_json,
    preflight_pinned_files,
    sha256_bytes,
)


ALPHA_REPLAY_LOCK_SCHEMA = "ode-edit-direct-z-alpha-replay-lock/v1"
DEFAULT_ALPHA_REPLAY_LOCK_PATH = Path(__file__).with_name(
    "direct_z_alpha_replay_lock.json"
)
_MODEL_ALIASES = frozenset({"llama3-8b-inst", "qwen2.5-7b-inst"})
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def _exact_mapping(
    raw: Any,
    *,
    name: str,
    fields: set[str],
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ContractError(f"{name} must be a JSON object")
    actual = set(raw)
    unknown = actual - fields
    missing = fields - actual
    if unknown:
        raise ContractError(f"{name} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ContractError(f"{name} is missing fields: {sorted(missing)}")
    return raw


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a 64-character hexadecimal string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in string.hexdigits for char in normalized):
        raise ContractError(f"{name} must be a 64-character hexadecimal string")
    return normalized


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _positive_energy(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a positive finite number")
    energy = float(value)
    if not math.isfinite(energy) or energy <= 0.0:
        raise ContractError(f"{name} must be a positive finite number")
    return energy


def _safe_token(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value):
        raise ContractError(
            f"{name} must be a non-empty safe token of at most 128 ASCII characters"
        )
    if value in {".", ".."}:
        raise ContractError(f"{name} must not be a relative-path component")
    return value


def _case_order(raw: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ContractError(f"{name} must be a non-empty JSON array")
    values = tuple(_safe_token(f"{name}[{index}]", value) for index, value in enumerate(raw))
    if len(values) != len(set(values)):
        raise ContractError(f"{name} contains duplicate case ids")
    return values


def _positive_identity(name: str, raw: Any) -> ExpectedFileIdentity:
    try:
        identity = ExpectedFileIdentity.from_value(raw)
    except ContractError as exc:
        raise ContractError(f"{name} must contain exactly sha256 and positive size") from exc
    if identity.size <= 0:
        raise ContractError(f"{name}.size must be positive")
    return identity


@dataclass(frozen=True, slots=True)
class AlphaReplayArtifact:
    """One read-only direct-z artifact pinned by the replay lock."""

    case_id: str
    request_id: str
    artifact_identity: ExpectedFileIdentity
    tensor_sha256: str
    origin_lineage_id: str
    target_token_sha256: str
    reference_c_energy: float

    @classmethod
    def from_mapping(cls, raw: Any, *, name: str) -> "AlphaReplayArtifact":
        value = _exact_mapping(
            raw,
            name=name,
            fields={
                "case_id",
                "request_id",
                "artifact_sha256",
                "artifact_size",
                "tensor_sha256",
                "origin_lineage_id",
                "target_token_sha256",
                "reference_c_energy",
            },
        )
        artifact_size = _positive_int(f"{name}.artifact_size", value["artifact_size"])
        return cls(
            case_id=_safe_token(f"{name}.case_id", value["case_id"]),
            request_id=_sha256(f"{name}.request_id", value["request_id"]),
            artifact_identity=ExpectedFileIdentity(
                sha256=_sha256(f"{name}.artifact_sha256", value["artifact_sha256"]),
                size=artifact_size,
            ),
            tensor_sha256=_sha256(f"{name}.tensor_sha256", value["tensor_sha256"]),
            origin_lineage_id=_sha256(f"{name}.origin_lineage_id", value["origin_lineage_id"]),
            target_token_sha256=_sha256(
                f"{name}.target_token_sha256", value["target_token_sha256"]
            ),
            reference_c_energy=_positive_energy(
                f"{name}.reference_c_energy", value["reference_c_energy"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "request_id": self.request_id,
            "artifact_sha256": self.artifact_identity.sha256,
            "artifact_size": self.artifact_identity.size,
            "tensor_sha256": self.tensor_sha256,
            "origin_lineage_id": self.origin_lineage_id,
            "target_token_sha256": self.target_token_sha256,
            "reference_c_energy": self.reference_c_energy,
        }


@dataclass(frozen=True, slots=True)
class AlphaReplayModel:
    """Pinned source-run metadata and ordered direct-z records for one model."""

    model_alias: str
    source_run_id: str
    source_manifest_identity: ExpectedFileIdentity
    source_summary_identity: ExpectedFileIdentity
    artifacts: tuple[AlphaReplayArtifact, ...]

    @classmethod
    def from_mapping(
        cls,
        raw: Any,
        *,
        model_alias: str,
        case_order: tuple[str, ...],
    ) -> "AlphaReplayModel":
        value = _exact_mapping(
            raw,
            name=f"models.{model_alias}",
            fields={"source_run_id", "source_manifest", "source_summary", "artifacts"},
        )
        raw_artifacts = value["artifacts"]
        if not isinstance(raw_artifacts, list):
            raise ContractError(f"models.{model_alias}.artifacts must be a JSON array")
        artifacts = tuple(
            AlphaReplayArtifact.from_mapping(
                artifact,
                name=f"models.{model_alias}.artifacts[{index}]",
            )
            for index, artifact in enumerate(raw_artifacts)
        )
        if not artifacts:
            raise ContractError(f"models.{model_alias}.artifacts must not be empty")
        case_ids = tuple(artifact.case_id for artifact in artifacts)
        request_ids = tuple(artifact.request_id for artifact in artifacts)
        if case_ids != case_order:
            raise ContractError(
                f"models.{model_alias}.artifacts case order must exactly match case_order"
            )
        if len(request_ids) != len(set(request_ids)):
            raise ContractError(f"models.{model_alias}.artifacts contains duplicate request ids")
        for field_name, values in (
            ("artifact sha256 values", tuple(item.artifact_identity.sha256 for item in artifacts)),
            ("tensor sha256 values", tuple(item.tensor_sha256 for item in artifacts)),
            ("origin lineage ids", tuple(item.origin_lineage_id for item in artifacts)),
        ):
            if len(values) != len(set(values)):
                raise ContractError(
                    f"models.{model_alias}.artifacts contains duplicate {field_name}"
                )
        return cls(
            model_alias=model_alias,
            source_run_id=_safe_token(
                f"models.{model_alias}.source_run_id", value["source_run_id"]
            ),
            source_manifest_identity=_positive_identity(
                f"models.{model_alias}.source_manifest", value["source_manifest"]
            ),
            source_summary_identity=_positive_identity(
                f"models.{model_alias}.source_summary", value["source_summary"]
            ),
            artifacts=artifacts,
        )

    def record_for_request(self, request_id: str) -> AlphaReplayArtifact:
        normalized = _sha256("request_id", request_id)
        for artifact in self.artifacts:
            if artifact.request_id == normalized:
                return artifact
        raise ContractError(
            f"request_id {normalized} is not pinned for model {self.model_alias}"
        )

    def record_for_case(self, case_id: str) -> AlphaReplayArtifact:
        normalized = _safe_token("case_id", case_id)
        for artifact in self.artifacts:
            if artifact.case_id == normalized:
                return artifact
        raise ContractError(f"case_id {normalized!r} is not pinned for model {self.model_alias}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_run_id": self.source_run_id,
            "source_manifest": {
                "sha256": self.source_manifest_identity.sha256,
                "size": self.source_manifest_identity.size,
            },
            "source_summary": {
                "sha256": self.source_summary_identity.sha256,
                "size": self.source_summary_identity.size,
            },
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class DirectZAlphaReplayLock:
    """Strict, immutable direct-z replay lock for the AlphaEdit follow-up."""

    schema_version: str
    selection_sha256: str
    rank_slice: tuple[int, int]
    case_order: tuple[str, ...]
    models: tuple[AlphaReplayModel, ...]

    @classmethod
    def from_mapping(cls, raw: Any) -> "DirectZAlphaReplayLock":
        value = _exact_mapping(
            raw,
            name="direct-z alpha replay lock",
            fields={"schema_version", "selection_sha256", "rank_slice", "case_order", "models"},
        )
        if value["schema_version"] != ALPHA_REPLAY_LOCK_SCHEMA:
            raise ContractError(
                f"unsupported replay lock schema: {value['schema_version']!r}"
            )
        case_order = _case_order(value["case_order"], name="case_order")
        raw_rank_slice = value["rank_slice"]
        if not isinstance(raw_rank_slice, list) or len(raw_rank_slice) != 2:
            raise ContractError("rank_slice must contain exactly [start, stop]")
        start = _nonnegative_int("rank_slice[0]", raw_rank_slice[0])
        stop = _positive_int("rank_slice[1]", raw_rank_slice[1])
        if start >= stop:
            raise ContractError("rank_slice must satisfy start < stop")
        if stop - start != len(case_order):
            raise ContractError("rank_slice length must equal the case_order length")

        raw_models = value["models"]
        if not isinstance(raw_models, Mapping):
            raise ContractError("models must be a JSON object")
        model_aliases = set(raw_models)
        unknown_models = model_aliases - _MODEL_ALIASES
        missing_models = _MODEL_ALIASES - model_aliases
        if unknown_models:
            raise ContractError(f"models contains unsupported aliases: {sorted(unknown_models)}")
        if missing_models:
            raise ContractError(f"models is missing fixed aliases: {sorted(missing_models)}")
        models = tuple(
            AlphaReplayModel.from_mapping(
                raw_models[alias], model_alias=alias, case_order=case_order
            )
            for alias in sorted(_MODEL_ALIASES)
        )
        source_run_ids = tuple(model.source_run_id for model in models)
        if len(source_run_ids) != len(set(source_run_ids)):
            raise ContractError("models must not reuse a source_run_id")
        request_orders = tuple(tuple(item.request_id for item in model.artifacts) for model in models)
        if len(set(request_orders)) != 1:
            raise ContractError("models must use an identical ordered request-id sequence")
        return cls(
            schema_version=ALPHA_REPLAY_LOCK_SCHEMA,
            selection_sha256=_sha256("selection_sha256", value["selection_sha256"]),
            rank_slice=(start, stop),
            case_order=case_order,
            models=models,
        )

    def model_for_alias(self, model_alias: str) -> AlphaReplayModel:
        if not isinstance(model_alias, str):
            raise ContractError("model_alias must be a string")
        for model in self.models:
            if model.model_alias == model_alias:
                return model
        raise ContractError(f"model_alias is not pinned by replay lock: {model_alias!r}")

    def record_for_request(self, model_alias: str, request_id: str) -> AlphaReplayArtifact:
        return self.model_for_alias(model_alias).record_for_request(request_id)

    def record_for_case(self, model_alias: str, case_id: str) -> AlphaReplayArtifact:
        return self.model_for_alias(model_alias).record_for_case(case_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selection_sha256": self.selection_sha256,
            "rank_slice": list(self.rank_slice),
            "case_order": list(self.case_order),
            "models": {model.model_alias: model.to_dict() for model in self.models},
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json(self.to_dict()).encode("utf-8")

    @property
    def canonical_identity(self) -> ExpectedFileIdentity:
        """Content identity of the normalized lock, independent of JSON whitespace."""

        return ExpectedFileIdentity(
            sha256=sha256_bytes(self.canonical_bytes),
            size=len(self.canonical_bytes),
        )

    @property
    def lock_id(self) -> str:
        return self.canonical_identity.sha256


@dataclass(frozen=True, slots=True)
class ModelReplayPreflight:
    """A verified, read-only source run usable for immediate direct-z replay."""

    lock: DirectZAlphaReplayLock
    model: AlphaReplayModel
    source_run_root: Path
    provenance: ProvenanceManifest

    def record_for_request(self, request_id: str) -> AlphaReplayArtifact:
        return self.model.record_for_request(request_id)

    def artifact_path_for_request(self, request_id: str) -> Path:
        artifact = self.record_for_request(request_id)
        return self.source_run_root / "direct_z" / f"{artifact.request_id}.pt"

    def artifact_identity_for_request(self, request_id: str) -> ExpectedFileIdentity:
        return self.record_for_request(request_id).artifact_identity

    def assert_current(self) -> None:
        """Re-hash all source bytes before any later tensor read."""

        self.provenance.assert_current()


def _read_json_lock(path: Path) -> Any:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ContractError(f"replay lock must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ContractError(f"replay lock is missing: {candidate}") from exc
    if not resolved.is_file():
        raise ContractError(f"replay lock is not a regular file: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_object_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
    except json.JSONDecodeError as exc:
        raise ContractError(f"replay lock is not valid JSON: {resolved}") from exc


def load_alpha_replay_lock(
    path: str | Path = DEFAULT_ALPHA_REPLAY_LOCK_PATH,
) -> DirectZAlphaReplayLock:
    """Load the bundled replay lock without inspecting any source-run result."""

    return DirectZAlphaReplayLock.from_mapping(_read_json_lock(Path(path)))


def _normalize_expected_case_order(expected_case_order: Sequence[str]) -> tuple[str, ...]:
    if isinstance(expected_case_order, (str, bytes)):
        raise ContractError("expected_case_order must be a sequence of case ids")
    return tuple(
        _safe_token(f"expected_case_order[{index}]", case_id)
        for index, case_id in enumerate(expected_case_order)
    )


def _source_run_root(output_root: str | Path, source_run_id: str) -> Path:
    raw_output_root = Path(output_root).expanduser()
    if raw_output_root.is_symlink():
        raise ContractError(f"output_root must not be a symlink: {raw_output_root}")
    try:
        output_root_resolved = raw_output_root.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ContractError(f"output_root is missing: {raw_output_root}") from exc
    if not output_root_resolved.is_dir():
        raise ContractError(f"output_root is not a directory: {output_root_resolved}")
    candidate = output_root_resolved / source_run_id
    if candidate.is_symlink():
        raise ContractError(f"source run root must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ContractError(f"source run root is missing: {candidate}") from exc
    if not resolved.is_dir() or resolved.parent != output_root_resolved:
        raise ContractError("source run root escaped output_root or is not a directory")
    return resolved


def _secure_relative_file(source_run_root: Path, relative_path: str) -> None:
    """Reject a source path which is absent, escaped, or touches a symlink."""

    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ContractError(f"invalid relative source artifact path: {relative_path!r}")
    candidate = source_run_root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ContractError(f"source artifact path must not use symlinks: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        # The identity preflight below reports a missing artifact alongside all
        # other mismatches.  Lexical containment was already checked above.
        return
    except OSError as exc:
        raise ContractError(f"could not safely resolve source artifact: {candidate}") from exc
    try:
        resolved.relative_to(source_run_root)
    except ValueError as exc:
        raise ContractError(f"source artifact escaped source run root: {candidate}") from exc
    if not resolved.is_file():
        raise ContractError(f"source artifact is not a regular file: {resolved}")


def preflight_model_replay(
    output_root: str | Path,
    model_alias: str,
    expected_case_order: Sequence[str],
    *,
    replay_lock: DirectZAlphaReplayLock | None = None,
    lock_path: str | Path = DEFAULT_ALPHA_REPLAY_LOCK_PATH,
) -> ModelReplayPreflight:
    """Byte-verify the pinned source-run inputs without any write or fallback.

    ``manifest.json``, ``summary.json``, and every requested direct-z artifact
    are enumerated explicitly.  This function never calls a direct-z producer,
    creates a directory, or deserializes an artifact.
    """

    if replay_lock is not None and lock_path != DEFAULT_ALPHA_REPLAY_LOCK_PATH:
        raise ContractError("supply either replay_lock or lock_path, not both")
    lock = replay_lock if replay_lock is not None else load_alpha_replay_lock(lock_path)
    expected_order = _normalize_expected_case_order(expected_case_order)
    if expected_order != lock.case_order:
        raise ContractError("expected_case_order must exactly match the replay lock")
    model = lock.model_for_alias(model_alias)
    source_root = _source_run_root(output_root, model.source_run_id)
    expected_files: dict[str, ExpectedFileIdentity] = {
        "manifest.json": model.source_manifest_identity,
        "summary.json": model.source_summary_identity,
    }
    for artifact in model.artifacts:
        relative_path = f"direct_z/{artifact.request_id}.pt"
        if relative_path in expected_files:
            raise ContractError("replay lock resolved two identities to one source path")
        expected_files[relative_path] = artifact.artifact_identity
    for relative_path in expected_files:
        _secure_relative_file(source_root, relative_path)
    provenance = preflight_pinned_files(
        expected_files,
        label=f"ODE-Edit Alpha direct-z replay ({model.model_alias})",
        base_dir=source_root,
    )
    return ModelReplayPreflight(
        lock=lock,
        model=model,
        source_run_root=source_root,
        provenance=provenance,
    )
