"""Frozen direct-z artifacts with fail-closed local cache reuse."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .contracts import (
    ContractError,
    ExpectedFileIdentity,
    FileRecord,
    SnapshotManifest,
    canonical_json,
    preflight_pinned_files,
    sha256_bytes,
    sha256_file,
)
from .hooks import tensor_sha256


DIRECT_Z_SCHEMA = "ode-edit-direct-z/v1"


@dataclass(frozen=True, slots=True)
class FrozenDirectZ:
    """Direct-z tensor computed once for an immutable entry state."""

    values: torch.Tensor
    source_snapshot_id: str
    source_state_id: str
    model_id: str
    context_id: str
    request_ids: tuple[str, ...]
    z_layer: int
    tensor_sha256: str
    artifact: FileRecord

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ContractError("direct-z values must be [hidden_size, request_count]")
        if self.values.shape[1] != len(self.request_ids) or not self.request_ids:
            raise ContractError("direct-z request count does not match request_ids")
        if not self.values.is_floating_point() or not torch.isfinite(self.values).all():
            raise ContractError("direct-z values must be finite floating-point tensors")
        frozen = self.values.detach().cpu().contiguous().clone()
        actual_hash = tensor_sha256(frozen)
        if actual_hash != self.tensor_sha256:
            raise ContractError(
                f"direct-z tensor hash mismatch: expected {self.tensor_sha256}, got {actual_hash}"
            )
        object.__setattr__(self, "values", frozen)

    @property
    def artifact_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DIRECT_Z_SCHEMA,
            "source_snapshot_id": self.source_snapshot_id,
            "source_state_id": self.source_state_id,
            "model_id": self.model_id,
            "context_id": self.context_id,
            "request_ids": list(self.request_ids),
            "z_layer": self.z_layer,
            "tensor_sha256": self.tensor_sha256,
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype),
            "artifact": self.artifact.to_dict(),
        }


class DirectZCache:
    """Single-file cache constrained to one caller-designated local root."""

    def __init__(self, local_cache_root: str | Path, cache_path: str | Path) -> None:
        root = Path(local_cache_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve(strict=True)
        candidate = Path(cache_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        self.path = candidate.resolve(strict=False)
        try:
            self.path.relative_to(self.root)
        except ValueError as exc:
            raise ContractError(
                f"direct-z cache path must stay under local root {self.root}: {self.path}"
            ) from exc

    @staticmethod
    def _metadata(snapshot: SnapshotManifest, z_layer: int) -> dict[str, Any]:
        return {
            "schema": DIRECT_Z_SCHEMA,
            "source_snapshot_id": snapshot.snapshot_id,
            "source_state_id": snapshot.state_id,
            "model_id": snapshot.model_id,
            "context_id": snapshot.context_id,
            "request_ids": list(snapshot.request_ids),
            "z_layer": int(z_layer),
        }

    @staticmethod
    def _validate_values(values: Any, request_count: int) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise ContractError("direct-z producer must return a tensor")
        frozen = values.detach().cpu().contiguous()
        if frozen.ndim != 2 or frozen.shape[1] != request_count:
            raise ContractError(
                "direct-z producer must return [hidden_size, request_count]"
            )
        if not frozen.is_floating_point() or not torch.isfinite(frozen).all():
            raise ContractError("direct-z producer returned invalid values")
        return frozen.clone()

    def _build(
        self,
        *,
        values: torch.Tensor,
        metadata: dict[str, Any],
        record: FileRecord,
    ) -> FrozenDirectZ:
        return FrozenDirectZ(
            values=values,
            source_snapshot_id=metadata["source_snapshot_id"],
            source_state_id=metadata["source_state_id"],
            model_id=metadata["model_id"],
            context_id=metadata["context_id"],
            request_ids=tuple(metadata["request_ids"]),
            z_layer=int(metadata["z_layer"]),
            tensor_sha256=metadata["tensor_sha256"],
            artifact=record,
        )

    def _load(
        self,
        *,
        expected_identity: ExpectedFileIdentity,
        expected_metadata: dict[str, Any],
    ) -> FrozenDirectZ:
        manifest = preflight_pinned_files(
            {str(self.path): expected_identity},
            label="direct-z cache",
        )
        try:
            payload = torch.load(
                self.path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as exc:
            raise ContractError(
                "safe direct-z loading requires torch.load(..., weights_only=True)"
            ) from exc
        if not isinstance(payload, dict) or set(payload) != {"metadata", "values"}:
            raise ContractError("direct-z cache payload has an unexpected schema")
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise ContractError("direct-z cache metadata must be a mapping")
        required_metadata = {
            **expected_metadata,
            "tensor_sha256": metadata.get("tensor_sha256"),
            "shape": metadata.get("shape"),
            "dtype": metadata.get("dtype"),
        }
        if metadata != required_metadata:
            raise ContractError("direct-z cache metadata does not match the requested state")
        values = self._validate_values(payload["values"], len(expected_metadata["request_ids"]))
        if list(values.shape) != metadata["shape"] or str(values.dtype) != metadata["dtype"]:
            raise ContractError("direct-z cache tensor shape or dtype does not match metadata")
        actual_tensor_hash = tensor_sha256(values)
        if actual_tensor_hash != metadata["tensor_sha256"]:
            raise ContractError(
                "direct-z cache tensor bytes do not match its embedded tensor hash"
            )
        manifest.assert_current()
        return self._build(
            values=values,
            metadata=metadata,
            record=manifest.files[0],
        )

    def _store(
        self,
        *,
        values: torch.Tensor,
        metadata: dict[str, Any],
    ) -> FrozenDirectZ:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = self.path.parent.resolve(strict=True)
        try:
            resolved_parent.relative_to(self.root)
        except ValueError as exc:
            raise ContractError("direct-z cache parent escaped the designated local root") from exc
        full_metadata = {
            **metadata,
            "tensor_sha256": tensor_sha256(values),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        }
        temporary = resolved_parent / f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        try:
            torch.save(
                {"metadata": full_metadata, "values": values},
                temporary,
            )
            try:
                self.path.hardlink_to(temporary)
            except FileExistsError as exc:
                raise ContractError(
                    "direct-z cache appeared concurrently; refusing to overwrite"
                ) from exc
        finally:
            if temporary.exists():
                temporary.unlink()
        record = FileRecord(
            path=str(self.path.resolve(strict=True)),
            sha256=sha256_file(self.path),
            size=self.path.stat().st_size,
        )
        return self._build(values=values, metadata=full_metadata, record=record)

    def load_or_compute(
        self,
        *,
        source_snapshot: SnapshotManifest,
        z_layer: int,
        compute: Callable[[], torch.Tensor],
        expected_identity: ExpectedFileIdentity | None = None,
    ) -> FrozenDirectZ:
        """Load a fully pinned artifact or invoke ``compute`` exactly once."""

        metadata = self._metadata(source_snapshot, z_layer)
        if self.path.exists():
            if expected_identity is None:
                raise ContractError(
                    "existing direct-z cache requires full expected sha256 and size"
                )
            return self._load(
                expected_identity=expected_identity,
                expected_metadata=metadata,
            )
        if expected_identity is not None:
            raise ContractError("pinned direct-z artifact is missing; refusing to recompute")
        values = self._validate_values(compute(), len(source_snapshot.request_ids))
        if self.path.exists():
            raise ContractError("direct-z cache appeared concurrently; refusing to overwrite")
        return self._store(values=values, metadata=metadata)
