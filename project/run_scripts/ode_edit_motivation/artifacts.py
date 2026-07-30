"""Strict JSONL artifact envelopes for motivation diagnostic runs."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Mapping

import torch

from .contracts import (
    ContextManifest,
    ContractError,
    ProvenanceManifest,
    SnapshotManifest,
)


def to_jsonable(value: Any) -> Any:
    """Convert diagnostic values to strict JSON primitives."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("artifact values must not contain NaN or infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return to_jsonable(detached.item())
        return to_jsonable(detached.tolist())
    if dataclasses.is_dataclass(value):
        if hasattr(value, "to_dict"):
            return to_jsonable(value.to_dict())
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise ContractError(f"artifact mapping key is not JSON-compatible: {key!r}")
            converted[str(key)] = to_jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    raise ContractError(f"artifact value is not JSON-compatible: {type(value).__name__}")


class JsonlArtifactWriter:
    """One-run JSONL writer bound to immutable provenance and contexts."""

    schema_version = "ode-edit-motivation/v1"

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        provenance: ProvenanceManifest,
        contexts: ContextManifest,
        snapshot: SnapshotManifest | None = None,
        fsync: bool = False,
        verify_each_write: bool = False,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ContractError("artifact run_id must be a non-empty string")
        self.path = Path(path)
        self.run_id = run_id
        self.provenance = provenance
        self.contexts = contexts
        self.snapshot = snapshot
        self.fsync = fsync
        self.verify_each_write = verify_each_write
        self._handle: Any = None
        self._sequence = 0
        self._lock = threading.Lock()

    def __enter__(self) -> "JsonlArtifactWriter":
        if self._handle is not None:
            raise RuntimeError("artifact writer is already open")
        self.provenance.assert_current()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally mixing snapshots or runs.
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        self.write(
            "run_manifest",
            {
                "provenance": self.provenance,
                "contexts": self.contexts,
                "snapshot": self.snapshot,
            },
        )
        return self

    def write(self, event: str, payload: Mapping[str, Any] | Any) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("artifact writer must be used as a context manager")
        if not isinstance(event, str) or not event.strip():
            raise ContractError("artifact event must be a non-empty string")
        with self._lock:
            if self.verify_each_write:
                self.provenance.assert_current()
            record = {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "event": event,
                "provenance_id": self.provenance.manifest_id,
                "context_id": self.contexts.manifest_id,
                "snapshot_id": None if self.snapshot is None else self.snapshot.snapshot_id,
                "payload": to_jsonable(payload),
            }
            encoded = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self._handle.write(encoded + "\n")
            self._handle.flush()
            if self.fsync:
                os.fsync(self._handle.fileno())
            self._sequence += 1
            return record

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False
