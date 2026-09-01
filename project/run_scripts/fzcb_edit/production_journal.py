"""Create-once arm journal for method-qualified production cohorts."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import TechnicalBoundary
from .hashing import canonical_hash, file_sha256


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "NOT_RECORDED_NONFINITE_OR_EMPTY"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise TechnicalBoundary(f"refusing to overwrite production journal member: {path}")
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        body = (json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode()
        os.write(descriptor, body)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


@dataclass(slots=True)
class ProductionJournal:
    root: Path
    model: str
    case_ids: tuple[int, ...]
    members: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, root: Path, model: str, case_ids: tuple[int, ...]) -> "ProductionJournal":
        if root.exists() or root.is_symlink():
            raise TechnicalBoundary(f"refusing to reuse production journal: {root}")
        root.mkdir(parents=True, mode=0o700)
        return cls(root, model, case_ids)

    def append(self, arm: str, payload: dict[str, Any]) -> dict[str, Any]:
        ordinal = len(self.members)
        payload = _json_safe(payload)
        envelope = {
            "schema": "odeedit.s06.fzcb.atomic-b10.arm-journal.v1",
            "ordinal": ordinal,
            "model": self.model,
            "case_ids": list(self.case_ids),
            "arm": arm,
            "payload": payload,
        }
        envelope["identity"] = canonical_hash(envelope)
        path = self.root / f"{ordinal:02d}-{arm.lower()}.json"
        _atomic_once(path, envelope)
        member = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "identity": envelope["identity"],
            "arm": arm,
            "status": payload.get("status"),
        }
        self.members.append(member)
        return member

    def seal(self) -> dict[str, Any]:
        payload = {
            "schema": "odeedit.s06.fzcb.atomic-b10.arm-journal-index.v1",
            "model": self.model,
            "case_ids": list(self.case_ids),
            "members": list(self.members),
            "member_root": canonical_hash(self.members),
        }
        path = self.root / "journal-index.json"
        _atomic_once(path, payload)
        return {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "member_root": payload["member_root"],
            "member_count": len(self.members),
        }
