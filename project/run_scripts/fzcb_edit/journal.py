"""Create-once atomic per-arm journals that survive later-arm failures."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import Arm, TechnicalBoundary
from .hashing import canonical_hash, file_sha256


def _atomic_json_once(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise TechnicalBoundary(f"refusing to overwrite journal member: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise TechnicalBoundary(f"journal temporary member already exists: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        body = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        os.write(descriptor, body)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.exists() or path.is_symlink():
        temporary.unlink(missing_ok=True)
        raise TechnicalBoundary(f"journal destination raced: {path}")
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@dataclass(slots=True)
class ArmJournal:
    root: Path
    model: str
    case_ids: tuple[int, ...]
    _members: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, root: Path, model: str, case_ids: tuple[int, ...]) -> "ArmJournal":
        if root.exists() or root.is_symlink():
            raise TechnicalBoundary(f"refusing to reuse arm journal root: {root}")
        root.mkdir(parents=True, mode=0o700)
        return cls(root=root, model=model, case_ids=case_ids)

    def append(self, ordinal: int, arm: Arm, payload: dict[str, Any]) -> dict[str, Any]:
        if ordinal != len(self._members):
            raise TechnicalBoundary("arm journal ordinal is not append-only")
        envelope = {
            "schema": "odeedit.s06.fzcb-tech-r1.arm-journal.v1",
            "ordinal": ordinal,
            "model": self.model,
            "case_ids": list(self.case_ids),
            "arm": arm.value,
            "payload": payload,
        }
        envelope["identity"] = canonical_hash(envelope)
        path = self.root / f"{ordinal:02d}-{arm.value.lower()}.json"
        _atomic_json_once(path, envelope)
        member = {
            "path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size,
            "identity": envelope["identity"], "arm": arm.value,
            "status": payload.get("status"),
        }
        self._members.append(member)
        return member

    def seal(self) -> dict[str, Any]:
        payload = {
            "schema": "odeedit.s06.fzcb-tech-r1.arm-journal-index.v1",
            "model": self.model,
            "case_ids": list(self.case_ids),
            "members": list(self._members),
            "member_root": canonical_hash(self._members),
        }
        path = self.root / "journal-index.json"
        _atomic_json_once(path, payload)
        return {
            "path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size,
            "member_root": payload["member_root"], "member_count": len(self._members),
        }

