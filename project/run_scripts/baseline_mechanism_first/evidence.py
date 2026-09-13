"""Read-only evidence inventory; byte availability never implies model validation."""
from __future__ import annotations

from collections import Counter
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Iterable, Mapping


class EvidenceUse(str, Enum):
    EXISTING_OBSERVATION = "EXISTING_COMPLETED_OBSERVATION"
    ACCESSIBLE_RAW_SOURCE = "ACCESSIBLE_RAW_OR_SOURCE"
    NEW_CASE_CALCULATION = "NEW_CPU_CASE_CALCULATION_FROM_RAW"
    NEW_GPU_REQUIRED = "NEW_GPU_INSTRUMENTATION_REQUIRED"
    MISSING = "MISSING_OR_UNVERIFIED"


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def inspect_evidence(path, *, kind: EvidenceUse, expected_sha256=None,
                     expected_bytes=None, declared_scope="", source_host="server1") -> dict:
    """lstat every component and hash regular bytes only; do not read tensors.

    This returns an inventory result, not a scientific PASS. A full-read assertion
    requires a separate human/agent read receipt and is deliberately always false.
    """
    p = Path(os.path.abspath(os.fspath(path)))
    result = dict(path=str(p), evidence_use=EvidenceUse(kind).value,
                  declared_scope=declared_scope, source_host=source_host,
                  full_read_claim=False, tensor_deserialized=False,
                  model_forward_count=0, verification="NOT_CHECKED")
    try:
        for component in (*reversed(p.parents), p):
            if stat.S_ISLNK(component.lstat().st_mode):
                return dict(result, status="SYMLINK_REJECTED", offending_path=str(component))
        before = p.stat()
        if not stat.S_ISREG(before.st_mode):
            return dict(result, status="NOT_REGULAR_FILE")
        digest = hashlib.sha256()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = p.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after):
            return dict(result, status="FILE_CHANGED_DURING_HASH")
        sha = digest.hexdigest()
        match = ((expected_sha256 is None or sha == expected_sha256)
                 and (expected_bytes is None or before.st_size == expected_bytes))
        return dict(result, status="FILE_BYTES_AVAILABLE" if match else "INTEGRITY_MISMATCH",
                    bytes=before.st_size, mode=f"{stat.S_IMODE(before.st_mode):04o}",
                    sha256=sha, expected_sha256=expected_sha256,
                    expected_bytes=expected_bytes, verification="LOCAL_FULL_BYTE_SHA_ONLY")
    except FileNotFoundError as exc:
        return dict(result, status="ABSENT_LOCAL", missing_path=str(exc.filename))
    except PermissionError as exc:
        return dict(result, status="ACCESS_DENIED", inaccessible_path=str(exc.filename))


def inventory_members(members: Iterable[Mapping], *, path_map: Mapping[str, str]) -> list[dict]:
    """Explicit exact source→local mapping only; never guess an S4/S2 mirror.

    Source member records carry path/sha256/bytes. Unmapped remote paths remain
    unavailable; existence of a same-named aggregate does not satisfy a raw member.
    """
    result = []
    for member in members:
        source = str(member["path"])
        if source not in path_map:
            result.append(dict(source_path=source, status="NO_EXPLICIT_LOCAL_BINDING",
                               evidence_use=EvidenceUse.MISSING.value,
                               expected_sha256=member["sha256"], expected_bytes=int(member["bytes"])))
            continue
        item = inspect_evidence(path_map[source], kind=EvidenceUse.ACCESSIBLE_RAW_SOURCE,
                                expected_sha256=member["sha256"], expected_bytes=int(member["bytes"]),
                                declared_scope=member.get("scope", "raw member"))
        result.append(dict(item, source_path=source))
    return result


def reuse_manifest(records: Iterable[Mapping]) -> dict:
    items = [dict(r) for r in records]
    return dict(schema="baseline-mechanism-evidence-v1", members=items,
                member_identity=canonical_sha(items),
                statuses=dict(Counter(r["status"] for r in items)),
                evidence_uses=dict(Counter(r["evidence_use"] for r in items)),
                aggregate_to_case_reconstruction=0, imputation=0,
                gpu_action_count=0, scientific_promotion=False)
