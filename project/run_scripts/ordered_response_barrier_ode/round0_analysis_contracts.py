"""Analysis-only contracts for the ORBODE round-0 B100 review.

This module intentionally has no torch, model, Slurm, or experiment-runtime
dependency.  It defines the sealed lineage and deterministic file helpers used
by the canonical analysis package.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


INSTRUCTION_ID = "ODEEDIT-ORRBODE-ROUND0-DETAILED-REVIEW-20260904-R1"
NONCE = "ODEEDIT-ORRBODE-ROUND0-DETAILED-REVIEW-20260904-R1"
RAW_SOURCE_HEAD = "84e6cde74db9b85bb9ffa16685ff80f50f2e6e06"
RAW_SOURCE_TREE = "32d996fd3c7e124c6068d713e6b0abb80dfb9105"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
ORDER_ROOT = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
ROUND_JOB_ID = "35694"

CELL_BINDINGS = {
    0: ("llama3-8b-inst", "MEMIT", "35737"),
    1: ("llama3-8b-inst", "AlphaEdit", "35752"),
    2: ("qwen2.5-7b-inst", "MEMIT", "35889"),
    3: ("qwen2.5-7b-inst", "AlphaEdit", "35694"),
}
PRIMARY_ARMS = ("O", "QCL", "NQFIX", "ORBFH", "JAC")
STAGE_ORDER = ("PRE_EDIT",) + PRIMARY_ARMS
DYNAMIC_ARMS = ("QCL", "NQFIX", "ORBFH", "JAC")
LAYERS = (4, 5, 6, 7, 8)
SWEEPS = (0, 1, 2, 3)
KIND_COUNTS = {
    "rewrite_target_new": 100,
    "rewrite_target_true": 100,
    "rephrase_target_new": 200,
    "rephrase_target_true": 200,
    "locality_target_true": 1000,
}
EXPECTED_EVALUATION_ROWS = 1600
EXPECTED_PRIMARY_ENDPOINTS = 4 * 5 * 100
EXPECTED_REQUESTS = 4 * 100
EXPECTED_DYNAMIC_STEPS = 4 * 4 * 20
EXPECTED_REQUEST_STEPS = 4 * 4 * 20 * 100

AUTHORITATIVE_DOCUMENTS = (
    (
        "/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-03-ordered-response-barrier-ode-edit-proposal.md",
        "03a6fc61258e7643fc281fab8ab0d3ca700bb80f09aaa3eb34591ce5827087be",
        55900,
        1359,
    ),
    (
        "/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-03-ordered-response-barrier-ode-edit-gh-fast-main-prompt.md",
        "8548d016fda343f8b8917bcbe43647b58ad99ab6f796ece2fe4ef661faa4c431",
        13143,
        287,
    ),
    (
        "/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-03-ordered-response-barrier-ode-edit-fast-main-table.md",
        "e134ac708c556482b912d7103e12a12347a78958943e7fa3d3da0a473908aa40",
        18113,
        451,
    ),
)


class AnalysisBoundary(RuntimeError):
    """A sealed identity, denominator, or analysis-only boundary failed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise AnalysisBoundary(f"missing regular file: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise AnalysisBoundary(f"not a regular non-symlink file: {path}")
    return info


def member(path: Path, *, relative_to: Path | None = None, kind: str = "member") -> dict[str, Any]:
    info = regular_file(path)
    shown = path.relative_to(relative_to).as_posix() if relative_to is not None else str(path)
    return {
        "path": shown,
        "kind": kind,
        "bytes": int(info.st_size),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "sha256": sha256_file(path),
    }


def verify_canonical_identity(payload: Mapping[str, Any], field: str = "identity_sha256") -> None:
    stripped = dict(payload)
    expected = stripped.pop(field, None)
    if not isinstance(expected, str) or expected != canonical_hash(stripped):
        raise AnalysisBoundary(f"canonical identity mismatch: {field}")


def write_json_once(path: Path, payload: Any, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")

