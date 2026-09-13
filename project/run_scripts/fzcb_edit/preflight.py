"""Essential identity, artifact, stream, and source preflight."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

from project.run_scripts.fixed_z_nonuniqueness.preflight import build as fixed_z_artifact_preflight

from .contracts import (
    CONTRACT_BYTES, CONTRACT_LINES, CONTRACT_PATH, CONTRACT_SHA256,
    EASYEDIT_HEAD, EASYEDIT_ROOT, EASYEDIT_TREE,
    EXPECTED_BASE_HEAD, EXPECTED_BASE_TREE, NumericalLock,
    PROPOSAL_BYTES, PROPOSAL_LINES, PROPOSAL_PATH, PROPOSAL_SHA256,
    TechnicalBoundary,
)
from .data import load_sealed_rows
from .forbidden_imports import scan
from .hashing import file_sha256, write_json_once


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _document(path: Path, sha256: str, size: int, lines: int, *, mode: int | None = None) -> dict[str, Any]:
    observed = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise TechnicalBoundary(f"authority is not regular non-symlink: {path}")
    if observed.st_size != size or file_sha256(path) != sha256 or path.read_bytes().count(b"\n") != lines:
        raise TechnicalBoundary(f"authority identity mismatch: {path}")
    if mode is not None and stat.S_IMODE(observed.st_mode) != mode:
        raise TechnicalBoundary(f"authority mode mismatch: {path}")
    return {"path": str(path), "sha256": sha256, "bytes": size, "lines": lines, "full_read": True}


def build(source_root: Path, expected_source_head: str | None = None) -> dict[str, Any]:
    source_root = source_root.resolve()
    head = _git(source_root, "rev-parse", "HEAD")
    tree = _git(source_root, "rev-parse", "HEAD^{tree}")
    if expected_source_head is not None and head != expected_source_head:
        raise TechnicalBoundary("execution source HEAD differs from launcher lock")
    if _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise TechnicalBoundary("execution source has tracked changes")
    if _git(source_root, "merge-base", head, EXPECTED_BASE_HEAD) != EXPECTED_BASE_HEAD:
        raise TechnicalBoundary("execution source is not based on required main")
    if _git(source_root, "rev-parse", f"{EXPECTED_BASE_HEAD}^{{tree}}") != EXPECTED_BASE_TREE:
        raise TechnicalBoundary("required base tree differs")
    if _git(EASYEDIT_ROOT, "rev-parse", "HEAD") != EASYEDIT_HEAD or _git(EASYEDIT_ROOT, "rev-parse", "HEAD^{tree}") != EASYEDIT_TREE:
        raise TechnicalBoundary("pinned stock EasyEdit identity differs")
    if _git(EASYEDIT_ROOT, "status", "--porcelain", "--untracked-files=no"):
        raise TechnicalBoundary("pinned stock EasyEdit tracked tree is dirty")
    proposal = _document(source_root / PROPOSAL_PATH, PROPOSAL_SHA256, PROPOSAL_BYTES, PROPOSAL_LINES)
    contract = _document(CONTRACT_PATH, CONTRACT_SHA256, CONTRACT_BYTES, CONTRACT_LINES, mode=0o600)
    _, b1 = load_sealed_rows(1)
    _, b10 = load_sealed_rows(10)
    violations = scan(source_root / "project/run_scripts/fzcb_edit")
    if violations:
        raise TechnicalBoundary(f"forbidden superseded-controller imports: {violations}")
    artifact = fixed_z_artifact_preflight(source_root)
    return {
        "schema": "odeedit.s06.fzcb-edit-main-method.pre-gpu.v1",
        "status": "PRE_GPU_PASS",
        "source": {"head": head, "tree": tree, "base_head": EXPECTED_BASE_HEAD, "base_tree": EXPECTED_BASE_TREE, "tracked_clean": True},
        "authority": {"proposal": proposal, "contract": contract},
        "easyedit": {"root": str(EASYEDIT_ROOT), "head": EASYEDIT_HEAD, "tree": EASYEDIT_TREE, "tracked_clean": True},
        "stream": {"B1": b1, "B10": b10},
        "artifact_preflight": artifact,
        "numerical_lock": NumericalLock().payload(),
        "full_fp32": True,
        "dense_inverse_count": 0,
        "explicit_kronecker_count": 0,
        "controller_output_metric_access_count": 0,
        "forbidden_controller_import_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-head")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.source_root, args.expected_source_head)
    write_json_once(args.output, payload)


if __name__ == "__main__":
    main()
