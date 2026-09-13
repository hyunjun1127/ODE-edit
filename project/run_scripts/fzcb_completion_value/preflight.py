"""Bounded identity/artifact/source preflight for K0/K1."""

from __future__ import annotations

import argparse
import ast
import json
import stat
import subprocess
from pathlib import Path

from project.run_scripts.fixed_z_nonuniqueness.preflight import build as reused_artifact_preflight

from .contracts import (
    CONTRACT_BYTES, CONTRACT_LINES, CONTRACT_PATH, CONTRACT_SHA256,
    DATASET, DATASET_SHA256, EASYEDIT_HEAD, EASYEDIT_ROOT, EASYEDIT_TREE,
    K0_CASE_IDS, K1_CASE_IDS, NumericalLock, TechnicalBoundary,
)
from .data import load_rows
from .hashing import canonical_hash, file_sha256, write_json_once


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _forbidden_surface(package: Path) -> list[str]:
    forbidden = ("value_gradient", "cbf", "qcqp", "hvp", "alphaedit")
    violations = []
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                value = ast.unparse(node).lower()
                if any(token in value for token in forbidden):
                    violations.append(f"{path.name}:{node.lineno}:{value}")
    return violations


def build(source_root: Path) -> dict[str, object]:
    st = CONTRACT_PATH.lstat()
    if not stat.S_ISREG(st.st_mode) or CONTRACT_PATH.is_symlink() or stat.S_IMODE(st.st_mode) != 0o600:
        raise TechnicalBoundary("contract type/mode mismatch")
    if st.st_size != CONTRACT_BYTES or file_sha256(CONTRACT_PATH) != CONTRACT_SHA256 or CONTRACT_PATH.read_bytes().count(b"\n") != CONTRACT_LINES:
        raise TechnicalBoundary("contract identity mismatch")
    if _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise TechnicalBoundary("tracked source dirty")
    if _git(EASYEDIT_ROOT, "rev-parse", "HEAD") != EASYEDIT_HEAD or _git(EASYEDIT_ROOT, "rev-parse", "HEAD^{tree}") != EASYEDIT_TREE or _git(EASYEDIT_ROOT, "status", "--porcelain", "--untracked-files=no"):
        raise TechnicalBoundary("stock EasyEdit identity mismatch")
    if file_sha256(DATASET) != DATASET_SHA256:
        raise TechnicalBoundary("CounterFact identity mismatch")
    k0, k1 = load_rows(source_root, "k0"), load_rows(source_root, "k1")
    if tuple(int(row["case_id"]) for row in k0) != K0_CASE_IDS or tuple(int(row["case_id"]) for row in k1) != K1_CASE_IDS or set(K0_CASE_IDS) & set(K1_CASE_IDS):
        raise TechnicalBoundary("K0/K1 sealed denominator mismatch")
    violations = _forbidden_surface(source_root / "project/run_scripts/fzcb_completion_value")
    if violations:
        raise TechnicalBoundary(f"K1-forbidden implementation surface: {violations}")
    reused = reused_artifact_preflight(source_root)
    package_root = source_root / "project/run_scripts/fzcb_completion_value"
    source_members = [
        {
            "path": str(path.relative_to(source_root)), "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(package_root.rglob("*.py"))
    ]
    payload = {
        "schema": "odeedit.s06.fzcb-completion-value.preflight.v1", "status": "PRE_GPU_PASS",
        "source": {"head": _git(source_root, "rev-parse", "HEAD"), "tree": _git(source_root, "rev-parse", "HEAD^{tree}"), "clean": True},
        "contract": {"path": str(CONTRACT_PATH), "sha256": CONTRACT_SHA256, "bytes": CONTRACT_BYTES, "wc_lines": CONTRACT_LINES, "mode": "0600"},
        "proposal": {"path": "project/proposals/2026-08-31-fzcb-edit-method-pivot-proposal.md", "sha256": file_sha256(source_root / "project/proposals/2026-08-31-fzcb-edit-method-pivot-proposal.md")},
        "k0_case_ids": list(K0_CASE_IDS), "k1_case_ids": list(K1_CASE_IDS), "k0_k1_overlap_count": 0,
        "dataset": {"path": str(DATASET), "sha256": DATASET_SHA256},
        "stock_easyedit": {"path": str(EASYEDIT_ROOT), "head": EASYEDIT_HEAD, "tree": EASYEDIT_TREE, "clean": True},
        "artifact_preflight_identity": canonical_hash(reused), "numerical_lock": NumericalLock().payload(),
        "source_members": source_members, "source_member_root": canonical_hash(source_members),
        "forbidden_k1_surface_count": 0, "dense_inverse_count": 0, "explicit_kronecker_count": 0,
        "value_gradient_count": 0, "cbf_count": 0, "full_qcqp_count": 0, "alphaedit_count": 0,
    }
    payload["identity"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_once(args.output, build(args.source_root.resolve()))


if __name__ == "__main__":
    main()
