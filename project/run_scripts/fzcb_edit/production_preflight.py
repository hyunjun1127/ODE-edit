"""Essential source, artifact, stream, and focused-gate preflight."""

from __future__ import annotations

import argparse
import json
import stat
import subprocess
from pathlib import Path
from typing import Any

from project.run_scripts.fixed_z_nonuniqueness.preflight import build as fixed_z_artifact_preflight

from .contracts import EASYEDIT_HEAD, EASYEDIT_ROOT, EASYEDIT_TREE, EXPECTED_BASE_HEAD, EXPECTED_BASE_TREE, TechnicalBoundary
from .data import load_sealed_rows
from .hashing import file_sha256, write_json_once
from .production_contracts import (
    CONTRACT_BYTES,
    CONTRACT_LINES,
    CONTRACT_PATH,
    CONTRACT_SHA256,
    OLD_TECH_R1_EVIDENCE_CLASS,
    OLD_TECH_R1_HEAD,
    OLD_TECH_R1_REPORT_SHA256,
    OLD_TECH_R1_TREE,
    ProductionNumericalLock,
)


OLD_REPORT = Path(
    "experiment-reports/servers/server4/"
    "fzcb-tech-r1-joint-b1-controller-validity-audit-2026-09-01-v2/"
    "fzcb-tech-r1-joint-b1-controller-validity-factual-ko.md"
)
NUMERICAL_LOCK = Path("project/run_scripts/fzcb_edit/config/atomic-b10-production-numerical-lock-v1.json")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _regular(path: Path, *, mode: int | None = None) -> dict[str, Any]:
    observed = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise TechnicalBoundary(f"preflight member is not regular non-symlink: {path}")
    if mode is not None and stat.S_IMODE(observed.st_mode) != mode:
        raise TechnicalBoundary(f"preflight member mode mismatch: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": observed.st_size,
        "mode": oct(stat.S_IMODE(observed.st_mode)),
        "sha256": file_sha256(path),
    }


def build(source_root: Path, expected_head: str, focused_tests: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    head = _git(source_root, "rev-parse", "HEAD")
    tree = _git(source_root, "rev-parse", "HEAD^{tree}")
    if head != expected_head:
        raise TechnicalBoundary("production source HEAD differs from launcher lock")
    if _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise TechnicalBoundary("production source tracked tree is dirty")
    if _git(source_root, "merge-base", head, EXPECTED_BASE_HEAD) != EXPECTED_BASE_HEAD:
        raise TechnicalBoundary("production source is not descended from required origin/main")
    if _git(source_root, "rev-parse", f"{EXPECTED_BASE_HEAD}^{{tree}}") != EXPECTED_BASE_TREE:
        raise TechnicalBoundary("required origin/main tree identity differs")
    if _git(EASYEDIT_ROOT, "rev-parse", "HEAD") != EASYEDIT_HEAD or _git(EASYEDIT_ROOT, "rev-parse", "HEAD^{tree}") != EASYEDIT_TREE:
        raise TechnicalBoundary("pinned stock EasyEdit identity differs")
    if _git(EASYEDIT_ROOT, "status", "--porcelain", "--untracked-files=no"):
        raise TechnicalBoundary("pinned stock EasyEdit tracked tree is dirty")
    contract = _regular(CONTRACT_PATH, mode=0o600)
    if (
        contract["sha256"] != CONTRACT_SHA256
        or contract["bytes"] != CONTRACT_BYTES
        or CONTRACT_PATH.read_bytes().count(b"\n") != CONTRACT_LINES
    ):
        raise TechnicalBoundary("atomic-B10 authority identity differs")
    old_report = _regular(source_root / OLD_REPORT)
    if old_report["sha256"] != OLD_TECH_R1_REPORT_SHA256:
        raise TechnicalBoundary("immutable TECH-R1 evidence report differs")
    numerical = _regular(source_root / NUMERICAL_LOCK)
    lock_payload = json.loads((source_root / NUMERICAL_LOCK).read_text(encoding="utf-8"))
    if lock_payload["authority"] != ProductionNumericalLock().authority or lock_payload["sketch_ladder"] != [2, 8, 32, 128]:
        raise TechnicalBoundary("production numerical lock differs from typed contract")
    focused = _regular(focused_tests)
    focused_payload = json.loads(focused_tests.read_text(encoding="utf-8"))
    if focused_payload.get("status") != "PASS" or focused_payload.get("tests_passed") != 12:
        raise TechnicalBoundary("focused 12-gate receipt is not PASS")
    rows, stream = load_sealed_rows(10)
    artifact = fixed_z_artifact_preflight(source_root)
    return {
        "schema": "odeedit.s06.fzcb.atomic-b10-production-pilot.pre-gpu.v1",
        "status": "PRE_GPU_PASS",
        "source": {
            "root": str(source_root),
            "head": head,
            "tree": tree,
            "base_head": EXPECTED_BASE_HEAD,
            "base_tree": EXPECTED_BASE_TREE,
            "tracked_clean": True,
        },
        "authority": {**contract, "lines": CONTRACT_LINES, "full_read": True},
        "old_tech_r1": {
            "head": OLD_TECH_R1_HEAD,
            "tree": OLD_TECH_R1_TREE,
            "report": old_report,
            "evidence_class": OLD_TECH_R1_EVIDENCE_CLASS,
            "main_method_denominator": 0,
            "FZCB_terminal_valid": "0/2",
        },
        "easyedit": {
            "root": str(EASYEDIT_ROOT),
            "head": EASYEDIT_HEAD,
            "tree": EASYEDIT_TREE,
            "tracked_clean": True,
        },
        "stream": {**stream, "case_ids": [int(row["case_id"]) for row in rows]},
        "artifact_preflight": artifact,
        "numerical_lock": {"member": numerical, "payload": lock_payload},
        "focused_tests": focused,
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "methods": ["memit", "alphaedit"],
        "batch": "atomic-B10-joint",
        "full_fp32": True,
        "scalar_reduction_dtype": "float64",
        "sequential_submit_count": 0,
        "scientific_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--focused-tests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_once(args.output, build(args.source_root, args.expected_head, args.focused_tests))


if __name__ == "__main__":
    main()
