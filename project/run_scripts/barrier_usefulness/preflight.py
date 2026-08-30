"""Essential G0 identity, source, manifest and leakage preflight."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

from .contracts import BarrierLock, TechnicalBoundary
from .firewall import forbidden_imports
from .hashing import canonical_hash, file_sha256, write_json_once


EXPECTED_CONTRACT = {
    "sha256": "042fc456a57d1c8c672636116aebdce6580209efbd74f72d9bd21d502f716595",
    "bytes": 21249,
    "lines": 632,
    "mode": "0600",
}
EXPECTED_EASYEDIT = {
    "root": "/data/janghj/EasyEdit-stock-14cea824",
    "head": "14cea8245f06715684592ab55184939b99d70784",
    "tree": "9c52aadbc0883da422badf0a730fff21aaa3a8a7",
}
REUSED_MEMBERS = (
    "project/run_scripts/fixed_z_nonuniqueness/contracts.py",
    "project/run_scripts/fixed_z_nonuniqueness/evaluation.py",
    "project/run_scripts/fixed_z_nonuniqueness/official.py",
    "project/run_scripts/fixed_z_nonuniqueness/padding.py",
    "project/run_scripts/fixed_z_nonuniqueness/config/artifact-lock-v1.json",
    "project/run_scripts/fixed_z_nonuniqueness/config/case-manifest-v1.json",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _identity(path: Path) -> dict[str, Any]:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or path.is_symlink():
        raise TechnicalBoundary(f"not regular non-symlink: {path}")
    return {"path": str(path), "sha256": file_sha256(path), "bytes": st.st_size, "mode": format(stat.S_IMODE(st.st_mode), "04o")}


def build(source_root: Path, contract: Path) -> dict[str, Any]:
    contract_id = _identity(contract)
    if contract_id["sha256"] != EXPECTED_CONTRACT["sha256"] or contract_id["bytes"] != EXPECTED_CONTRACT["bytes"] or contract_id["mode"] != EXPECTED_CONTRACT["mode"] or len(contract.read_text().splitlines()) != EXPECTED_CONTRACT["lines"]:
        raise TechnicalBoundary("authoritative contract identity mismatch")
    if _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise TechnicalBoundary("tracked source worktree not clean")
    easyedit = Path(EXPECTED_EASYEDIT["root"])
    if _git(easyedit, "rev-parse", "HEAD") != EXPECTED_EASYEDIT["head"] or _git(easyedit, "rev-parse", "HEAD^{tree}") != EXPECTED_EASYEDIT["tree"] or _git(easyedit, "status", "--porcelain", "--untracked-files=no"):
        raise TechnicalBoundary("stock EasyEdit identity/clean mismatch")
    package = source_root / "project/run_scripts/barrier_usefulness"
    violations = forbidden_imports(package)
    if violations:
        raise TechnicalBoundary(f"forbidden scientific authority imports: {violations}")
    proposal = _identity(source_root / "project/proposals/barrier-usefulness-f2-2026-08-31.md")
    fresh_path = package / "config/fresh-case-manifest-v1.json"
    anchor_path = package / "config/anchor-pool-manifest-v1.json"
    fresh = json.loads(fresh_path.read_text())
    anchor = json.loads(anchor_path.read_text())
    if fresh["root_digest"] != "ef670ab365978a966ea7cd1eba533048f55335eaabe48e05686001112962aa82" or anchor["root_digest"] != "57b914a8b5ab68a39700e3dfc04d96895f660375b441c8a7430d5ea528031a5a":
        raise TechnicalBoundary("preregistered manifest root mismatch")
    if fresh["replacement_count"] != 0 or fresh["outcome_influence_count"] != 0 or fresh["final_audit_open_count"] != 0:
        raise TechnicalBoundary("fresh-case leakage boundary failed")
    reused = {relative: _identity(source_root / relative) for relative in REUSED_MEMBERS}
    return {
        "schema": "odeedit.s06.barrier-usefulness.g0-preflight.v1",
        "status": "G0_PASS",
        "source": {"head": _git(source_root, "rev-parse", "HEAD"), "tree": _git(source_root, "rev-parse", "HEAD^{tree}"), "clean": True},
        "contract": contract_id,
        "proposal": proposal,
        "fresh_manifest": {**_identity(fresh_path), "root": fresh["root_digest"], "case_ids": [row["case_id"] for row in fresh["fresh_cases"]]},
        "anchor_pool_manifest": {**_identity(anchor_path), "root": anchor["root_digest"], "ctrl_gate_overlap_count": sum(row["ctrl_gate_overlap_count"] for row in anchor["pools"])},
        "stock_easyedit": EXPECTED_EASYEDIT,
        "reused_fixed_z_infrastructure": reused,
        "fixed_z_prior_mutation_count": 0,
        "forbidden_import_count": 0,
        "q_gate_edited_access_count": 0,
        "phase_c_submit_count": 0,
        "full_fp32": True,
        "numerical_lock": BarrierLock().payload(),
        "receipt_identity": canonical_hash({"contract": contract_id["sha256"], "proposal": proposal["sha256"], "fresh": fresh["root_digest"], "anchor": anchor["root_digest"]}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_once(args.output, build(args.source_root.resolve(), args.contract.resolve()))


if __name__ == "__main__":
    main()
