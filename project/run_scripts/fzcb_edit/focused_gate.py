"""Run and seal only the TECH-R1 focused CPU/source gates."""

from __future__ import annotations

import argparse
import io
import py_compile
import subprocess
import unittest
from pathlib import Path
from typing import Any

from .hashing import canonical_hash, file_sha256, write_json_once


TEST_MODULES = (
    "project.run_scripts.fzcb_edit.tests.test_core",
    "project.run_scripts.fzcb_edit.tests.test_tech_r1",
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _source_members(root: Path) -> list[dict[str, Any]]:
    members = []
    package = root / "project/run_scripts/fzcb_edit"
    paths = sorted(path for path in package.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    paths.append(root / "project/run_scripts/session06_fzcb_tech_r1_joint_b1_server4.sbatch")
    paths.append(root / "plans/updates/server4/fzcb-tech-r1-joint-b1-controller-validity-delta.md")
    for path in paths:
        members.append({
            "path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        })
    return members


def build(source_root: Path, expected_head: str) -> dict[str, Any]:
    source_root = source_root.resolve()
    head = _git(source_root, "rev-parse", "HEAD")
    if head != expected_head or _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("focused gate requires exact clean committed source")
    members = _source_members(source_root)
    for member in members:
        if member["path"].endswith(".py"):
            py_compile.compile(str(source_root / member["path"]), doraise=True)
    subprocess.check_call([
        "bash", "-n", str(source_root / "project/run_scripts/session06_fzcb_tech_r1_joint_b1_server4.sbatch")
    ])
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    payload = {
        "schema": "odeedit.s06.fzcb-tech-r1.focused-gate.v1",
        "status": "FOCUSED_GATE_PASS" if result.wasSuccessful() else "FOCUSED_GATE_FAIL",
        "source": {
            "head": head,
            "tree": _git(source_root, "rev-parse", "HEAD^{tree}"),
            "tracked_clean": True,
        },
        "tests": {
            "modules": list(TEST_MODULES),
            "run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "output": stream.getvalue(),
        },
        "compile_member_count": sum(member["path"].endswith(".py") for member in members),
        "bash_syntax": "PASS",
        "members": members,
        "member_root": canonical_hash(members),
        "broad_unrelated_test_count": 0,
    }
    if not result.wasSuccessful():
        raise RuntimeError(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_once(args.output, build(args.source_root, args.expected_head))


if __name__ == "__main__":
    main()
