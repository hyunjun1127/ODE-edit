"""Run only lifelong and inherited observer/continuity focused tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


TESTS = (
    "project/run_scripts/official_layer_realization_debt/tests/test_observer.py",
    "project/run_scripts/official_layer_realization_debt/tests/test_sequential.py",
    "project/run_scripts/official_layer_realization_debt/tests/test_lifelong.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pytest-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite lifelong focused gate")
    python = args.pytest_python.resolve(strict=True)
    completed = subprocess.run(
        [str(python), "-m", "pytest", "-q", *TESTS],
        text=True,
        capture_output=True,
        check=False,
    )
    passed = 0
    for token in completed.stdout.replace("\n", " ").split():
        if token.isdigit():
            passed = int(token)
    payload = {
        "schema": "odeedit.s06.layer-realization-debt.lifelong-focused-gate.v1",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "passed": passed,
        "failed": 0 if completed.returncode == 0 else 1,
        "tests": list(TESTS),
        "pytest_python": str(args.pytest_python),
        "pytest_python_resolved": str(python),
        "stdout_tail": completed.stdout.splitlines()[-12:],
        "stderr_tail": completed.stderr.splitlines()[-12:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(raw)
    os.chmod(args.output, 0o600)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
