"""Run only lifelong and inherited observer/continuity focused tests."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
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
    import pytest

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        returncode = int(pytest.main(["-q", *TESTS]))
    stdout_value = stdout.getvalue()
    stderr_value = stderr.getvalue()
    match = re.search(r"(?:^|\s)(\d+) passed(?:\s|,)", stdout_value)
    passed = int(match.group(1)) if match else 0
    payload = {
        "schema": "odeedit.s06.layer-realization-debt.lifelong-focused-gate.v1",
        "status": "PASS" if returncode == 0 else "FAIL",
        "passed": passed,
        "failed": 0 if returncode == 0 else 1,
        "tests": list(TESTS),
        "pytest_python": str(args.pytest_python),
        "pytest_python_resolved": str(sys.executable),
        "stdout_tail": stdout_value.splitlines()[-12:],
        "stderr_tail": stderr_value.splitlines()[-12:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(raw)
    os.chmod(args.output, 0o600)
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
