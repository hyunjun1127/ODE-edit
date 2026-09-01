"""Run only the compact observer/recurrence/interface test suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite focused gate receipt")
    test = "project/run_scripts/official_layer_realization_debt/tests/test_observer.py"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        test,
    ]
    completed = subprocess.run(command, cwd=args.source_root, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    last = next((line for line in reversed(output.splitlines()) if "passed" in line or "failed" in line), "")
    passed = int(last.split()[0]) if completed.returncode == 0 and last.split() and last.split()[0].isdigit() else 0
    payload = {
        "schema": "odeedit.s06.official-layer-realization-debt.focused-gate.v1",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "passed": passed,
        "failed": 0 if completed.returncode == 0 else 1,
        "command": command,
        "summary": last,
        "broad_unrelated_test_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(args.output, 0o600)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
