"""Run only observer and cumulative-continuity focused tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite focused gate receipt")
    tests = [
        "project/run_scripts/official_layer_realization_debt/tests/test_observer.py",
        "project/run_scripts/official_layer_realization_debt/tests/test_sequential.py",
    ]
    command = ["python", "-m", "pytest", "-q", *tests]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    passed = 0
    for token in completed.stdout.replace("\n", " ").split():
        if token.isdigit():
            passed = int(token)
    payload = {
        "schema": "odeedit.s06.official-layer-realization-debt.sequential-focused-gate.v1",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "passed": passed,
        "failed": 0 if completed.returncode == 0 else 1,
        "tests": tests,
        "stdout_tail": completed.stdout.splitlines()[-8:],
        "stderr_tail": completed.stderr.splitlines()[-8:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(raw)
    os.chmod(args.output, 0o600)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
