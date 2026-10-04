"""Run the CPU contract suite and write reproducible, JSON-safe evidence."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import unittest

import numpy as np
import scipy

from witnesses import non_kkt_witness, sparse_first_step_witness


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checks = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.checks.append({"test": test.id(), "status": "PASS"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.checks.append({"test": test.id(), "status": "FAIL", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.checks.append({"test": test.id(), "status": "ERROR", "detail": self._exc_info_to_string(err, test)})

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.checks.append({"test": subtest.id(), "status": "FAIL", "detail": self._exc_info_to_string(err, test)})


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "audit.json")
    args = parser.parse_args()
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(root), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordingResult).run(suite)
    source_paths = sorted([*root.glob("*.py"), root / "README-ko.md", root / "requirements.txt"])
    report = {
        "schema_version": 1,
        "scope": "CPU float64 contract witnesses; no production model, GPU, experiment dispatch, or scientific performance claim.",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "reproduce": "python -B plans/global/2026-10-04-jlz-v12-marginal-allocation/reference/audit.py",
        "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "checks": sorted(result.checks, key=lambda x: x["test"]),
        "diagnostic_tolerances": {
            "reference_cpu_active_tol": 1e-12,
            "reference_cpu_boundary_tol": 1e-12,
            "method_caller_active_tol": 1e-10,
            "method_caller_boundary_tol": "1e-6 * max(1, radius)",
            "role": "Diagnostic classification only; no parameter pruning or scientific acceptance gate.",
        },
        "terminal_candidate_contract": {
            "tag": "NO_BACKWARD_TERMINAL", "kkt": None,
            "scope": "Specification only. This reference has no model forward/backward implementation and does not test production stopping behavior.",
        },
        "expected_limitations": [non_kkt_witness(), sparse_first_step_witness()],
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(log.getvalue(), end="")
    print(f"Audit: {args.output}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
