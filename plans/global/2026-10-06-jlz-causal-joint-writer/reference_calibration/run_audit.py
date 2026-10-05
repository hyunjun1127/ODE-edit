import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest

import numpy as np

import test_math_contract


def main():
    root = Path(__file__).resolve().parent
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_math_contract))
    paths = [root / name for name in ("math_contract.py", "test_math_contract.py", "run_audit.py")]
    report = {
        "schema": "JLZ_RADIAL_CALIBRATION_MATH_REFERENCE_V1",
        "scope": "synthetic algebra only; no model, data, fitting, or experimental coefficient selection",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "seconds": time.perf_counter() - started,
        "python": platform.python_version(), "numpy": np.__version__, "dtype": "float64",
        "measurements": test_math_contract.MEASUREMENTS,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "limits": [
            "Radial balance does not imply clamp equivalence, full KKT, or per-request balance.",
            "Fixed-key d=2Q does not describe arbitrary all-layer causal radial paths.",
            "A native subject-injection reference need not have a positive price under the actual candidate objective.",
            "Negative, zero, and degenerate prices are reported rather than clipped or replaced.",
            "Proximal tests do not qualify a production optimizer or backtracking implementation.",
            "No PS, NS, ACC, benchmark, or production data enters calibration or tests.",
        ],
    }
    output = root / "audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": str(output), "status": report["status"], "tests": result.testsRun}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
