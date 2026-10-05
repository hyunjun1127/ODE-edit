"""Run synthetic CPU checks and write a compact, data-free audit JSON."""

import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest

import numpy as np

import test_reference


def main():
    root = Path(__file__).resolve().parent
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_reference)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    paths = [root / "geometry.py", root / "test_reference.py", root / "run_audit.py"]
    payload = {
        "schema": "JLZ_CD_CUMULATIVE_ALLOCATION_CPU_REFERENCE_V1",
        "scope": "synthetic FP64 algebra and accumulation contracts only; no model, dataset, GPU, or production qualification",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dtype": "float64",
        "seed": test_reference.SEED,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "elapsed_seconds": time.perf_counter() - started,
        "measurements": test_reference.MEASUREMENTS,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "limits": [
            "No production tolerances or allocation-price calibration prescribed.",
            "No semantic locality, accuracy, planner convergence, or fresh-key commit parity certified.",
            "Fixed entry geometry; actual sequential upper-layer key drift remains outside these identities.",
            "Logical batch size changes geometry; microbatch accumulation equality is not logical-batch invariance.",
            "B1 arm identity requires zero cumulative entry displacement and otherwise identical inputs.",
        ],
    }
    output = root / "audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": str(output), "status": payload["status"], "tests_run": result.testsRun}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
