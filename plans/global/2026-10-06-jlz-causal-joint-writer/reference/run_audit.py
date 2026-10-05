"""Run tiny CPU proofs without loading data, models, or CUDA tensors."""

import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest

import numpy as np

import test_causal_memit


def main():
    root = Path(__file__).resolve().parent
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_causal_memit)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = [root / "causal_memit.py", root / "test_causal_memit.py", root / "run_audit.py"]
    audit = {
        "schema": "JLZ_CAUSAL_MEMIT_CPU_MATH_REFERENCE_V1",
        "scope": "one MEMIT writer, actual candidate-weight forward, full causal key derivatives; synthetic FP64 only",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "seconds": time.perf_counter() - start,
        "python": platform.python_version(), "numpy": np.__version__, "seed": test_causal_memit.SEED,
        "measurements": test_causal_memit.MEASUREMENTS,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limits": [
            "No model/benchmark data, production code, or experiment changed.",
            "Ideal symmetric SPD A and FP64 weights; production cast/add and asymmetric-input contracts need separate validation.",
            "No production optimizer, memory/time qualification, stopping policy, semantic accuracy, or convergence certified.",
            "Microbatch loss adjoints preserve full logical-batch geometry; logical-batch partition invariance is not claimed.",
            "Calibration toy value is not a proposed experimental coefficient.",
            "Exact mean interpolation appears only as a mathematical support witness, not an experimental arm.",
        ],
    }
    output = root / "audit.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": str(output), "status": audit["status"], "tests": result.testsRun}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
