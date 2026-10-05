"""CPU-only production regression receipt; never imports a model or scheduler."""
import argparse
import contextlib
import io
import json
import os
import time
import unittest
from pathlib import Path
from . import ROOT, NONCE, TASK, DESIGN, member, write, require


def run(out):
    require(not Path(out).exists(), 'CREATE_ONCE_CPU_PREFLIGHT')
    folder = ROOT / 'project/run_scripts/causal_allocation_editing'
    files = sorted(folder.glob('*.py'))
    # Syntax validation does not generate bytecode or rewrite source.
    for path in files:
        compile(path.read_text(), str(path), 'exec')
    start = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(folder), 'test_*.py', str(ROOT))
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
    record = dict(instruction_id=NONCE, task_id=TASK, status='PASS' if result.wasSuccessful() else 'FAIL',
        passed=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), skipped=len(result.skipped), seconds=time.monotonic()-start,
        source=[member(p) for p in files], model_load=0, model_forward=0,
        new_GPU=0, Slurm_write=0, actual_model_qualified=False,
        evidence_scope='새 production CPU 회귀/fixture만; actual 모델 qualification은 sealed GPU job의 별도 READY',
        mathematical_reuse=[member(ROOT / DESIGN / p) for p in
            ('reference/audit.json', 'reference_calibration/audit.json')],
        owner_tests_not_independent_red=True)
    write(out, record)
    # Full CPU diagnostics remain ignored-local, not broadcast into Git.
    Path(str(out)+'.txt').write_text(capture.getvalue())
    require(result.wasSuccessful(), 'PRODUCTION_CPU_PREFLIGHT_FAILED')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out)))
