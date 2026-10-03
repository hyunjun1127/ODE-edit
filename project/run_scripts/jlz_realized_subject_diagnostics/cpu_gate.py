"""Run actual production-linked CPU suites and preserve compact provenance."""
import argparse
import contextlib
import io
import json
import time
import unittest
from pathlib import Path
from .common import write,member,ROOT,NONCE


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    names=['project.run_scripts.jlz_realized_subject.test_core',
        'project.run_scripts.jlz_realized_subject_diagnostics.test_diagnostics',
        'project.run_scripts.jlz_realized_subject_diagnostics.test_workflow']
    suite=unittest.TestLoader().loadTestsFromNames(names);log=io.StringIO();start=time.monotonic()
    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'cpu-full-local.log').open('x') as f:f.write(log.getvalue())
    files=list((ROOT/'project/run_scripts/jlz_realized_subject_diagnostics').glob('*.py'))
    files+=[ROOT/'project/run_scripts/jlz_realized_subject/optimize.py']
    receipt=dict(instruction=NONCE,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),passed=result.wasSuccessful(),seconds=time.monotonic()-start,suites=names,
        source=[member(x) for x in sorted(files)],actual_pretrained_GPU='NOT_RUN',independent_reviewer_agent=False)
    write(a.out/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='source'}))
    if not result.wasSuccessful():print(log.getvalue());raise SystemExit(1)


if __name__=='__main__':main()
