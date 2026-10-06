"""Owner-only telemetry/horizon checks; no repeated CPU/GPU math campaign."""
import ast
import io
import json
import subprocess
import time
import unittest
from . import *

def main():
    started=time.monotonic();prior=LOCAL/'cpu-ready/receipt.json'
    old=json.loads(prior.read_text());require(old['passed'] and old['tests']==7,'PRIOR_CPU7')
    # Callback addition is explicitly excluded; original adapter equations exact.
    base=subprocess.check_output(['git','show','d1e197f8:project/run_scripts/fe_baseline/adapter.py'],cwd=ROOT,text=True)
    actual=(ROOT/'project/run_scripts/fe_baseline/adapter.py').read_text()
    stripped=actual.replace('from .telemetry import fit as log_fit\n','').replace("            log_fit(getattr(self,'tracking',None),index,vals)\n",'')
    require(stripped==base,'MATH_SOURCE_CHANGED')
    for p in (ROOT/'project/run_scripts/fe_baseline').glob('*.py'):ast.parse(p.read_text())
    buffer=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('project.run_scripts.fe_baseline.test_resume')
    result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    out=LOCAL/'cpu-login-10k-r1';out.mkdir(exist_ok=False);(out/'tests.txt').write_text(buffer.getvalue())
    receipt=dict(passed=result.wasSuccessful(),tests=result.testsRun,prior_CPU7=member(prior),math_source_exact_except_scalar_callback=True,
        source=[member(p) for p in sorted((ROOT/'project/run_scripts/fe_baseline').glob('*.py'))],
        seconds=time.monotonic()-started,owner_review=True,independent_reviewer=0,actual_model='NOT_RUN',GPU=0,
        scope='10k horizon, numeric batch ordering, scalar whitelist, startup-before-model, resources/noCP')
    write(out/'receipt.json',receipt);print(buffer.getvalue());require(receipt['passed'],'CPU_RESUME_FAILURE')

if __name__=='__main__':main()
