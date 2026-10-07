"""Narrow import/counter/asset/config tests, not actual GPU qualification."""
import json
import sys
import unittest
from .common import *
from . import test_native
from . import test_bridge

def main():
    authority();out=LOCAL/'preparation-r1';c=json.loads((out/'config.json').read_text())
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_native)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_bridge))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(),'NATIVE_CPU_TESTS')
    import torch
    from .run import NativeTransaction
    from .collect import endpoint
    from .metrics import ObservationView
    require(not torch.cuda.is_initialized(),'CPU_ONLY')
    # W0 input receipt is reused, never results monitored while preparing.
    from scripts.fixed_counterfact import load_prefix
    records=load_prefix(Path(c['stream']).parent,2000)
    c['packs']=[dict(ids=[r['case_id'] for r in current]) for _,current,_ in batches(records)]
    previous=json.loads(Path('/mnt/raid5/janghj/ODE-edit/local/base-model-gptj-w0-cohort-curves/preparation-r1/config.json').read_text())
    c['assets'].append(previous['input_source'])
    write(out/'config-submission.json',c)
    files=sorted((ROOT/'project/run_scripts/gptj_native_baselines').glob('*.py'))
    for p in files:compile(p.read_text(),str(p),'exec')
    write(LOCAL/'cpu-tests-r2.json',dict(passed=True,tests=result.testsRun,actual_GPU=False,
        owner_review=True,independent_reviewer=False,native_math_changes=0,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},config=member(out/'config-submission.json')))

if __name__=='__main__':main()
