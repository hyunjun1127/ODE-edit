"""Bounded CPU/fake-SDK gate and immutable prepared-config extension."""
import argparse
import contextlib
import io
import json
import sys
import time
import unittest
from .common import *

TESTS=('project.run_scripts.gpt2xl_native_baselines.test_native',
       'project.run_scripts.gpt2xl_native_baselines.test_metrics',
       'project.run_scripts.gpt2xl_native_baselines.test_run_submit',
       'project.run_scripts.experiment_tracking.test_tracking',
       'project.run_scripts.experiment_tracking.test_job_identity',
       'project.run_scripts.experiment_tracking.test_method')

def run(config,out=None):
    c=json.loads(config.read_text());out=config.parent if out is None else out
    require(not (out/'cpu-preflight.json').exists(),'CREATE_ONCE_CPU_RECEIPT');out.mkdir(parents=True,exist_ok=True)
    authority();log=io.StringIO();started=time.monotonic()
    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(TESTS))
    import torch
    require(not torch.cuda.is_initialized(),'CPU_NO_GPU_INITIALIZATION')
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-started,
        GPU_model_validation='NOT_RUN',native_apply_actual=0,online_validation='NOT_RUN',
        source=[member(p) for p in sorted((ROOT/'project/run_scripts/gpt2xl_native_baselines').glob('*.py'))],
        helper_source=[member(p) for p in sorted((ROOT/'project/run_scripts/experiment_tracking').glob('*.py'))],
        tests_list=list(TESTS),담당='SH1 owner CPU gate + native/metrics 독립 구현자 fixtures',checkpoint_saved=False)
    write(out/'cpu-test-local.json',dict(output=log.getvalue(),**receipt))
    write(out/'cpu-preflight.json',receipt)
    require(result.wasSuccessful(),'CPU_REGRESSION_FAILURE')
    c['cpu_preflight']=member(out/'cpu-preflight.json')
    write(out/'config-sealed.json',c)
    print(json.dumps(dict(status=receipt['status'],tests=receipt['tests'],config=str(out/'config-sealed.json')),ensure_ascii=False))
    return out/'config-sealed.json'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args();run(a.config.resolve(),a.out.resolve() if a.out else None)
