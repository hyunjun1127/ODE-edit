"""Bounded actual production CPU functions; no model/GPU/scientific fit."""
import argparse
import contextlib
import io
import json
import time
import unittest
from .common import *

TESTS=(
 'project.run_scripts.experiment_generation_eval.test_generator',
 'project.run_scripts.experiment_generation_eval.test_kv_generator',
 'project.run_scripts.experiment_generation_eval.test_compatibility',
 'project.run_scripts.experiment_generation_eval.test_progress',
 'project.run_scripts.experiment_generation_eval.test_metrics',
 'project.run_scripts.experiment_generation_eval.test_observer',
 'project.run_scripts.experiment_generation_eval.test_assets',
 'project.run_scripts.gpt2xl_generation_baselines.test_run',
 'project.run_scripts.gpt2xl_generation_baselines.test_cache_repair',
 'project.run_scripts.gpt2xl_generation_baselines.test_registration',
 'project.run_scripts.gpt2xl_generation_baselines.test_collect',
 'project.run_scripts.gpt2xl_generation_baselines.test_submit',
 'project.run_scripts.gpt2xl_native_baselines.test_native',
 'project.run_scripts.gpt2xl_cake_blue.test_native',
 'project.run_scripts.gpt2xl_prune_rect.test_native',
 'project.run_scripts.gpt2xl_prune_rect.test_terminal',
 'project.run_scripts.experiment_tracking.test_tracking',
 'project.run_scripts.experiment_tracking.test_job_identity',
 'project.run_scripts.experiment_tracking.test_method',
 'project.run_scripts.experiment_tracking.test_generation')

def run(config,out=None):
    c=read(config);out=config.parent if out is None else out;authority()
    require(not (out/'cpu-preflight.json').exists(),'CREATE_ONCE_CPU_RECEIPT')
    log=io.StringIO();started=time.monotonic()
    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(TESTS))
    import torch
    require(not torch.cuda.is_initialized(),'CPU_NO_GPU_INITIALIZATION')
    paths=('gpt2xl_generation_baselines','experiment_generation_eval','experiment_tracking')
    sources=[member(p) for d in paths for p in sorted((ROOT/'project/run_scripts'/d).glob('*.py'))]
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        seconds=time.monotonic()-started,GPU_model_validation='NOT_RUN',online_validation='NOT_RUN',
        native_apply_actual=0,source=sources,helper_source=[],tests_list=list(TESTS),
        independent_reviews=c.get('independent_reviews',[]),review='Owner actual CPU fixtures; independent review scope in separate receipt')
    write(out/'cpu-test-local.json',dict(output=log.getvalue(),**receipt));write(out/'cpu-preflight.json',receipt)
    require(result.wasSuccessful(),'CPU_REGRESSION_FAILURE')
    c['cpu_preflight']=member(out/'cpu-preflight.json');write(out/'config-sealed.json',c)
    print(json.dumps(dict(status=receipt['status'],tests=receipt['tests'],skipped=receipt['skipped'])))
    return out/'config-sealed.json'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();run(a.config.resolve(),a.out.resolve() if a.out else None)
