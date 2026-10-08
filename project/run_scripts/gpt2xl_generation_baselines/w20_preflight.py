"""Bounded W20-only caller/collector/identity regressions; no model/GPU run."""
import argparse
import contextlib
import io
import json
import time
import unittest
from .w20_common import *

TESTS=(
    'project.run_scripts.gpt2xl_generation_baselines.test_w20_control',
    'project.run_scripts.gpt2xl_generation_baselines.test_w20_run',
    'project.run_scripts.gpt2xl_generation_baselines.test_w20_collect',
    'project.run_scripts.experiment_generation_eval.test_observer',
    'project.run_scripts.experiment_generation_eval.test_generator',
    'project.run_scripts.experiment_generation_eval.test_kv_generator',
    'project.run_scripts.experiment_generation_eval.test_metrics',
    'project.run_scripts.experiment_tracking.test_w20_generation',
    'project.run_scripts.experiment_tracking.test_generation',
    'project.run_scripts.experiment_tracking.test_job_identity',
    'project.run_scripts.gpt2xl_native_baselines.test_native',
    'project.run_scripts.gpt2xl_cake_blue.test_native',
    'project.run_scripts.gpt2xl_prune_rect.test_native',
    'project.run_scripts.gpt2xl_prune_rect.test_terminal')

def run(config,out=None):
    value=read(config);validate_config(value);authority();out=config.parent if out is None else out
    require(not (out/'cpu-preflight.json').exists(),'W20_CREATE_ONCE_CPU_RECEIPT')
    log=io.StringIO();started=time.monotonic()
    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(TESTS))
    import torch
    require(not torch.cuda.is_initialized(),'W20_CPU_NO_GPU_INITIALIZATION')
    directories=('gpt2xl_generation_baselines','experiment_generation_eval','experiment_tracking')
    source=[member(path) for directory in directories
        for path in sorted((ROOT/'project/run_scripts'/directory).glob('*.py'))]
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',instruction_id=NONCE,task_id=TASK,
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        seconds=time.monotonic()-started,source=source,helper_source=[],tests_list=list(TESTS),
        GPU_model_validation='NOT_RUN',online_validation='NOT_RUN',native_apply_actual=0,
        qualification_actual='NOT_RUN; prelocked PLAN only',generation_schedule=SCHEDULE,
        review='Owner CPU fixtures; independent review recorded separately',
        independent_reviews=value.get('independent_reviews',[]))
    write(out/'cpu-test-local.json',dict(output=log.getvalue(),**receipt));write(out/'cpu-preflight.json',receipt)
    require(result.wasSuccessful(),'W20_CPU_REGRESSION_FAILURE')
    value['cpu_preflight']=member(out/'cpu-preflight.json');write(out/'config-sealed.json',value)
    print(json.dumps(dict(status=receipt['status'],tests=receipt['tests'],skipped=receipt['skipped'])))
    return out/'config-sealed.json'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path);args=parser.parse_args()
    run(args.config.resolve(),args.out.resolve() if args.out else None)
