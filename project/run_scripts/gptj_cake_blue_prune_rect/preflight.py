"""Narrow CPU tests and source/config binding; no model or GPU qualification."""
import io
import argparse
import time
import unittest
import torch
from .common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--metadata-r2',action='store_true');args=parser.parse_args()
    authority();out=LOCAL/('cpu-tests-r2.json' if args.metadata_r2 else 'cpu-tests-r1.json');require(not out.exists(),'CREATE_ONCE_CPU_RECEIPT')
    config=LOCAL/'preparation-r1/config.json';require(config.is_file(),'PREPARED_INPUTS')
    names=['test_runner'] if args.metadata_r2 else ['test_cake_blue','test_prune_rect','test_prepare','test_runner']
    prior=None
    if args.metadata_r2:
        prior=read(LOCAL/'cpu-tests-r1.json');require(prior['passed'],'PRIOR_CPU72')
        for name in ('native_cake_blue.py','native_prune_rect.py','prepare.py','common.py','metrics.py','collect.py',
            'test_cake_blue.py','test_prune_rect.py','test_prepare.py'):
            path='project/run_scripts/gptj_cake_blue_prune_rect/'+name
            require(sha(ROOT/path)==prior['source_sha256'][path],'REUSED_UNCHANGED_CPU_NATIVE_INPUT:'+name)
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        'project.run_scripts.gptj_cake_blue_prune_rect.'+name for name in names])
    log=io.StringIO();started=time.monotonic()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    elapsed=time.monotonic()-started
    require(not torch.cuda.is_initialized(),'CPU_ONLY_NO_CUDA_CONTEXT')
    source={str(path.relative_to(ROOT)):sha(path) for path in sorted((ROOT/'project/run_scripts/gptj_cake_blue_prune_rect').glob('*.py'))}
    log_path=out.with_suffix('.txt')
    with log_path.open('x') as stream:stream.write(log.getvalue())
    receipt=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),seconds=elapsed,source_sha256=source,config=member(config),
        no_model=True,GPU=0,new_fits=0,new_Slurm=0,actual_GPU_PASS=False,
        review='OWNER_SCOPED_CHECKS_AND_BOUNDED_WORKER_TESTS; no independent red certification',
        full_test_log=member(log_path),prior_unchanged_native_input_tests=member(LOCAL/'cpu-tests-r1.json') if prior else None)
    write(out,receipt)
    print(dict(status='CPU_PASS' if receipt['passed'] else 'CPU_FAILED',tests=receipt['tests'],seconds=elapsed,receipt=str(out)))
    if not receipt['passed']:raise SystemExit(1)

if __name__=='__main__':main()
