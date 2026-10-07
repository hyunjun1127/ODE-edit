"""Final narrow caller integration fixtures; preserve prior CPU/native evidence."""
import argparse
import ast
import io
import json
import resource
import time
import unittest
from contextlib import redirect_stdout, redirect_stderr
import torch
from .generation_common import *
from .generation_plan import ready
MODULES=(
 'project.run_scripts.gptj_native_baselines.test_generation_bridge',
 'project.run_scripts.gptj_native_baselines.test_generation_tracking_shared',
 'project.run_scripts.gptj_native_baselines.test_generation_run',
 'project.run_scripts.gptj_native_baselines.test_generation_submit',
 'project.run_scripts.gptj_native_baselines.test_generation_collect',
)
def main():
    target=LOCAL/'cpu-integration-r1.json'
    require(not target.exists(),'PRESERVE_INTEGRATION_RECEIPT')
    c=read(LOCAL/'preparation-r2/config.json');ready(c)
    previous=read(LOCAL/'cpu-checks-preparation-r2.json')
    unchanged=('native_cake_blue.py','test_bias_guard.py','generation_common.py',
               'generation_plan.py','test_generation_plan.py')
    reused=[r for r in previous['source'] if Path(r['path']).name in unchanged]
    require(len(reused)==5,'REUSED_FOCUSED_CPU_SOURCE')
    for row in reused:verify(row)
    torch.set_num_threads(1);start=time.monotonic()
    suite=unittest.TestLoader().loadTestsFromNames(MODULES)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sources=sorted((ROOT/'project/run_scripts/gptj_native_baselines').glob('generation_*.py'))
    sources+=sorted((ROOT/'project/run_scripts/gptj_native_baselines').glob('test_generation_*.py'))
    sources+=[ROOT/'project/run_scripts/gptj_cake_blue_prune_rect/native_cake_blue.py']
    for p in sources:ast.parse(p.read_text(),filename=str(p))
    receipt=dict(status='PASS_CPU_INTEGRATION' if result.wasSuccessful() else 'FAILED_CPU_INTEGRATION',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),modules=MODULES,
        seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        CPU_threads=torch.get_num_threads(),CUDA_initialized=torch.cuda.is_initialized(),
        model_loaded=False,actual_generation=False,actual_native_fit=False,new_Slurm=0,
        actual_online_validation=False,source=[small_member(p) for p in sources],
        config=small_member(LOCAL/'preparation-r2/config.json'),
        reference_binding=small_member(LOCAL/'preparation-r2/binding.json'),
        reused_unchanged_prior_CPU=small_member(LOCAL/'cpu-checks-preparation-r2.json'),
        reused_source=reused,shared_generation_source=c['generation']['source_sha'],
        independent_reducer='Separate worker-owned CPU reducer; no actual GPU red PASS',
        source_not_frozen_until_committed=True)
    write(target,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source','reused_source')}))
    require(result.wasSuccessful() and not torch.cuda.is_initialized(),'CPU_INTEGRATION_FAILED')
if __name__=='__main__':main()

