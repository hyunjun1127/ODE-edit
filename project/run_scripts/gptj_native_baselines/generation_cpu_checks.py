"""Bounded task-private CPU fixtures; never asserts actual GPU/source READY."""
import json
import os
import argparse
import re
import resource
import time
import unittest
from pathlib import Path
import torch
from .generation_common import LOCAL, ROOT, require, small_member, write

MODULES=(
 'project.run_scripts.gptj_cake_blue_prune_rect.test_bias_guard',
 'project.run_scripts.gptj_native_baselines.test_generation_run',
 'project.run_scripts.gptj_native_baselines.test_generation_plan',
 'project.run_scripts.gptj_native_baselines.test_generation_tracking',
)
def checks(receipt_name='cpu-checks-preparation-r1.json'):
    require(re.fullmatch(r'cpu-checks-preparation-r[1-9][0-9]*\.json',receipt_name), 'EXACT_CPU_RECEIPT_NAME')
    require(not (LOCAL/receipt_name).exists(), 'PRESERVE_PRIOR_CPU_RECEIPT')
    torch.set_num_threads(1)
    started=time.monotonic()
    suite=unittest.TestLoader().loadTestsFromNames(MODULES)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    source=[ROOT/'project/run_scripts/gptj_cake_blue_prune_rect/native_cake_blue.py',
            ROOT/'project/run_scripts/gptj_cake_blue_prune_rect/test_bias_guard.py']
    source+=sorted((ROOT/'project/run_scripts/gptj_native_baselines').glob('generation_*.py'))
    source+=sorted((ROOT/'project/run_scripts/gptj_native_baselines').glob('test_generation_*.py'))
    receipt=dict(status='PASS_CPU_FIXTURES' if result.wasSuccessful() else 'FAILED_CPU_FIXTURES',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        modules=MODULES,seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        CPU_threads=torch.get_num_threads(),CUDA_initialized=torch.cuda.is_initialized(),
        actual_model_loaded=False,actual_native_fit=False,actual_generation=False,
        actual_network_or_online_validation=False,new_Slurm_jobs=0,
        source=[small_member(p) for p in source],
        execution_source_frozen=False,SH1_source_API_reference_bound=False,
        fixture_PASS_is_not_GPU_PASS=True)
    write(LOCAL/receipt_name,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='source'}))
    if not result.wasSuccessful() or torch.cuda.is_initialized():raise SystemExit(1)
    return receipt
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt-name',default='cpu-checks-preparation-r1.json')
    checks(parser.parse_args().receipt_name)
