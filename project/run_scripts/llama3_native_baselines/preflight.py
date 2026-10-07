"""Bounded CPU software checks; no toy science, model or GPU qualification."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import unittest

import torch

from .common import ROOT, TASK, authority, member, require, verify, write
from .generation import require_binding
from .native_binding import verify as native_verify
from .native import load_native
from .producer import transport_support

TESTS = ('test_native_binding','test_producer','test_observer','test_run',
         'test_submit','test_tracking','test_collect')


def check(config, output):
    c=json.loads(Path(config).read_text())
    authority();require(c['task_id']==TASK,'EXACT_TASK')
    require(not torch.cuda.is_initialized(),'CPU_ONLY_INITIAL_STATE')
    require_binding(c)
    support=transport_support()
    require(support['status']=='KEYS_PRESENT_REQUIRES_WORKER_AND_AXIS_CHECK','SHARED_KEYS')
    bindings=[]
    for method,binding in c['native'].items():
        for row in binding['files']+[binding['hparams']]:native_verify(row)
        package=load_native(binding)
        bindings.append(dict(method=method,source_files=len(binding['files']),
            layers=list(package.hp.layers),v_lr=package.hp.v_lr,
            v_num_grad_steps=package.hp.v_num_grad_steps, no_model_load=True))
    suite=unittest.defaultTestLoader.loadTestsFromNames(
        ['project.run_scripts.llama3_native_baselines.'+name for name in TESTS])
    result_stream=io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        result=unittest.TextTestRunner(stream=result_stream,verbosity=2).run(suite)
    require(result.wasSuccessful(),'CPU_SOFTWARE_REGRESSION_FAILED:'+result_stream.getvalue())
    require(not torch.cuda.is_initialized(),'CPU_ONLY_FINAL_STATE')
    paths=[]
    for namespace in ('llama3_native_baselines','experiment_generation_eval','experiment_tracking'):
        paths += sorted((ROOT/'project/run_scripts'/namespace).glob('*.py'))
    paths += [ROOT/row['relative'] for row in c['W0_evaluator']]
    proof=dict(status='CPU_SOURCE_READY',task_id=TASK,tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),source_members=[member(p) for p in paths],
        private_native_imports=bindings,generation=require_binding(c),tracking=support,
        owner_review='software/source owner check; focused separate source reviewer scope recorded separately',
        actual_GPU='NOT_OBSERVED',model_loads=0,scientific_fits=0,toy_experiments=0,
        Slurm_writes=0,online_delivery='NOT_OBSERVED; actual new runner startup/finish only',
        noCP=True,CPU_stdout_Git=False)
    write(output,proof)
    return proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=check(a.config,a.output)
    print(json.dumps({k:r[k] for k in ('status','tests','actual_GPU','model_loads','Slurm_writes')}))


if __name__=='__main__':main()
