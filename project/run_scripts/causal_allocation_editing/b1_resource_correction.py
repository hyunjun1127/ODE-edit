"""Create-once six-CPU binding; reuse CPU68, inputs and original rejection."""
import ast
import copy
import json
import subprocess
import unittest
from pathlib import Path
from . import ROOT, member, require, sha, verify, write
from .b1_prepare import LOCAL
from . import b1_submit
from .test_b1 import B1Tests

def main():
    out=LOCAL/'preparation-cpu6'
    require(not out.exists(),'CREATE_ONCE_RESOURCE_CORRECTION')
    old=LOCAL/'preparation/configuration.json'
    c=json.loads(old.read_text());before=copy.deepcopy(c)
    authority=ROOT/'audits/global/causal-allocation-editing-b1-dispatch/resource-correction.json'
    correction=json.loads(authority.read_text())
    require(correction['effective_GPU_job_CPUs']==6 and correction['science_changed'] is False,'RESOURCE_AUTHORITY')
    prior=json.loads(verify(c['cpu_preflight']).read_text())
    require(prior['passed'] and prior['tests']==68,'PRIOR_CPU68')
    changed={'b1_submit.py','run.py','test_b1.py'}
    for row in prior['source']:
        if Path(row['path']).name not in changed:verify(row)
    for name in changed:
        ast.parse((ROOT/'project/run_scripts/causal_allocation_editing'/name).read_text())
    suite=unittest.TestSuite(B1Tests(n) for n in (
        'test_single_runner_qualification_and_main_not_skipped','test_resources_exact_name_cap_one'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(),'RESOURCE_ROUTING_TESTS')
    c['resources'].update(cpu=6,collector_cpu=6)
    c['attempt']=str(LOCAL/'attempt-cpu6');c['run_instance']['attempt']='attempt-cpu6'
    for row in c['authority_members']:
        if row['path']==str(ROOT/'messages/head/causal-allocation-editing-b1.json'):
            row.update(member(Path(row['path'])))
        else:verify(row)
    c['authority_members'].append(member(authority))
    for role in b1_submit.ROLES:
        script=b1_submit.launcher(Path('/source'),'commit',role,Path(c['attempt']),c)
        require('OMP_NUM_THREADS=6' in script and 'MKL_NUM_THREADS=6' in script,'SIX_THREADS')
        subprocess.run(['bash','-n'],input=script,text=True,check=True)
    run=(ROOT/'project/run_scripts/causal_allocation_editing/run.py').read_text()
    require("torch.set_num_threads(c['resources'].get('cpu',8))" in run,'TORCH_THREADS_DEFAULT2K_RETAINED')
    require(all(c[k]==before[k] for k in before if k not in ('resources','attempt','run_instance','authority_members','cpu_preflight')),'RESOURCE_ONLY_CONFIG')
    out.mkdir()
    receipt=dict(passed=True,status='RESOURCE_ROUTING_PASS_PRIOR_CPU68_REUSED',tests=2,
        prior_CPU68=member(LOCAL/'preparation/cpu-ready.json'),full_suite_rerun=False,
        authority=member(authority),actual_GPU='NOT_RUN',math_changed=False,
        prior_rejection=member(LOCAL/'attempt/submission-rejection.json'),
        source=[member(p) for p in sorted((ROOT/'project/run_scripts/causal_allocation_editing').glob('*.py'))])
    write(out/'cpu-resource.json',receipt);c['cpu_preflight']=member(out/'cpu-resource.json')
    write(out/'configuration.json',c)
    print(json.dumps(dict(config=str(out/'configuration.json'),tests=2,prior_tests_reused=68)))

if __name__=='__main__':main()
