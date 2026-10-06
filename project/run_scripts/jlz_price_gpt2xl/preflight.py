"""Bounded CPU source/parser/caller audit; actual B1 remains unobserved."""
import argparse,ast,importlib,json,os,socket,subprocess,sys,unittest
from .common import *

def check(out):
    require(socket.gethostname()=='devbox','HOST_BOUNDARY')
    require(os.environ.get('CODEX_THREAD_ID')=='01a04939-f93a-7b50-bca0-65438eab2062','SESSION_BOUNDARY')
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==
        'codex/server1-jlz-price-gpt2xl-2k','OWN_BRANCH')
    require('hyunjun1127/ODE-edit' in subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True),'ORIGIN')
    paths=sorted((ROOT/'project/run_scripts/jlz_price_gpt2xl').glob('*.py'))
    for p in paths:
        ast.parse(p.read_text());importlib.import_module('project.run_scripts.jlz_price_gpt2xl.'+p.stem)
    from .prepare import authority
    authority()
    modules=['project.run_scripts.jlz_price_gpt2xl.test_contract',
        'project.run_scripts.experiment_tracking.test_method',
        'project.run_scripts.experiment_tracking.test_job_identity',
        'project.run_scripts.experiment_tracking.test_tracking']
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m) for m in modules)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    require(result.wasSuccessful(),'CPU_REGRESSION')
    from .tracking import contract_ready
    contract_ready()
    from project.run_scripts.experiment_tracking.schema import load_env
    load_env('/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env')
    import torch
    require(not torch.cuda.is_initialized(),'CPU_ONLY')
    helper=[member(p) for p in sorted((ROOT/'project/run_scripts/experiment_tracking').glob('*.py'))]
    value=dict(task=TASK,nonce=NONCE,passed=True,status='CPU_SOURCE_CONTRACT_CHECKED',source=[member(p) for p in paths],
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        source_review_level='OWNER_CPU_AUDIT; independent bounded static reviewers recorded separately, not target-model PASS',
        numeric_tests='CPU pure operator fixtures only; zero target-model qualification',toy_runs=0,
        model_load=False,actual_B1='NOT_OBSERVED',tracking_ready=True,helper_ready=True,
        integration='FAKE_SDK_PAYLOAD_AXES_IDENTITY_PASS',helper_sources=helper,
        tracking_contracts=[member(ROOT/p) for p in ('control/wandb-policy.json',
            'control/wandb-method-metric-schema.json','messages/head/2026-10-07-wandb-method-metrics-all-sh.json')],
        online_scientific_validation='NOT_OBSERVED',old_jobs_unchanged=True)
    write(out,value);return dict(status=value['status'],tests=result.testsRun,actual_B1='NOT_OBSERVED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(check(a.out)))
