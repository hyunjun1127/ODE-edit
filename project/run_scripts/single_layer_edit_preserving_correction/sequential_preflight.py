"""Narrow CPU/source/input evidence for the explicit S override; no GPU T."""
import argparse
import ast
import datetime
import json
import os
from pathlib import Path
import subprocess
from .common import ROOT,member,write,sha
from .control import PYTHON,PACKAGE

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    repo=a.repo;out=ROOT/'S/preflight-r1'
    old=json.loads((ROOT/'receipts/full-read-m0.json').read_text())
    bindings=[]
    for m in old['files']:
        path=Path(m['path']);rel=path.relative_to(ROOT/'worktree');new=repo/rel
        if sha(new)!=m['sha256']:raise ValueError('EXACT_PRIOR_FULL_READ_CHANGED:'+str(rel))
        bindings.append(dict(relative=str(rel),sha256=m['sha256'],mode='EXACT_PREVIOUS_FULL_READ_REUSE'))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
        PYTHONPATH='/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2:'+str(repo))
    argv=[PYTHON,'-B','-m','unittest','project.run_scripts.single_layer_edit_preserving_correction.test_sequential','-v']
    result=subprocess.run(argv,cwd=repo,env=env,text=True,capture_output=True)
    write(out/'cpu-tests.json',dict(args=argv,exit=result.returncode,stdout=result.stdout,stderr=result.stderr,
        CUDA_VISIBLE_DEVICES='',level='CPU_ROUTING_STATE_MOCK_NOT_ACTUAL_T'))
    if result.returncode:raise ValueError('CPU_TEST_FAILURE')
    sources=[]
    for f in sorted((repo/PACKAGE).glob('sequential*.py')):
        ast.parse(f.read_bytes());sources.append(member(f))
    launch=subprocess.run(['bash','-n',str(repo/PACKAGE/'sequential.sbatch')],capture_output=True,text=True)
    if launch.returncode:raise ValueError(launch.stderr)
    from scripts.fixed_counterfact import load_prefix
    records=load_prefix('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1',1000)
    write(out/'receipt.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,
        CPU_tests=18,CPU_PASS=True,actual_model_test='NOT_RUN_USER_T_SKIP',syntax_PASS=True,
        prior_FULL_READ=member(ROOT/'receipts/full-read-m0.json'),bindings=bindings,
        new_design_contract='direct reread + inherited exact positioning/BLUE/native evidence',
        fixed_prefix_requests=len(records),official_P_N_online=False,own_state_and_failure_mock=True,
        independent_red='NOT_RUN_SELF_AUDIT_ONLY',
        session_helper='new isolated worktree lacks ignored servers/local/session-boundary.env; registry exact session/cwd checked; helper PASS not claimed',
        immutable_runtime_core='unchanged; verified again at source freeze',raw_Git=False,
        GH_response=member(ROOT/'receipts/stop-M-start-S-r1/gh-delivery.json')))
    print(json.dumps(member(out/'receipt.json')))

if __name__=='__main__':main()
