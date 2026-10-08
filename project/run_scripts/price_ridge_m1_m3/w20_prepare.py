"""Prepare new immutable W20-only config; no cancellation/submission/model IO."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
from .subset_rerun import ROOT
from .run import TASK
from .submit import job

def reconcile():
    previous=ROOT/'attempt-subset-repair-r1'
    path=ROOT/'w20-only-cancellation.json'
    if path.exists():raise RuntimeError('W20_RECONCILIATION_ALREADY_RECORDED')
    roles={'61401':'collector','61400':'GPT2XL_M1_M2','61399':'GPTJ_M1','61398':'LLAMA_REPRO'}
    def check(j):
        r=job(j)
        assert r['UserId']=='janghj(1025)' and r['ReqNodeList']=='server4'
        assert r['JobName']==TASK+'-'+roles[j]
        assert r['Command']==str(previous/(roles[j]+'.sh')) and r['WorkDir']==str(previous/'source')
        return r
    data=dict(authority='USER_EXPLICIT_CANCEL_THREE_AND_COLD_W20_ONLY_RERUN',
        before={j:check(j) for j in roles},actions=[])
    def persist():path.write_text(json.dumps(data,indent=2)+'\n')
    persist()
    for j in roles:
        r=check(j)
        if r['JobState'] in ('PENDING','RUNNING','CONFIGURING','COMPLETING'):
            subprocess.run(['scancel',j],check=True)
            data['actions'].append(dict(job_id=j,before=r,action='scancel'));persist()
    data['after']={j:check(j) for j in roles};persist()
    print(json.dumps({j:r['JobState'] for j,r in data['after'].items()}))

def prepare(attempt):
    from project.run_scripts.jlz_interference_l1.cap_common import verify,require,member
    previous=ROOT/'attempt-subset-repair-r1'
    data=json.loads((previous/'config.json').read_text())
    for row in data['input_members']:verify(row)
    for c in data['cells'].values():
        profile=copy.deepcopy(c['arm_profiles'])
        c['run_instance']={'attempt':'generation-w20-only-20261008'}
        c['generation']['schedule']='W20_ONLY'
        c['generation']['qualification_state']='POST_W20_EDIT_ONLY_NOT_W0'
        require(c['arm_profiles']==profile,'EDIT_PROFILE_UNCHANGED')
    data['previous_attempt']=str(previous)
    data['repair']='USER_W20_ONLY_GENERATION'
    data['input_members'].append(member(ROOT/'w20-only-cancellation.json'))
    data['storage']['free_bytes']=shutil.disk_usage(ROOT).free
    require(data['storage']['free_bytes']>=data['storage']['reserve_bytes'],'STORAGE_W20_ADMISSION')
    attempt.mkdir(exist_ok=False)
    with (attempt/'config.json').open('x') as f:json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2)
    print(json.dumps({'attempt':str(attempt),'new_model_forward':0,'new_W0':0,'submission':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path);p.add_argument('--reconcile',action='store_true')
    args=p.parse_args()
    if args.reconcile:reconcile()
    elif args.attempt:prepare(args.attempt.resolve())
    else:p.error('--attempt or --reconcile required')
