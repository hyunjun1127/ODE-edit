"""Explicit user rerun: reconcile exact old graph and bind unchanged inputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from .submit import job
from .run import TASK

ROOT=Path('/data/janghj/ODE-edit/local')/TASK
OLD=ROOT/'attempt-generation-repair-r1'
NEW=ROOT/'attempt-subset-repair-r1'

def reconcile():
    receipt=ROOT/'subset-repair-cancellation.json'
    if receipt.exists():raise RuntimeError('RECONCILIATION_ALREADY_RECORDED')
    roles={'61359':'collector','61358':'GPT2XL_M1_M2','61357':'GPTJ_M1','61356':'LLAMA_REPRO'}
    def check(j):
        r=job(j)
        assert r['UserId']=='janghj(1025)' and r['ReqNodeList']=='server4'
        assert r['JobName']==TASK+'-'+roles[j]
        assert r['Command']==str(OLD/(roles[j]+'.sh')) and r['WorkDir']==str(OLD/'source')
        return r
    data=dict(authority='USER_ALL_THREE_RERUN',before={j:check(j) for j in roles},actions=[])
    def persist():receipt.write_text(json.dumps(data,indent=2)+'\n')
    persist()
    for j in roles:  # collector before all upstream jobs
        r=check(j)
        if r['JobState'] in ('PENDING','RUNNING','CONFIGURING','COMPLETING'):
            subprocess.run(['scancel',j],check=True)
            data['actions'].append(dict(job_id=j,before=r,action='scancel'));persist()
    data['after']={j:check(j) for j in roles};persist()
    print(json.dumps({j:r['JobState'] for j,r in data['after'].items()}))

def prepare():
    from project.run_scripts.jlz_interference_l1.cap_common import verify,member,require
    data=json.loads((OLD/'config.json').read_text())
    require(set(data['cells'])=={'LLAMA_REPRO','GPTJ_M1','GPT2XL_M1_M2'},'EXACT_THREE_CELLS')
    for row in data['input_members']:verify(row)
    for c in data['cells'].values():
        for row in c['assets']:
            s=Path(row['path']).stat()
            require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_STAT_CHANGED')
        c['run_instance']=dict(attempt='subset-identity-repair-20261008')
        c['resources']['project_cap']=3;c['resources']['task_cap']=3
    free=shutil.disk_usage(ROOT).free
    require(free>=data['storage']['reserve_bytes'],'STORAGE_RERUN_ADMISSION')
    data.update(repair='USER_ALL_THREE_SUBSET_IDENTITY_REPAIR',project_gpu_cap=3)
    data['storage']['free_bytes']=free
    data['previous_attempt']=str(OLD)
    data['input_members'].append(member(ROOT/'subset-repair-cancellation.json'))
    NEW.mkdir(exist_ok=False)
    with (NEW/'config.json').open('x') as f:json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2)
    print(json.dumps(dict(attempt=str(NEW),cells=list(data['cells']),free_bytes=free,new_W0=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['reconcile','prepare'])
    globals()[p.parse_args().action]()
