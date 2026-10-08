"""One deliberate, exact-source baseline replacement authorized by direct USER."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
from .common import read, write, sha, require, LOCAL

ROOT=Path(__file__).resolve().parents[3]
AUTHORITY=ROOT/'plans/updates/server1/gpt2xl-baselines-native-generation-repair/user-authority.json'
OLD=LOCAL/'attempt-w20-only-three-r1'
OUT=LOCAL/'native-repo-repair-r1'
ROLES={'collector':'61439','RECT':'61438','PRUNE':'61437','ALPHAEDIT_BLUE':'61436'}
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'}

def command(argv):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=25)
    return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)

def check(role, job, lock):
    response=command(['scontrol','show','job','-o',job])
    require(response['returncode']==0,'EXACT_BASELINE_SCHEDULER_RECEIPT')
    d=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)',response['stdout']))
    script=OLD/(role+'.sh')
    require(os.getuid()==1025 and d.get('JobId')==job and d.get('UserId')=='janghj(1025)'
        and d.get('JobName')=='gpt2xl-blue-prune-rect-w20-generation-'+role
        and d.get('Command')==str(script) and d.get('WorkDir')==str(OLD/'source')
        and d.get('ReqNodeList')=='devbox','EXACT_OWNER_SOURCE_TASK_TARGET')
    item=next(x for x in lock['launchers'] if x['path']==str(script))
    require(sha(script)==item['sha256'] and script.stat().st_size==item['bytes'],
            'FROZEN_ORIGINAL_LAUNCHER')
    submitted=read(OLD/('submitted-'+role+'.json'))
    require(submitted['job']==job and submitted['role']==role,'ORIGINAL_SUBMISSION_BINDING')
    return {k:d.get(k) for k in ('JobId','JobName','UserId','JobState','RunTime','StartTime',
        'AllocTRES','ReqTRES','Dependency','Command','WorkDir','NodeList','ReqNodeList')}

def transition(cancel=False):
    authority=read(AUTHORITY)
    require(authority['server1_cancel_scope']==[int(x) for x in ROLES.values()],
        'DIRECT_USER_CANCEL_SCOPE')
    lock=read(OLD/'execution.lock.json')
    require(lock['source_commit']=='9a8c7ebfae197cc0d8dba994ff1208cdb24635f8'
        and lock['config_sha256']==sha(OLD/'config.json')
        and lock['owner']=='janghj','ORIGINAL_FROZEN_IDENTITY')
    before={role:check(role,job,lock) for role,job in ROLES.items()}
    counts={role:dict(committed_batches=len(list((OLD/role).glob('batch-*/commit.json'))),
        generation_final_receipt=(OLD/role/'generation-W20.json').exists(),raw_preserved=True)
        for role in ROLES if role!='collector'}
    result=dict(instruction_id=authority['instruction_id'],authority_sha256=sha(AUTHORITY),
        owner=authority['owner'],at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        old_source=lock['source_commit'],old_config=lock['config_sha256'],before=before,
        partial_before=counts,order=list(ROLES),unrelated_jobs_mutated=False,raw_source_preserved=True)
    if not cancel:return result
    require(not (OUT/'cancel-started.json').exists(),'ONE_DELIBERATE_CANCEL_PASS')
    write(OUT/'cancel-started.json',result)
    actions=[]
    for role,job in ROLES.items():
        current=check(role,job,lock)
        action=dict(role=role,job=job,before=current)
        if current['JobState'] in ACTIVE:
            action.update(action='USER_BASELINE_REPLACEMENT_CANCEL',response=command(['scancel',job]))
            write(OUT/('cancel-'+role+'.json'),action)
            require(action['response']['returncode']==0,'EXACT_CANCEL_FAILED')
        else:action['action']='TERMINAL_KEEP_NO_CANCEL'
        actions.append(action)
    result.update(actions=actions,post_queue=command(['squeue','-h','-j',','.join(ROLES.values()),
        '-o','%i|%u|%T|%j|%N']),accounting=command(['sacct','-X','-n','-P','-j',','.join(ROLES.values()),
        '--format=JobIDRaw,JobName%64,User,State,ElapsedRaw,AllocTRES,ExitCode']))
    write(OUT/'cancellation.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cancel',action='store_true');a=p.parse_args()
    r=transition(a.cancel)
    print(json.dumps(dict(actions=r.get('actions',[]),partial=r['partial_before'],
        post_queue=r.get('post_queue'),preserved=r['raw_source_preserved']),ensure_ascii=False))
