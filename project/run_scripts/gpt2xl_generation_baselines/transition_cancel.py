"""Exact USER-authorized old baseline DAG transition; no other job mutation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
from .common import ROOT,LOCAL,SESSION,read,write,sha,require

NONCE='USER-GH-SH1-SH2-BASELINE-GENERATION-KV-BATCH-REPAIR-20261008-R1'
AUTHORITY='66cddb8fa9c09e475f0ae3f423635091c6a71e3d'
ENVELOPE='messages/head/2026-10-08-baseline-generation-kv-batch-repair.json'
ENVELOPE_SHA='6902af1cb866fde3e6e01a0cd21cac7642db9c20de1bbdb86d18d72646371546'
OLD=LOCAL/'attempt-register-r1'
OLD_SOURCE='6bc51602632b5a2dfb4c832479002b30b604b8eb'
OLD_CONFIG='9738bb2a05f28cff347f260789d32db667d6bcb542142a5e9c4528f8e13cb4f2'
JOBS={'BASE_MEMIT':'60928','BASE_ALPHAEDIT':'60929','CAKE':'60930',
      'ALPHAEDIT_BLUE':'60931','PRUNE':'60932','RECT':'60933','collector':'60934'}
ORDER=('collector','RECT','PRUNE','ALPHAEDIT_BLUE','CAKE','BASE_ALPHAEDIT','BASE_MEMIT')
CONTROL=LOCAL/'cache-repair-20261008-r1'
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'}

def command(argv):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=20)
    return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)

def scheduler(job):
    result=command(['scontrol','show','job','-o',job])
    require(result['returncode']==0,'EXACT_JOB_CURRENT_STATE_AVAILABLE')
    values=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)',result['stdout']))
    return {k:values.get(k) for k in ('JobId','JobName','UserId','JobState','Reason',
        'NodeList','ReqNodeList','Command','WorkDir','Dependency','StartTime','RunTime',
        'AllocTRES','ReqTRES','ExitCode','Restarts')}

def check(role,job,lock):
    current=scheduler(job);script=OLD/(role+'.sh')
    require(current['JobId']==job and current['UserId']=='janghj(1025)'
        and os.getuid()==1025 and current['JobName']=='gpt2xl-baselines-fluency-consistency-2k-'+role
        and current['Command']==str(script) and current['WorkDir']==str(OLD/'source')
        and current['ReqNodeList']=='devbox','EXACT_OWN_PARENT_BASELINE_TARGET')
    item=next(x for x in lock['launchers'] if x['path']==str(script))
    require(script.stat().st_size==item['bytes'] and sha(script)==item['sha256'],'ORIGINAL_SCRIPT_IDENTITY')
    submitted=read(OLD/('submitted-'+role+'.json'))
    require(submitted['job']==job and submitted['role']==role and submitted['argv'][-1]==str(script),
        'EXACT_SUBMISSION_RECEIPT')
    return current

def phase():
    c=read(OLD/'config.json');root=Path(c['generation']['shared_W0_root'])
    result={}
    for role in JOBS:
        if role=='collector':continue
        out=OLD/role;obs=out/'generation/observations'
        commits=list(out.glob('B*/commit.json'))
        result[role]=dict(completed_case_observation_files=sum(1 for p in obs.glob('*.json')
            if re.fullmatch(r'[0-9a-f]{64}\.json',p.name) and p.is_file() and not p.is_symlink()),
            durable_batch_commits=len(commits),generation_W0_ready=(root/'READY.json').is_file(),
            W0_generation_work_receipt=(out/'generation-work/W0.json').is_file(),
            terminal_receipt=(out/'terminal.json').is_file(),
            scientific_completion_not_inferred=True)
    return result

def transition(cancel=False,continue_cancel=False):
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA,'LATEST_EXACT_AUTHORITY_BYTES')
    e=read(ROOT/ENVELOPE);owner=e['owners']['server1']
    require(e['instruction_id']==NONCE and owner['session']==SESSION and owner['model']=='gpt2xl',
        'LATEST_OWNER_CANCEL_PERMISSION')
    lock=read(OLD/'execution.lock.json')
    require(lock['owner']=='janghj' and lock['source_commit']==OLD_SOURCE
        and lock['config_sha256']==OLD_CONFIG and sha(OLD/'config.json')==OLD_CONFIG,
        'ORIGINAL_SOURCE_CONFIG_OWNER_BINDING')
    if continue_cancel:
        saved=read(CONTROL/'cancel-started.json')
        require(cancel and saved['instruction_id']==NONCE and saved['source']==OLD_SOURCE
            and saved['config_sha256']==OLD_CONFIG,'EXACT_INTERRUPTED_CONTROL_PASS_ONLY')
    else:
        require(not (CONTROL/'cancel-started.json').exists(),'ONE_DELIBERATE_TRANSITION_NO_RETRY')
    before={role:check(role,job,lock) for role,job in JOBS.items()}
    proof=dict(schema=1,instruction_id=NONCE,authority=AUTHORITY,session=SESSION,
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=OLD_SOURCE,
        config_sha256=OLD_CONFIG,before=before,phase_before=phase(),cancel_order=list(ORDER),
        other_jobs_mutated=False,old_source_raw_preserved=True,checkpoint_save=False)
    if not cancel:return proof
    CONTROL.mkdir(exist_ok=True)
    write(CONTROL/('cancel-control-resume.json' if continue_cancel else 'cancel-started.json'),proof)
    results=[]
    for role in ORDER:
        job=JOBS[role];now=check(role,job,lock)
        if now['JobState'] not in ACTIVE:
            result=dict(role=role,job=job,current=now,action='TERMINAL_RETAINED_NO_CANCEL')
        else:
            result=dict(role=role,job=job,current=now,action='EXACT_USER_CANCEL',
                response=command(['scancel',job]))
            results.append(result);write(CONTROL/('cancel-'+role+'.json'),result)
            require(result['response']['returncode']==0,'EXACT_CANCEL_COMMAND_FAILED_STOP')
            continue
        results.append(result);write(CONTROL/('cancel-'+role+'.json'),result)
    post=command(['squeue','-h','-j',','.join(JOBS.values()),'-o','%i|%u|%T|%j|%N'])
    require(post['returncode']==0,'EXACT_POST_QUEUE_AVAILABLE')
    proof.update(results=results,post_queue=post,phase_after=phase(),
        afterany_downstream_first=True,new_jobs_submitted=0)
    write(CONTROL/'cancellation.json',proof)
    return dict(status='EXACT_OLD_BASELINE_DAG_CANCEL_REQUESTED',
        IDs=[JOBS[x] for x in ORDER],post_queue=post['stdout'],phase=proof['phase_after'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cancel',action='store_true')
    p.add_argument('--continue-cancel',action='store_true');args=p.parse_args()
    print(json.dumps(transition(args.cancel,args.continue_cancel),ensure_ascii=False))
