"""Single explicit held-inspect-release admission, never touches other jobs."""
import argparse
import fnmatch
import os
import re
import subprocess
from pathlib import Path
from .common import *


def run(args):
    result=subprocess.run(args,text=True,capture_output=True,check=True)
    return result.stdout.strip()


def project_allocation(detail):
    """Include unassigned pending requests; squeue -w can omit these jobs."""
    fields=dict(x.split('=',1) for x in detail.split() if '=' in x)
    assigned=fields.get('NodeList','(null)')
    requested=fields.get('ReqNodeList','(null)')
    known=[x for x in (assigned,requested) if x not in ('(null)','None','')]
    if not known:
        raise RuntimeError('PROJECT_NODE_ADMISSION_UNRESOLVED:'+fields.get('JobId','unknown'))
    # Project launchers require an explicit single-node request. Do not guess
    # expansion of heterogeneous/multi-node host lists at admission.
    if any('[' in x or ',' in x for x in known):
        raise RuntimeError('PROJECT_NODE_ADMISSION_UNRESOLVED:'+fields.get('JobId','unknown'))
    if 'server2' not in known:return 0
    tres=fields.get('AllocTRES','(null)')
    if tres in ('(null)',''):tres=fields.get('ReqTRES','')
    gpu=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
    if not gpu:raise RuntimeError('PROJECT_GPU_COUNT_UNRESOLVED:'+fields.get('JobId','unknown'))
    return int(gpu.group(1))


def require_recall(recall):
    pause=ATTEMPT/'receipts/user-implementation-only-20260920.json'
    if pause.exists():
        if recall is None:raise RuntimeError('USER_RECALL_REQUIRED_IMPLEMENTATION_ONLY')
        r=read(recall)
        assert r.get('pause_receipt_sha256')==sha256(pause)
        assert r.get('instruction_id')=='ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1'
        assert r.get('explicit_user_recall') is True and r.get('user_message')


def submit(execution,recall=None):
    require_recall(recall)
    execution=Path(execution);lock=read(execution/'execution.lock.json')
    receipt=execution/'admission.json'
    assert not receipt.exists() and not (execution/'submission.json').exists(),'DUPLICATE_SUBMISSION'
    env=os.environ.copy();env['AGENT_GPU_CAPS_FILE']=str(ROOT/'servers/local/gpu-caps.tsv')
    cap=subprocess.run(['bash',str(REPO/'scripts/check-slurm-resource-cap.sh'),'server2','1','60416M'],
        env=env,text=True,capture_output=True)
    queue=run(['squeue','-h','-t','RUNNING,COMPLETING,CONFIGURING,PENDING','-o','%i|%j|%b|%T'])
    total=0;records=[]
    for line in queue.splitlines():
        job,name,tres,state=line.split('|')
        if not any(fnmatch.fnmatch(name,pat) for pat in ('odeedit_*','odealloc_*')):continue
        detail=run(['scontrol','show','job',job,'--oneliner'])
        count=project_allocation(detail)
        if not count:continue
        total+=count;records.append(dict(job=job,name=name,state=state,gpus=count))
    admitted=cap.returncode==0 and total+1<=2
    write_json(receipt,dict(status='ALLOW' if admitted else 'WAITING_FOR_ISOLATED_RESOURCE',
        active_and_admitted_project_gpus=total,new_gpus=1,cap=2,records=records,
        resource_helper_exit=cap.returncode,resource_helper_stdout=cap.stdout,resource_helper_stderr=cap.stderr,
        memory_mib=60416,source_head=lock['source_head']))
    if not admitted:
        print('WAITING_FOR_ISOLATED_RESOURCE');return
    log=ATTEMPT/'logs';log.mkdir(exist_ok=True)
    mode=lock.get('mode','gate');assert mode in ('gate','keys','operator','activation')
    source=Path(lock['source']);script=source/'project/run_scripts/checkpoint_mechanism_audit'/('run.sbatch' if mode=='gate' else 'analysis.sbatch')
    args=['sbatch','--hold','--parsable','--job-name=odeedit_checkpoint_mechanism_s2_'+mode,
        '--chdir='+str(source),'--output='+str(log/(mode+'-%j.out')),
        '--error='+str(log/(mode+'-%j.err')),str(script),str(source),str(execution/'execution.lock.json')]
    if mode=='gate':args.append(lock['expected_output'])
    raw=run(args);job=raw.split(';')[0];assert job.isdigit()
    write_json(execution/'submitted-held.json',dict(job=job,args=args,source_head=lock['source_head']))
    detail=run(['scontrol','show','job',job,'--oneliner'])
    expected=['JobState=PENDING','Reason=JobHeldUser','Requeue=0','NumCPUs=8','ReqNodeList=server2','WorkDir='+str(source)]
    assert all(s in detail for s in expected),'HELD_INSPECTION_FAILED'
    assert 'gres/gpu=1' in detail and ('mem=59G' in detail or 'mem=60416M' in detail),detail
    write_json(execution/'held-inspection.json',dict(status='PASS',job=job,job_record=detail,
        source_lock_sha256=sha256(execution/'execution.lock.json'),script_sha256=sha256(script)))
    released=run(['scontrol','release',job])
    write_json(execution/'submission.json',dict(status='RELEASED',job=job,release_response=released,
        source_head=lock['source_head'],source_tree=lock['source_tree'],output=lock['expected_output'],
        lock_sha256=sha256(execution/'execution.lock.json'),new_gpu_jobs=1,other_job_mutation=0))
    print('SUBMITTED_RELEASED',job,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',required=True);p.add_argument('--recall-receipt')
    a=p.parse_args();submit(a.execution,a.recall_receipt)
