"""Explicit held-submit / inspect-release control; no loops or follow-up jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
from .preflight import TASK_ROOT, file_identity


def command(args):
    return subprocess.check_output(args,text=True,timeout=30).strip()


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,sort_keys=True);f.write('\n')


def snapshot():
    return command(['squeue','-h','-u','janghj','-w','server4','-t','RUNNING,PENDING,CONFIGURING,COMPLETING','-o','%i|%j|%u|%T|%b|%E|%R'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['submit-held','inspect-release'],required=True)
    ap.add_argument('--job');a=ap.parse_args()
    assert socket.gethostname()=='server4' and pwd.getpwuid(os.getuid()).pw_name=='janghj'
    task=TASK_ROOT;lock_path=task/'execution.lock.json';lock=json.loads(lock_path.read_bytes())
    assert lock['resource']['project_cap']==2 and lock['checkpoint_storage']=='DISABLED_USER_DIRECTED_NO_W_M_TENSORS'
    queue=snapshot();lines=[l.split('|') for l in queue.splitlines() if l]
    # Conservative: all this owner's GPU jobs on the node, including pending.
    other=[x for x in lines if x[0]!=a.job]
    reserved=0
    for x in other:
        m=re.search(r'gpu(?::[^:|]+)?:(\d+)',x[4])
        assert m, 'UNKNOWN_GPU_RESERVATION:'+repr(x)
        reserved+=int(m.group(1))
    assert reserved+1<=2, 'PROJECT_CAP_PENDING_ADMISSION_HOLD'
    fs=os.statvfs(task);available=fs.f_bavail*fs.f_frsize
    assert available>=lock['raw_reserve_bytes'], 'DISK_HOLD'
    receipt=dict(time_utc=datetime.now(timezone.utc).isoformat(),mode=a.mode,host='server4',owner='janghj',
        source_lock=file_identity(lock_path),queue_snapshot=queue,other_active_plus_pending_reserved_gpus=reserved,
        requested_gpu=1,project_cap=2,available_bytes=available,free_inodes=fs.f_favail,
        resource=dict(gpus=1,cpus=8,mem_mib=60416,wall='2-00:00:00',export='NONE',requeue=0))
    script=task/'source/cake_native_lifelong/run.sbatch'
    if a.mode=='submit-held':
        assert not (task/'submission-held.json').exists(), 'ALREADY_REGISTERED_NO_DUPLICATE'
        (task/'logs').mkdir(mode=0o700,exist_ok=False)
        argv=['sbatch','--hold','--parsable','--output',str(task/'logs/%j.out'),
              '--error',str(task/'logs/%j.err'),'--chdir',str(task/'execution/CAKE'),str(script),str(task)]
        job=command(argv).split(';')[0];assert job.isdigit()
        receipt.update(job_id=job,argv=argv,script=file_identity(script))
        save(task/'submission-held.json',receipt)
    else:
        old=json.loads((task/'submission-held.json').read_bytes())
        assert a.job==old['job_id'] and file_identity(script)==old['script']
        assert receipt['source_lock']==old['source_lock']
        text=command(['scontrol','show','job',a.job,'--oneliner'])
        required=['JobName=odeedit_cake_native_lifelong_s4','UserId=janghj(',
                  'JobState=PENDING','Reason=JobHeldUser','Requeue=0','TimeLimit=2-00:00:00',
                  'NumCPUs=8','MinMemoryNode=59G',str(script),str(task)]
        for value in required:assert value in text, 'HELD_INSPECTION:'+value
        assert 'gres/gpu=1' in text or 'gres/gpu:rtx_pro_6000=1' in text
        receipt.update(job_id=a.job,held_scontrol=text,inspection='PASS')
        save(task/'held-inspection.json',receipt)
        command(['scontrol','release',a.job])
        last=command(['scontrol','show','job',a.job,'--oneliner'])
        state=re.search(r'JobState=(\S+)',last).group(1)
        save(task/'release.json',dict(job_id=a.job,status=state,scontrol=last,
             time_utc=datetime.now(timezone.utc).isoformat(),monitoring_active=False,automatic_resume=False,
             resume_trigger='explicit_user_call',initial_gpu_observation='NOT_RUN_OR_NOT_OBSERVED'))
        receipt.update(last_state=state,released=True)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
