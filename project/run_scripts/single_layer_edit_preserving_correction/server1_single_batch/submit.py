"""One-shot held/inspect/release. No polling, retry, cancellation, or callback."""
import argparse
import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess
from ..common import member,write

def command(args,**kwargs):
    return subprocess.check_output(args,text=True,**kwargs).strip()

def admitted():
    raw=command(['squeue','-h','-w','devbox','-t','RUNNING,PENDING,COMPLETING,CONFIGURING','-o','%i|%j|%b'])
    total=0;rows=[]
    for line in raw.splitlines():
        job,name,tres=line.split('|')
        if not any(fnmatch.fnmatch(name,p) for p in ('odeedit_*','motivation_*','session01_*')):continue
        m=re.search(r'gpu(?::[^,:=]+)?:(\d+)',tres)
        if not m:raise RuntimeError('UNRESOLVED_ADMITTED_GPU:'+line)
        total+=int(m.group(1));rows.append(dict(id=job,name=name,gpus=int(m.group(1))))
    return dict(gpus=total,rows=rows,cap=2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();repo=Path.cwd();submission=root/'submission-r1'
    submission.mkdir(exist_ok=False)
    if command(['git','status','--porcelain','--untracked-files=no']):raise ValueError('DIRTY_EXECUTION_SOURCE')
    head=command(['git','rev-parse','HEAD']);tree=command(['git','rev-parse','HEAD^{tree}'])
    if os.uname().nodename!='devbox':raise ValueError('WRONG_HOST')
    command(['bash','scripts/check-session-boundary.sh','01a04939-f93a-7b50-bca0-65438eab2062'])
    env=dict(os.environ,AGENT_GPU_CAPS_FILE='/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    cap=command(['bash','scripts/check-slurm-resource-cap.sh','server1','1','182272M'],env=env)
    launcher=repo/'project/run_scripts/single_layer_edit_preserving_correction/server1_single_batch/launch.sh'
    memory=command(['python3','scripts/slurm_memory_policy.py','audit',str(launcher)])
    before=admitted()
    if before['gpus']+1>2:raise RuntimeError('WAITING_FOR_ISOLATED_RESOURCE')
    lock=json.loads((root/'locks/runtime-draft.json').read_text())
    lock['execution']=dict(head=head,tree=tree,frozen_M='87f65ea2abcbe7e77e04367f73a001d63443734b',
        source_root=str(repo),scope='S1_REUSE_FIRST_B001',launcher=member(launcher))
    locked=write(submission/'execution.lock.json',lock)
    job=command(['sbatch','--hold','--parsable','--chdir',str(repo),'--output',str(submission/'stdout.log'),
        '--error',str(submission/'stderr.log'),str(launcher),str(repo),head,locked['path']]).split(';')[0]
    if not job.isdecimal():raise RuntimeError('JOB_ID_UNRESOLVED')
    write(submission/'held.json',dict(job=job,source=head,tree=tree,lock=locked,pre_admission=before,cap_check=cap,memory=memory))
    detail=command(['scontrol','show','job',job,'--oneliner'])
    required=['JobName=odeedit_enfc_m_b001_s1','UserId=janghj(','JobState=PENDING','Reason=JobHeldUser',
        'Dependency=(null)','Requeue=0','NumCPUs=8','MinMemoryNode=178G','ReqNodeList=devbox',
        'TimeLimit=1-00:00:00','gres/gpu=1','Command='+str(launcher),'WorkDir='+str(repo)]
    if not all(x in detail for x in required):
        write(submission/'inspection-HOLD.json',dict(job=job,detail=detail,missing=[x for x in required if x not in detail]))
        raise RuntimeError('HELD_INSPECTION_HOLD_NO_RELEASE')
    after=admitted()
    if after['gpus']>2:raise RuntimeError('HELD_CAP_HOLD_NO_RELEASE')
    write(submission/'inspection.json',dict(job=job,detail=detail,post_held_admission=after,pass_exact=True))
    command(['scontrol','release',job])
    state=command(['scontrol','show','job',job,'--oneliner'])
    write(submission/'released.json',dict(job=job,detail=state,dependency=None,source=head,lock=locked,
        initial_gate='NOT_YET',remaining='B1 missing3 plus exact endpoint observations',S4_job_action=0))
    print(json.dumps(dict(job=job,source=head,tree=tree,lock=locked,status='RELEASED_INITIAL_NOT_YET',dependency=None)))

if __name__=='__main__':main()
