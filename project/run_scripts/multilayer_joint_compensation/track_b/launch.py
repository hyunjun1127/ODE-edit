"""One explicit B-OS admission/held-inspection/release, without monitoring loops."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from ..contracts import sha,save,digest

SESSION='01a0493a-074c-7f91-9a13-769116326fef'

def command(args):
    r=subprocess.run(args,text=True,capture_output=True)
    return dict(command=args,exit=r.returncode,stdout=r.stdout,stderr=r.stderr)

def admission(exclude_job=None):
    gate=command(['scripts/check-slurm-resource-cap.sh','server2','1','60416M'])
    queue=command(['squeue','-h','-u','janghj','-w','server2','-t','RUNNING,COMPLETING,CONFIGURING,PENDING','-o','%i|%j|%b|%T'])
    if gate['exit'] or queue['exit']:return False,dict(gate=gate,queue=queue)
    jobs=[];gpu_count=0
    for line in queue['stdout'].splitlines():
        job,name,tres,state=line.split('|')
        if job==exclude_job:continue
        if name=='odeedit_multilayer_b_s2':return False,dict(gate=gate,queue=queue,reason='DUPLICATE_B_JOB_ACTIVE_OR_PENDING')
        if not name.startswith(('odeedit_','odealloc_')):continue
        if '[' in job or 'gpu:' not in tres:return False,dict(gate=gate,queue=queue,reason='UNRESOLVED_ARRAY_OR_GPU_COUNT')
        count=sum(int(t.split(':')[-1]) for t in tres.split(',') if t.startswith('gres/gpu:') or t.startswith('gpu:'))
        if count<1:return False,dict(gate=gate,queue=queue,reason='UNRESOLVED_GPU_COUNT')
        gpu_count+=count;jobs.append(dict(job=job,name=name,GPUs=count,state=state))
    return gpu_count+1<=2,dict(gate=gate,queue=queue,project_jobs=jobs,existing_including_pending=gpu_count,requested=1,cap=2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-lock',required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--test-repair-lock');a=p.parse_args()
    root=Path.cwd();attempt=Path(a.attempt).absolute();attempt.mkdir(parents=True,exist_ok=False)
    boundary=command(['scripts/check-session-boundary.sh',SESSION]);save(attempt/'session.json',boundary)
    if boundary['exit']:raise RuntimeError('SESSION_BOUNDARY')
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True)
    if dirty:raise RuntimeError('EXECUTION_SOURCE_NOT_CLEAN')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    inp=Path(a.input_lock).absolute();inputs=json.loads(inp.read_text())
    if inputs['status']!='COMMON_CPU_SCHEMA_AND_LOCAL_ASSET_PASS':raise RuntimeError('CPU_INPUT_GATE')
    paths=subprocess.check_output(['git','ls-files','-z','project/run_scripts','scripts','control/gpu-concurrency-policy.tsv','servers/slurm-memory-policy.tsv'],text=True).split('\0')
    members=[dict(path=str(root/n),sha256=sha(root/n)) for n in paths if n and (root/n).is_file()]
    execution=dict(source_head=head,source_tree=tree,worktree=str(root),source_members=members,source_root=digest(members),
        input_lock=dict(path=str(inp),sha256=sha(inp)),arm='B-OS',entry='Middle',GPU=1,mem_mib=60416,CPUs=8,cap=2,
        output=str(attempt/'output'),monitoring_policy='ODEEDIT-INITIAL-GATE-ONLY-USER-RECALL-20260911',
        agent_pauses_after_initial_valid=True,no_automatic_followup_submission=True,scientific_promotion=False)
    if a.test_repair_lock:
        repair=Path(a.test_repair_lock).absolute()
        execution['test_repair_lock']=dict(path=str(repair),sha256=sha(repair))
    execution_path=attempt/'execution.lock.json';save(execution_path,execution)
    allowed,cap=admission();save(attempt/'admission.json',cap)
    if not allowed:
        save(attempt/'pause.json',dict(status='WAITING_FOR_ISOLATED_RESOURCE',submission=0,initial_gate='NOT_RUN',admission=cap))
        print('WAITING_FOR_ISOLATED_RESOURCE',flush=True);return
    launcher=root/'project/run_scripts/multilayer_joint_compensation/track_b/run.sbatch'
    cmd=['sbatch','--hold','--parsable','--output='+str(attempt/'slurm-%j.out'),'--error='+str(attempt/'slurm-%j.err'),
         str(launcher),str(root),head,'--common',inputs['common'],'--entry','Middle','--arm','B-OS','--source-head',head,
         '--execution-lock',str(execution_path),'--output',str(attempt/'output')]
    submitted=command(cmd);save(attempt/'submission.json',submitted)
    if submitted['exit']:raise RuntimeError('SBATCH_FAILED')
    job=submitted['stdout'].strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('UNRESOLVED_JOB_ID')
    inspected=command(['scontrol','show','job',job,'--oneliner']);save(attempt/'held-inspection.json',inspected)
    text=inspected['stdout']
    fields=dict(token.split('=',1) for token in text.split() if '=' in token)
    expected={'JobId':job,'JobName':'odeedit_multilayer_b_s2','JobState':'PENDING','Reason':'JobHeldUser',
        'WorkDir':str(root),'Command':str(launcher),'NumCPUs':'8','ReqNodeList':'server2'}
    if inspected['exit'] or any(fields.get(k)!=v for k,v in expected.items()) or fields.get('TresPerNode')!='gres/gpu:1' or fields.get('MinMemoryNode') not in ('60416M','59G'):
        save(attempt/'hold.json',dict(status='HELD_INSPECTION_MISMATCH',job=job,expected=expected,observed=fields))
        raise RuntimeError('OWN_NEW_JOB_REMAINS_HELD_INSPECTION_MISMATCH')
    # The held job itself is now in pending. Check others + this requested GPU.
    allowed,release_gate=admission(exclude_job=job)
    save(attempt/'release-admission.json',release_gate)
    if not allowed:raise RuntimeError('OWN_NEW_JOB_REMAINS_HELD_RESOURCE_CHANGE')
    released=command(['scontrol','release',job]);save(attempt/'release.json',released)
    if released['exit']:raise RuntimeError('RELEASE_FAILED')
    save(attempt/'submitted-receipt.json',dict(status='SUBMITTED_RELEASED',job=job,source_head=head,source_tree=tree,
        execution_lock_sha=sha(execution_path),input_lock_sha=sha(inp),common=inputs['common'],output=str(attempt/'output'),
        initial_gate='NOT_YET_OBSERVED',policy=execution['monitoring_policy'],scientific_promotion=False))
    print(json.dumps(dict(status='SUBMITTED_RELEASED',job=job,output=str(attempt/'output'),source_head=head)),flush=True)

if __name__=='__main__':main()
