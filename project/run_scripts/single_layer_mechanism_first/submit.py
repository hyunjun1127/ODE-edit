"""Exact task-owned held/inspection/release; no unrelated job mutation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from .config import ROOT,check_lock
from .preflight import resource_snapshot
from project.run_scripts.single_layer_edit_preserving_correction.common import write,member


def run(lock_path):
    lock_path=Path(lock_path);lock=json.loads(lock_path.read_text());check_lock(lock)
    if lock.get('save_checkpoints') is not True or not lock.get('checkpoint_exception_authority'):
        raise ValueError('LEGACY_T0_SUBMISSION_NOT_AUTHORIZED_BY_CURRENT_NOCP')
    if lock['stage']!='T0' or lock['phase']!='HOOK':raise ValueError('ONLY_FROZEN_T0_HOOK_SUBMISSION_IMPLEMENTED')
    attempt=lock_path.parent;source=Path(lock['execution']['source'])
    if (attempt/'submission.json').exists():raise ValueError('DUPLICATE_SUBMISSION')
    snap=resource_snapshot()
    # This first common T0 uses one slot. A nonempty owner's admitted queue is
    # not assumed safe: the parent must explicitly resolve its resource lane.
    if snap['resource_only_queue']:raise RuntimeError('NONEMPTY_ADMISSION_REQUIRES_EXACT_CAPACITY_LANE')
    if snap['free_bytes']<24*2**30:raise OSError('TASK_NEW_STORAGE_RESERVE_UNAVAILABLE')
    env=dict(os.environ,AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    cap=subprocess.run([str(source/'scripts/check-slurm-resource-cap.sh'),'server4','1','60416M'],
        env=env,check=True,capture_output=True,text=True)
    write(attempt/'admission.json',dict(snapshot=snap,capacity=dict(existing_running=0,existing_admitted=0,new=1,cap=2),
        memory_cap_check=cap.stdout,lock=member(lock_path)))
    (attempt/'logs').mkdir(exist_ok=False)
    command=['sbatch','--parsable','--hold',f'--output={attempt}/logs/%j.out',f'--error={attempt}/logs/%j.err',
             str(source/'project/run_scripts/single_layer_mechanism_first/run.sbatch'),str(source),str(lock_path)]
    job=subprocess.check_output(command,cwd=source,text=True).strip().split(';')[0]
    if not re.fullmatch(r'\d+',job):raise RuntimeError('SBATCH_ID_SCHEMA')
    write(attempt/'submission.json',dict(job_id=job,command=command,held=True,source=lock['execution']['commit'],lock=member(lock_path)))
    observed=subprocess.check_output(['scontrol','show','job',job,'--oneliner'],text=True)
    conditions={
        'owner':'UserId=janghj(' in observed,'name':'JobName=odeedit_slmf_T0_s4' in observed,
        'held':'JobState=PENDING' in observed and 'Reason=JobHeldUser' in observed,
        'GPU':'gres/gpu=1' in observed,'CPU':'NumCPUs=8' in observed,
        'memory':'mem=60416M' in observed or 'MinMemoryNode=60416M' in observed or 'mem=59G' in observed,
        'node':'ReqNodeList=server4' in observed,'requeue':'Requeue=0' in observed,
        'script':str(source/'project/run_scripts/single_layer_mechanism_first/run.sbatch') in observed,
        'cwd':f'WorkDir={source}' in observed,
        'no_dependency':'Dependency=(null)' in observed or 'Dependency=' not in observed,
        'source_argv':command[-2]==lock['execution']['source'],
        'lock_argv':Path(command[-1]).resolve()==lock_path.resolve()}
    write(attempt/'held-inspection.json',dict(job_id=job,checks=conditions,scontrol=observed,full_argv=command))
    if not all(conditions.values()):raise RuntimeError('HELD_INSPECTION_FAILED_NO_RELEASE')
    subprocess.run(['scontrol','release',job],check=True)
    state=subprocess.check_output(['scontrol','show','job',job,'--oneliner'],text=True)
    write(attempt/'release.json',dict(job_id=job,released=True,scontrol=state,scientific_initial_gate='NOT_OBSERVED',
        technical_scope='FOUR_REQUEST_NATIVE_Z_HOOK_ONLY',full_T0_ready=False))
    print(json.dumps(dict(job_id=job,source=lock['execution']['commit'],release=True,output=lock['output'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True)
    run(p.parse_args().lock)
