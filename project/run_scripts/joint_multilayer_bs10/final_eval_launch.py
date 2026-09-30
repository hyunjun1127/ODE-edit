"""Freeze final evaluation and admit one new GPU lane under existing cap3."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
from .common import read,record,save,sha,require
from .server2_entry import PYTHON

BASE=Path('/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1')
ROOT=BASE/'final-full-eval-20260930-v1'
FILES=['common.py','observations.py','reduce.py','server2_entry.py','final_eval.py','final_eval_launch.py','test_final_eval.py']

def command(a,cwd=None):return subprocess.check_output(a,cwd=cwd,text=True).strip()

def freeze(repo):
    repo=Path(repo).resolve();require(not command(['git','status','--porcelain'],repo),'DIRTY_SOURCE')
    ROOT.mkdir(exist_ok=False);src=ROOT/'source';members=[]
    for rel in ['scripts/fixed_counterfact.py']+['project/run_scripts/joint_multilayer_bs10/'+f for f in FILES]:
        dst=src/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/rel,dst);members.append(record(dst))
    old=BASE/'attempt-s2-r1';cfg=old/'configuration.json';arms=[]
    for arm in ('NATIVE','JOINT_STEP','JOINT_CUM'):
        d=old/'output'/('B010-'+arm);term=read(d/'terminal.json');require(term['status']=='COMPLETED' and term['batches']==100,'ORIGINAL_TERMINAL')
        snap=d/'weights'/('B010-'+arm)/'T100.receipt.json';r=read(snap);s=Path(r['file']['path']).stat()
        require(s.st_size==r['file']['bytes'] and s.st_ino==r['file']['inode'] and s.st_mtime_ns==r['file']['mtime_ns'],'CP_CURRENT_STAT')
        arms.append(dict(arm=arm,snapshot_receipt=record(snap),snapshot_file=r['file'],original_terminal=record(d/'terminal.json'),
            old_catalog=record(d/'token-catalog.json'),old_final=record(d/'B100/all-offered.json')))
    lock=dict(root=str(ROOT),source=command(['git','rev-parse','HEAD'],repo),tree=command(['git','rev-parse','HEAD^{tree}'],repo),
        original_source=read(old/'execution.lock.json')['source'],source_members=members,configuration=record(cfg),arms=arms,
        scope='B010 T100 final: three arms, each rewrite100/paraphrase200/neighborhood1000',
        resources=dict(project_cap=3,array='0-2%1',gpus_per_job=1,cpus=8,mem_mib=60416,export='NONE',requeue=0,wall='02:00:00'),
        no_edit=True,no_native_fit=True,save_checkpoints=False,monitor='first valid evaluation then pause; no automatic terminal monitoring',
        snapshot_verification='prior B010 review full SHA + current stat; each job verifies file/tensor SHA again before restore',
        estimate='prior model load32-47s, short-row observer about0.065s/call; 2600calls/arm plus hash/restore/I-O, 2h wall allowance not measured ETA',
        projected_gpu_peak_gib=38,NO_BROADCAST_NOT_REQUIRED=True)
    save(ROOT/'execution.lock.json',lock)
    script='''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=server2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60416M
#SBATCH --export=NONE
#SBATCH --no-requeue
#SBATCH --time=02:00:00
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
'''+f'cd {src}\nexec {PYTHON} -B -m project.run_scripts.joint_multilayer_bs10.final_eval --lock {ROOT}/execution.lock.json --index "$SLURM_ARRAY_TASK_ID"\n'
    with (ROOT/'eval.sbatch').open('x') as f:f.write(script)
    save(ROOT/'freeze.json',dict(lock=record(ROOT/'execution.lock.json'),script=record(ROOT/'eval.sbatch')))
    print(ROOT)

def submit():
    require(not (ROOT/'submitted.json').exists(),'DUPLICATE_SUBMIT')
    f=read(ROOT/'freeze.json')
    for r in f.values():require(sha(r['path'])==r['sha256'],'FROZEN_CHANGED')
    q=command(['squeue','-h','-r','-u','janghj','-w','server2','-o','%i|%j|%T|%b|%R'])
    rows=[r.split('|') for r in q.splitlines() if r]
    # Restrict admission to observed original two GPU jobs; an unknown job requires fresh review.
    gpu=[r for r in rows if 'gpu' in r[3]]
    require(all(r[0] in ('55116_3','55116_4') and r[3]=='gres/gpu:1' for r in gpu),'UNEXPECTED_GPU_ADMISSION')
    require(len(gpu)+1<=3,'CAP3')
    require(shutil.disk_usage(ROOT).free>2*(1024**3),'OUTPUT_SPACE')
    save(ROOT/'admission.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=q,existing_reserved=len(gpu),new_lane=1,project_cap=3,
        node=command(['scontrol','show','node','server2']),disk=shutil.disk_usage(ROOT)._asdict(),other_jobs_mutated=False))
    argv=['sbatch','--parsable','--hold','--array=0-2%1','--job-name=odeedit_joint_finaleval_s2',
        '--output='+str(ROOT/'%A_%a.out'),'--error='+str(ROOT/'%A_%a.err'),str(ROOT/'eval.sbatch')]
    job=command(argv).split(';')[0];require(job.isdigit(),'JOB_ID')
    save(ROOT/'submitted.json',dict(job_id=job,argv=argv,held=True))
    info=command(['scontrol','show','job','-o',job])
    for token in ('UserId=janghj(','JobState=PENDING','Reason=JobHeldUser','ArrayTaskId=0-2%1','Requeue=0','NumCPUs=8','gres/gpu=1','MinMemoryNode=59G','ReqNodeList=server2',str(ROOT/'eval.sbatch')):
        require(token in info,'HELD_INSPECTION:'+token)
    save(ROOT/'held-inspection.json',dict(job_id=job,info=info,mapping={str(i):a['arm'] for i,a in enumerate(read(ROOT/'execution.lock.json')['arms'])}))
    subprocess.run(['scontrol','release',job],check=True)
    save(ROOT/'released.json',dict(job_id=job,status='RELEASED',all_three_registered=True,initial='NOT_OBSERVED'))
    print(job)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo');p.add_argument('--submit',action='store_true');a=p.parse_args()
    if a.submit:submit()
    else:freeze(a.repo)
