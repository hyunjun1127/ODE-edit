"""Create-once CPU reuse pipeline, exact held inspection then release; no monitor."""
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
from .common import *

def command(args):
    return subprocess.check_output(args,text=True).strip()

def main():
    require(getpass.getuser()=='janghj','OWNER')
    require(command(['hostname'])=='devbox','HOST')
    require(command(['git','branch','--show-current'])=='codex/server1-gpt2-xl-stats-projector-20261007','BRANCH')
    require(not command(['git','status','--porcelain','--','project/run_scripts/gpt2_xl_asset_prep','project/run_scripts/experiment_tracking']),'SOURCE_DIRTY')
    root=LOCAL/'submission-r1';root.mkdir(exist_ok=False)
    source=command(['git','rev-parse','HEAD']);tree=command(['git','rev-parse','HEAD^{tree}'])
    archive=root/'source.tar'
    subprocess.run(['git','archive','--format=tar','--output='+str(archive),source,
                    'project/run_scripts/gpt2_xl_asset_prep','project/run_scripts/experiment_tracking',
                    'control/wandb-policy.json'],check=True)
    frozen=root/'source';frozen.mkdir()
    with tarfile.open(archive) as tf:
        require(all(not m.issym() and not m.islnk() and not Path(m.name).is_absolute()
                    and '..' not in Path(m.name).parts for m in tf.getmembers()),'ARCHIVE_PATHS')
        tf.extractall(frozen,filter='data')
    inputs=member(LOCAL/'preparation-r1/input-lock.json')
    require(inputs['sha256']=='9b6c25b687b316c652d9cd66bb4f6bbfb40107a7c76093791066cd3ab8edbd99','INPUT_LOCK')
    free=shutil.disk_usage(LOCAL).free;require(free>=4*1024**3,'DISK_RESERVE')
    queue=command(['squeue','-h','-u','janghj','-w','devbox','-o','%i|%j|%T|%C|%m|%b|%R'])
    require('odeedit_gpt2xl_reuse' not in queue,'DUPLICATE_TASK_JOB')
    partition=command(['scontrol','show','partition','gpu'])
    require('devbox' in partition and 'lab_gpu_s1' in partition,'PARTITION_QOS')
    write(root/'admission.json',dict(owner='janghj',host='devbox',queue_resource_only=queue,
        partition=partition,disk_free_bytes=free,inodes_free=os.statvfs(LOCAL).f_favail,
        new_GPU=0,CPU_lanes=2,max_concurrent_CPU=16,max_concurrent_memory_MiB=65536,
        GPU_cap=2,old_jobs_mutation=False,stats_or_P_generation=0))
    lock=root/'execution.lock.json'
    write(lock,dict(task_id=TASK,nonce=NONCE,source_commit=source,source_tree=tree,archive=member(archive),
        source_members=[member(p) for p in sorted(frozen.rglob('*')) if p.is_file()],input_lock=inputs,
        tracking_job_identity_policy='USER-GH-ALL-SH-WANDB-JOB-ID-20261007',
        mode='REUSE_ALL_STATS_AND_P_CPU_REVALIDATION',GPU=0,save_checkpoints=False,
        exact_resume='NOT_NEEDED_READONLY_ASSET_VALIDATION',broadcast='NO_BROADCAST_NOT_REQUIRED'))
    jobs=[]
    for role,lane in [('verify',0),('verify',1),('pack',0),('collect',0)]:
        label=f'{role}-{lane}';launcher=root/(label+'.sbatch')
        deps=[] if role=='verify' else [j['job_id'] for j in jobs]
        dependency=(('afterok:' if role=='pack' else 'afterany:')+':'.join(deps)) if deps else None
        wall='02:00:00' if role=='collect' else '12:00:00'
        cpus=4 if role=='collect' else 8;mem=8192 if role=='collect' else 32768
        argv=[PYTHON,'-u','-m','project.run_scripts.gpt2_xl_asset_prep.worker','--lock',str(lock),'--role',role,'--lane',str(lane)]
        body='\n'.join(['#!/bin/bash','set -euo pipefail',
            '# No credentials, model loading, new stats or projector tensors in this launcher.',
            'export SLURM_EXPORT_ENV=ALL',
            'export PYTHONPATH='+shlex.quote(str(frozen)),
            'export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""',
            f'export OMP_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus}',
            'cd '+shlex.quote(str(frozen)), 'exec '+shlex.join(argv),''])
        with launcher.open('x') as f:f.write(body)
        name='odeedit_gpt2xl_reuse_'+label.replace('-','_')+'_s1'
        args=['sbatch','--parsable','--hold','--account=lab','--partition=gpu','--qos=lab_gpu_s1',
            '--nodelist=devbox','--nodes=1','--ntasks=1',f'--cpus-per-task={cpus}',f'--mem={mem}M',
            '--time='+wall,'--export=NONE','--no-requeue','--job-name='+name,
            '--chdir='+str(frozen),'--output='+str(root/(label+'-%j.log'))]
        if dependency:args+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
        args.append(str(launcher));job=command(args).split(';')[0]
        require(bool(re.fullmatch('[0-9]+',job)),'SUBMIT_ID')
        row=dict(job_id=job,role=role,lane=lane,name=name,dependency=dependency,argv=argv,
            CPU=cpus,mem_MiB=mem,wall=wall,GPU=0,launcher=member(launcher),submit_argv=args)
        jobs.append(row);write(root/(label+'-submitted.json'),row)
        snapshot=command(['scontrol','show','job','-o',job])
        checks=[f'JobId={job}',f'JobName={name}','UserId=janghj(', 'JobState=PENDING','Reason=JobHeldUser',
            'Requeue=0','Partition=gpu','QOS=lab_gpu_s1','ReqNodeList=devbox',f'NumCPUs={cpus}',
            'Command='+str(launcher),'WorkDir='+str(frozen),'TimeLimit='+wall]
        require(all(x in snapshot for x in checks),'HELD_INSPECTION:'+job)
        require('gres/gpu' not in snapshot and 'TresPerNode=gres/gpu' not in snapshot,'CPU_ONLY')
        require(f'mem={mem//1024}G' in snapshot or f'mem={mem}M' in snapshot,'MEMORY')
        if dependency:require(all(j in snapshot for j in deps),'DEPENDENCY')
        write(root/(label+'-held-inspection.json'),dict(job_id=job,snapshot=snapshot,
            inspection='PASS',export_NONE_in_submit_argv=True,source_lock=member(lock),GPU=0))
    # Entire exact DAG registered and inspected before any worker release.
    write(root/'registered.json',dict(source=source,lock=member(lock),jobs=jobs))
    released=[]
    for row in reversed(jobs):
        subprocess.run(['scontrol','release',row['job_id']],check=True);released.append(row['job_id'])
    write(root/'released.json',dict(status='RELEASED_INITIAL_NOT_OBSERVED',jobs=jobs,released=released,
        source=source,execution_lock=member(lock),no_gpu_submission=True))
    print(json.dumps(dict(status='RELEASED_INITIAL_NOT_OBSERVED',jobs=[(j['job_id'],j['role'],j['dependency']) for j in jobs],
        source=source,lock_sha256=sha(lock))))

if __name__=='__main__':main()
