"""One deliberate held registration pass. Never cancels or modifies existing jobs."""
import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
from project.run_scripts.jlz_interference_l1.cap_common import require,verify,member
from project.run_scripts.jlz_interference_l1.cap_storage import write,guard


def command(args):
    return subprocess.check_output(args,text=True,timeout=45).strip()


def info(job):
    return json.loads(command(['scontrol','show','job',str(job),'--json']))['jobs'][0]


def job_state(detail):
    value=detail['job_state']
    return value[0] if isinstance(value,list) and len(value)==1 else value


def submit(root):
    root=Path(root).resolve();lock=json.loads((root/'execution.lock.json').read_text())
    require(not (root/'submission.json').exists(),'NO_DUPLICATE_REGISTRATION')
    verify(lock['archive'])
    for row in lock['launchers']+lock['input_members']:verify(row)
    guard(root,lock['storage_reserve_bytes'])
    # Count ALL same-owner node allocations, not just historic job-name prefixes.
    snapshot=json.loads(command(['squeue','-u',getpass.getuser(),'-w','server4','--json']))['jobs']
    write(root/'queue-before.json',snapshot,limit=2*1024**2)
    active=[]
    for row in snapshot:
        job=str(row['job_id']);details=info(job)
        gpu=int(next((v.split('=')[-1] for v in details.get('tres_req_str','').split(',') if v.startswith('gres/gpu=')),'0'))
        if gpu:
            # Fail closed on unrelated/new pending DAG until manually reconciled.
            require(details['user_id']==os.getuid(),'OWN_JOB_REQUIRED')
            require(job_state(details) in ('RUNNING','COMPLETING','CONFIGURING'),'UNRECONCILED_PENDING_GPU')
            active.append((job,gpu))
    require(sum(g for _,g in active)<=2,'LEGACY_OVERCAP_NO_NEW_ADMISSION')
    write(root/'admission.json',dict(effective_combined_cap=2,user_task_requested_GPUs=2,
        allocated=active,warmup_GPU=1,sweep_GPU=2,
        rule='warmup uses one free slot else waits all existing; sweep waits all existing and successful warmup',
        existing_jobs_mutated=False,storage=guard(root,lock['storage_reserve_bytes'])))
    # A previous interrupted submit attempt is detected by exact frozen Command.
    for row in snapshot:
        detail=info(row['job_id'])
        require(detail.get('command') not in [str(root/'warmup.sh'),str(root/'sweep.sh')],'DUPLICATE_FROZEN_COMMAND')
    jobs={};held=[]
    for phase,gpu,cpu,mem,wall in [('warmup',1,8,59392,'04:00:00'),('sweep',2,16,118784,'24:00:00')]:
        dependencies=[]
        if phase=='warmup' and sum(g for _,g in active)>=2:
            dependencies=['afterany:'+':'.join(j for j,_ in active)]
        if phase=='sweep':
            dependencies=['afterok:'+jobs['warmup']]
            if active:dependencies.append('afterany:'+':'.join(j for j,_ in active))
        argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4',
            '--nodes=1','--ntasks=1',f'--cpus-per-task={cpu}',f'--mem={mem}M',
            f'--gres=gpu:rtx_pro_6000:{gpu}',f'--time={wall}','--export=NONE','--no-requeue',
            f'--job-name=qwen-price-tuning-{phase}',f'--chdir={root}',
            f'--output={root}/{phase}-%j.out',f'--error={root}/{phase}-%j.err']
        if dependencies:argv.append('--dependency='+','.join(dependencies))
        argv.append(str(root/(phase+'.sh')))
        result=subprocess.run(argv,text=True,capture_output=True,timeout=45)
        write(root/(phase+'-sbatch.json'),dict(argv=argv,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        require(result.returncode==0 and result.stdout.strip().split(';')[0].isdigit(),'SBATCH_REJECTED_NO_RETRY')
        job=result.stdout.strip().split(';')[0];jobs[phase]=job
        detail=info(job);write(root/(phase+'-held.json'),detail,limit=1024**2)
        # SubmitLine and batch Command must both identify the exact frozen launcher.
        require(detail['user_id']==os.getuid() and job_state(detail)=='PENDING'
            and str(root/(phase+'.sh')) in detail.get('submit_line','')
            and detail.get('command')==str(root/(phase+'.sh'))
            and detail.get('name')=='qwen-price-tuning-'+phase,'HELD_OWNER_ARGV_NAME')
        require(f'gres/gpu={gpu}' in detail['tres_req_str'] and f'cpu={cpu}' in detail['tres_req_str'],
                'HELD_GPU_CPU')
        require(detail['requeue'] is False or detail['requeue']==0,'NO_REQUEUE')
        require(detail['memory_per_node']['number']==mem and detail['cpus_per_task']['number']==cpu,
                'HELD_HOST_MEMORY_CPU')
        require('--export=NONE' in detail['submit_line'] and detail['partition']=='gpu'
                and detail['qos']=='lab_gpu_s4','HELD_EXPORT_PARTITION_QOS')
        for dep in dependencies:
            kind,*ids=dep.split(':')
            require(all(kind+':'+j in detail['dependency'] for j in ids),'HELD_DEPENDENCY')
        held.append(job)
    write(root/'submission.json',dict(jobs=jobs,lock=member(root/'execution.lock.json'),
        status='HELD_INSPECTED',existing_jobs_mutated=False))
    for job in held:
        require(job_state(info(job))=='PENDING','HELD_STATE_BEFORE_RELEASE')
        command(['scontrol','release',job])
    snapshot=json.loads(command(['squeue','-j',','.join(held),'--json']))
    write(root/'postrelease.json',snapshot,limit=2*1024**2)
    print(json.dumps(dict(jobs=jobs,status='RELEASED',attempt=str(root))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    submit(p.parse_args().attempt)
