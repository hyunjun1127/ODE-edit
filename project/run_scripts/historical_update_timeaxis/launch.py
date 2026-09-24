"""Immutable source/lock, held exact inspection, two bounded workers and CPU join."""
import argparse
import datetime
import shutil
import subprocess
import tarfile
from .common import *

def command(argv):return subprocess.check_output(argv,text=True).strip()

def freeze(attempt):
    dest=ROOT/attempt;dest.mkdir(parents=True,exist_ok=False)
    source=dest/'source';namespace=Path('project/run_scripts/historical_update_timeaxis')
    # Copy only this approved namespace and required fixed-dataset verifier, immutable commit first.
    tracked=command(['git','ls-files',str(namespace),'scripts/fixed_counterfact.py']).splitlines()
    assert tracked and not command(['git','status','--porcelain','--',str(namespace)])
    for name in tracked:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,p)
    members=[record(source/n) for n in tracked];sourcedigest=digest([{**m,'path':str(Path(m['path']).relative_to(source))} for m in members])
    archive=dest/'source.tar'
    with tarfile.open(archive,'x') as tf:
        for name in tracked:tf.add(source/name,arcname=name,recursive=False)
    t0=ROOT/'inputs/t0-v1/runtime-binding.json';bind=read(t0)
    st=shutil.disk_usage(ROOT);assert st.free>=20*2**30,('STORAGE_BLOCKED',st.free)
    lock=dict(instruction_id=INSTRUCTION,attempt=attempt,output=str(dest/'output'),source_commit=command(['git','rev-parse','HEAD']),source_tree=command(['git','rev-parse','HEAD^{tree}']),
        source_sha256=sourcedigest,source_members=members,archive=record(archive),T0=record(t0),T0_sha256=sha(t0),token_sha256=bind['token_manifest']['sha256'],contract_sha256=sha(DESIGN/'experiment-contract.json'),
        evaluator_signature=dict(evaluator=sha(DESIGN/'source-evidence/alphaedit_strength_neutral_barrier__evaluator.py'),contracts=sha(DESIGN/'source-evidence/alphaedit_strength_neutral_barrier__contracts.py'),token_manifest=bind['token_manifest']['sha256'],
            torch='2.9.1+cu128',transformers='4.44.2',precision='FP32/eager/autocastOFF/matmulTF32OFF/cuDNN_TF32ON',microbatch=16,model_shards=bind['model_shards']),
        resources=dict(gpus_per_job=1,cpus_per_job=8,mem_mib=60416,task_cap=2,project_cap=2,wall='7-00:00:00',wall_basis='conservative partition-bounded registration; pilot padded-token forecast pending, not measured',host_peak_estimate_gib=44,host_components='32GiB transient CPU model load + 1.094GiB W0 + <=5 selected mmap endpoints + <3GiB per-layer FP64 scratch + <3GiB Python/input/allocator; steady CPU far below loading peak',disk_reserve_gib=20,free_observed_bytes=st.free),
        barrier_timeout_seconds=6*86400,save_checkpoints=False,native_fitting=0,history_append=0,scope=['T0','T1','T2P','T2F','T3A','T3B','T4'],
        science_effect_gate=False,automatic_scientific_retry=False,NO_BROADCAST_NOT_REQUIRED='same-host readonly inputs and local scores; no remote payload consumer required')
    save(dest/'execution.lock.json',lock)
    # Generated script is task-owned derived source; explicit job-total memory and clean env.
    script=dest/'gpu.sbatch'
    script.write_text('#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --nodelist=server4\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=60416M\n#SBATCH --time=7-00:00:00\n#SBATCH --export=NONE\n#SBATCH --no-requeue\nset -euo pipefail\nexport OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1\ncd '+str(source)+'\nexec /data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.historical_update_timeaxis.worker --lock '+str(dest/'execution.lock.json')+' --family "$1"\n')
    cpu=dest/'collector.sbatch';cpu.write_text('#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --nodelist=server4\n#SBATCH --partition=gpu\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=24576M\n#SBATCH --time=04:00:00\n#SBATCH --export=NONE\n#SBATCH --no-requeue\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1\ncd '+str(source)+'\nexec /data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.historical_update_timeaxis.reduce --lock '+str(dest/'execution.lock.json')+'\n')
    save(dest/'source-and-lock-receipt.json',dict(lock=record(dest/'execution.lock.json'),gpu_script=record(script),cpu_script=record(cpu),archive=record(archive)))
    return dest

def submit(dest):
    assert not (dest/'submission.json').exists(),'DUPLICATE_SUBMISSION'
    queue=command(['squeue','-u','janghj','-w','server4','-h','-o','%i|%j|%T|%b|%R'])
    # Do not race admitted project capacity. Exact owner queue must be empty here.
    assert not queue,('RESOURCE_ADMISSION_REQUIRES_EXACT_DEPENDENCY',queue)
    save(dest/'admission.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=queue,node=command(['scontrol','show','node','server4']),disk=shutil.disk_usage(ROOT)._asdict(),project_task_cap=2))
    jobs={};inspections=[]
    for family in FAMILIES:
        job=command(['sbatch','--parsable','--hold','--job-name=odeedit_hist_'+('alpha' if family==FAMILIES[0] else 'memit')+'_s4','--output='+str(dest/('%j.out')),'--error='+str(dest/('%j.err')),str(dest/'gpu.sbatch'),family]).split(';')[0]
        assert job.isdigit();jobs[family]=job
        save(dest/('submitted-'+family+'.json'),dict(job_id=job,source=record(dest/'execution.lock.json'),held=True))
        info=command(['scontrol','show','job','-o',job]);assert 'UserId=janghj(' in info and 'JobState=PENDING' in info and 'Dependency=(null)' in info and 'Requeue=0' in info
        assert 'MinMemoryNode=59G' in info or 'MinMemoryNode=60416M' in info
        assert 'NumCPUs=8' in info and str(dest/'gpu.sbatch') in info and 'gres/gpu=1' in info;inspections.append(dict(job_id=job,scontrol=info))
    collector=command(['sbatch','--parsable','--hold','--job-name=odeedit_hist_T4_s4','--dependency=afterany:'+':'.join(jobs.values()),'--output='+str(dest/('%j.out')),'--error='+str(dest/('%j.err')),str(dest/'collector.sbatch')]).split(';')[0]
    ci=command(['scontrol','show','job','-o',collector]);assert 'UserId=janghj(' in ci and 'Requeue=0' in ci and 'NumCPUs=8' in ci and 'afterany:' in ci;inspections.append(dict(job_id=collector,scontrol=ci))
    save(dest/'submission.json',dict(family_jobs=jobs,collector=collector,held_inspection=inspections,source=record(dest/'execution.lock.json'),internal_DAG='matching atomic PASS joins across both persistent family workers; collector only reduces after all science PASS',status='HELD_INSPECTION_COMPLETE'))
    for job in list(jobs.values())+[collector]:subprocess.run(['scontrol','release',job],check=True)
    snap=command(['squeue','-j',','.join(list(jobs.values())+[collector]),'-h','-o','%i|%j|%T|%R|%b'])
    save(dest/'release.json',dict(job_ids=list(jobs.values())+[collector],status='RELEASED',snapshot=snap,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(family_jobs=jobs,collector=collector,lock=record(dest/'execution.lock.json'),snapshot=snap)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',default='attempt-v1');p.add_argument('--submit',action='store_true');args=p.parse_args();dest=freeze(args.attempt)
    if args.submit:submit(dest)
    else:print(dest)
