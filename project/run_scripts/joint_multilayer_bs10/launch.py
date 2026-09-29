"""Create-once source/lock; held array0-8%2 and afterany CPU collector."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from .common import *


def command(args,cwd=None):return subprocess.check_output(args,text=True,cwd=cwd).strip()


def write_new(path,content):
    with Path(path).open('x') as f:f.write(content);f.flush();os.fsync(f.fileno())


def freeze(repo,attempt,configuration):
    repo=Path(repo).resolve();dest=ROOT/attempt;dest.mkdir(exist_ok=False);source=dest/'source'
    namespace='project/run_scripts/joint_multilayer_bs10'
    require(not command(['git','status','--porcelain'],repo),'SOURCE_WORKTREE_NOT_CLEAN')
    tracked=command(['git','ls-files',namespace,'plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1',
        'plans/global/2026-09-29-joint-multilayer-preservation-method-ko.md',
        'messages/head/2026-09-29-joint-multilayer-bs10-sh4.md',
        'audits/global/2026-09-29-joint-multilayer-bs10-sh4-dispatch/input-manifest.json',
        'audits/global/2026-09-22-alphaedit-original-failure-audit/official/AlphaEdit/AlphaEdit_main.py'],repo).splitlines()
    require(bool(tracked),'NO_SOURCE')
    for name in tracked:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target)
    members=[dict(path=name,sha256=sha(source/name),bytes=(source/name).stat().st_size) for name in tracked]
    archive=dest/'source.tar'
    with tarfile.open(archive,'x') as tf:
        for name in tracked:tf.add(source/name,arcname=name,recursive=False)
    config=read(configuration)
    validate_execution(config)
    require(sha(config['user_override']['authority']['path'])==config['user_override']['authority']['sha256'],'USER_OVERRIDE_EVIDENCE')
    shutil.copyfile(config['user_override']['authority']['path'],dest/'user-override.md')
    config['user_override']['authority']=record(dest/'user-override.md')
    config['design']=str(source/'plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1')
    require((Path(config['design'])/'contract.json').is_file(),'FROZEN_DESIGN_MISSING')
    config['official_native']['path']=str(source/'audits/global/2026-09-22-alphaedit-original-failure-audit/official/AlphaEdit/AlphaEdit_main.py')
    config['settings'].update(snapshot_answer_logp='BITWISE_SAME_LAYOUT')
    cfg=save(dest/'configuration.json',config)
    require(shutil.disk_usage(ROOT).free>=config['resources']['output_reserve_bytes'],'STORAGE_BLOCKED')
    lock=dict(instruction_id=NONCE,attempt=str(dest),source_root=str(source),
        source=command(['git','rev-parse','HEAD'],repo),source_tree=command(['git','rev-parse','HEAD^{tree}'],repo),
        source_members=[dict(r,path=str(source/r['path'])) for r in members],archive=record(archive),configuration=cfg,output=str(dest/'output'),
        resources=dict(gpus_per_job=1,cpus=8,host_mem_mib=60416,array='0-8%2',task_cap=2,project_cap=2,
            node='server4',partition='gpu',export='NONE',Requeue=0,wall=config['resources']['wall']),
        snapshot_policy=config['contract']['weight_snapshots'],automatic_retry=False,
        user_override=config['user_override'],
        initial_gate='9 trajectories released AND representative joint B1 accepted or normal rejection -> B2 state/anchor/dual/RNG entry; native history5 separate; OR verified GPU resource pending',
        monitoring_after_handoff=False,NO_BROADCAST_NOT_REQUIRED='same-host existing parent assets; no downstream remote raw consumer')
    lr=save(dest/'execution.lock.json',lock)
    for p in source.rglob('*'):
        if p.is_file():p.chmod(0o444)
    (dest/'output').mkdir()
    common='#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --nodelist=server4\n#SBATCH --partition=gpu\n#SBATCH --cpus-per-task=8\n#SBATCH --export=NONE\n#SBATCH --no-requeue\n'
    env='set -euo pipefail\nexport OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1\ncd '+str(source)+'\n'
    write_new(dest/'branch.sbatch',common+'#SBATCH --gres=gpu:1\n#SBATCH --mem=60416M\n#SBATCH --time='+config['resources']['wall']+'\n'+env+
        'exec '+PYTHON+' -m project.run_scripts.joint_multilayer_bs10.runner --lock '+str(dest/'execution.lock.json')+' --index "$SLURM_ARRAY_TASK_ID"\n')
    write_new(dest/'collector.sbatch',common+'#SBATCH --mem=24576M\n#SBATCH --time=04:00:00\n'+env+'export CUDA_VISIBLE_DEVICES=""\n'+
        'exec '+PYTHON+' -m project.run_scripts.joint_multilayer_bs10.reduce --lock '+str(dest/'execution.lock.json')+'\n')
    save(dest/'frozen.json',dict(lock=lr,branch_script=record(dest/'branch.sbatch'),collector_script=record(dest/'collector.sbatch')))
    return dest


def submit(dest):
    dest=Path(dest);require(not (dest/'submission.json').exists(),'DUPLICATE_SUBMIT')
    queue=command(['squeue','-h','-u','janghj','-w','server4','-o','%i|%j|%T|%b|%R'])
    # Conservative admission: never race an unrelated already-admitted user job.
    require(not queue,'NONEMPTY_OWNER_QUEUE_REQUIRES_EXACT_CAP_SAFE_DEPENDENCY')
    lock=read(dest/'execution.lock.json');cfg=read(lock['configuration']['path'])
    require(shutil.disk_usage(ROOT).free>=cfg['resources']['output_reserve_bytes'],'STORAGE_BLOCKED')
    save(dest/'admission.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_queue=queue,
        node=command(['scontrol','show','node','server4']),partition=command(['scontrol','show','partition','gpu']),
        disk=shutil.disk_usage(ROOT)._asdict(),project_cap=2,task_cap=2))
    argv=['sbatch','--parsable','--hold','--array=0-8%2','--job-name=odeedit_joint_bs1_s4',
        '--output='+str(dest/'%A_%a.out'),'--error='+str(dest/'%A_%a.err'),str(dest/'branch.sbatch')]
    job=command(argv).split(';')[0];require(job.isdigit(),'SBATCH_RESPONSE')
    save(dest/'submitted-array.json',dict(job_id=job,argv=argv,held=True))
    info=command(['scontrol','show','job','-o',job]);
    for token in ('UserId=janghj(','JobState=PENDING','Reason=JobHeldUser','Requeue=0','NumCPUs=8','gres/gpu=1','ReqNodeList=server4','Partition=gpu',str(dest/'branch.sbatch')):
        require(token in info,'HELD_INSPECTION:'+token)
    require('MinMemoryNode=59G' in info or 'MinMemoryNode=60416M' in info,'HELD_MEMORY')
    require('ArrayTaskId=0-8%2' in info and 'Dependency=(null)' in info,'ARRAY_THROTTLE')
    argv2=['sbatch','--parsable','--hold','--dependency=afterany:'+job,'--job-name=odeedit_joint_collect_s4',
        '--output='+str(dest/'%j-collector.out'),'--error='+str(dest/'%j-collector.err'),str(dest/'collector.sbatch')]
    collector=command(argv2).split(';')[0];require(collector.isdigit(),'COLLECTOR_RESPONSE')
    save(dest/'submitted-collector.json',dict(job_id=collector,argv=argv2,held=True))
    ci=command(['scontrol','show','job','-o',collector])
    for token in ('UserId=janghj(','JobState=PENDING','Requeue=0','NumCPUs=8','ReqNodeList=server4',str(dest/'collector.sbatch'),'afterany:'+job):
        require(token in ci,'COLLECTOR_INSPECTION:'+token)
    require('MinMemoryNode=24G' in ci and 'gres/gpu=' not in ci,'COLLECTOR_RESOURCE')
    mapping=[dict(array_index=i,job_id=f'{job}_{i}',**spec,edits=100,output=str(Path(lock['output'])/spec['name'])) for i,spec in enumerate(cfg['trajectories'])]
    save(dest/'submission.json',dict(array_job=job,collector=collector,mapping=mapping,held_inspection=[info,ci],
        lock=record(dest/'execution.lock.json'),status='HELD_INSPECTION_COMPLETE',all_9_registered=True))
    for j in (job,collector):
        subprocess.run(['scontrol','release',j],check=True)
        save(dest/f'release-{j}.json',dict(job_id=j,release_returncode=0))
    save(dest/'release.json',dict(array_job=job,collector=collector,status='RELEASED',all_9_registered=True,initial='NOT_OBSERVED'))
    print(dict(array_job=job,collector=collector,initial='NOT_OBSERVED'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo');p.add_argument('--attempt');p.add_argument('--configuration');p.add_argument('--submit-existing')
    a=p.parse_args()
    if a.submit_existing:submit(a.submit_existing)
    else:print(freeze(a.repo,a.attempt,a.configuration))
