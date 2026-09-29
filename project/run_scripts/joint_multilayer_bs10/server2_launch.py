"""S2-only held array/CPU collector, exact cancellation binding before release."""
import argparse
import datetime
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from .server2_entry import bind, ROOT, NONCE, PYTHON
bind()
from .common import read,record,save,sha,require,validate_execution

def command(argv,cwd=None):return subprocess.check_output(argv,text=True,cwd=cwd).strip()
def write_new(path,text):
    with Path(path).open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())

def cancellation(cfg):
    r=cfg['cancellation'];require(sha(r['path'])==r['sha256'],'CANCELLATION_CHANGED')
    c=read(r['path']);require(c['nonce']==NONCE and c['active_pending_empty'],'S4_AUTHORITY')
    require({x['job_id'] for x in c['jobs']}=={'55091'}|{f'55090_{i}' for i in range(9)},'S4_JOB_SET')
    require(all(x['state'].startswith('CANCELLED') and x['elapsed_seconds']==0 and not x['allocated_tres'] for x in c['jobs']),'S4_ACTIVE')
    sub=read(cfg['s4_provenance']['submission.json']['path'])
    require(sub['array_job']=='55090' and sub['collector']=='55091','S4_SUBMISSION')
    require([(r['array_index'],r['checkpoint'],r['arm']) for r in sub['mapping']]==[(i,r['checkpoint'],r['arm']) for i,r in enumerate(cfg['trajectories'])],'S4_MAPPING')
    return record(r['path'])

def freeze(repo,attempt):
    repo=Path(repo).resolve();require(not command(['git','status','--porcelain'],repo),'DIRTY_EXECUTION_SOURCE')
    cfg=read(ROOT/'preflight/configuration.json');validate_execution(cfg);cr=cancellation(cfg)
    dest=ROOT/attempt;dest.mkdir(exist_ok=False);source=dest/'source'
    namespaces=['project/run_scripts/joint_multilayer_bs10','plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1',
        'plans/global/2026-09-29-joint-multilayer-preservation-method-ko.md',
        'messages/head/2026-09-29-joint-multilayer-bs1-sh2-migration.md','messages/head/2026-09-29-joint-multilayer-bs10-sh4.md',
        'transfers/approvals/2026-09-29-joint-multilayer-bs1-s4-to-s2.json',
        'audits/global/2026-09-29-joint-multilayer-bs10-sh4-dispatch/input-manifest.json',
        'audits/global/2026-09-22-alphaedit-original-failure-audit/official/AlphaEdit/AlphaEdit_main.py']
    tracked=command(['git','ls-files',*namespaces],repo).splitlines();members=[]
    for name in tracked:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,p);members.append(record(p))
    with tarfile.open(dest/'source.tar','x') as tf:
        for name in tracked:tf.add(source/name,arcname=name,recursive=False)
    cfg['design']=str(source/'plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1')
    cfg['official_native']['path']=str(source/'audits/global/2026-09-22-alphaedit-original-failure-audit/official/AlphaEdit/AlphaEdit_main.py')
    cf=save(dest/'configuration.json',cfg)
    lock=dict(instruction_id=NONCE,attempt=str(dest),source_root=str(source),source=command(['git','rev-parse','HEAD'],repo),
        source_tree=command(['git','rev-parse','HEAD^{tree}'],repo),source_members=members,archive=record(dest/'source.tar'),
        configuration=cf,output=str(dest/'output'),cancellation=cr,
        resources=dict(node='server2',partition='gpu',array='0-8%2',gpus_per_job=1,cpus=8,host_mem_mib=60416,
            task_cap=2,project_cap=2,export='NONE',Requeue=0,wall=cfg['resources']['wall']),
        snapshot_policy=cfg['contract']['weight_snapshots'],automatic_retry=False,monitoring_after_handoff=False,
        initial_gate='representative joint accepted or normal reject B1 -> B2 W/M/anchor/dual/RNG; OR all9 released verified resource pending',
        NO_BROADCAST_NOT_REQUIRED='S2 original CP/model; only approved S4 small handoff received')
    save(dest/'execution.lock.json',lock);(dest/'output').mkdir()
    common='#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --nodelist=server2\n#SBATCH --partition=gpu\n#SBATCH --cpus-per-task=8\n#SBATCH --export=NONE\n#SBATCH --no-requeue\n'
    env='set -euo pipefail\nexport OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\ncd '+str(source)+'\n'
    entry=PYTHON+' -B -m project.run_scripts.joint_multilayer_bs10.server2_entry '
    write_new(dest/'branch.sbatch',common+'#SBATCH --gres=gpu:1\n#SBATCH --mem=60416M\n#SBATCH --time='+cfg['resources']['wall']+'\n'+env+
        'exec '+entry+'run --lock '+str(dest/'execution.lock.json')+' --index "$SLURM_ARRAY_TASK_ID"\n')
    write_new(dest/'collector.sbatch',common+'#SBATCH --mem=24576M\n#SBATCH --time=04:00:00\n'+env+'export CUDA_VISIBLE_DEVICES=""\n'+
        'exec '+entry+'collect --lock '+str(dest/'execution.lock.json')+'\n')
    save(dest/'frozen.json',dict(lock=record(dest/'execution.lock.json'),branch_script=record(dest/'branch.sbatch'),collector_script=record(dest/'collector.sbatch')))
    for p in source.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print(dest)

def submit(dest):
    dest=Path(dest);require(not (dest/'submitted-array.json').exists(),'DUPLICATE_SUBMIT')
    lock=read(dest/'execution.lock.json');cfg=read(lock['configuration']['path']);cancellation(cfg)
    require(sha(lock['configuration']['path'])==lock['configuration']['sha256'],'CONFIG_CHANGED')
    for member in lock['source_members']:require(sha(member['path'])==member['sha256'],'SOURCE_CHANGED')
    frozen=read(dest/'frozen.json')
    for key in ('branch_script','collector_script'):require(sha(frozen[key]['path'])==frozen[key]['sha256'],'SCRIPT_CHANGED')
    q=command(['squeue','-h','-u','janghj','-w','server2','-o','%i|%j|%T|%b|%R'])
    require(not q,'NONEMPTY_QUEUE_ADMISSION_REQUIRES_DEPENDENCY')
    require(shutil.disk_usage(ROOT).free>=cfg['resources']['output_reserve_bytes'],'DISK_RESERVE')
    save(dest/'admission.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_queue=q,
        node=command(['scontrol','show','node','server2']),partition=command(['scontrol','show','partition','gpu']),
        disk=shutil.disk_usage(ROOT)._asdict(),project_cap=2,task_cap=2))
    argv=['sbatch','--parsable','--hold','--array=0-8%2','--job-name=odeedit_joint_bs1_s2',
        '--output='+str(dest/'%A_%a.out'),'--error='+str(dest/'%A_%a.err'),str(dest/'branch.sbatch')]
    job=command(argv).split(';')[0];require(job.isdigit(),'SBATCH_ID');save(dest/'submitted-array.json',dict(job_id=job,argv=argv,held=True))
    info=command(['scontrol','show','job','-o',job])
    for token in ('UserId=janghj(','JobState=PENDING','Reason=JobHeldUser','Requeue=0','NumCPUs=8','gres/gpu=1','ReqNodeList=server2','Partition=gpu',str(dest/'branch.sbatch'),'ArrayTaskId=0-8%2','Dependency=(null)'):
        require(token in info,'HELD_INSPECTION:'+token)
    require('MinMemoryNode=59G' in info or 'MinMemoryNode=60416M' in info,'MEMORY')
    argv2=['sbatch','--parsable','--hold','--dependency=afterany:'+job,'--job-name=odeedit_joint_collect_s2',
        '--output='+str(dest/'%j-collector.out'),'--error='+str(dest/'%j-collector.err'),str(dest/'collector.sbatch')]
    collector=command(argv2).split(';')[0];require(collector.isdigit(),'COLLECTOR_ID');save(dest/'submitted-collector.json',dict(job_id=collector,argv=argv2,held=True))
    ci=command(['scontrol','show','job','-o',collector])
    for token in ('UserId=janghj(','JobState=PENDING','Requeue=0','NumCPUs=8','ReqNodeList=server2',str(dest/'collector.sbatch'),'afterany:'+job):require(token in ci,'COLLECTOR:'+token)
    require('MinMemoryNode=24G' in ci and 'gres/gpu=' not in ci,'COLLECTOR_RESOURCE')
    mapping=[dict(array_index=i,job_id=f'{job}_{i}',**spec,edits=100,output=str(Path(lock['output'])/spec['name'])) for i,spec in enumerate(cfg['trajectories'])]
    save(dest/'submission.json',dict(array_job=job,collector=collector,mapping=mapping,held_inspection=[info,ci],lock=record(dest/'execution.lock.json'),all_9_registered=True))
    cancellation(cfg)
    for j in (job,collector):
        subprocess.run(['scontrol','release',j],check=True);save(dest/f'release-{j}.json',dict(job_id=j,returncode=0))
    save(dest/'release.json',dict(array_job=job,collector=collector,status='RELEASED',all_9_registered=True,initial='NOT_OBSERVED'))
    print(job,collector,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo');p.add_argument('--attempt');p.add_argument('--submit');a=p.parse_args()
    if a.submit:submit(a.submit)
    else:freeze(a.repo,a.attempt)
