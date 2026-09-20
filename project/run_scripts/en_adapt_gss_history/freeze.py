"""Freeze both jobs, inspect both while held, release; zero subsequent queries."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time
from .prepare import BASE, ARMS
from .io import save


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    repo=Path(__file__).resolve().parents[3]
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise ValueError('CLEAN_COMMITTED_SOURCE_REQUIRED')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip()
    frozen=BASE/'frozen-source-v1';frozen.mkdir(exist_ok=False)
    source=frozen/'source';source.mkdir()
    prefixes=['project/run_scripts','scripts',
        'audits/global/2026-09-20-en-adaptive-nullspace-design-v1',
        'audits/global/2026-09-20-en-adapt-gss-history-design-v1']
    files=subprocess.check_output(['git','ls-files',*prefixes],cwd=repo,text=True).splitlines()
    files=[p for p in files if Path(p).suffix in ('.py','.json','.sbatch')]
    members=[]
    with tarfile.open(frozen/'source.tar.gz','x:gz') as archive:
        for name in files:
            p=repo/name
            if p.is_symlink():
                raise ValueError('SOURCE_SYMLINK')
            data=p.read_bytes();dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as stream:stream.write(data)
            archive.add(p,arcname=name,recursive=False)
            members.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    execution=dict(commit=commit,tree=tree,source=str(source),members=members,
        archive=dict(path=str(frozen/'source.tar.gz'),bytes=(frozen/'source.tar.gz').stat().st_size,sha256=sha(frozen/'source.tar.gz')))
    preparation=BASE/'preparation-v1'
    resource_plan=json.loads((preparation/'resource-plan.json').read_text())
    reference=dict(path=str(preparation/'reference-stat-binding.json'),sha256=sha(preparation/'reference-stat-binding.json'))
    map_seal=dict(path=str(preparation/'fixed-map-logical-seal.json'),sha256=sha(preparation/'fixed-map-logical-seal.json'))
    authority=[]
    for receipt_name in ('authority-full-read.json','twoarm-full-read.json'):
        p=repo/'audits/servers/server3/2026-09-20-en-adapt-gss-history'/receipt_name
        authority.append(dict(path=str(p.relative_to(repo)),sha256=sha(p)))
    outputs=[]
    for arm in ARMS:
        attempt=BASE/arm.removeprefix('EN_ADAPT_H_').lower()/'attempt-v1'
        sealed=attempt/'execution-inputs';sealed.mkdir(exist_ok=False)
        inputs={}
        for p in sorted((attempt/'inputs').glob('*.json')):
            save(sealed/p.name,json.loads(p.read_text()));inputs[p.name]=sha(sealed/p.name)
        lock=dict(instruction='ODEEDIT-S06-EN-ADAPT-GSS-HISTORY-10K-SH3-V1',
            nonce='ODEEDIT-GH-SH3-GSS-TWOARMS-CAP2-20260920-R1',
            authority_main='7216cc96c0436fcbd1d3b7cba2c69c09d613bd93',authority_receipts=authority,
            arm=arm,session='01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3',
            execution=execution,input_seals=inputs,reference_stat_binding=reference,map_seal=map_seal,
            sequence_identity=dict(path=str(preparation/'sequence-token-identity.json'),sha256=sha(preparation/'sequence-token-identity.json')),
            resource_plan=dict(path=str(preparation/'resource-plan.json'),sha256=sha(preparation/'resource-plan.json')),
            resources=dict(GPU=1,CPU=8,mem_MiB=121856,node='ubuntu',partition='gpu',
                export='NONE',requeue=0,wall_hours=168,project_cap=2,task_cap=2,
                aggregate_new_GPU=2,aggregate_CPU=16,aggregate_mem_MiB=243712),
            scope=dict(batches=100,batch_size=100,entry='FRESH_W0_ZERO_M4',
                independent_job=True,cross_job_edited_state_or_teacher_sharing=False,
                B2='CONTINUE_SAME_RAM_NO_QUALITY_GATE',maximum_candidates=200),
            save_checkpoints=False,exact_crash_resume='NOT_AVAILABLE',
            parent_actual_execution='5d452221288f3b924e1737578f11aaa654594422',
            parent_analysis='abd08ee1dde24ca975aace49c925a4ba00211004',
            post_release_monitoring=False,actual_initial='NOT_OBSERVED',actual_terminal='NOT_OBSERVED',
            output=str(attempt/'output'))
        save(attempt/'execution.lock.json',lock)
        outputs.append(dict(arm=arm,attempt=str(attempt),lock_sha256=sha(attempt/'execution.lock.json')))
    save(BASE/'frozen-source-v1/freeze-receipt.json',dict(execution=execution,arms=outputs,
        required_free_bytes=resource_plan['storage']['required_free_bytes']))
    print(json.dumps(dict(source=commit,tree=tree,arms=outputs)))


def held_checks(text,command,source,attempt,job_name):
    def field(name):
        m=re.search(r'(?:^|\s)'+re.escape(name)+r'=(.*?)(?=\s+[A-Za-z][A-Za-z0-9_]*=|$)',text)
        return m.group(1) if m else None
    return dict(owner=field('UserId')=='janghj(1025)',name=field('JobName')==job_name,
        held=field('JobState')=='PENDING' and field('Reason')=='JobHeldUser',
        GPU='gres/gpu=1' in (field('ReqTRES') or ''),CPU=field('NumCPUs')=='8',
        memory=field('MinMemoryNode') in ('121856M','119G'),
        partition=field('Partition')=='gpu',time_limit=field('TimeLimit')=='7-00:00:00',
        node=field('ReqNodeList')=='ubuntu',requeue=field('Requeue')=='0',
        dependency=field('Dependency')=='(null)',
        command=field('Command')==str(source/'project/run_scripts/en_adapt_gss_history/run.sbatch'),
        args=field('SubmitLine') is not None and shlex.split(field('SubmitLine'))==command,
        cwd=field('WorkDir')==str(source),
        exact_attempt=command[-2:]==[str(attempt),str(source)],export_NONE='--export=NONE' in command)


def project_capacity(queue):
    patterns=('odeedit_*','bfode_*','motivation_*','session01_*','project_*')
    admitted=[]
    for line in queue.splitlines():
        job,name,state,gres=line.split('|',3)
        if not any(fnmatch.fnmatchcase(name,p) for p in patterns):
            continue
        if not re.fullmatch(r'\d+(?:_\d+)?',job):
            raise ValueError('UNRESOLVED_OWN_PROJECT_ARRAY_CAPACITY')
        match=re.fullmatch(r'(?:gres/)?gpu(?::[^:,]+)?:(\d+)',gres)
        if not match:
            raise ValueError('UNRESOLVED_OWN_PROJECT_GPU_CAPACITY')
        admitted.append(dict(job_id=job,name=name,state=state,GPU=int(match.group(1))))
    return admitted


def submit():
    attempts=[BASE/arm.removeprefix('EN_ADAPT_H_').lower()/'attempt-v1' for arm in ARMS]
    if any((a/'submission.json').exists() for a in attempts):
        raise ValueError('DUPLICATE_OR_PARTIAL_REGISTRATION_REQUIRES_EXACT_HANDOFF')
    locks=[json.loads((a/'execution.lock.json').read_text()) for a in attempts]
    source=Path(locks[0]['execution']['source'])
    if any(lock['execution']!=locks[0]['execution'] for lock in locks):
        raise ValueError('TWO_JOB_SOURCE_MISMATCH')
    archive=locks[0]['execution']['archive']
    if sha(archive['path'])!=archive['sha256']:
        raise ValueError('SOURCE_ARCHIVE_SHA')
    for member in locks[0]['execution']['members']:
        p=source/member['path']
        if p.stat().st_size!=member['bytes'] or sha(p)!=member['sha256']:
            raise ValueError('FROZEN_SOURCE_CHANGED')
    for attempt,lock in zip(attempts,locks):
        for name,value in lock['input_seals'].items():
            if sha(attempt/'execution-inputs'/name)!=value:
                raise ValueError('FROZEN_INPUT_CHANGED')
    for lock in locks:
        for name in ('resource_plan','reference_stat_binding','map_seal','sequence_identity'):
            binding=lock[name]
            if sha(binding['path'])!=binding['sha256']:
                raise ValueError('SEALED_EXTERNAL_INPUT_CHANGED:'+name)
    resource_plan=json.loads(Path(locks[0]['resource_plan']['path']).read_text())
    stat=os.statvfs(BASE);required=resource_plan['storage']['required_free_bytes']
    if stat.f_bavail*stat.f_frsize<required or stat.f_favail<150000:
        raise RuntimeError('TWO_JOB_STORAGE_ADMISSION_BLOCK')
    # Single resource-only admission snapshot for the whole two-job bundle.
    queue=subprocess.check_output(['squeue','-h','-u','janghj','-w','ubuntu','-o','%i|%j|%T|%b'],text=True)
    others=project_capacity(queue)
    if sum(x['GPU'] for x in others)+2>2:
        save(BASE/'admission-blocked.json',dict(others=others,requested_GPU=2,cap=2,
            status='PRE_SUBMISSION_RESOURCE_BLOCK',registered=0))
        raise RuntimeError('PROJECT_CAP2_PRE_SUBMISSION_BLOCK_NO_WAIT')
    node=subprocess.check_output(['scontrol','show','node','ubuntu','--oneliner'],text=True)
    extract=lambda key:int(re.search(r'(?:^|\s)'+key+r'=(\d+)',node).group(1))
    if extract('RealMemory')<243712 or extract('CPUTot')<16:
        raise RuntimeError('AGGREGATE_NODE_CAPACITY_BLOCK')
    save(BASE/'admission.json',dict(time=time.time(),resource_only_queue=queue.splitlines(),
        own_project_other_admitted=others,requested_GPU=2,project_cap=2,task_cap=2,
        requested_CPU=16,requested_memory_MiB=243712,node_metadata=node,
        free_bytes=stat.f_bavail*stat.f_frsize,required_free_bytes=required,
        existing_jobs_modified=0,existing_model_reference_copied=0,post_release_monitoring=False))
    jobs=[]
    # Both exact held inspections happen BEFORE either release.
    for attempt,lock in zip(attempts,locks):
        name='odeedit_gss_'+lock['arm'].removeprefix('EN_ADAPT_H_').lower()+'_10k_s3'
        (attempt/'logs').mkdir(exist_ok=False)
        command=['sbatch','--parsable','--hold','--export=NONE',f'--job-name={name}',
            f'--output={attempt}/logs/%j.out',f'--error={attempt}/logs/%j.err',
            str(source/'project/run_scripts/en_adapt_gss_history/run.sbatch'),str(attempt),str(source)]
        job=subprocess.check_output(command,cwd=source,text=True).strip().split(';')[0]
        if not re.fullmatch(r'\d+',job):raise ValueError('SBATCH_JOB_ID_SCHEMA')
        save(attempt/'submission.json',dict(job_id=job,arm=lock['arm'],argv=command,held=True,
            source=lock['execution']['commit'],lock_sha256=sha(attempt/'execution.lock.json')))
        state=subprocess.check_output(['scontrol','show','job',job,'--oneliner'],text=True)
        checks=held_checks(state,command,source,attempt,name)
        save(attempt/'held-inspection.json',dict(job_id=job,scontrol=state,checks=checks))
        if not all(checks.values()):raise RuntimeError('HELD_EXACT_INSPECTION_FAILED_NO_RELEASE')
        jobs.append((attempt,lock,job))
    for attempt,lock,job in jobs:
        command=['scontrol','release',job]
        result=subprocess.run(command,capture_output=True,text=True,check=True)
        # NO scheduler/log/result/availability queries or waits below this point.
        save(attempt/'release.json',dict(job_id=job,arm=lock['arm'],released=True,argv=command,
            returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
            source=lock['execution']['commit'],lock_sha256=sha(attempt/'execution.lock.json'),
            status='MONITORING_PAUSED_AWAITING_USER',actual_initial='NOT_OBSERVED',
            actual_terminal='NOT_OBSERVED',monitoring=False))
    print(json.dumps(dict(jobs=[dict(arm=lock['arm'],job_id=job) for _,lock,job in jobs],
        source=locks[0]['execution']['commit'],status='MONITORING_PAUSED_AWAITING_USER',
        actual_initial='NOT_OBSERVED',actual_terminal='NOT_OBSERVED')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args()
    (submit if a.submit else freeze)()
