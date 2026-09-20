"""Source/input freeze and resource-only held inspection for one S4 lane."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
from .prepare_server4 import sha, write, BASE


def freeze(attempt):
    attempt=Path(attempt).absolute();repo=Path(__file__).resolve().parents[3]
    if not attempt.is_relative_to(BASE):raise ValueError('SCOPE')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise ValueError('CLEAN_SOURCE_REQUIRED')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip()
    source=attempt/'source';source.mkdir(exist_ok=False)
    files=subprocess.check_output(['git','ls-files','project/run_scripts','scripts','audits/global/2026-09-20-en-adaptive-nullspace-design-v1'],cwd=repo,text=True).splitlines()
    files=[p for p in files if Path(p).suffix in ('.py','.json','.sbatch')]
    members=[]
    with tarfile.open(attempt/'source.tar.gz','x:gz') as archive:
        for name in files:
            p=repo/name
            if p.is_symlink():raise ValueError('SOURCE_SYMLINK')
            archive.add(p,arcname=name,recursive=False)
            dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(p.read_bytes())
            members.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p)))
    # Preserve preparation inputs; the execution copy binds frozen import paths.
    sealed=attempt/'execution-inputs';sealed.mkdir(exist_ok=False)
    mapping={}
    for p in (attempt/'inputs').glob('*.json'):
        value=json.loads(p.read_text())
        def relocate(x):
            if isinstance(x,str):return x.replace(str(repo)+'/',str(source)+'/').replace(str(attempt/'inputs')+'/',str(sealed)+'/')
            if isinstance(x,list):return [relocate(v) for v in x]
            if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
            return x
        write(sealed/p.name,relocate(value));mapping[p.name]=sha(sealed/p.name)
    manifest=json.loads((sealed/'manifest.json').read_text())
    gm=Path(manifest['generated_root'])/'manifest.json';generated=json.loads(gm.read_text())
    ref=[]
    for document in generated['documents']:
        for kind in ('capsule','keys','residual','logp'):
            d=document[kind];p=gm.parent/d['path'];st=p.stat()
            if st.st_size!=d['bytes']:raise ValueError('REFERENCE_SIZE_CHANGED')
            ref.append(dict(path=str(p),bytes=st.st_size,prior_sha256=d['sha256'],
                stat=[st.st_dev,st.st_ino,st.st_mtime_ns],verification='COMPLETED_PREP_SEAL_PLUS_CURRENT_SIZE_STAT'))
    write(sealed/'reference-stat-binding.json',ref)
    stat=os.statvfs(attempt)
    if stat.f_bavail*stat.f_frsize<4*2**30 or stat.f_favail<10000:raise RuntimeError('ACTUAL_STORAGE_RESERVE_4GIB')
    lock=dict(instruction='ODEEDIT-S06-EN-ADAPTIVE-NULLSPACE-B300-SH4-MIGRATION-V1',
        nonce='ODEEDIT-GH-SH4-EN-ADAPT-MIGRATION-20260920-R1',
        session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',node='server4',
        SH3_handoff='43904c13def0aa06bb37dcd7574df5df101db135',SH3_actual_submissions=0,
        execution=dict(commit=commit,tree=tree,source=str(source),members=members,
            archive=dict(path=str(attempt/'source.tar.gz'),sha256=sha(attempt/'source.tar.gz'))),
        input_seals=mapping,reference_stat_binding=sha(sealed/'reference-stat-binding.json'),
        resources=dict(GPU=1,CPU=8,mem_MiB=60416,node='server4',export='NONE',requeue=0,wall_hours=24,project_cap=2,task_cap=1,GPUhour_hardcap=None),
        scope=dict(maximum_batch=3,B1=['N4','EN_EXACT','EN_NUM','EN_ADAPT'],B2_B3=['N4','EN_EXACT','EN_ADAPT'],persistent_lane=True),
        save_checkpoints=False,exact_crash_resume='NOT_AVAILABLE',prior_waivers_inherited=False,
        memory_plan=dict(estimated_peak_GiB=52,measured=False,limit_GiB=59,reference_prefix_GiB=17,
            branch_P_basis_GiB=7,current_and_geometry_GiB=15,controller_history_GiB=7,allocator_margin_GiB=6,
            source_changes='release_current_oracle_before_SVD;lazy_default_M_allocation;no_scientific_reduction'),
        storage=dict(free_bytes=stat.f_bavail*stat.f_frsize,new_reserve_bytes=4*2**30,edited_checkpoint_bytes=0,exclusive_reserved=False),
        expected_counters=dict(native_batches=7,native_requests=700,reference_gradients=5,max_reference_candidates=14),
        no_broadcast='NO_BROADCAST_NOT_REQUIRED',output=str(attempt/'output'))
    write(attempt/'execution.lock.json',lock)
    print(json.dumps(dict(source=commit,tree=tree,archive=lock['execution']['archive'],lock_sha256=sha(attempt/'execution.lock.json'))))


def submit(attempt):
    attempt=Path(attempt).absolute();lock_path=attempt/'execution.lock.json';lock=json.loads(lock_path.read_text())
    if (attempt/'submission.json').exists():raise RuntimeError('DUPLICATE_SUBMISSION')
    source=Path(lock['execution']['source'])
    queue=subprocess.check_output(['squeue','-h','-u','janghj','-w','server4','-o','%i|%j|%T|%b'],text=True)
    from project.run_scripts.single_layer_mechanism_first.submit_program import other_capacity
    others=other_capacity(queue,excluded_jobs=())
    if sum(x['GPU'] for x in others)+1>2:raise RuntimeError('PROJECT_CAP_RESOURCE_BLOCK')
    if sha(lock['execution']['archive']['path'])!=lock['execution']['archive']['sha256']:raise ValueError('ARCHIVE_SHA')
    st=os.statvfs(attempt)
    if st.f_bavail*st.f_frsize<4*2**30:raise RuntimeError('ACTUAL_STORAGE_RESERVE')
    write(attempt/'admission.json',dict(time=time.time(),queue=queue.splitlines(),others=others,new_GPU=1,cap=2,task_cap=1,free_bytes=st.f_bavail*st.f_frsize))
    (attempt/'logs').mkdir(exist_ok=False)
    command=['sbatch','--parsable','--hold',f'--output={attempt}/logs/%j.out',f'--error={attempt}/logs/%j.err',str(source/'project/run_scripts/en_adaptive_nullspace/run-server4.sbatch'),str(attempt)]
    job=subprocess.check_output(command,cwd=source,text=True).strip().split(';')[0]
    if not re.fullmatch(r'\d+',job):raise ValueError('JOB_ID')
    write(attempt/'submission.json',dict(job_id=job,argv=command,lock_sha256=sha(lock_path),source=lock['execution']['commit'],held=True))
    state=subprocess.check_output(['scontrol','show','job',job,'--oneliner'],text=True)
    import shlex
    match=re.search(r'(?:^|\s)SubmitLine=(.*?)(?=\s+[A-Za-z][A-Za-z0-9_]*=|$)',state)
    checks=dict(owner='UserId=janghj(' in state,name='JobName=odeedit_en_adapt_B300_s4' in state,
        held='Reason=JobHeldUser' in state,GPU='gres/gpu=1' in state,CPU='NumCPUs=8' in state,
        memory=any(s in state for s in ('mem=60416M','mem=59G','MinMemoryNode=60416M')),
        node='ReqNodeList=server4' in state,requeue='Requeue=0' in state,dependency='Dependency=(null)' in state,
        command=str(source/'project/run_scripts/en_adaptive_nullspace/run-server4.sbatch') in state,
        argv=bool(match and shlex.split(match.group(1))==command),cwd=f'WorkDir={source}' in state)
    write(attempt/'held-inspection.json',dict(job_id=job,scontrol=state,checks=checks))
    if not all(checks.values()):raise RuntimeError('HELD_INSPECTION_FAIL_NO_RELEASE')
    subprocess.run(['scontrol','release',job],check=True)
    released=subprocess.check_output(['scontrol','show','job',job,'--oneliner'],text=True)
    write(attempt/'release.json',dict(job_id=job,released=True,scontrol=released,actual_T0='NOT_OBSERVED',actual_science='NOT_OBSERVED'))
    print(json.dumps(dict(job_id=job,released=True,source=lock['execution']['commit'],output=lock['output'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--submit',action='store_true');a=p.parse_args()
    (submit if a.submit else freeze)(a.attempt)
