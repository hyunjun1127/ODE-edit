"""CPU source/input freeze and one held/inspected cap1 queue. No polling."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime,timezone
from .common import ROOT,TASK,ARMS,EXPECTED,save,identity,sha,digest

PREFIX='project/run_scripts/local_z_adaptive_allocation/'
OLD=Path('/data/janghj/ODE-edit/local/ep-tw1-alpha-cap-sweep/20260915-v1/CAP10/attempt-v1/execution.lock.json')
ENV='messages/head/2026-09-16-sh4-local-z-adaptive-allocation.md'
MAN='plans/global/2026-09-16-local-z-adaptive-allocation-sh4-dispatch/source-input-manifest.json'

def command(args,**kw):
    r=subprocess.run(args,text=True,capture_output=True,**kw)
    if r.returncode:raise RuntimeError(dict(args=args,stdout=r.stdout,stderr=r.stderr,exit=r.returncode))
    return r.stdout.strip()

def git(w,*args):return command(['git','-C',str(w),*args])

def copied(source,target):
    p=Path(target);p.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb') as inp,p.open('xb') as out:shutil.copyfileobj(inp,out)
    assert sha(source)==sha(p);return identity(p)

def initialize(w):
    assert command(['hostname'])=='server4' and git(w,'rev-parse','HEAD')=='cee9447330e0eabe097775ca6185ea4c45cf267d'
    documents=[ENV,MAN,'PROTOCOL.md','control/gpu-concurrency-policy.tsv',
        'plans/global/2026-09-16-local-z-adaptive-allocation-sh4-dispatch.json',
        'plans/global/2026-09-16-local-z-adaptive-allocation-design-v1.md',
        'plans/global/2026-09-16-local-z-adaptive-allocation-contract-v1.json',
        'plans/global/2026-09-16-local-z-adaptive-allocation-cells-v1.csv',
        'audits/global/2026-09-16-local-z-adaptive-allocation/validate_design.py',
        'audits/global/2026-09-16-local-z-adaptive-allocation/checks.json',
        'plans/global/2026-09-13-low-cost-write-donor-pilot-design.md',
        'audits/global/2026-09-14-lowcost-seq10-review-ko.md',
        'plans/global/2026-09-15-bg-tw-reference-data-contract.json','plans/global/fixed-counterfact-10k-policy.md']
    refs=[]
    for rel in documents:
        refs.append(dict(copied(w/rel,ROOT/'authoritative'/rel),repo_path=rel,
            lines=len((w/rel).read_bytes().splitlines()),reading='FULL_READ'))
    old=json.loads(OLD.read_text())
    native=[]
    for rel in ('AlphaEdit/AlphaEdit_main.py','AlphaEdit/compute_z.py','AlphaEdit/compute_ks.py'):
        p=Path(old['blue_root'])/rel;native.append(dict(identity(p),reading='FULL_READ'))
    for rel in ('fitting.py','sequential_runtime.py'):
        p=w/'project/run_scripts/low_cost_write_donor_pilot'/rel;native.append(dict(identity(p),reading='FULL_READ'))
    save(ROOT/'full-read-receipt.json',dict(instruction_id=TASK,host='server4',session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',
        source_head=git(w,'rev-parse','HEAD'),tree=git(w,'rev-parse','HEAD^{tree}'),worktree=str(w),
        time=datetime.now(timezone.utc).isoformat(),documents=refs,native_sources=native,prior_asset_lock=identity(OLD),
        session_helper='BLOCK_MISSING_WORKTREE_LOCAL_CONFIG; actual host/CWD/registry/session independently checked; no helper PASS',
        new_GPU_actions=0,cap=1,monitoring='INITIAL_GATE_ONLY_OR_PENDING_HANDOFF'))
    cap=Path('/data/janghj/ODE-edit/local/state/server4-gpu-cap1-20260916-v1')
    save(cap/'received.json',dict(instruction_id=TASK,gpu_cap=1,authority='server4 gpu cap은 1이다.',other_servers_changed=False))
    print('FULL_READ_RECEIPT',identity(ROOT/'full-read-receipt.json'))

def check(w):
    for p in (w/PREFIX).glob('*.py'):ast.parse(p.read_text())
    command(['bash','-n',str(w/(PREFIX+'run.sbatch'))])
    result=subprocess.run(['/data/janghj/EasyEdit/.venv/bin/python','-B','-m','unittest',PREFIX.replace('/','.').rstrip('.')+'.test_local_z','-v'],cwd=w,text=True,capture_output=True)
    assert result.returncode==0,(result.stdout,result.stderr)
    source=[identity(p) for p in sorted((w/PREFIX).glob('*')) if p.is_file()]
    path=ROOT/'checks'/(digest(source)+'.json')
    save(path,dict(status='CPU_ONLY_PASS',output=result.stdout+result.stderr,source=source,GPU_forwards=0))
    print('CPU_CHECKS',identity(path))

def freeze(w):
    from scripts.fixed_counterfact import load_prefix
    assert not git(w,'status','--porcelain','--',PREFIX),'RUNTIME_NOT_COMMITTED'
    checked=[identity(p) for p in sorted((w/PREFIX).glob('*')) if p.is_file()]
    checks=ROOT/'checks'/(digest(checked)+'.json')
    assert json.loads(checks.read_text())['source']==checked,'CURRENT_SOURCE_CPU_CHECK_REQUIRED'
    old=json.loads(OLD.read_text());source=ROOT/'source-v1';source.mkdir(exist_ok=False)
    oldroot=Path(old['source_root']);relative=set();external=[]
    stats={x['path']:x for x in old['member_stats']}
    for m in old['members']:
        p=Path(m['path'])
        if p.is_relative_to(oldroot):
            rel=str(p.relative_to(oldroot))
            if rel.startswith(('project/','scripts/')) and p.suffix=='.py':relative.add(rel)
        else:
            assert p.exists(),('MISSING_SEALED_ASSET',str(p))
            s=p.stat();prior=stats[str(p)]
            assert (s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size)==(prior['dev'],prior['inode'],prior['mtime_ns'],prior['bytes']),('ASSET_STAT_DRIFT',str(p))
            external.append(dict(m,verification='PRIOR_FULL_SHA_STABLE_STAT',stat=[s.st_dev,s.st_ino,s.st_mtime_ns]))
    relative.update(str(p.relative_to(w)) for p in (w/PREFIX).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    relative.update((ENV,MAN,'scripts/fixed_counterfact.py'))
    members=[];lineage=[]
    for rel in sorted(relative):
        p=w/rel if (w/rel).is_file() else oldroot/rel
        members.append(copied(p,source/rel));lineage.append(dict(relative=rel,input=identity(p)))
        (source/rel).chmod(0o400)
    archive=ROOT/'source-v1.tar'
    with archive.open('xb') as handle,tarfile.open(fileobj=handle,mode='w') as tar:
        for rel in sorted(relative):
            info=tar.gettarinfo(str(source/rel),arcname=rel);info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o400
            with (source/rel).open('rb') as f:tar.addfile(info,f)
    records=load_prefix(old['dataset_root'],1000)
    sample=json.loads(Path(old['dataset_root'],'source-sample.lock.json').read_text())
    prefix_root=digest(sample['records'][:1000])
    assert prefix_root=='40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd'
    config=json.loads(Path(old['config4']).read_text())
    assert config['layers']==[4] and config['L2']==1
    disk=shutil.disk_usage(ROOT);reserve=212*(1<<30);assert disk.free>reserve,'STORAGE_RESERVE'
    keys=('blue_root','snapshot','projector','config4','historical_evaluator_root','helper_scripts_root','editor_sha256','reference_root','teacher_manifest','dataset_root','torch','transformers','model_revision')
    lock={k:old[k] for k in keys}
    lock.update(instruction_id=TASK,source_head=git(w,'rev-parse','HEAD'),source_tree=git(w,'rev-parse','HEAD^{tree}'),
        source_root=str(source),source_archive=identity(archive),members=members+external+[identity(checks),identity(ROOT/'full-read-receipt.json')],
        CPU_checks=identity(checks),prior_asset_full_SHA=identity(OLD),
        arms=list(ARMS),seed=20260916,gpu_cap=1,batches=10,batch_size=100,tf32_matmul=False,tf32_cudnn=False,
        records_digest=digest(records),prefix1000_ordered_root=prefix_root,whole_ordered_root=sample['ordered_root'],sample_order=[r['case_id'] for r in records],
        batch_locks=[dict(batch=b+1,ordinals=[100*b,100*(b+1)],digest=digest(records[100*b:100*(b+1)])) for b in range(10)],
        common_ready=str(ROOT/'technical/attempt-v1/READY.json'),
        numerical=dict(E_H_repeat=5e-5,D_repeat=5e-7,current_plateau=.05,E_H_allowance=1e-4,D_tie=1e-6),
        state_policy='Restore batch-entry RNG before fits/evaluations and after commit; only W/M/received ledger advance',
        event_id_encoding='SHA256 canonical ASCII JSON {ordinal,case_id,subject,relation,target}; priority prefix UTF-8 LZ-ALLOC-v1|20260916|past|',
        storage=dict(free_bytes=disk.free,reserve_bytes=reserve,CP_schedule=[1,5,10],CP_count=21,
            fit_native_weights_and_targets=True,candidate_weights='RECONSTRUCT_FROM_EPISODE_ENTRY_AND_SAVED_FITS_GATES',
            delta_replay='FP32 deltas alone NOT_CLAIMED_EXACT; selected weights in snapshots',optional_teacher_bytes=12608077824),
        resource=dict(gpus=1,cpus=8,mem_MiB=60416,wall_hours=12,hour_cap=None,
            estimate='Per-arm up to 3 native target sweeps plus up to 7 candidate E/H/D evaluations per batch; 12h reserve, NOT_MEASURED',
            queue='technical then afterok science array%1; no overlapping reservation capacity'),
        after_gate='MONITORING_PAUSED_AWAITING_USER',automatic_resume=False)
    ref=save(ROOT/'execution.lock.json',lock)
    save(ROOT/'source-lineage.json',dict(source_head=lock['source_head'],tree=lock['source_tree'],members=lineage,
        old_warm_runner='READ_ONLY_REFERENCE_NOT_LAUNCHED',old_EP_policy='NOT_CALLED',archive=identity(archive)))
    for i,arm in enumerate(ARMS):
        save(ROOT/'arms'/arm/'attempt-v1'/'arm.lock.json',dict(arm=arm,array_index=i,execution_lock=ref,
            target_calls_per_batch=EXPECTED[arm][0],solves_per_batch=EXPECTED[arm][1],candidates_per_batch=EXPECTED[arm][2],
            W0_cold=True,seed=20260916,new_requests=1000))
    print('FROZEN',ref)

def submit(w):
    control=ROOT/'submission-v1';control.mkdir(exist_ok=False)
    # One fresh resource-only query, no unrelated result/log access.
    queue=command(['squeue','-h','-u','janghj','-t','RUNNING,PENDING,CONFIGURING,COMPLETING','-o','%i|%j|%T|%b|%R'])
    reservations=[]
    for row in queue.splitlines():
        fields=row.split('|');jid=fields[0]
        detail=command(['scontrol','show','job',jid,'-o'])
        if 'server4' in detail and ('gres/gpu' in detail or 'gres:gpu' in detail or 'gpu:' in fields[3]):reservations.append(dict(row=row,detail=detail))
    save(control/'admission.json',dict(cap=1,existing=reservations,resource_only=True,time=datetime.now(timezone.utc).isoformat()))
    if reservations:raise RuntimeError('CAP1_OCCUPIED_NO_NEW_ADMISSION')
    lock=ROOT/'execution.lock.json';source=ROOT/'source-v1';script=source/(PREFIX+'run.sbatch')
    jobs=[]
    def sbatch(mode,name,extra):
        args=['sbatch','--parsable','--hold','--no-requeue','--job-name='+name,'--output='+str(control/(mode+'-%A_%a.out')),
              '--error='+str(control/(mode+'-%A_%a.err')),*extra,str(script),str(source),str(lock),mode]
        job=command(args).split(';')[0];assert job.isdigit();jobs.append(job)
        command(['scontrol','update','JobId='+job,'Requeue=0'])
        details=command(['scontrol','show','job',job,'-o'])
        save(control/(mode+'-held.json'),dict(job_id=job,args=args,inspection=details))
        assert 'UserId=janghj(' in details and 'NumCPUs=8' in details and ('MinMemoryNode=59G' in details or 'MinMemoryNode=60416M' in details)
        assert 'ReqNodeList=server4' in details and 'Requeue=0' in details and 'JobState=PENDING' in details
        assert str(script) in details and 'gpu:rtx_pro_6000:1' in details and 'TimeLimit=12:00:00' in details
        if mode=='science':assert 'afterok:'+technical in details and 'ArrayTaskThrottle=1' in details
        return job
    technical=sbatch('technical','odeedit_lz_prepare_s4',[])
    science=sbatch('science','odeedit_lz_seq1000_s4',['--array=0-6%1','--dependency=afterok:'+technical])
    save(control/'registered.json',dict(technical=technical,science_array=science,mapping=dict(enumerate(ARMS)),
        aggregate_concurrent_capacity=1,technical_READY_required=True,all_seven_registered=True,technical_PASS=False))
    command(['scontrol','release',science]);command(['scontrol','release',technical])
    observed=command(['squeue','-h','-j',technical+','+science,'-o','%i|%j|%T|%b|%R'])
    save(control/'released.json',dict(jobs=jobs,last_observation=observed,time=datetime.now(timezone.utc).isoformat(),
        monitoring_policy='INITIAL_GATE_ONLY_OR_PENDING_HANDOFF',agent_callback=False))
    print(observed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','check','freeze','submit']);p.add_argument('--worktree',required=True)
    a=p.parse_args();globals()[a.action](Path(a.worktree).resolve())
