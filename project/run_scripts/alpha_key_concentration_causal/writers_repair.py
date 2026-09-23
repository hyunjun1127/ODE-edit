"""One user-invoked writer repair registration; no periodic watcher/retry.

Keeps running geometry and its original reducer untouched. A new CPU reducer
joins the same geometry with the replacement writer via afterany dependencies.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import tarfile
import time
import traceback

from .common import file_sha, save
from .control import command, dependency_matches
from .numerical_policy import OBSERVE, USER_TEXT

PARENT_LOCK_SHA='5c461288fc77ae7071e84b0264cc240cb845b8cb4862ac47ecf37e874fcbc38b'
NONCE='ODEEDIT-GH-SH4-ALPHA-KEY-WRITERS-REPAIR-20260923-R1'


def paths(root):
    root=Path(root).resolve()
    return dict(root=root,control=root/'controls/attempt-r4',out=root/'execution/attempt-r4',
                frozen=root/'execution-source-r4',logs=root/'logs/attempt-r4')


def size_tree(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink())


def remaining_storage(plan,root):
    # Geometry is a live *resource-only* co-tenant. Already written payload
    # consumes current free space and is not charged a second time as future.
    current=size_tree(Path(root)/'execution/attempt-r3/geometry')
    remaining=max(0,plan['geometry_total_tensor_payload']+3*2**30-current)
    p=paths(root)
    archive=p['control']/'source.tar.gz'
    source_written=(archive.stat().st_size if archive.exists() else 0)+size_tree(p['frozen'])
    return (plan['required_future_bytes']-plan['components']['geometry_remaining']+remaining
            -min(source_written,plan['components']['source_archive_unpack']))


def prepare(root):
    """CPU-only exact inventory of completed data; running sibling size only."""
    p=paths(root);root=p['root'];dest=root/'writers-repair-r1/preparation';dest.mkdir(exist_ok=False)
    parent=root/'controls/attempt-r3/execution.lock.json'
    assert file_sha(parent)==PARENT_LOCK_SHA
    old=json.loads(parent.read_text());oldout=root/'execution/attempt-r3'
    members=[]
    for directory in (oldout/'gate',oldout/'writers/W050/NATIVE',oldout/'writers/W050/SHAM'):
        for f in sorted(directory.rglob('*')):
            assert not f.is_symlink()
            if f.is_file():
                st=f.stat();members.append(dict(path=str(f),bytes=st.st_size,sha256=file_sha(f),
                    device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns))
    for branch in ('NATIVE','SHAM'):
        b=oldout/'writers/W050'/branch
        t=json.loads((b/'write/terminal.json').read_text())
        assert t['status']=='COMPLETED' and t['counters']['history_appends']==5
        for stage in ('entry','z','W4','W5','W6','W7','W8','history'):
            assert (b/'stages'/stage/'observation.json').is_file()
        # Existing artifact-level seals must agree, not just newly hashed bytes.
        for layer in (4,5,6,7,8):
            r=json.loads((b/f'write/L{layer}-factor-receipt.json').read_text())
            for name in ('native_artifact','artifact'):
                a=r[name];row=next(x for x in members if x['path']==a['path'])
                assert (row['bytes'],row['sha256'])==(a['bytes'],a['sha256'])
    ready=oldout/'gate/READY.json';assert json.loads(ready.read_text())['status']=='PASS'
    zsource=Path(json.loads((root/'inputs/design/evidence/audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-execution.lock.json').read_text())['blue_root'])/'AlphaEdit/compute_z.py'
    manifest=dict(status='CPU_HASH_BOUND_ACTUAL_RECONSTRUCTION_REQUIRED',writers=str(oldout/'writers'),
        gate_dir=str(oldout/'gate'),gate_ready_sha256=file_sha(ready),parent_lock_sha256=PARENT_LOCK_SHA,
        parent_source=old['source_commit'],failed_job_id='52565',failed_allocation_gpu_seconds=1494,
        gate_job_id='52563',gate_allocation_gpu_seconds=196,geometry_job_id='52564',old_reducer_job_id='52566',
        compute_z_source=dict(path=str(zsource),sha256=file_sha(zsource),bytes=zsource.stat().st_size),
        RNG_after_z='NOT_RECORDED; source-deterministic zeros/Adam/eval; use exact entry RNG',
        members=members,logical_reused_bytes=sum(r['bytes'] for r in members))
    save(dest/'reuse-manifest.json',manifest)
    # Same output payloads, scoped to what remains. No output/coverage reduction.
    geo_existing=size_tree(oldout/'geometry')  # allocation footprint only, no scientific row inspection
    geometry_payload=7*4*1000*5*14336*4*7 + 18*512*5*14336*4*7 + 18*3488*5*14336*4
    branch_bytes=max(size_tree(oldout/'writers/W050'/b) for b in ('NATIVE','SHAM'))
    GiB=2**30
    future=dict(geometry_remaining=max(0,geometry_payload+3*GiB-geo_existing),
        writer_remaining_22_branches=int(22*branch_bytes*1.10),components_diagnostics=12_000_000_000,
        source_archive_unpack=3*GiB,atomic_and_metadata=4*GiB,shared_volume_safety=20*GiB)
    fs=os.statvfs(root);needed=sum(future.values());free=fs.f_bavail*fs.f_frsize
    save(dest/'resource-plan.json',dict(status='PASS' if free>=needed else 'STORAGE_BLOCKED',
        free_bytes=free,required_future_bytes=needed,components=future,
        geometry_observed_bytes=geo_existing,geometry_total_tensor_payload=geometry_payload,
        measured_max_completed_branch_bytes=branch_bytes,writer_new_branches=22,
        previous_50GB_safety='prior whole-task estimate preserved; new scoped plan has explicit 20GiB margin plus 7GiB source/atomic',
        outputs_or_observers_reduced=False,storage_waiver=False,exclusive_reservation=False,
        host_mem_MiB=60416,host_peak_estimate_GiB=49,task_peak_gpus=2,new_job_gpus=1,
        native_targets_reused=100,native_targets_new=300,wall='7-00:00:00',
        wall_basis='prior two-branch 1489s inclusive; 22 remaining branches plus 4 component panels, generous original scheduler limit; not measured total',
        GPUhour_hardcap=None))
    print(json.dumps(dict(reuse_members=len(members),free_bytes=free,required_future_bytes=needed,storage_PASS=free>=needed)))


def freeze(root,repo):
    p=paths(root);root=p['root'];repo=Path(repo).resolve();control=p['control']
    assert not command(['git','-C',str(repo),'status','--porcelain'])
    assert not any(p[k].exists() for k in ('control','frozen','out'))
    prep=root/'writers-repair-r1/preparation';plan=json.loads((prep/'resource-plan.json').read_text())
    needed=remaining_storage(plan,root)
    assert plan['status']=='PASS' and os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>=needed
    control.mkdir();source=command(['git','-C',str(repo),'rev-parse','HEAD']);tree=command(['git','-C',str(repo),'rev-parse','HEAD^{tree}'])
    archive=control/'source.tar.gz'
    with archive.open('xb') as f:
        proc=subprocess.Popen(['git','-C',str(repo),'archive','--format=tar',source],stdout=subprocess.PIPE)
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=1) as z:
            while True:
                b=proc.stdout.read(8<<20)
                if not b:break
                z.write(b)
        assert proc.wait()==0;f.flush();os.fsync(f.fileno())
    p['frozen'].mkdir()
    with tarfile.open(archive,'r:gz') as tf:tf.extractall(p['frozen'],filter='data')
    old=json.loads((root/'controls/attempt-r3/execution.lock.json').read_text())
    members=[]
    for name in command(['git','-C',str(repo),'ls-tree','-r','--name-only',source,'project/run_scripts/alpha_key_concentration_causal']).splitlines():
        f=p['frozen']/name;members.append(dict(relative_path=name,bytes=f.stat().st_size,sha256=file_sha(f)))
    lock=dict(old,attempt='attempt-r4',source_commit=source,source_tree=tree,repo=str(p['frozen']),publication_repo=str(repo),
        archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=file_sha(archive)),execution_source_members=members,
        execution_lock_path=str(control/'execution.lock.json'),report_subdir='generated-r4',log_directory=str(p['logs']),
        allowed_phases=['writers','reduce'],override_nonce=NONCE,numerical_comparison_policy=OBSERVE,
        user_numerical_override=USER_TEXT,full_numerical_equivalence='NOT_ESTABLISHED',
        full_read_authority='ac44d4d9 writers recall + direct user numerical comparison observer-only override',
        reused_parent_lock=dict(path=str(root/'controls/attempt-r3/execution.lock.json'),sha256=PARENT_LOCK_SHA),
        reuse_manifest=dict(path=str(prep/'reuse-manifest.json'),sha256=file_sha(prep/'reuse-manifest.json')),
        resource_plan=dict(path=str(prep/'resource-plan.json'),sha256=file_sha(prep/'resource-plan.json')),
        inherited_output_roots=[str(root/'execution/attempt-r3/gate'),str(root/'execution/attempt-r3/geometry')],
        parent_failed_execution=dict(source=old['source_commit'],job_id='52565',allocated_gpu_seconds=1494,lock_sha256=PARENT_LOCK_SHA),
        gate_policy='reused actual 52563 READY; numerical comparisons observational; integrity still blocking',
        source_native_unchanged=True,monitoring_handoff='confirmed GPU resource pending or repaired path initial valid; no completion wait')
    save(control/'execution.lock.json',lock)
    for phase in ('writers','reduce'):
        env=dict(PYTHONPATH=old['dependencies']+':'+str(p['frozen']),PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='8' if phase=='writers' else '4',MKL_NUM_THREADS='8' if phase=='writers' else '4',OPENBLAS_NUM_THREADS='8' if phase=='writers' else '4')
        if phase=='reduce':env['CUDA_VISIBLE_DEVICES']=''
        args=[lock['python'],'-u','-m','project.run_scripts.alpha_key_concentration_causal.writers_repair','run','--root',str(root),'--phase',phase]
        script=['#!/bin/bash','set -euo pipefail','cd '+shlex.quote(str(p['frozen']))]
        script+=['export '+k+'='+shlex.quote(v) for k,v in env.items()];script+=['exec '+shlex.join(args)]
        with (control/(phase+'.sh')).open('x') as f:f.write('\n'.join(script)+'\n')
    print(json.dumps(dict(source=source,tree=tree,archive=lock['archive'],lock_sha256=file_sha(control/'execution.lock.json'))))


def submit(root):
    p=paths(root);control=p['control'];lock=json.loads((control/'execution.lock.json').read_text())
    assert not (control/'submission-events.jsonl').exists(),'ALREADY_REGISTERING_REUSE_RECEIPT_NO_DUPLICATE'
    owner=command(['id','-un']);active=command(['squeue','-h','-u',owner,'-o','%i|%j|%T|%b|%E|%R'])
    # Exact still-valid sibling and old CPU reducer only; unfamiliar admissions
    # require explicit cap routing instead of silent overbooking or cancellation.
    for row in active.splitlines():
        fields=row.split('|');jid=fields[0]
        assert jid in ('52564','52566'),'OTHER_ADMISSION_REQUIRES_CAP_ROUTING:'+row
        if jid=='52564':assert fields[1]=='odeedit_alpha_key_geometry_s4' and fields[3] in ('gres/gpu:1','gpu:1')
        else:assert fields[1]=='odeedit_alpha_key_reduce_s4' and fields[3] in ('N/A','(null)','')
    fs=os.statvfs(p['root']);plan=json.loads(Path(lock['resource_plan']['path']).read_text())
    needed=remaining_storage(plan,p['root'])
    assert fs.f_bavail*fs.f_frsize>=needed and fs.f_favail>10000,'STORAGE_ADMISSION_FAILED'
    save(control/'admission.json',dict(epoch=time.time(),owner_queue=active,node=command(['scontrol','show','node','server4','--oneliner']),
        project_cap=2,existing_task_gpu_max=1,new_gpu_max=1,free_bytes=fs.f_bavail*fs.f_frsize,needed_bytes=needed))
    p['logs'].mkdir();jobs={};inspection={}
    with (control/'submission-events.jsonl').open('x') as journal:
        def record(x):journal.write(json.dumps(x)+'\n');journal.flush();os.fsync(journal.fileno())
        for phase in ('writers','reduce'):
            cpu=phase=='reduce';dependency='afterany:52564:'+jobs['writers'] if cpu else None
            args=['sbatch','--parsable','--hold','--job-name=odeedit_alpha_key_'+phase+'_r4_s4','--partition=gpu','--nodelist=server4',
                '--nodes=1','--ntasks=1','--cpus-per-task='+str(4 if cpu else 8),'--mem='+str(8192 if cpu else 60416)+'M',
                '--time='+('2-00:00:00' if cpu else '7-00:00:00'),'--export=NONE','--no-requeue',
                '--output='+str(p['logs']/(phase+'-%j.out')),'--error='+str(p['logs']/(phase+'-%j.err'))]
            if dependency:args+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
            if not cpu:args+=['--gres=gpu:1']
            args+=[str(control/(phase+'.sh'))];answer=command(args);jid=answer.split(';')[0];assert jid.isdigit()
            jobs[phase]=jid;record(dict(phase=phase,job_id=jid,argv=args,dependency=dependency,epoch=time.time()))
            s=command(['scontrol','show','job',jid,'--oneliner']);inspection[phase]=s
            assert f'JobId={jid} ' in s and f'UserId={owner}(' in s and 'JobState=PENDING' in s and 'Priority=0' in s
            assert f'Command={control}/{phase}.sh' in s and 'Requeue=0' in s and 'ReqNodeList=server4' in s and 'Partition=gpu' in s
            assert f'NumCPUs={4 if cpu else 8}' in s and dependency_matches(s,dependency)
            assert ('mem=8G' in s or 'mem=8192M' in s) if cpu else ('gres/gpu=1' in s and ('mem=59G' in s or 'mem=60416M' in s))
            assert 'TimeLimit='+('2-00:00:00' if cpu else '7-00:00:00') in s
            script=command(['scontrol','write','batch_script',jid,'-'])
            assert script==(control/(phase+'.sh')).read_text().strip(),'STORED_SCRIPT_DRIFT'
            record(dict(job_id=jid,inspection='PASS',script_sha256=hashlib.sha256(script.encode()).hexdigest()))
        save(control/'held-inspection.json',dict(status='PASS',jobs=jobs,inspection=inspection))
        for phase,jid in jobs.items():record(dict(action='release',job_id=jid,result=command(['scontrol','release',jid]),epoch=time.time()))
    receipt=dict(status='REGISTERED_INSPECTED_RELEASED',jobs=jobs,observed_epoch=time.time(),
        queue=command(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%j|%T|%b|%E|%R']),
        node=command(['scontrol','show','node','server4','--oneliner']),project_cap=2,
        untouched_jobs=['52564','52566'],new_GPU_job_count=1,new_CPU_reducer_count=1,
        dependencies={'writers':None,'reduce':'afterany:52564:'+jobs['writers']},repair_initial='NOT_OBSERVED',automatic_retry=False)
    save(control/'submission.json',receipt);print(json.dumps(receipt,indent=2))


def run(root,phase):
    p=paths(root);lockpath=p['control']/'execution.lock.json';lock=json.loads(lockpath.read_text())
    assert phase in lock['allowed_phases']==['writers','reduce'] and lock['attempt']=='attempt-r4'
    assert Path(__file__).resolve().is_relative_to(p['frozen']) and os.uname().nodename=='server4'
    assert lock['numerical_comparison_policy']==OBSERVE and not lock['save_new_resume_checkpoints'] and not lock['followup_submissions']
    for row in lock['execution_source_members']:
        f=p['frozen']/row['relative_path'];assert f.stat().st_size==row['bytes'] and file_sha(f)==row['sha256']
    for name in ('reuse_manifest','resource_plan','reused_parent_lock'):
        item=lock[name];assert file_sha(item['path'])==item['sha256']
    oldout=p['root']/'execution/attempt-r3';out=p['out'];out.mkdir(exist_ok=True)
    t=time.monotonic()
    try:
        if phase=='reduce':
            for name in ('gate','geometry','gate-program.json','geometry-program.json'):
                original=oldout/name
                if original.exists() and not (out/name).exists():(out/name).symlink_to(original,target_is_directory=original.is_dir())
            if (oldout/'geometry-failure.json').exists():(out/'geometry-failure.json').symlink_to(oldout/'geometry-failure.json')
            from .reduce import reduce_package
            reduce_package(p['root'],out,Path(lock['publication_repo']),lock)
            return
        from .writer_reuse import verify_members
        reuse=json.loads(Path(lock['reuse_manifest']['path']).read_text());verify_members(reuse)
        assert json.loads((oldout/'gate/READY.json').read_text())['status']=='PASS'
        from .runtime import Runtime
        from .writer_runner import run_writers
        rt=Runtime(p['root'],p['frozen']);rt.numerical_comparison_policy=OBSERVE
        run_writers(rt,out/'writers',oldout/'gate',reuse=reuse)
        import torch
        save(out/'writers-program.json',dict(status='COMPLETED',phase=phase,job_id=os.environ.get('SLURM_JOB_ID'),
            wall_seconds=time.monotonic()-t,source_commit=lock['source_commit'],lock_sha256=file_sha(lockpath),
            numerical_comparison_policy=OBSERVE,model_load_seconds=rt.model_load_seconds,
            host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    except BaseException as exc:
        save(out/(phase+'-failure.json'),dict(status='TECHNICAL_FAILED',phase=phase,message=str(exc),exception_type=type(exc).__name__,
            traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-t,job_id=os.environ.get('SLURM_JOB_ID'),
            source_commit=lock['source_commit'],lock_sha256=file_sha(lockpath),automatic_retry=False))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','freeze','submit','run'))
    parser.add_argument('--root',required=True);parser.add_argument('--repo');parser.add_argument('--phase',choices=('writers','reduce'))
    a=parser.parse_args()
    if a.action=='prepare':prepare(a.root)
    elif a.action=='freeze':freeze(a.root,a.repo)
    elif a.action=='submit':submit(a.root)
    else:run(a.root,a.phase)


if __name__=='__main__':main()
