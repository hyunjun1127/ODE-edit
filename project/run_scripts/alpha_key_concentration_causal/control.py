"""Create-once source lock and explicit upfront held/inspect/release launcher.

No runtime callback submits jobs. Only this user-invoked control entrypoint
registers the four finite phases. No scancel/hold/requeue of existing jobs.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time
from .common import save,file_sha

def command(argv):return subprocess.check_output(argv,text=True).strip()

def dependency_matches(job_text, expected):
    """Check exact dependency IDs while tolerating Slurm status annotations."""
    match=re.search(r'(?:^|\s)Dependency=(\S+)',job_text)
    if not match:return False
    value=match.group(1)
    if expected is None:return value in ('(null)','None')
    value=re.sub(r'\([^)]*\)','',value)
    kind,*ids=expected.split(':')
    found=[]
    for group in value.split(','):
        actual_kind,*actual_ids=group.split(':')
        if actual_kind!=kind:return False
        found.extend(actual_ids)
    return len(found)==len(ids) and set(found)==set(ids)

def attempt_paths(root,attempt):
    if not re.fullmatch(r'attempt-r[1-9][0-9]*',attempt):
        raise ValueError('INVALID_IMMUTABLE_ATTEMPT')
    root=Path(root).resolve();suffix=attempt.removeprefix('attempt-')
    return dict(control=root/'control' if attempt=='attempt-r1' else root/'controls'/attempt,
                frozen=root/('execution-source-'+suffix),output=root/'execution'/attempt,
                logs=root/'logs'/attempt,report_subdir='generated-'+suffix)

def freeze(root,repo,attempt='attempt-r1'):
    root=Path(root).resolve();repo=Path(repo).resolve();paths=attempt_paths(root,attempt);control=paths['control']
    assert not command(['git','-C',str(repo),'status','--porcelain']), 'FREEZE_REQUIRES_CLEAN_WORKTREE'
    assert not paths['output'].exists() and not paths['frozen'].exists(),'ATTEMPT_ALREADY_USED'
    control.mkdir(parents=True,exist_ok=False)
    source=command(['git','-C',str(repo),'rev-parse','HEAD']);tree=command(['git','-C',str(repo),'rev-parse','HEAD^{tree}'])
    preflight=root/'receipts/preflight-r1/preflight.json'
    assert json.loads(preflight.read_text())['status']=='PASS'
    resource=root/'receipts/preflight-r1/resource-plan.json'
    assert json.loads(resource.read_text())['status']=='PASS'
    fs=os.statvfs(root);needed=json.loads(resource.read_text())['planned_future_total']
    assert fs.f_bavail*fs.f_frsize>=needed,'STORAGE_RESERVE_UNAVAILABLE_NO_WAIVER'
    assert fs.f_favail>10000,'INSUFFICIENT_INODES'
    save(control/'storage-preflight.json',dict(epoch=time.time(),free_bytes=fs.f_bavail*fs.f_frsize,
        free_inodes=fs.f_favail,planned_future_bytes=needed,exclusive_reservation=False,status='PASS',storage_waiver=False))
    archive=control/'source.tar.gz'
    with archive.open('xb') as f:
        proc=subprocess.Popen(['git','-C',str(repo),'archive','--format=tar',source],stdout=subprocess.PIPE)
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=1) as z:
            while True:
                b=proc.stdout.read(8<<20)
                if not b:break
                z.write(b)
        assert proc.wait()==0
        f.flush();os.fsync(f.fileno())
    frozen=paths['frozen'];frozen.mkdir(exist_ok=False)
    with tarfile.open(archive,'r:gz') as tf:tf.extractall(frozen,filter='data')
    paths=command(['git','-C',str(repo),'ls-tree','-r','--name-only',source,'project/run_scripts/alpha_key_concentration_causal']).splitlines()
    members=[dict(relative_path=p,bytes=(frozen/p).stat().st_size,sha256=file_sha(frozen/p)) for p in paths]
    old=json.loads((root/'inputs/design/evidence/audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-execution.lock.json').read_text())
    binding=root/'receipts/native-input-binding-r2.json'
    if binding.exists():
        assert file_sha(binding)=='500cd93224d4d2cb0daf880dad67aa18acf91bc0a12d127a2a23c42e7805ded1','REUSED_INPUT_BINDING_DRIFT'
        bound=json.loads(binding.read_text());assert bound['status']=='PASS' and not bound['unresolved']
    else:
        prior=json.loads((root/'receipts/native-input-binding-r1.json').read_text());recovered=[]
        for m in prior['unresolved']:
            rel=m['path'].split('/policy-source/',1)[1]
            blob=subprocess.check_output(['git','-C',str(repo),'show','6fd7f1482c395b9ea7271c15c94967120dffca7e:'+rel])
            assert len(blob)==m['bytes'] and hashlib.sha256(blob).hexdigest()==m['sha256']
            dest=root/'inputs/frozen-policy'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(blob)
            recovered.append(dict(m,resolved_path=str(dest),verification='EXACT_GIT_BLOB_FULL_SHA'))
        save(binding,dict(status='PASS',members=prior['members']+recovered,unresolved=[],
            parent_receipt_sha256=file_sha(root/'receipts/native-input-binding-r1.json'),recovery_commit='6fd7f1482c395b9ea7271c15c94967120dffca7e'))
    from .token_binding import load_reference,reference_path,REFERENCE_SHA
    load_reference(root)
    lock=dict(instruction_id='ODEEDIT-GH-SH4-ALPHA-KEY-CAUSAL-20260923-R1',
        override_nonce='ODEEDIT-GH-SH4-ALPHA-KEY-AUTONOMOUS-RESUME-20260923-R1',
        actor_session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',source_commit=source,source_tree=tree,
        archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=file_sha(archive)),
        execution_source_members=members,root=str(root),repo=str(frozen),publication_repo=str(repo),attempt=attempt,
        execution_lock_path=str(control/'execution.lock.json'),report_subdir=paths['report_subdir'],log_directory=str(paths['logs']),
        native_token_reference=dict(path=str(reference_path(root)),sha256=REFERENCE_SHA),
        parent_failed_execution=dict(source='a95876f8e5c4cf59df9cd9d7f824d1ac99f8bc77',job_id='52527',allocated_gpu_seconds=44,
            lock_sha256='4a80070051cbbcc1b5e1d01f0e94124ddc92f79bedaba3772c42dcbbc7530725'),
        project_gpu_cap=2,task_gpu_cap=2,allowed_phases=['gate','geometry','writers','reduce'],
        scientific_contrast_families=94,followup_submissions=[],save_new_resume_checkpoints=False,
        diagnostic_storage='approved K/R/delta/targets/timestamp-current banks and key geometry only',
        input_binding_receipt=str(root/'receipts/native-input-binding-r2.json'),
        input_binding_sha256=file_sha(root/'receipts/native-input-binding-r2.json'),
        preflight_sha256=file_sha(preflight),resource_plan_sha256=file_sha(resource),
        python='/data/janghj/EasyEdit/.venv/bin/python',dependencies=old['dependencies'],
        resources=dict(node='server4',partition='gpu',gpus_each=1,cpus=8,mem_MiB=60416,export='NONE',requeue=False,wall='7-00:00:00'),
        reducer_resources=dict(cpus=4,mem_MiB=8192,gpus=0,wall='2-00:00:00'),
        full_read_authority='base4da5514 + cap2 pending33e7bb7 + autonomous0498b22; no scientific waiver',
        gate_policy='internal actual READY; afterok gates; incomplete actual is never PASS',
        source_native_unchanged=True,monitoring_handoff='actual GPU resource pending after full registration or actual G0-G3')
    save(control/'execution.lock.json',lock)
    for phase in lock['allowed_phases']:
        env=dict(PYTHONPATH=old['dependencies']+':'+str(frozen),PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='8' if phase!='reduce' else '4',OPENBLAS_NUM_THREADS='8' if phase!='reduce' else '4',MKL_NUM_THREADS='8' if phase!='reduce' else '4')
        if phase=='reduce':env['CUDA_VISIBLE_DEVICES']=''
        args=[lock['python'],'-u','-m','project.run_scripts.alpha_key_concentration_causal.runner','--lock',str(control/'execution.lock.json'),'--phase',phase]
        lines=['#!/bin/bash','set -euo pipefail',f'cd {shlex.quote(str(frozen))}']
        lines += ['export '+k+'='+shlex.quote(v) for k,v in env.items()]
        lines += ['exec '+shlex.join(args)]
        with (control/(phase+'.sh')).open('x') as f:f.write('\n'.join(lines)+'\n')
    return lock

def submit(root,attempt='attempt-r1'):
    root=Path(root).resolve();paths=attempt_paths(root,attempt);control=paths['control'];lock=json.loads((control/'execution.lock.json').read_text())
    assert lock['attempt']==attempt and lock['repo']==str(paths['frozen']),'IMMUTABLE_ATTEMPT_ROUTING_DRIFT'
    assert os.uname().nodename=='server4' and os.getuid()==int(command(['id','-u']))
    if (control/'submission.json').exists() or (control/'submission-events.jsonl').exists():
        raise RuntimeError('EXISTING_REGISTRATION_MUST_BE_REUSED_NO_DUPLICATE_SUBMIT')
    active=command(['squeue','-h','-u',command(['id','-un']),'-o','%i|%j|%T|%b|%E|%R'])
    # This submission graph needs both project slots after gate. Never mutate
    # another task to create capacity; exact outstanding admissions must be routed.
    if active:raise RuntimeError('OTHER_OWNER_ADMISSIONS_REQUIRE_EXACT_CAP_ROUTING:'+active)
    node_before=command(['scontrol','show','node','server4','--oneliner'])
    fs=os.statvfs(root)
    needed=json.loads((root/'receipts/preflight-r1/resource-plan.json').read_text())['planned_future_total']
    assert fs.f_bavail*fs.f_frsize>=needed and fs.f_favail>10000,'ACTUAL_STORAGE_ADMISSION_FAILED'
    save(control/'admission.json',dict(owner=command(['id','-un']),epoch=time.time(),
        exact_owner_queue=active,node=node_before,free_bytes=fs.f_bavail*fs.f_frsize,free_inodes=fs.f_favail,
        lock_sha256=file_sha(control/'execution.lock.json'),project_cap=2,task_cap=2))
    paths['logs'].mkdir(parents=True,exist_ok=False)
    jobs={};inspection={}
    with (control/'submission-events.jsonl').open('x') as journal:
        for phase in lock['allowed_phases']:
            cmd=['sbatch','--parsable','--hold','--job-name=odeedit_alpha_key_'+phase+'_s4',
                '--partition=gpu','--nodelist=server4','--nodes=1','--ntasks=1',
                '--cpus-per-task='+str(4 if phase=='reduce' else 8),
                '--mem='+str(8192 if phase=='reduce' else 60416)+'M',
                '--time='+('2-00:00:00' if phase=='reduce' else '7-00:00:00'),
                '--export=NONE','--no-requeue','--output='+str(paths['logs']/(phase+'-%j.out')),
                '--error='+str(paths['logs']/(phase+'-%j.err'))]
            dependency=None
            if phase in ('geometry','writers'):dependency='afterok:'+jobs['gate']
            if phase=='reduce':dependency='afterany:'+':'.join(jobs.values())
            if dependency:cmd+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
            if phase!='reduce':cmd+=['--gres=gpu:1']
            cmd += [str(control/(phase+'.sh'))]
            answer=command(cmd);job=answer.split(';')[0]
            assert job.isdigit(),('SBATCH_NOT_JOBID',answer)
            jobs[phase]=job
            journal.write(json.dumps(dict(phase=phase,job_id=job,argv=cmd,dependency=dependency,epoch=time.time()))+'\n');journal.flush();os.fsync(journal.fileno())
            text=command(['scontrol','show','job',job,'--oneliner']);inspection[phase]=text
            assert f'JobId={job} ' in text and 'JobState=PENDING' in text and 'Priority=0' in text
            assert f'UserId={command(["id","-un"])}(' in text and 'Requeue=0' in text
            assert f'Command={control}/{phase}.sh' in text and 'ReqNodeList=server4' in text
            assert f'NumCPUs={4 if phase=="reduce" else 8}' in text
            assert 'Partition=gpu' in text
            assert 'TimeLimit='+('2-00:00:00' if phase=='reduce' else '7-00:00:00') in text
            assert dependency_matches(text,dependency),'SLURM_DEPENDENCY_ID_OR_TYPE_DRIFT'
            if phase!='reduce':assert 'gres/gpu=1' in text and ('mem=59G' in text or 'mem=60416M' in text)
            else:assert 'mem=8G' in text or 'mem=8192M' in text
            actual_script=subprocess.check_output(['scontrol','write','batch_script',job,'-'],text=True)
            assert actual_script==(control/(phase+'.sh')).read_text(),'SLURM_STORED_SCRIPT_OR_FULL_ARGV_DRIFT'
            journal.write(json.dumps(dict(phase=phase,job_id=job,stored_script_sha256=hashlib.sha256(actual_script.encode()).hexdigest(),
                export_NONE_submitted='--export=NONE' in cmd,full_argv_inside_script=actual_script,inspection='PASS'))+'\n');journal.flush();os.fsync(journal.fileno())
        save(control/'held-inspection.json',dict(jobs=jobs,inspection=inspection,lock_sha256=file_sha(control/'execution.lock.json'),status='PASS'))
        # All source/args/resources/owner/dependencies inspected before ANY release.
        for phase,job in jobs.items():
            result=command(['scontrol','release',job])
            journal.write(json.dumps(dict(phase=phase,job_id=job,action='release',result=result,epoch=time.time()))+'\n');journal.flush();os.fsync(journal.fileno())
    rows=command(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%j|%T|%b|%E|%R'])
    node_after=command(['scontrol','show','node','server4','--oneliner'])
    receipt=dict(status='REGISTERED_INSPECTED_RELEASED',jobs=jobs,after_queue=rows,node_after=node_after,
        observed_epoch=time.time(),G0_G1_G2='NOT_OBSERVED',G3='FINITE_AUTONOMOUS_GRAPH_REGISTERED',
        dependency_graph={'gate':None,'geometry':'afterok:'+jobs['gate'],'writers':'afterok:'+jobs['gate'],'reduce':'afterany:'+':'.join(jobs[p] for p in ('gate','geometry','writers'))},
        peak_task_gpus=2,peak_project_gpus=2,other_owner_admissions=0,automatic_agent_resume=False)
    save(control/'submission.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','submit'));p.add_argument('--root',required=True);p.add_argument('--repo');p.add_argument('--attempt',default='attempt-r1')
    a=p.parse_args()
    if a.action=='freeze':freeze(a.root,a.repo,a.attempt)
    else:submit(a.root,a.attempt)

if __name__=='__main__':main()
