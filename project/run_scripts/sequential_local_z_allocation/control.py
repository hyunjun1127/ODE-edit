"""CPU checks and immutable runtime/asset/technical contract freeze."""
import argparse
import ast
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from .common import ROOT,COLD,TASK,ARMS,ARM_LAYERS,LAYERS,save,identity,sha,digest
from .preparation import ENVELOPE,PREFIX as AUTH_PREFIX

PACKAGE='project/run_scripts/sequential_local_z_allocation/'
PYTHON='/data/janghj/EasyEdit/.venv/bin/python'
DEPS='/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2'

def command(args,**kw):
    p=subprocess.run(args,text=True,capture_output=True,**kw)
    if p.returncode:raise RuntimeError(dict(args=args,stdout=p.stdout,stderr=p.stderr,exit=p.returncode))
    return p.stdout.strip()

def git(w,*args):return command(['git','-C',str(w),*args])

def copy_once(src,dst):
    src,dst=Path(src),Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
    assert sha(src)==sha(dst);return identity(dst)

def source_list(w):
    return [dict(relative=str(p.relative_to(w)),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted((w/PACKAGE).rglob('*')) if p.is_file() and '__pycache__' not in str(p)]

def check(w):
    for p in (w/PACKAGE).rglob('*.py'):ast.parse(p.read_text())
    command(['bash','-n',str(w/(PACKAGE+'run.sbatch'))])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=DEPS+':'+str(w))
    # unittest writes to stderr; preserve one actual invocation, not a rerun.
    modules=['project.run_scripts.sequential_local_z_allocation.'+p.stem for p in sorted((w/PACKAGE).glob('test_*.py'))]
    tests=subprocess.run([PYTHON,'-B','-m','unittest',*modules,'-v'],cwd=w,env=env,text=True,capture_output=True)
    assert tests.returncode==0,(tests.stdout,tests.stderr)
    source=source_list(w);path=ROOT/'checks'/(digest(source)+'.json')
    save(path,dict(status='CPU_ONLY_PASS',source=source,output=tests.stdout+tests.stderr,
        model_loads=0,GPU_forwards=0,actual_technical='NOT_RUN',time=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(identity(path)))

def freeze(w):
    from scripts.fixed_counterfact import load_prefix
    assert not git(w,'status','--porcelain','--',PACKAGE),'UNCOMMITTED_RUNTIME'
    current=source_list(w);checkref=ROOT/'checks'/(digest(current)+'.json')
    assert json.loads(checkref.read_text())['source']==current
    oldpath=COLD/'execution.lock.json';old=json.loads(oldpath.read_text());oldroot=Path(old['source_root'])
    root=ROOT/'source-v1';root.mkdir(exist_ok=False)
    relative={}
    # Read-only dependency closure from the actual cold7 archive, not a latest
    # main replacement for native/evaluator code. Unused prior modules are not launched.
    external=[]
    admitted=[Path(old[k]) for k in ('blue_root','snapshot','projector','config4','historical_evaluator_root',
        'helper_scripts_root','reference_root','dataset_root')]+[Path(old['teacher_manifest']['path']).parent]
    for m in old['members']:
        p=Path(m['path'])
        if p.is_relative_to(oldroot):
            rel=str(p.relative_to(oldroot))
            if rel.startswith(('project/','scripts/')) and p.suffix=='.py':relative[rel]=p
        elif any(p==a or p.is_relative_to(a) for a in admitted):
            s=p.stat();assert s.st_size==m['bytes'],('PRIOR_ASSET_SIZE',str(p))
            if m.get('verification')=='PRIOR_FULL_SHA_STABLE_STAT':
                assert [s.st_dev,s.st_ino,s.st_mtime_ns]==m['stat'],('PRIOR_ASSET_DRIFT',str(p))
            else:assert sha(p)==m['sha256'],('PRIOR_ASSET_SHA',str(p))
            external.append(dict(m,verification='PRIOR_FULL_SHA_STABLE_STAT',stat=[s.st_dev,s.st_ino,s.st_mtime_ns]))
    # Only explicitly reused cold7 files and the new package are overlaid.
    for rel in ('common.py','policy.py','technical.py','model.py','engine.py'):
        rp='project/run_scripts/local_z_adaptive_allocation/'+rel
        assert sha(w/rp)==sha(oldroot/rp),'READ_ONLY_COLD_HELPER_DRIFT'
        relative[rp]=oldroot/rp
    for member in current:relative[member['relative']]=w/member['relative']
    relative[ENVELOPE]=w/ENVELOPE
    members=[];lineage=[]
    for rel,p in sorted(relative.items()):
        ref=copy_once(p,root/rel);members.append(ref);lineage.append(dict(relative=rel,input=identity(p)))
        (root/rel).chmod(0o400)
    archive=ROOT/'source-v1.tar'
    with archive.open('xb') as handle,tarfile.open(fileobj=handle,mode='w') as tar:
        for rel in sorted(relative):
            info=tar.gettarinfo(str(root/rel),arcname=rel);info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o400
            with (root/rel).open('rb') as f:tar.addfile(info,f)
    records=load_prefix(old['dataset_root'],1000)
    sample=json.loads(Path(old['dataset_root'],'source-sample.lock.json').read_text())
    prefix=digest(sample['records'][:1000]);assert prefix=='40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd'
    cold=identity(COLD/'technical/attempt-v1/cold-capsule.json')
    assert cold['sha256']=='2d5d5c45bbbdf36ed859451242d7d84874d08cda4008d585945e565d9b3b1cb7'
    teacher=json.loads(Path(cold['path']).read_text())['teacher_manifest']
    assert identity(teacher['path'])==teacher
    disk=shutil.disk_usage(ROOT);reserve=64*(1<<30);assert disk.free>reserve,'DISK_RESERVE'
    import scipy,scipy.optimize._cobyla,scipy.optimize._cobyla_py,torch,transformers
    assert scipy.__version__=='1.15.3' and torch.__version__==old['torch'] and transformers.__version__==old['transformers']
    dependencies=[identity(Path(m.__file__)) for m in (scipy,scipy.optimize._cobyla,scipy.optimize._cobyla_py,torch,transformers)]
    keys=('blue_root','snapshot','projector','config4','historical_evaluator_root','helper_scripts_root','editor_sha256',
        'reference_root','dataset_root','torch','transformers','model_revision')
    lock={k:old[k] for k in keys}
    lock.update(instruction_id=TASK,source_head=git(w,'rev-parse','HEAD'),source_tree=git(w,'rev-parse','HEAD^{tree}'),
        source_root=str(root),source_archive=identity(archive),members=members+external+dependencies+[cold,teacher],
        CPU_checks=identity(checkref),FULL_READ=identity(ROOT/'full-read-m0.json'),prior_asset_lock=identity(oldpath),
        scipy='1.15.3',python=command([PYTHON,'--version']),dependencies=dependencies,
        arms=list(ARMS),arm_layers={k:list(v) for k,v in ARM_LAYERS.items()},seed=20260916,gpu_cap=2,
        batches=10,batch_size=100,history_layers=list(LAYERS),tf32_matmul=False,tf32_cudnn=False,
        records_digest=digest(records),prefix1000_ordered_root=prefix,whole_ordered_root=sample['ordered_root'],
        sample_order=[r['case_id'] for r in records],
        batch_locks=[dict(batch=b+1,ordinals=[b*100,(b+1)*100],digest=digest(records[b*100:(b+1)*100])) for b in range(10)],
        cold_capsule=cold,teacher_manifest=teacher,W0_observation=identity(COLD/'technical/attempt-v1/W0-first1000.json'),
        common_ready=str(ROOT/'technical/attempt-v1/READY.json'),
        numerical=dict(E_H_repeat=5e-5,B_repeat=5e-7,E_H_allowance=1e-4,B_tie=1e-6,current_plateau=False,
            native_weight_max_abs=5e-6,native_weight_relative_Frobenius=5e-5,strict_pair_repeat='EXACT_ID_SET'),
        budgets=dict(search_suffix_fits=32,prune_suffix_fits=8,search_endpoints=24,prune_endpoints=4,
            extra_Adam=9600,pre_fit_Adam_reserve=2400,maxiter_function_evaluations=64,native_N4_separate=True),
        coverage=dict(C45678_nonN4_completed_gate_vectors_min=7,distinct_a4_min=2,actual_endpoint_count_separate=True),
        state_policy='ENTRY_RNG_RESTORED_FOR_EACH_BRANCH_AND_AFTER_COMMIT; actual selected W/M only advance',
        checkpoint_policy='NO_DISK_W_M_CHECKPOINT',crash_resume='NOT_AVAILABLE',
        storage=dict(free_bytes=disk.free,reserve_bytes=reserve,structurally_shared_RAM=True,
            disk_native_target_key_loss=True,disk_W_M=False,optional_teacher_bytes=12608077824),
        resource=dict(gpus=1,cpus=8,mem_MiB=60416,technical_wall_hours=12,science_wall='AFTER_TECHNICAL_MEASUREMENT',
            hour_cap=None,estimated_component_cost='NOT_MEASURED',max_project_concurrent_capacity=2),
        planned_science_upper_bounds=dict(target=89000,Adam=408000,loss=497000,solve=890,score=960,history=300),
        prior49238='ADMISSION_ONLY_COMPLETED_LAST_OBSERVATION',
        after_gate='MONITORING_PAUSED_AWAITING_USER',automatic_resume=False)
    ref=save(ROOT/'execution.lock.json',lock)
    save(ROOT/'source-lineage.json',dict(source_head=lock['source_head'],tree=lock['source_tree'],members=lineage,
        immutable_prior_closure=identity(oldpath),archive=identity(archive),old_repair_runtime_imported=False))
    for i,arm in enumerate(ARMS):
        save(ROOT/'arms'/arm/'attempt-v1'/'arm.lock.json',dict(arm=arm,array_index=i,execution_lock=ref,
            layers=list(ARM_LAYERS[arm]),history_layers=list(LAYERS),W0_cold=True,seed=20260916,requests=1000,
            technical_READY_required=True,registered=False))
    print(json.dumps(dict(lock=ref,source=lock['source_head'],archive=lock['source_archive'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','freeze']);p.add_argument('--worktree',required=True)
    a=p.parse_args();globals()[a.action](Path(a.worktree))
