"""Own-scope source freeze + single-job held inspection, B1 only."""
import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from .config import NUMERIC,require_scope
from .provenance import ROOT,sha,now,create_json,create_bytes
from .qp import DEFAULT_QP_POLICY
from project.run_scripts.single_layer_edit_preserving_correction.common import digest

PACKAGE='project/run_scripts/base_choice_constrained_write'
PYTHON='/data/janghj/EasyEdit/.venv/bin/python'
DEPS='/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2'
OLD=Path('/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/S/attempt-v1/execution.lock.json')

def call(args,cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True).strip()

def member(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def preflight(repo):
    directory=ROOT/'CPU-preflight-v1';directory.mkdir(exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=DEPS+':'+str(repo),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
    cmd=[PYTHON,'-B','-m','unittest',PACKAGE.replace('/','.')+'.test_qp',PACKAGE.replace('/','.')+'.test_boundaries','-v']
    start=time.monotonic();p=subprocess.run(cmd,cwd=repo,env=env,text=True,capture_output=True)
    create_json(directory/'tests.json',dict(args=cmd,exit=p.returncode,stdout=p.stdout,stderr=p.stderr,
        seconds=time.monotonic()-start,model_forwards=0))
    if p.returncode:raise ValueError('CPU_TESTS_FAILED')
    call(['bash','-n',str(repo/PACKAGE/'run.sbatch')])
    import torch,transformers,numpy,scipy
    runner=importlib.import_module(PACKAGE.replace('/','.')+'.runner')
    assert Path(runner.__file__).resolve()==(repo/PACKAGE/'runner.py').resolve()
    from scripts.fixed_counterfact import load_prefix
    records=load_prefix('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1',100)
    final=json.loads((ROOT/'reference-inputs-v2/manifest.json').read_text())
    old=json.loads((ROOT/'reference-inputs-v1/manifest.json').read_text())
    create_json(directory/'input-firewall-correction.json',dict(
        original_policy=member(ROOT/'reference-inputs-v1/selection-policy.json'),
        final_policy=member(ROOT/'reference-inputs-v2/selection-policy.json'),
        reason='Legacy overlap checker included CounterFact future/official P/N; final selector only uses Wiki128 and old reference fingerprints',
        same_selected_input_bytes=old['inputs_sha256']==final['inputs_sha256'],model_evaluations_before_fix=0,
        old_inputs_preserved=True,selection_used_for_execution='reference-inputs-v2'))
    r=dict(status='CPU_IMPLEMENTATION_CHECKS_PASS_ACTUAL_MODEL_NOT_RUN',time=now(),tests=27,
        torch=str(torch.__version__),transformers=transformers.__version__,numpy=numpy.__version__,scipy=scipy.__version__,
        records_digest=digest(records),sample_order=[r['case_id'] for r in records],
        reference=member(ROOT/'reference-inputs-v2/manifest.json'),source_import=str(runner.__file__),
        independent_work='bounded QP implementation/19 tests and independent small-factor audit by bpcw_qp worker; parent integration/boundary tests',
        full_model_numerical_validation='NOT_ESTABLISHED_PENDING_B1_INTEGRATED_CHECKS',
        test_receipt=member(directory/'tests.json'),syntax=True,sequential_authorized=False,max_batches=1)
    create_json(directory/'receipt.json',r);print(json.dumps(r,indent=2))

def freeze(repo,attempt):
    repo=repo.resolve();parent=ROOT/'B1'/attempt
    if parent.exists():raise FileExistsError(parent)
    if call(['git','status','--porcelain'],repo):raise ValueError('SOURCE_MUST_BE_COMMITTED_CLEAN')
    pre=json.loads((ROOT/'CPU-preflight-v1/receipt.json').read_text())
    old=json.loads(OLD.read_text())
    cleanup=json.loads((ROOT/'en-cleanup/removal-receipt.json').read_text())
    if cleanup['removed_files']!=52 or cleanup['EN_exact_checkpoint_resume']!='UNAVAILABLE_AFTER_USER_DIRECTED_REMOVAL':
        raise ValueError('EN_CLEANUP_RECEIPT_NOT_BOUND')
    free=shutil.disk_usage(ROOT).free
    # 1536 x (4096*144 FP32 A + 14336*144 FP64 QK), bounded technical dense4,
    # two W/M CP, native/Q/current keys, temporary serializations and compact/raw observer reserve.
    factors_max=1536*(4096*144*4+14336*144*8)
    storage=dict(factor_upper_bytes=factors_max,checkpoint_two_bytes=2*(4096*14336*4+14336**2*4),
        technical_gradient4_bytes=4*4096*14336*8,geometry_native_keys_raw_temporary_reserve_bytes=12*2**30,
        required_free_bytes=48*2**30,observed_free_bytes=free,free_inodes=os.statvfs(ROOT).f_favail,
        original_EN_waiver_inherited=False,exclusive_reservation=False,actual_IO_fatal=True)
    if free<storage['required_free_bytes']:raise ValueError('BPCW_B1_STORAGE_RESERVE_INSUFFICIENT')
    parent.mkdir(parents=True);source=parent/'source';source.mkdir()
    head=call(['git','rev-parse','HEAD'],repo);tree=call(['git','rev-parse','HEAD^{tree}'],repo)
    # Compact transitive Python source, not historical raw/reports.
    archive=parent/'source.tar'
    paths=['project/run_scripts','scripts/fixed_counterfact.py']
    with archive.open('xb') as f:
        subprocess.run(['git','archive','--format=tar',head,*paths],cwd=repo,stdout=f,check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(source,filter='data')
    members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()]
    execution=dict(commit=head,tree=tree,archive=member(archive),source_root=str(source),members=members)
    # Reuse only named immutable asset identities, never old runtime controls/native/teacher.
    keys=['snapshot','config4','blue_root','dataset_root','projector','cold_capsule','editor_sha256',
          'historical_evaluator_root','helper_scripts_root','P_star_basis']
    lock={k:old[k] for k in keys}
    for key in ('cold_capsule','P_star_basis'):
        if member(lock[key]['path'])!=lock[key]:raise ValueError('IMMUTABLE_ASSET_'+key)
    assets=[]
    for path in [Path(lock['config4']),*Path(lock['blue_root'],'AlphaEdit').glob('*.py')]:assets.append(member(path))
    # Large pretrained/P bytes: reuse prior sealed identity; current stat separately; actual W0/P tensor hash checked on load.
    identity_stats=[]
    for path in [Path(lock['projector']),*Path(lock['snapshot']).glob('*')]:
        if path.is_file():identity_stats.append(dict(path=str(path),realpath=str(path.resolve()),bytes=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns,level='PRIOR_IMMUTABLE_BINDING_PLUS_CURRENT_STAT; W0/P TENSOR CHECK_ON_LOAD'))
    ref=json.loads((ROOT/'reference-inputs-v2/manifest.json').read_text())
    lock.update(max_batches=1,sequential_authorized=False,arms=['N4','BPCW512'],batch_size=100,
        native_fit_reuse=False,native_policy='FRESH_SAME_HOST_SHARED_ONCE',task_gpu_cap=1,project_gpu_cap=2,
        numeric={**NUMERIC,'pair_rows':list(NUMERIC['pair_rows'])},qp_policy=asdict(DEFAULT_QP_POLICY),
        seed=20260916,torch=pre['torch'],transformers=pre['transformers'],numpy=pre['numpy'],scipy=pre['scipy'],
        model_revision='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2',
        corpus_revision=json.loads((ROOT/'reference-inputs-v2/selection-policy.json').read_text())['corpus_revision'],
        records_digest=pre['records_digest'],sample_order=pre['sample_order'],reference_inputs=member(ROOT/'reference-inputs-v2/inputs.json'),
        fixed8=ref['fixed8'],nested256=ref['nested256'],storage=storage,immutable_asset_stats=identity_stats,
        external_members=assets,prior_asset_locator=member(OLD),full_read=member(ROOT/'receipts/full-read-m0.json'),
        cleanup=member(ROOT/'en-cleanup/removal-receipt.json'),CPU_checks=member(ROOT/'CPU-preflight-v1/receipt.json'),
        execution=execution,output=str(parent/'output'),resources=dict(GPU=1,CPU=8,mem_MiB=60416,wall_hours=24,
            node='server4',export='NONE',requeue=0,hour_hardcap=None),
        cost_plan=dict(status='PRE_EXECUTION_ESTIMATE_NOT_MEASURED',W0_generation_max_full_prefix_forwards=640*16,
            fresh_native_targets=100,maximum_pair_scalar_backward=1536,maximum_local_QP=2,
            maximum_candidate_bank_scans=2,history_appends=2,setup_hours_estimate=[.5,2],
            B1_editing_hours_estimate=[.1,3],observer_technical_hours_estimate=[.5,3],
            GPU_peak_estimate_GiB=[45,75],host_peak_estimate_GiB=[40,58],
            historical_EN_measurement_not_new_actual=True),
        monitoring='CONTINUE_TO_B1_REPORT; NO_B2_WITHOUT_NEW_USER_AUTHORITY')
    require_scope(lock);lock['lock_identity']=digest(lock)
    create_json(parent/'execution.lock.json',lock);print(json.dumps(member(parent/'execution.lock.json'),indent=2))

def submit(path):
    lock=json.loads(path.read_text());require_scope(lock);parent=path.parent
    if (parent/'submission.json').exists() or (parent/'held-inspection.json').exists():raise ValueError('NO_DUPLICATE_JOB')
    queue=call(['squeue','-h','-u','janghj','-w','server4','-o','%i|%j|%T|%b|%R'])
    # Empty queue is the simple fully verified admission case; never cancel unrelated work.
    if queue:raise ValueError('EXACT_OTHER_CAPACITY_ACCOUNTING_REQUIRED:'+queue)
    free=shutil.disk_usage(ROOT).free
    if free<lock['storage']['required_free_bytes']:raise ValueError('B1_STORAGE_RESERVE')
    for item in lock['external_members']+lock['execution']['members']:
        if member(item['path'])!=item:raise ValueError('SOURCE_ASSET_CHANGED:'+item['path'])
    create_bytes(parent/'write-probe.bin',b'BPCW512 atomic/create-once admission probe\n')
    audit=dict(time=now(),project_queue=queue,node=call(['scontrol','show','node','server4']),
        disk_free=free,inodes_free=os.statvfs(ROOT).f_favail,meminfo=Path('/proc/meminfo').read_text(),
        new_task_concurrency=1,total_project_admitted_capacity=1,project_cap=2,cleanup=lock['cleanup'],
        actual_model_checks='PENDING_B1',sequential_authorized=False)
    create_json(parent/'resource-admission.json',audit);(parent/'logs').mkdir()
    script=str(Path(lock['execution']['source_root'])/PACKAGE/'run.sbatch')
    args=['sbatch','--parsable','--hold','--job-name=odeedit_bpcw512_B1_s4',
        f'--output={parent}/logs/%j.out',f'--error={parent}/logs/%j.err',script,lock['execution']['source_root'],str(path)]
    job=call(args).split(';')[0];text=call(['scontrol','show','job',job])
    create_json(parent/'held-inspection.json',dict(job=job,args=args,text=text,time=now(),lock=member(path)))
    for needle in ['UserId=janghj','JobState=PENDING','JobHeldUser','NumCPUs=8','Requeue=0','server4',
                   'Command='+script,str(path),'Dependency=(null)','TimeLimit=1-00:00:00','gres/gpu=1']:
        if needle not in text:raise ValueError('HELD_INSPECTION:'+needle)
    if 'mem=59G' not in text and 'mem=60416M' not in text:raise ValueError('MEMORY_INSPECTION')
    if 'ArrayTaskId=' in text:raise ValueError('B1_MUST_NOT_BE_ARRAY')
    call(['scontrol','release',job]);result=dict(job=job,release_time=now(),lock=member(path),args=args,
        batch_count=1,arms=['N4','BPCW512'],GPU=1,sequential_authorized=False,B2_submitted=False,actual_initial='NOT_OBSERVED')
    create_json(parent/'submission.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','freeze','submit'])
    p.add_argument('--repo',type=Path);p.add_argument('--attempt',default='attempt-v1');p.add_argument('--lock',type=Path)
    a=p.parse_args()
    if a.action=='preflight':preflight(a.repo)
    elif a.action=='freeze':freeze(a.repo,a.attempt)
    else:submit(a.lock)
