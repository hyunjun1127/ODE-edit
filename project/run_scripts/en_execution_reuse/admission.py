"""Explicit single-job admission. No autonomous submission or B2 path."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import re
import shlex
from .config import Scope, runtime_policy
from .model import require_lock, verify_large_asset_stats
from .preparation import ROOT, DEPS, member, sha, create_json, storage_plan, resource_snapshot
from project.run_scripts.single_layer_edit_preserving_correction.common import digest

PKG = 'project/run_scripts/en_execution_reuse'
PRIOR = Path('/data/janghj/ODE-edit/local/bpcw512/20260918-v2/B1/attempt-r1/execution.lock.json')


def call(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def inspect_held(text, command, dependency_job=None):
    match=re.search(r'^\s*Command=(.*?)\s*$',text,re.M)
    submitted=re.search(r'^\s*SubmitLine=(.*?)\s*$',text,re.M)
    if (match is None or shlex.split(match.group(1))!=command[:1] or submitted is None or
        shlex.split(submitted.group(1))[-len(command):]!=command):
        raise ValueError('HELD_EXACT_COMMAND_SOURCE_LOCK_STAGE')
    fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)',text))
    exact=dict(JobState='PENDING',Reason='JobHeldUser',NumCPUs='8',Requeue='0',
               ReqNodeList='server4',TimeLimit='1-00:00:00',
               TresPerNode='gres/gpu:rtx_pro_6000:1')
    if any(fields.get(k)!=v for k,v in exact.items()):
        raise ValueError('HELD_EXACT_STATE_RESOURCE_NODE')
    dependency = fields.get('Dependency')
    if dependency_job is None:
        if dependency != '(null)':raise ValueError('HELD_UNEXPECTED_DEPENDENCY')
    elif dependency not in (f'afterok:{dependency_job}',f'afterok:{dependency_job}(unfulfilled)',
                             f'afterok:{dependency_job}(fulfilled)'):
        raise ValueError('HELD_EXACT_PREPARATION_AFTEROK')
    if not re.fullmatch(r'janghj\([0-9]+\)',fields.get('UserId','')):
        raise ValueError('HELD_OWNER')
    tres=dict(item.split('=',1) for item in fields.get('ReqTRES','').split(',') if '=' in item)
    if (tres.get('mem') not in ('59G','60416M') or tres.get('gres/gpu')!='1' or
        tres.get('cpu')!='8' or 'ArrayTaskId' in fields):
        raise ValueError('RESOURCE_OR_ARRAY_MISMATCH')


def freeze(repo, stage, attempt, cpu_receipt, teacher_ready=None, preparation_lock=None):
    repo = Path(repo).resolve()
    if stage not in ('GENERATED_REFERENCE_PREPARATION', 'MATCHED_B1'):
        raise ValueError('B1_STAGE_ALLOWLIST')
    if call(['git', 'status', '--porcelain'], repo):
        raise ValueError('COMMITTED_CLEAN_SOURCE_REQUIRED')
    cpu = json.loads(Path(cpu_receipt).read_text())
    if cpu['status'] != 'CPU_TESTS_PASS_ACTUAL_LLAMA_NOT_RUN':
        raise ValueError('CPU_RECEIPT_REQUIRED')
    checked={Path(i['path']).name:i['sha256'] for i in cpu['sources']}
    for p in sorted((repo/PKG).glob('*.py')):
        if checked.get(p.name)!=sha(p):
            raise ValueError('CPU_RECEIPT_SOURCE_DRIFT:'+p.name)
    # Current input bound is computed with the exact pinned tokenizer/context,
    # before any model load. Add explicit raw/log/serializer operational margin.
    ready = None
    pending = None
    if preparation_lock is not None:
        if stage != 'MATCHED_B1' or teacher_ready is not None:
            raise ValueError('DEFERRED_READY_MATCHED_B1_ONLY')
        from .dependent_start import pending_spec
        pending = pending_spec(preparation_lock)
        plan = storage_plan(current_valid_token_upper=10416)
        plan['incremental_phase'] = 'DEPENDENT_B1_CONSERVATIVE_FULL_PREP_PAYLOAD_PLUS_B1'
        plan['conservative_duplicate_payload_reserve_while_preparation_runs'] = plan['payload_teacher_key_bytes']
        plan['runtime_incremental_bytes_without_margin'] = plan['estimated_bytes_without_unrecorded_overhead']-plan['payload_teacher_key_bytes']
    elif stage == 'MATCHED_B1':
        if teacher_ready is None:
            raise ValueError('SEALED_GENERATED256_TEACHER_REQUIRED')
        ready = json.loads(Path(teacher_ready).read_text())
        if ready['status'] != 'GENERATED_REFERENCE_READY_NOT_CORRECTION_VALIDATION':
            raise ValueError('REFERENCE_READY_STATUS')
        manifest = ready['manifest']
        if sha(manifest['path']) != manifest['sha256']:
            raise ValueError('PREP_MANIFEST_BYTES')
        stored = json.loads(Path(manifest['path']).read_text())
        lengths = [d['logp']['shape'][0] for d in stored['documents']]
        plan = storage_plan(lengths, current_valid_token_upper=10416)
        retained = plan['payload_teacher_key_bytes']
        plan['estimated_bytes_without_unrecorded_overhead'] -= retained
        plan['already_retained_payload_bytes_excluded_from_new_free_requirement'] = retained
        plan['incremental_phase'] = 'MATCHED_B1_NO_TEACHER_OR_CACHE_COPY'
    else:
        plan = storage_plan(current_valid_token_upper=10416)
        plan['incremental_phase'] = 'PREP_INCLUDES_NEW_TEACHER_CACHE_AND_FUTURE_B1_RESERVE'
    plan['operational_compact_raw_serialization_margin_bytes'] = 8*2**30
    plan['required_free_bytes'] = plan['estimated_bytes_without_unrecorded_overhead'] + 8*2**30
    snap = resource_snapshot(ROOT, plan)
    if snap['available_bytes'] < plan['required_free_bytes']:
        raise ValueError('STORAGE_RESERVE_INSUFFICIENT_NO_OLD_WAIVER:'+json.dumps(snap))
    parent = ROOT/('PREP' if stage == 'GENERATED_REFERENCE_PREPARATION' else 'B1')/attempt
    parent.mkdir(parents=True, exist_ok=False)
    source = parent/'source'
    source.mkdir()
    head = call(['git','rev-parse','HEAD'],repo)
    tree = call(['git','rev-parse','HEAD^{tree}'],repo)
    archive = parent/'source.tar'
    with archive.open('xb') as f:
        subprocess.run(['git','archive','--format=tar',head,'project/run_scripts','scripts/fixed_counterfact.py'],
                       cwd=repo,stdout=f,check=True)
    with tarfile.open(archive) as tf:
        tf.extractall(source,filter='data')
    execution = dict(commit=head,tree=tree,archive=member(archive),source_root=str(source),
                     members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()])
    prior = json.loads(PRIOR.read_text())
    keys = ('snapshot','config4','blue_root','dataset_root','projector','cold_capsule','editor_sha256',
            'historical_evaluator_root','helper_scripts_root','P_star_basis','reference_inputs',
            'model_revision','records_digest','sample_order','torch','transformers','numpy','scipy')
    lock = {key:prior[key] for key in keys}
    from scripts.fixed_counterfact import load_prefix
    records = load_prefix(lock['dataset_root'],100)
    if digest(records) != lock['records_digest'] or [r['case_id'] for r in records] != lock['sample_order']:
        raise ValueError('PREMODEL_DATA_IDENTITY')
    assets = list({item['path']: item for item in prior['external_members']}.values())
    for item in assets:
        if Path(item['path']).stat().st_size != item['bytes'] or sha(item['path']) != item['sha256']:
            raise ValueError('EXTERNAL_ASSET_SHA_CHANGED:'+item['path'])
    # Task-local pinned overlay, not the default mutable transformers4.57.1.
    # Include existing bytecode as well as source/native extension bytes.
    dependency_members=[member(p) for p in sorted(DEPS.rglob('*')) if p.is_file()]
    assets=list({i['path']:i for i in assets+dependency_members}.values())
    large = []
    for item in prior['prior_immutable_members']:
        p = Path(item['path'])
        if str(p).startswith(lock['snapshot']+'/') or str(p) == lock['projector']:
            st = p.stat()
            if st.st_size != item['bytes'] or ('stat' in item and list((st.st_dev,st.st_ino,st.st_mtime_ns)) != item['stat']):
                raise ValueError('PRIOR_LARGE_ASSET_STAT_CHANGED:'+str(p))
            large.append(dict(prior=item,current_stat=[st.st_dev,st.st_ino,st.st_mtime_ns],
                              verification='PRIOR_FULL_SHA_PLUS_CURRENT_STAT_NOT_NEW_FULL_REHASH'))
    model = [x['prior'] for x in large if x['prior']['path'].endswith('.safetensors')]
    if not model:
        raise ValueError('PRETRAINED_MODEL_PRIOR_FULL_SHA_MANIFEST_MISSING')
    tokenizer = [member(Path(lock['snapshot'])/p) for p in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json')]
    for key in ('cold_capsule','P_star_basis','reference_inputs'):
        if sha(lock[key]['path']) != lock[key]['sha256']:
            raise ValueError('IMMUTABLE_INPUT_'+key)
    native_meta_path = PRIOR.parent/'output/native/native-binding.json'
    native = json.loads(native_meta_path.read_text())
    if (native['case_ids'] != lock['sample_order'] or native['receipt']['history_append'] != 0 or
        native['entry']['independent_cold'] is not True or native['receipt']['source']['source_sha256'] != lock['editor_sha256']):
        raise ValueError('NATIVE_REUSE_SOURCE_COLD_ORDER_HISTORY')
    native_file = native['source']
    if Path(native_file['path']).stat().st_size != native_file['bytes'] or sha(native_file['path']) != native_file['sha256']:
        raise ValueError('RETAINED_NATIVE_CAPSULE_SHA')
    lock.update(scope=asdict(Scope()),runtime_policy=runtime_policy(),stage=stage,seed=20260916,
        execution=execution,external_members=assets,prior_large_asset_binding=large,
        prior_asset_locator=member(PRIOR),CPU=member(cpu_receipt),
        model_config_sha256=sha(Path(lock['snapshot'])/'config.json'),
        model_weights_identity_sha256=digest(model),tokenizer_identity_sha256=digest(tokenizer),
        transformers_import=str((DEPS/'transformers/__init__.py').resolve()),
        reused_native_b1=native_file['path'],reused_native_binding=dict(native=native_file,b1_endpoint_verified=native['endpoint']),
        native_reuse_lineage=dict(binding=member(native_meta_path),prior_seconds=native['receipt']['seconds'],
            new_native_fit_calls=0,new_native_targets=0,previous_state='W0_ZERO_M4_CURRENT_SAME_INPUT',
            old_output_readonly=True,actual_new_model_binding='CHECK_ON_LOAD'),
        storage=plan,resource_preflight=snap,output=str(parent/'output'),
        resources=dict(GPU=1,CPU=8,mem_MiB=60416,wall_hours=24,node='server4',export='NONE',requeue=0,GPUhour_hardcap=None),
        estimate=dict(status='NOT_MEASURED_NEW_TASK',generation_full_prefix_forward_upper=640*256,
            generation_processed_tokens_upper=640*sum(range(129,385)),
            preparation_hours_estimate=[2,10],matched_two_schedule_hours_estimate=[1,8],
            host_limit_MiB=60416,preparation_model_validation='PENDING_ACTUAL',
            source='Planning range; old max16 timing not a generated256 measurement or speedup claim'),
        sequential_authorized=False,auto_continue=False,source_checks_not_actual_Llama_PASS=True)
    if stage == 'MATCHED_B1':
        if pending is None:
            lock['generated_ready'] = member(teacher_ready)
        else:
            lock['generated_ready_pending'] = pending
            from .dependent_start import compare_preparation_inputs
            compare_preparation_inputs(lock,json.loads(Path(pending['lock']['path']).read_text()))
        oldroot=PRIOR.parent/'output'
        oldsource=Path(prior['execution']['source_root'])/'project/run_scripts/single_layer_edit_preserving_correction/observer.py'
        lock['prior_observer_reuse']={k:member(p) for k,p in dict(lock=PRIOR,
            runtime=oldroot/'runtime-load.json',nonselected_before=oldroot/'nonselected-before.json',
            nonselected_after=oldroot/'nonselected-after.json',W0=oldroot/'W0-current.json',
            N4=oldroot/'arms/N4/current.json',observer_source=oldsource).items()}
    verify_large_asset_stats(lock)
    lock['lock_identity'] = digest(lock)
    require_lock(lock)
    path = parent/'execution.lock.json'
    create_json(path,lock)
    print(json.dumps(member(path),indent=2))


def submit(path):
    path = Path(path).resolve()
    lock = json.loads(path.read_text())
    require_lock(lock)
    verify_large_asset_stats(lock)
    parent = path.parent
    if (parent/'held-inspection.json').exists() or (parent/'submission.json').exists():
        raise ValueError('DUPLICATE_JOB_BLOCKED')
    # This conservative path accepts only the unambiguous empty project queue.
    # Nonempty capacity must be resolved explicitly, never cancelled here.
    queue = call(['squeue','-h','-u','janghj','-w','server4','-o','%i|%j|%T|%b|%R'])
    pending=lock.get('generated_ready_pending')
    if pending:
        from .dependent_start import verify_pending_members, inspect_preparation_job
        verify_pending_members(pending)
        prep_job=pending['job']
        if any(row.split('|')[0]!=prep_job for row in queue.splitlines() if row):
            raise ValueError('OTHER_PROJECT_ADMISSION_REQUIRES_CAP_ACCOUNTING:'+queue)
        prep_text=call(['scontrol','show','job',prep_job])
        inspect_preparation_job(prep_text,pending)
    elif queue:
        raise ValueError('EXACT_PROJECT_CAPACITY_ACCOUNTING_REQUIRED:'+queue)
    if shutil.disk_usage(ROOT).free < lock['storage']['required_free_bytes']:
        raise ValueError('STORAGE_NO_PRIOR_WAIVER')
    for item in lock['execution']['members']+lock['external_members']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('PRE_SUBMIT_SOURCE_ASSET_DRIFT')
    create_json(parent/'resource-admission.json',dict(queue=queue,node=call(['scontrol','show','node','server4']),
        free_bytes=shutil.disk_usage(ROOT).free,free_inodes=os.statvfs(ROOT).f_favail,
        task_admitted_capacity=1,project_admitted_capacity=1,project_cap=2,other_job_mutations=0,
        serialization=(dict(afterok=prep_job,preparation_inspection=prep_text,
            explanation='PREP and B1 cannot overlap; one task lane') if pending else None)))
    (parent/'logs').mkdir()
    script = str(Path(lock['execution']['source_root'])/PKG/'run.sbatch')
    tag = 'prep' if lock['stage']=='GENERATED_REFERENCE_PREPARATION' else 'B1'
    args = ['sbatch','--parsable','--hold',*([f'--dependency=afterok:{prep_job}'] if pending else []),
            '--job-name=odeedit_en_reuse_g256_'+tag+'_s4',
            f'--output={parent}/logs/%j.out',f'--error={parent}/logs/%j.err',
            script,lock['execution']['source_root'],str(path),lock['stage']]
    job = call(args).split(';')[0]
    text = call(['scontrol','show','job',job])
    create_json(parent/'held-inspection.json',dict(job=job,args=args,text=text,lock=member(path)))
    inspect_held(text,[script,lock['execution']['source_root'],str(path),lock['stage']],
                 pending['job'] if pending else None)
    call(['scontrol','release',job])
    receipt=dict(job=job,stage=lock['stage'],lock=member(path),args=args,
                 inspection='PASS_HELD_OWNER_SOURCE_ARGS_NODE_RESOURCES_DEPENDENCY',released=True,
                 actual_model_validation='NOT_OBSERVED_AT_RELEASE',max_batches=1,
                 sequential_authorized=False,automatic_resume=False)
    if pending:
        receipt.update(dependency=f'afterok:{prep_job}',teacher_ready='PENDING_RUNTIME_EXACT_BINDING',
            monitoring_active=False,submission_authority='USER_B1_PENDING_REQUEST_AFTER_MONITORING_PAUSE')
    create_json(parent/'submission.json',receipt)
    print(json.dumps(receipt,indent=2))


def release_inspected(path):
    """Narrow recovery of already-held OWN job after control-only inspection bug.

    No resubmission, new lock, changed science source, or job configuration.
    Slurm Command is script-only; SubmitLine holds the submitted positional args.
    """
    path=Path(path).resolve();parent=path.parent
    if (parent/'submission.json').exists():raise ValueError('ALREADY_RELEASED')
    lock=json.loads(path.read_text());require_lock(lock);verify_large_asset_stats(lock)
    held=json.loads((parent/'held-inspection.json').read_text())
    if sha(path)!=held['lock']['sha256']:raise ValueError('HELD_LOCK_CHANGED')
    job=held['job']
    if not job.isdecimal():raise ValueError('EXACT_JOB_ID')
    text=call(['scontrol','show','job',job])
    command=held['args'][-4:]
    if command!=[str(Path(lock['execution']['source_root'])/PKG/'run.sbatch'),lock['execution']['source_root'],str(path),lock['stage']]:
        raise ValueError('ORIGINAL_SUBMISSION_ARG_IDENTITY')
    inspect_held(text,command)
    if f'JobId={job} ' not in text:raise ValueError('EXACT_JOB_MAPPING')
    queue=call(['squeue','-h','-u','janghj','-w','server4','-o','%i|%j|%T|%b|%R'])
    if any(row.split('|')[0]!=job for row in queue.splitlines() if row):
        raise ValueError('OTHER_PROJECT_ADMISSION_REQUIRES_CAP_ACCOUNTING')
    if shutil.disk_usage(ROOT).free<lock['storage']['required_free_bytes']:
        raise ValueError('STORAGE_NO_WAIVER')
    for item in lock['execution']['members']+lock['external_members']:
        if sha(item['path'])!=item['sha256']:raise ValueError('PRE_RELEASE_SOURCE_ASSET_CHANGED')
    create_json(parent/'held-inspection-control-r1.json',dict(job=job,text=text,queue=queue,
        control_source_sha=sha(Path(__file__)),execution_source=lock['execution']['commit'],
        repair='Command is script-only; exact SubmitLine positional source/lock/stage checked',
        model_execution_before_repair=0,lock_unchanged=True,science_code_change=0))
    call(['scontrol','release',job])
    receipt=dict(job=job,stage=lock['stage'],lock=member(path),args=held['args'],released=True,
        inspection='PASS_EXACT_COMMAND_AND_SUBMITLINE_OWNER_NODE_RESOURCES',max_batches=1,
        actual_model_validation='NOT_OBSERVED_AT_RELEASE',sequential_authorized=False,automatic_resume=False)
    create_json(parent/'submission.json',receipt);print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['freeze','submit','release-inspected'])
    p.add_argument('--repo',type=Path);p.add_argument('--stage',choices=['GENERATED_REFERENCE_PREPARATION','MATCHED_B1'])
    p.add_argument('--attempt',default='attempt-v1');p.add_argument('--cpu-receipt',type=Path)
    p.add_argument('--teacher-ready',type=Path);p.add_argument('--lock',type=Path)
    p.add_argument('--preparation-lock',type=Path)
    a=p.parse_args()
    if a.action=='freeze':freeze(a.repo,a.stage,a.attempt,a.cpu_receipt,a.teacher_ready,a.preparation_lock)
    elif a.action=='submit':submit(a.lock)
    else:release_inspected(a.lock)
